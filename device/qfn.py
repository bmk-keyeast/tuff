"""Qwen3.8-Flash-Next decode on the card: ref.py's recurrence written in device/ops.py operations, with tuff's split of
the work between the p150a and the host.

    card   embedding row in, hyper-connections, Gated DeltaNet (state on the card), full attention with the QSA indexer
           (KV cache, indexer keys and pooled block keys on the card), router, shared expert, the routed experts that
           sit in the card's expert slots, the final mixer and the LM head
    host   token embedding and N-gram rows (host RAM / NVMe), the routed experts not in a slot (CPU kernels), the
           adaptive cache's bookkeeping: after each token up to `budget` missed experts are copied into slots (LRU)

Per layer only vectors cross the bus: the MoE input goes to the host when some routed expert misses, and the host's
partial sum comes back. ref.py is the specification; device/test_qfn.py checks Card against it.

ponytail: one token per step (no MTP verify batch yet), everything through ops.py's generic operations; fused ttnn ops,
traces and the C host side (tcache, tstore, moe_forward) come in once the card runs this.
"""
import math
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import plan  # noqa: E402

# dense weight name inside a layer -> plan.DENSE group (its format); first match wins (as trace.PATTERN)
PATTERN = {"lin_qkv": "linear_attn.in_proj_qkv", "lin_z": "linear_attn.in_proj_z", "lin_out": "linear_attn.out_proj",
           "lin_misc": "linear_attn.", "full": "self_attn.", "shared": "shared_expert", "router": "mlp.gate",
           "hc": "hyper_connection", "ple": "ple.", "lm_head": "lm_head"}


class Host:
    """The host side in numpy (the engine's is C: tstore, moe_forward, tcache). w: checkpoint names -> arrays."""

    def __init__(self, w, cfg):
        import ref
        self.w, self.c = w, cfg
        self.r = ref.Ref(w, cfg)                                # for the N-gram hashing

    def embed(self, tok):
        return np.asarray(self.w["embed_tokens.weight"][tok], np.float64)

    def ngram_rows(self, L, tokens):
        self.r.tokens = list(tokens)
        return np.asarray(self.w[f"layers.{L}.ple.ple_embedding.ngram_embedding.weight"][self.r.ngram_ids(L)]).ravel()

    def experts(self, L, x, es, wts):
        p, y = f"layers.{L}.mlp.experts.", 0
        for e, we in zip(es, wts):
            g, u = np.split(np.asarray(self.w[p + "gate_up_proj"][e], np.float64) @ x, 2)
            y = y + we * (np.asarray(self.w[p + "down_proj"][e], np.float64) @ (g / (1 + np.exp(-g)) * u))
        return y


class Card:
    def __init__(self, ops, w, cfg, host, dense_fmt="bfp8", expert_fmt="bfp4", slots=10_000, budget=25):
        self.o, self.w, self.c, self.host = ops, w, cfg, host
        self.dense_fmt, self.expert_fmt, self.n_slots, self.budget = dense_fmt, expert_fmt, slots, budget
        c = cfg
        self.d, self.hc, self.eps = c.hidden_size, c.hc_count, c.rms_norm_eps
        rp = c.rope_parameters
        self.rd = int(c.head_dim * rp.get("partial_rotary_factor", 1.0))
        self.inv_freq = 1.0 / (rp["rope_theta"] ** (np.arange(0, self.rd, 2) / self.rd))
        self.gated_act = (c.output_gate_type or c.hidden_act) == "sigmoid"
        self.dev, self.slots, self.missed, self.routes = {}, OrderedDict(), [], {}
        self.t, self.pos0, self.tokens = 0, 0, []
        self.hits = self.lookups = 0
        self.ple = {L - 1 for L in c.ple_layer_ids}
        self.max_t = 4096
        self.state = {L: self.fresh_state(L) for L in range(c.num_hidden_layers)}

    # ---- weights, uploaded on first use
    def fmt(self, name):
        g = next((g for g, s in PATTERN.items() if s in name), "lin_misc")
        return self.dense_fmt if isinstance(self.dense_fmt, str) else plan.fmt_of(self.dense_fmt, g)

    def lin(self, name):
        if name not in self.dev:
            self.dev[name] = self.o.put_weight(self.w[name + ".weight"], self.fmt(name))
        return self.dev[name]

    def norm(self, name, plus_one=True, rows=1):
        """Qwen4ExpTextRMSNorm multiplies by (1 + w); the DeltaNet's gated norm by w. rows > 1: a per-stream norm's
        weight as [streams, d], to scale x laid out as [streams, d]."""
        if (name, rows) not in self.dev:
            w = np.asarray(self.w[name], np.float64)
            self.dev[name, rows] = self.o.put(((1 + w) if plus_one else w).reshape(rows, -1))
        return self.dev[name, rows]

    def const(self, key, make):
        if key not in self.dev:
            self.dev[key] = self.o.put(make())
        return self.dev[key]

    def fresh_state(self, L):
        c, o = self.c, self.o
        s = {}
        if c.layer_types[L] == "linear_attention":
            C = 2 * c.linear_num_key_heads * c.linear_key_head_dim + c.linear_num_value_heads * c.linear_value_head_dim
            s.update(conv=o.zeros((c.linear_conv_kernel_dim - 1, C)),
                     S=o.zeros((c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim)))
        else:
            kvd = c.num_key_value_heads * c.head_dim
            s.update(K=o.zeros((self.max_t, kvd)), V=o.zeros((self.max_t, kvd)),
                     idx=o.zeros((self.max_t, c.indexer_head_dim)),
                     blk=o.zeros((self.max_t // c.indexer_compress_ratio, c.indexer_head_dim)))
        if L in self.ple:
            s["ple"] = o.zeros(((c.ple_conv_kernel_size - 1) * c.ngram_size, self.hc * self.d))
        return s

    # ---- building blocks
    def rms(self, x, w):
        o = self.o
        return o.mul(o.mul(x, o.rsqrt(o.add(o.mean(o.mul(x, x), 1), self.eps))), w)

    def rope(self, x, pos):
        o, rd, half = self.o, self.rd, self.rd // 2
        f = pos * self.inv_freq
        cos = self.const(("cos", pos), lambda: np.concatenate([np.cos(f)] * 2)[None])
        sin = self.const(("sin", pos), lambda: np.concatenate([np.sin(f)] * 2)[None])
        r = o.slice(x, 1, 0, rd)
        rot = o.concat([o.scale(o.slice(r, 1, half, rd), -1), o.slice(r, 1, 0, half)], 1)
        y = o.add(o.mul(r, cos), o.mul(rot, sin))
        return o.concat([y, o.slice(x, 1, rd, x.shape[1])], 1) if x.shape[1] > rd else y

    def hc_read(self, p, h4, combine=True):
        o, hc, d = self.o, self.hc, self.d
        hn = self.rms(o.reshape(h4, (hc, d)), self.norm(p + ".hc_norm.weight", rows=hc))   # [hc, d]
        flat = o.reshape(hn, (1, hc * d))
        mix = o.sigmoid(o.linear(o.silu(o.scale(o.linear(flat, self.lin(p + ".input_mix_weight_down")), 1 / hc)),
                                 self.lin(p + ".input_mix_weight_up")))
        mixed = o.mean(o.mul(o.reshape(mix, (hc, d)), hn), 0)                         # [1, d]
        if not combine:
            return mixed
        return mixed, o.scale(o.sigmoid(o.scale(o.linear(flat, self.lin(p + ".block_inject_weight")), 1 / hc)), 2)

    def hc_write(self, h4, inj, out):
        o = self.o
        return o.add(h4, o.reshape(o.mul(o.reshape(inj, (self.hc, 1)), out), (1, self.hc * self.d)))

    # ---- Gated DeltaNet: conv window and recurrent state stay on the card
    def deltanet(self, L, x):
        o, c, s, p = self.o, self.c, self.state[L], f"layers.{L}.linear_attn"
        K, V, dk, dv = c.linear_num_key_heads, c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim
        qkv = o.linear(x, self.lin(p + ".in_proj_qkv"))                                # [1, C]
        win = o.concat([s["conv"], qkv], 0)                                            # [kernel, C], oldest first
        s["conv"] = o.slice(win, 0, 1, win.shape[0])
        cw = self.const(p + ".conv", lambda: np.asarray(self.w[p + ".conv1d.weight"], np.float64)[:, 0, :].T)
        qkv = o.silu(o.sum(o.mul(win, cw), 0))
        l2 = lambda t: o.mul(t, o.rsqrt(o.add(o.sum(o.mul(t, t), 1), 1e-6)))
        q = l2(o.reshape(o.slice(qkv, 1, 0, K * dk), (K, dk)))
        k = l2(o.reshape(o.slice(qkv, 1, K * dk, 2 * K * dk), (K, dk)))
        v = o.reshape(o.slice(qkv, 1, 2 * K * dk, qkv.shape[1]), (V, dv))
        z = o.reshape(o.linear(x, self.lin(p + ".in_proj_z")), (V, dv))
        beta = o.sigmoid(o.linear(x, self.lin(p + ".in_proj_b")))
        negA = self.const(p + ".negA", lambda: -np.exp(np.asarray(self.w[p + ".A_log"], np.float64))[None])
        dtb = self.const(p + ".dt_bias", lambda: np.asarray(self.w[p + ".dt_bias"], np.float64)[None])
        g = o.mul(negA, o.softplus(o.add(o.linear(x, self.lin(p + ".in_proj_a")), dtb)))
        out, s["S"] = o.gdr(q, k, v, g, beta, s["S"], dk ** -0.5)
        gate = o.sigmoid(z) if self.gated_act else o.silu(z)
        out = o.mul(self.rms(out, self.norm(p + ".norm.weight", plus_one=False)), gate)
        return o.linear(o.reshape(out, (1, V * dv)), self.lin(p + ".out_proj"))

    # ---- full attention with the QSA indexer
    def attention(self, L, x):
        o, c, s, p = self.o, self.c, self.state[L], f"layers.{L}.self_attn"
        t, pos = self.t, self.t + self.pos0
        H, KV, hd = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        nh, ihd, ratio = c.indexer_n_heads, c.indexer_head_dim, c.indexer_compress_ratio
        qk = o.linear(x, self.lin(p + ".indexer.index_qk_proj"))
        s["idx"] = o.write_row(s["idx"], t, o.slice(qk, 1, nh * ihd, nh * ihd + ihd))   # raw key: no norm, no rope
        if (t + 1) % ratio == 0:                       # a block is complete: pool, norm and rope its key once
            b = t // ratio
            blk = o.mean(o.slice(s["idx"], 0, b * ratio, (b + 1) * ratio), 0)
            blk = self.rope(self.rms(blk, self.norm(p + ".indexer.k_layernorm.weight")), b * ratio + self.pos0)
            s["blk"] = o.write_row(s["blk"], b, blk)
        iq = self.rope(self.rms(o.reshape(o.slice(qk, 1, 0, nh * ihd), (nh, ihd)),
                                self.norm(p + ".indexer.q_layernorm.weight")), pos)
        nb = (t + 1) // ratio
        keep = list(range(nb * ratio, t + 1))
        if nb:
            score = o.scale(o.sum(o.relu(o.matmul(iq, o.transpose(o.slice(s["blk"], 0, 0, nb)))), 0), 1 / math.sqrt(ihd))
            _, top = o.topk(score, min(c.indexer_budget // ratio, nb))
            keep = sorted(int(i) * ratio + r for i in top for r in range(ratio)) + keep
        qg = o.reshape(o.linear(x, self.lin(p + ".q_proj")), (H, 2 * hd))
        q = self.rope(self.rms(o.slice(qg, 1, 0, hd), self.norm(p + ".q_norm.weight")), pos)
        gate = o.reshape(o.slice(qg, 1, hd, 2 * hd), (1, H * hd))
        k = self.rope(self.rms(o.reshape(o.linear(x, self.lin(p + ".k_proj")), (KV, hd)), self.norm(p + ".k_norm.weight")), pos)
        s["K"] = o.write_row(s["K"], t, o.reshape(k, (1, KV * hd)))
        s["V"] = o.write_row(s["V"], t, o.linear(x, self.lin(p + ".v_proj")))
        Ks, Vs = o.gather_rows(s["K"], keep), o.gather_rows(s["V"], keep)            # [n, KV*hd]
        G, outs = H // KV, []
        for kv in range(KV):
            a = o.softmax(o.scale(o.matmul(o.slice(q, 0, kv * G, (kv + 1) * G), o.transpose(o.slice(Ks, 1, kv * hd, (kv + 1) * hd))),
                                  1 / math.sqrt(hd)))
            outs.append(o.matmul(a, o.slice(Vs, 1, kv * hd, (kv + 1) * hd)))
        out = o.mul(o.reshape(o.concat(outs, 0), (1, H * hd)), o.sigmoid(gate))
        return o.linear(out, self.lin(p + ".o_proj"))

    # ---- MoE: router on the card; slot experts on the card, the rest on the host
    def slot(self, L, e):
        key = (L, e)
        if key not in self.slots:
            p = f"layers.{L}.mlp.experts."
            gu, dn = np.asarray(self.w[p + "gate_up_proj"][e]), np.asarray(self.w[p + "down_proj"][e])
            self.slots[key] = (self.o.put_weight(gu, self.expert_fmt), self.o.put_weight(dn, self.expert_fmt))
        return self.slots[key]

    def moe(self, L, x):
        o, c, p = self.o, self.c, f"layers.{L}.mlp"
        I = c.moe_intermediate_size
        vals, top = o.topk(o.softmax(o.linear(x, self.lin(p + ".gate"))), c.num_experts_per_tok)
        wts = self.o.get(vals)[0]
        wts = wts / wts.sum() if c.norm_topk_prob else wts
        self.routes[L, self.t] = top
        y, cold = None, []
        for e, we in zip(top, wts):
            self.lookups += 1
            if (L, int(e)) in self.slots:
                self.hits += 1
                self.slots.move_to_end((L, int(e)))
                gu, dn = self.slots[L, int(e)]
                h = o.linear(x, gu)
                ye = o.scale(o.linear(o.mul(o.silu(o.slice(h, 1, 0, I)), o.slice(h, 1, I, 2 * I)), dn), we)
                y = ye if y is None else o.add(y, ye)
            else:
                cold.append((int(e), we))
        sh = o.linear(o.mul(o.silu(o.linear(x, self.lin(p + ".shared_expert.gate_proj"))),
                            o.linear(x, self.lin(p + ".shared_expert.up_proj"))), self.lin(p + ".shared_expert.down_proj"))
        sh = o.mul(sh, o.sigmoid(o.linear(x, self.lin(p + ".shared_expert_gate"))))
        y = sh if y is None else o.add(y, sh)
        if cold:                                        # the only vectors that cross the bus
            yc = self.host.experts(L, o.get(x)[0], [e for e, _ in cold], [w for _, w in cold])
            y = o.add(y, o.put(yc[None]))
            self.missed += [(L, e) for e, _ in cold]
        return y

    def promote(self):
        """After a token: copy up to `budget` missed experts into slots, evicting the least recently used."""
        for key in sorted(set(self.missed))[:self.budget]:
            while len(self.slots) >= self.n_slots:
                self.slots.popitem(last=False)
            self.slot(*key)
        self.missed = []

    # ---- PLE: the host hashes the N-gram ids and sends the rows; the card gates and convolves
    def ple_layer(self, L, h4):
        o, c, p, hc, d, s = self.o, self.c, f"layers.{L}.ple", self.hc, self.d, self.state[L]
        emb = o.put(self.host.ngram_rows(L, self.tokens)[None])
        key = self.rms(o.reshape(o.linear(emb, self.lin(p + ".key_proj")), (hc, d)), self.norm(p + ".norm_key.weight", rows=hc))
        value = o.linear(emb, self.lin(p + ".value_proj"))
        query = self.rms(o.reshape(h4, (hc, d)), self.norm(p + ".norm_query.weight", rows=hc))
        gate = o.scale(o.sum(o.mul(key, query), 1), 1 / math.sqrt(d))                 # [hc, 1]
        gate = o.mul(o.sqrt(o.clamp_min(o.abs(gate), 1e-6)), o.sign(gate))
        gv = o.mul(o.sigmoid(gate), value)                                             # [hc, d]
        gvn = o.reshape(self.rms(gv, self.norm(p + ".norm_conv.weight", rows=hc)), (1, hc * d))
        win = o.concat([s["ple"], gvn], 0)                                             # last 9 + current
        s["ple"] = o.slice(win, 0, 1, win.shape[0])
        cw = self.const(p + ".conv", lambda: np.asarray(self.w[p + ".conv1d.weight"], np.float64)[:, 0, :].T)
        taps = o.concat([o.slice(win, 0, i, i + 1) for i in range(0, win.shape[0], c.ngram_size)], 0)
        return o.add(o.reshape(gv, (1, hc * d)), o.silu(o.sum(o.mul(taps, cw), 0)))

    # ---- one token through the model
    def step(self, token):
        o, c = self.o, self.c
        self.tokens.append(int(token))
        h4 = o.put(np.tile(self.host.embed(int(token)), self.hc)[None])
        for L in range(c.num_hidden_layers):
            if L in self.ple:
                h4 = o.add(h4, self.ple_layer(L, h4))
            x, inj = self.hc_read(f"layers.{L}.attn_hyper_connection", h4)
            h4 = self.hc_write(h4, inj, self.deltanet(L, x) if c.layer_types[L] == "linear_attention" else self.attention(L, x))
            x, inj = self.hc_read(f"layers.{L}.mlp_hyper_connection", h4)
            h4 = self.hc_write(h4, inj, self.moe(L, x))
        self.t += 1
        self.promote()
        self.h4 = h4
        return self.hc_read("hyper_connection_mixer", h4, combine=False)
