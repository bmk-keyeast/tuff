"""Qwen3.8-Flash-Next decode on the card: ref.py's recurrence written in device/ops.py operations, with tuff's split of
the work between the p150a and the host, and MTP speculative decoding.

    card   embedding rows in, hyper-connections, Gated DeltaNet (state on the card), full attention with the QSA indexer
           (KV cache, indexer keys and pooled block keys on the card), router, shared expert, the routed experts that
           sit in the card's expert slots, the final mixer, the LM head, and QFN's MTP drafter layer
    host   token embedding and N-gram rows (host RAM / NVMe), the routed experts not in a slot (CPU kernels), the
           adaptive cache's bookkeeping: after each step up to `budget` missed experts are copied into slots (LRU)

Card.verify(tokens) runs m tokens through every layer at once: every matmul takes the m rows together (one read of the
weights), only the recurrent parts (DeltaNet's conv and delta rule, the indexer's causal selection, attention) walk
the tokens in order. Card.commit(j) then keeps the first j tokens: DeltaNet's per-token states, the conv windows and
the cache positions roll back to exactly that point. generate() does greedy speculative decoding with it, and must
emit exactly the tokens plain greedy decoding emits (device/test_qfn.py checks this).

ponytail: everything through ops.py's generic operations; fused ttnn ops, traces and the C host side (tcache, tstore,
moe_forward_n) come in once the card runs this.
"""
import math
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import plan  # noqa: E402

# dense weight name -> plan.DENSE group (its format); first match wins (as trace.PATTERN)
PATTERN = {"lin_qkv": "linear_attn.in_proj_qkv", "lin_z": "linear_attn.in_proj_z", "lin_out": "linear_attn.out_proj",
           "lin_misc": "linear_attn.", "full": "self_attn.", "shared": "shared_expert", "router": "mlp.gate",
           "hc": "hyper_connection", "ple": "ple.", "lm_head": "lm_head"}


class Host:
    """The host side in numpy (the engine's is C: tstore, moe_forward_n, tcache). w: checkpoint names -> arrays."""

    def __init__(self, w, cfg):
        import ref
        self.w, self.c = w, cfg
        self.r = ref.Ref(w, cfg)                                # for the N-gram hashing

    def embed(self, tok):
        return np.asarray(self.w["embed_tokens.weight"][tok], np.float64)

    def ngram_rows(self, L, tokens):
        self.r.tokens = list(tokens)
        return np.asarray(self.w[f"layers.{L}.ple.ple_embedding.ngram_embedding.weight"][self.r.ngram_ids(L)]).ravel()

    def experts(self, L, X, es, G):
        """A layer's cold experts for m tokens at once: X [m, d], es: k experts, G [k, m] gate weights (0: the token
        does not use that expert). Returns [m, d]."""
        p, Y = f"layers.{L}.mlp.experts.", np.zeros_like(X)
        for e, gw in zip(es, G):
            gu, dn = np.asarray(self.w[p + "gate_up_proj"][e], np.float64), np.asarray(self.w[p + "down_proj"][e], np.float64)
            for i in np.flatnonzero(gw):
                g, u = np.split(gu @ X[i], 2)
                Y[i] += gw[i] * (dn @ (g / (1 + np.exp(-g)) * u))
        return Y


class PyCache:
    """The adaptive cache's policy in Python (the engine's is C: tcache, same policy): which experts sit in the card's
    slots; after each step up to `budget` of the step's misses (lowest (layer, expert) first) are copied in, evicting
    the least recently used."""

    def __init__(self, n_slots, budget):
        self.n, self.budget, self.lru, self.missed = n_slots, budget, OrderedDict(), []

    def lookup(self, keys):
        hit = []
        for k in keys:
            hit.append(k in self.lru)
            if hit[-1]:
                self.lru.move_to_end(k)
            else:
                self.missed.append(k)
        return hit

    def end_step(self):
        """-> [(expert to copy in, expert evicted or None)]"""
        out = []
        for key in sorted(set(self.missed))[:self.budget]:
            ev = self.lru.popitem(last=False)[0] if len(self.lru) >= self.n else None
            self.lru[key] = True
            out.append((key, ev))
        self.missed = []
        return out


class Card:
    def __init__(self, ops, w, cfg, host, dense_fmt="bfp8", expert_fmt="bfp4", slots=10_000, budget=25, max_t=4096,
                 cache=None):
        self.o, self.w, self.c, self.host = ops, w, cfg, host
        self.cache = cache or PyCache(slots, budget)
        self.dense_fmt, self.expert_fmt, self.max_t = dense_fmt, expert_fmt, max_t
        c = cfg
        self.d, self.hc, self.eps = c.hidden_size, c.hc_count, c.rms_norm_eps
        rp = c.rope_parameters
        self.rd = int(c.head_dim * rp.get("partial_rotary_factor", 1.0))
        self.inv_freq = 1.0 / (rp["rope_theta"] ** (np.arange(0, self.rd, 2) / self.rd))
        self.gated_act = (c.output_gate_type or c.hidden_act) == "sigmoid"
        self.dev, self.slots, self.routes = {}, {}, {}      # slots: (layer, expert) -> its weights on the card
        self.t, self.tokens = 0, []                    # committed tokens; the next one sits at position t
        self.hits = self.lookups = 0
        self.ple = {L - 1 for L in c.ple_layer_ids}
        self.state = {L: self.fresh_state(L) for L in range(c.num_hidden_layers)}
        self.pending = {}

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
        weight as [streams, d]."""
        if (name, rows) not in self.dev:
            w = np.asarray(self.w[name], np.float64)
            self.dev[name, rows] = self.o.put(((1 + w) if plus_one else w).reshape(rows, -1))
        return self.dev[name, rows]

    def const(self, key, make):
        if key not in self.dev:
            self.dev[key] = self.o.put(make())
        return self.dev[key]

    def fresh_state(self, L):
        c, o, s = self.c, self.o, {}
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

    # ---- building blocks (x: [m, ...], one row per token)
    def rms(self, x, w):
        o = self.o
        return o.mul(o.mul(x, o.rsqrt(o.add(o.mean(o.mul(x, x), -1), self.eps))), w)

    def rope(self, x, pos):
        """x [rows, D] of one token at RoPE position pos; partial rotary on the first rd features."""
        o, rd, half = self.o, self.rd, self.rd // 2
        f = pos * self.inv_freq
        cos = self.const(("cos", pos), lambda: np.concatenate([np.cos(f)] * 2)[None])
        sin = self.const(("sin", pos), lambda: np.concatenate([np.sin(f)] * 2)[None])
        r = o.slice(x, 1, 0, rd)
        rot = o.concat([o.scale(o.slice(r, 1, half, rd), -1), o.slice(r, 1, 0, half)], 1)
        y = o.add(o.mul(r, cos), o.mul(rot, sin))
        return o.concat([y, o.slice(x, 1, rd, x.shape[1])], 1) if x.shape[1] > rd else y

    def hc_read(self, p, h4, combine=True):
        o, hc, d, m = self.o, self.hc, self.d, h4.shape[0]
        hn = self.rms(o.reshape(h4, (m, hc, d)), self.norm(p + ".hc_norm.weight", rows=hc))      # [m, hc, d]
        flat = o.reshape(hn, (m, hc * d))
        mix = o.sigmoid(o.linear(o.silu(o.scale(o.linear(flat, self.lin(p + ".input_mix_weight_down")), 1 / hc)),
                                 self.lin(p + ".input_mix_weight_up")))
        mixed = o.reshape(o.mean(o.mul(o.reshape(mix, (m, hc, d)), hn), 1), (m, d))
        if not combine:
            return mixed
        return mixed, o.scale(o.sigmoid(o.scale(o.linear(flat, self.lin(p + ".block_inject_weight")), 1 / hc)), 2)

    def hc_write(self, h4, inj, out):
        o, m = self.o, h4.shape[0]
        return o.add(h4, o.reshape(o.mul(o.reshape(inj, (m, self.hc, 1)), o.reshape(out, (m, 1, self.d))),
                                   (m, self.hc * self.d)))

    # ---- Gated DeltaNet: conv window and recurrent state stay on the card
    def deltanet(self, L, x):
        o, c, s, p = self.o, self.c, self.state[L], f"layers.{L}.linear_attn"
        K, V, dk, dv = c.linear_num_key_heads, c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim
        m, kern = x.shape[0], c.linear_conv_kernel_dim
        qkv = o.linear(x, self.lin(p + ".in_proj_qkv"))                                # [m, C]
        win = o.concat([s["conv"], qkv], 0)                                            # [kern - 1 + m, C], oldest first
        cw = self.const(p + ".conv", lambda: np.asarray(self.w[p + ".conv1d.weight"], np.float64)[:, 0, :].T)
        qkv = o.silu(o.concat([o.sum(o.mul(o.slice(win, 0, i, i + kern), cw), 0) for i in range(m)], 0))
        l2 = lambda t: o.mul(t, o.rsqrt(o.add(o.sum(o.mul(t, t), -1), 1e-6)))
        q = l2(o.reshape(o.slice(qkv, 1, 0, K * dk), (m, K, dk)))
        k = l2(o.reshape(o.slice(qkv, 1, K * dk, 2 * K * dk), (m, K, dk)))
        v = o.reshape(o.slice(qkv, 1, 2 * K * dk, qkv.shape[1]), (m, V, dv))
        z = o.reshape(o.linear(x, self.lin(p + ".in_proj_z")), (m, V, dv))
        beta = o.sigmoid(o.linear(x, self.lin(p + ".in_proj_b")))
        negA = self.const(p + ".negA", lambda: -np.exp(np.asarray(self.w[p + ".A_log"], np.float64))[None])
        dtb = self.const(p + ".dt_bias", lambda: np.asarray(self.w[p + ".dt_bias"], np.float64)[None])
        g = o.mul(negA, o.softplus(o.add(o.linear(x, self.lin(p + ".in_proj_a")), dtb)))
        out, states = o.gdr(q, k, v, g, beta, s["S"], dk ** -0.5)
        self.pending[L] = dict(win=win, states=states)
        gate = o.sigmoid(z) if self.gated_act else o.silu(z)
        out = o.mul(self.rms(out, self.norm(p + ".norm.weight", plus_one=False)), gate)
        return o.linear(o.reshape(out, (m, V * dv)), self.lin(p + ".out_proj"))

    # ---- full attention with the QSA indexer; rows past the commit point are simply written again later
    def attention(self, L, x):
        o, c, s, p = self.o, self.c, self.state[L], f"layers.{L}.self_attn"
        m, t0 = x.shape[0], self.t
        H, KV, hd = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        nh, ihd, ratio = c.indexer_n_heads, c.indexer_head_dim, c.indexer_compress_ratio
        qk = o.linear(x, self.lin(p + ".indexer.index_qk_proj"))
        s["idx"] = o.write_rows(s["idx"], t0, o.slice(qk, 1, nh * ihd, nh * ihd + ihd))      # raw keys
        qg = o.linear(x, self.lin(p + ".q_proj"))
        kk = o.linear(x, self.lin(p + ".k_proj"))
        vv = o.linear(x, self.lin(p + ".v_proj"))
        s["V"] = o.write_rows(s["V"], t0, vv)
        G, outs = H // KV, []
        for i in range(m):
            t = t0 + i
            if (t + 1) % ratio == 0:                   # a block completes: pool, norm and rope its key once
                b = t // ratio
                blk = o.mean(o.slice(s["idx"], 0, b * ratio, (b + 1) * ratio), 0)
                s["blk"] = o.write_rows(s["blk"], b, self.rope(self.rms(blk, self.norm(p + ".indexer.k_layernorm.weight")),
                                                               b * ratio + self.pos0))
            iq = self.rope(self.rms(o.reshape(o.slice(o.slice(qk, 0, i, i + 1), 1, 0, nh * ihd), (nh, ihd)),
                                    self.norm(p + ".indexer.q_layernorm.weight")), t + self.pos0)
            nb = (t + 1) // ratio
            keep = list(range(nb * ratio, t + 1))
            if nb:
                score = o.scale(o.sum(o.relu(o.matmul(iq, o.transpose(o.slice(s["blk"], 0, 0, nb)))), 0),
                                1 / math.sqrt(ihd))
                _, top = o.topk(score, min(c.indexer_budget // ratio, nb))
                keep = sorted(int(j) * ratio + r for j in top[0] for r in range(ratio)) + keep
            qi = o.reshape(o.slice(qg, 0, i, i + 1), (H, 2 * hd))
            q = self.rope(self.rms(o.slice(qi, 1, 0, hd), self.norm(p + ".q_norm.weight")), t + self.pos0)
            k = self.rope(self.rms(o.reshape(o.slice(kk, 0, i, i + 1), (KV, hd)), self.norm(p + ".k_norm.weight")), t + self.pos0)
            s["K"] = o.write_rows(s["K"], t, o.reshape(k, (1, KV * hd)))
            Ks, Vs = o.gather_rows(s["K"], keep), o.gather_rows(s["V"], keep)          # [n, KV*hd]
            heads = []
            for kv in range(KV):
                a = o.softmax(o.scale(o.matmul(o.slice(q, 0, kv * G, (kv + 1) * G),
                                               o.transpose(o.slice(Ks, 1, kv * hd, (kv + 1) * hd))), 1 / math.sqrt(hd)))
                heads.append(o.matmul(a, o.slice(Vs, 1, kv * hd, (kv + 1) * hd)))
            outs.append(o.mul(o.reshape(o.concat(heads, 0), (1, H * hd)),
                              o.sigmoid(o.reshape(o.slice(qi, 1, hd, 2 * hd), (1, H * hd)))))
        return o.linear(o.concat(outs, 0), self.lin(p + ".o_proj"))

    # ---- MoE: router on the card; slot experts on the card (each read once for all m rows), the rest on the host
    def upload(self, L, e):
        # ponytail: from the float weights; on the card this is tstore_expert_tt's tiles, copied as they are
        p = f"layers.{L}.mlp.experts."
        gu, dn = np.asarray(self.w[p + "gate_up_proj"][e]), np.asarray(self.w[p + "down_proj"][e])
        return self.o.put_weight(gu, self.expert_fmt), self.o.put_weight(dn, self.expert_fmt)

    def moe(self, L, x):
        o, c, p = self.o, self.c, f"layers.{L}.mlp"
        m, I, E = x.shape[0], c.moe_intermediate_size, c.num_experts
        vals, top = o.topk(o.softmax(o.linear(x, self.lin(p + ".gate"))), c.num_experts_per_tok)
        wts = o.get(vals)
        wts = wts / wts.sum(1, keepdims=True) if c.norm_topk_prob else wts
        gate = np.zeros((m, E))                                  # gate[i, e]: token i's weight on expert e
        for i in range(m):
            gate[i, top[i]] = wts[i]
            self.routes[L, self.t + i] = top[i]
        y = o.mul(o.linear(o.mul(o.silu(o.linear(x, self.lin(p + ".shared_expert.gate_proj"))),
                                 o.linear(x, self.lin(p + ".shared_expert.up_proj"))), self.lin(p + ".shared_expert.down_proj")),
                  o.sigmoid(o.linear(x, self.lin(p + ".shared_expert_gate"))))
        es = sorted(set(top.ravel().tolist()))
        cold = []
        for e, hit in zip(es, self.cache.lookup([(L, e) for e in es])):
            n = int((gate[:, e] > 0).sum())
            self.lookups += n
            if hit:
                self.hits += n
                gu, dn = self.slots[L, e]
                h = o.linear(x, gu)
                ye = o.linear(o.mul(o.silu(o.slice(h, 1, 0, I)), o.slice(h, 1, I, 2 * I)), dn)
                y = o.add(y, o.mul(ye, o.put(gate[:, e:e + 1])))
            else:
                cold.append(e)
        if cold:                                          # the only vectors that cross the bus: X out, the sum back
            y = o.add(y, o.put(self.host.experts(L, o.get(x), cold, gate[:, cold].T)))
        return y

    def promote(self):
        """After a step: the cache's copies into the card's slots."""
        for (L, e), ev in self.cache.end_step():
            self.slots.pop(ev, None)
            self.slots[L, e] = self.upload(L, e)

    # ---- PLE: the host hashes the N-gram ids and sends the rows; the card gates and convolves
    def ple_layer(self, L, h4, tokens):
        o, c, p, hc, d, s = self.o, self.c, f"layers.{L}.ple", self.hc, self.d, self.state[L]
        m = h4.shape[0]
        emb = o.put(np.stack([self.host.ngram_rows(L, self.tokens + tokens[:i + 1]) for i in range(m)]))
        key = self.rms(o.reshape(o.linear(emb, self.lin(p + ".key_proj")), (m, hc, d)), self.norm(p + ".norm_key.weight", rows=hc))
        value = o.reshape(o.linear(emb, self.lin(p + ".value_proj")), (m, 1, d))
        query = self.rms(o.reshape(h4, (m, hc, d)), self.norm(p + ".norm_query.weight", rows=hc))
        gate = o.scale(o.sum(o.mul(key, query), -1), 1 / math.sqrt(d))                 # [m, hc, 1]
        gate = o.mul(o.sqrt(o.clamp_min(o.abs(gate), 1e-6)), o.sign(gate))
        gv = o.mul(o.sigmoid(gate), value)                                             # [m, hc, d]
        gvn = o.reshape(self.rms(gv, self.norm(p + ".norm_conv.weight", rows=hc)), (m, hc * d))
        win = o.concat([s["ple"], gvn], 0)                                             # last 9 + the m new
        self.pending[L, "ple"] = win
        cw = self.const(p + ".conv", lambda: np.asarray(self.w[p + ".conv1d.weight"], np.float64)[:, 0, :].T)
        span = (c.ple_conv_kernel_size - 1) * c.ngram_size
        conv = [o.sum(o.mul(o.concat([o.slice(win, 0, j, j + 1) for j in range(i, i + span + 1, c.ngram_size)], 0), cw), 0)
                for i in range(m)]
        return o.add(o.reshape(gv, (m, hc * d)), o.silu(o.concat(conv, 0)))

    # ---- m tokens through the model, then keep j of them
    pos0 = 0                                           # RoPE position of cache row 0 (the MTP layer starts at 1)

    def layers(self, h4, tokens=()):
        """h4 [m, hc*d] residual rows of m consecutive positions starting at self.t -> after the last layer."""
        o, c = self.o, self.c
        self.pending = {}
        for L in range(c.num_hidden_layers):
            if L in self.ple:
                h4 = o.add(h4, self.ple_layer(L, h4, list(tokens)))
            x, inj = self.hc_read(f"layers.{L}.attn_hyper_connection", h4)
            h4 = self.hc_write(h4, inj, self.deltanet(L, x) if c.layer_types[L] == "linear_attention" else self.attention(L, x))
            x, inj = self.hc_read(f"layers.{L}.mlp_hyper_connection", h4)
            h4 = self.hc_write(h4, inj, self.moe(L, x))
        return h4

    def verify(self, tokens):
        """The next m tokens (the last accepted token and m-1 drafts) in one pass; returns their mixer outputs [m, d].
        Nothing is kept until commit()."""
        tokens = [int(t) for t in tokens]
        h4 = self.o.put(np.stack([np.tile(self.host.embed(t), self.hc) for t in tokens]))
        self.h4 = self.layers(h4, tokens)
        self.verified = tokens
        return self.hc_read("hyper_connection_mixer", self.h4, combine=False)

    def commit(self, j):
        """Keep the first j verified tokens: states as if only they had run."""
        o, c = self.o, self.c
        for L, s in self.state.items():
            if L in self.pending:
                p, kern = self.pending[L], c.linear_conv_kernel_dim
                s["conv"] = o.slice(p["win"], 0, j, j + kern - 1)
                s["S"] = o.reshape(o.slice(p["states"], 0, j - 1, j), tuple(p["states"].shape[1:]))
            if (L, "ple") in self.pending:
                s["ple"] = o.slice(self.pending[L, "ple"], 0, j, j + (c.ple_conv_kernel_size - 1) * c.ngram_size)
        self.tokens += self.verified[:j]
        self.t += j
        self.promote()

    def step(self, token):
        out = self.verify([token])
        self.commit(1)
        return out

    def lm_argmax(self, mixed):
        return self.o.topk(self.o.linear(mixed, self.lin("lm_head")), 1)[1][:, 0]

    def generate(self, prompt, n, drafter=None, k=2):
        """Greedy decoding of n tokens after the prompt; with a drafter, speculative: each step verifies the last
        token plus k drafts and keeps the drafts up to the first one the model disagrees with."""
        nxt = int(self.lm_argmax(self.verify(prompt))[-1])
        self.commit(len(prompt))
        if drafter:
            drafter.catch_up(self, 0, len(prompt))
        out, self.steps = [], 0
        while len(out) < n:
            drafts = drafter.propose(self, nxt, k) if drafter else []
            start = self.t
            pred = self.lm_argmax(self.verify([nxt] + drafts))                # pred[i]: the model's pick after row i
            j = 1
            while j <= len(drafts) and drafts[j - 1] == pred[j - 1]:
                j += 1
            self.commit(j)
            out += self.tokens[-j:]
            nxt = int(pred[j - 1])
            self.steps += 1
            if drafter:
                drafter.catch_up(self, start, j)
        return out[:n]


class MTPDrafter:
    """QFN's MTP layer on the card (layout as in Basalt, see ref_real.py): input = per-stream fc_hidden of the main
    model's final residual (pre_fc_norm_hidden per stream) + fc_embedding of the next token's embedding, then one
    full-attention MoE layer with its own mixer, then the main model's LM head. Its KV cache holds one entry per
    position, built from the main model's residuals as tokens are accepted (catch_up); chained drafts feed its own
    residual and guess back in, on cache rows that the next catch_up overwrites."""

    def __init__(self, ops, w_mtp, cfg, host, **card_kw):
        import copy
        mc = copy.deepcopy(cfg)
        mc.num_hidden_layers, mc.layer_types, mc.ple_layer_ids = 1, ["full_attention"], []
        self.m = Card(ops, w_mtp, mc, host, **card_kw)
        self.m.pos0 = 1                                 # cache row r holds position r + 1
        self.w, self.R = w_mtp, None

    def run(self, R, tok, row):
        """One MTP step at cache row `row` from residual R [1, hc*d] and token tok; returns (its residual, its guess)."""
        m, o = self.m, self.m.o
        hc, d = m.hc, m.d
        hn = m.rms(o.reshape(R, (hc, d)), m.norm("pre_fc_norm_hidden.weight", rows=hc))
        h2 = o.linear(hn, m.lin("fc_hidden"))                                          # [hc, d], per stream
        e = o.put(m.host.embed(tok)[None])
        e2 = o.linear(m.rms(e, m.norm("pre_fc_norm_embedding.weight")), m.lin("fc_embedding"))
        m.t = row
        x = m.layers(o.reshape(o.add(h2, e2), (1, hc * d)))
        m.pending = {}
        m.promote()
        return x, int(self.lm.lm_argmax(m.hc_read("hyper_connection_mixer", x, combine=False))[0])

    def catch_up(self, card, start, j):
        """After card.commit(j) of a verify that started at position `start`: final cache entries for the accepted
        positions. Row p (position p + 1) comes from the main residual at p and the token at p + 1."""
        self.lm = card
        res = lambda p: self.R if p == start - 1 else card.o.slice(card.h4, 0, p - start, p - start + 1)
        for p in range(max(start - 1, 0), start + j - 1):
            self.run(res(p), card.tokens[p + 1], p)
        self.R = res(start + j - 1)                                                    # residual of the last accepted

    def propose(self, card, nxt, k):
        drafts, R, tok = [], self.R, nxt
        for i in range(k):
            R, g = self.run(R, tok, card.t - 1 + i)
            drafts.append(g)
            tok = g
        return drafts
