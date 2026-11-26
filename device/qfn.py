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

Speed (device/opcount.py counts it): a pass is a chain of traced segments (ops.trace: captured once per shape,
replayed from the card's trace buffer with no Python dispatch) broken only where the host must decide something:
    each DeltaNet layer   one segment: the previous layer's MoE write, hyper-connection read, DeltaNet, the MLP's
                          hyper-connection read, router, shared expert; then the routing read
    each attention layer  projections + norms + RoPE for all m tokens; per token the indexer's block scores (read
                          by the host, which picks the blocks) and attention over the chosen rows padded to a fixed
                          length; then o_proj + the MLP's dense part
    the slot experts      three gathered matmuls over per-device weight banks (ops.gather_matmul: only the chosen
                          slots are read), traced per number of experts
Inside, weights that read the same input are one linear per format (lins), RMSNorm one op, RoPE one matmul.
Caches and banks are updated in place, so segments read them by address.
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
    """The adaptive cache's policy in Python (the engine's is C: tcache, same policy and slot numbers): which experts
    sit in the accelerators' slots; after each step up to `budget` of the step's misses (lowest (layer, expert) first)
    are copied in, evicting the least recently used. Empty slots go first, highest number first, as in tcache."""

    def __init__(self, n_slots, budget):
        self.n, self.budget, self.lru, self.missed, self.free = n_slots, budget, OrderedDict(), [], list(range(n_slots))

    def lookup(self, keys):
        """-> slot of each key, or -1 (a miss: computed on the host)"""
        out = []
        for k in keys:
            if k in self.lru:
                self.lru.move_to_end(k)
                out.append(self.lru[k])
            else:
                self.missed.append(k)
                out.append(-1)
        return out

    def end_step(self):
        """-> [(expert to copy in, expert evicted or None, slot)]"""
        out = []
        for key in sorted(set(self.missed))[:self.budget]:
            if self.free:
                ev, slot = None, self.free.pop()
            else:
                ev, slot = self.lru.popitem(last=False)
            self.lru[key] = slot
            out.append((key, ev, slot))
        self.missed = []
        return out


class Card:
    def __init__(self, ops, w, cfg, host, dense_fmt="bfp8", expert_fmt="bfp4", slots=10_000, budget=25, max_t=4096,
                 cache=None, trace=8):
        self.o, self.w, self.c, self.host, self.trace = ops, w, cfg, host, trace
        self.cache = cache or PyCache(slots, budget)
        self.dense_fmt, self.expert_fmt, self.max_t = dense_fmt, expert_fmt, max_t
        c = cfg
        self.d, self.hc, self.eps = c.hidden_size, c.hc_count, c.rms_norm_eps
        rp = c.rope_parameters
        self.rd = int(c.head_dim * rp.get("partial_rotary_factor", 1.0))
        self.inv_freq = 1.0 / (rp["rope_theta"] ** (np.arange(0, self.rd, 2) / self.rd))
        self.gated_act = (c.output_gate_type or c.hidden_act) == "sigmoid"
        self.dev, self.slots, self.routes = {}, {}, {}      # slots: (layer, expert) -> its expert slot
        self.n_slots, self.banks = slots, None          # the slots' weights: one bank per matrix (gate, up, down)
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

    def lins(self, x, names):
        """x through several weights that read the same input: their rows stacked into one linear per format (one
        weight read, one dispatch), the output split back. -> [x @ W^T for each name]"""
        key = ("lins",) + tuple(names)
        if key not in self.dev:
            groups = {}
            for n in names:
                W, f = np.asarray(self.w[n + ".weight"], np.float32), self.fmt(n)
                if f in ("bfp8", "bfp4") and (W.shape[0] % 16 or W.shape[1] % 8):
                    f = "bf16"                          # put_weight falls back to bf16 for it alone, too
                groups.setdefault(f, []).append((n, W))
            self.dev[key] = [(self.o.put_weight(np.concatenate([W for _, W in g]), f), [(n, W.shape[0]) for n, W in g])
                             for f, g in groups.items()]
        out = {}
        for w, parts in self.dev[key]:
            y, i = self.o.linear(x, w), 0
            for n, r in parts:
                out[n] = y if len(parts) == 1 else self.o.slice(y, 1, i, i + r)
                i += r
        return [out[n] for n in names]

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
        return self.o.rms_norm(x, w, self.eps)

    def cs(self, pos, n=1):
        """RoPE cos and sin rows [n, rd] of positions pos .. pos+n-1, sliced from tables kept on the card"""
        f = np.arange(self.max_t + 1)[:, None] * self.inv_freq[None]
        cos = self.const("cos", lambda: np.concatenate([np.cos(f)] * 2, 1))
        sin = self.const("sin", lambda: np.concatenate([np.sin(f)] * 2, 1))
        return self.o.slice(cos, 0, pos, pos + n), self.o.slice(sin, 0, pos, pos + n)

    def rope(self, x, cos, sin):
        """x [..., D], partial rotary on the first rd features of the last dim; cos, sin broadcast to x[..., :rd]"""
        o, rd, half, dim, D = self.o, self.rd, self.rd // 2, len(x.shape) - 1, x.shape[-1]
        P = self.const("rot", lambda: np.block([[np.zeros((half, half)), np.eye(half)], [-np.eye(half), np.zeros((half, half))]]))
        r = o.slice(x, dim, 0, rd) if D > rd else x
        y = o.add(o.mul(r, cos), o.mul(o.matmul(r, P), sin))           # r @ P = [-r2, r1]
        return o.concat([y, o.slice(x, dim, rd, D)], dim) if D > rd else y

    def hc_read(self, p, h4, combine=True):
        o, hc, d, m = self.o, self.hc, self.d, h4.shape[0]
        hn = self.rms(o.reshape(h4, (m, hc, d)), self.norm(p + ".hc_norm.weight", rows=hc))      # [m, hc, d]
        flat = o.reshape(hn, (m, hc * d))
        down, *inj = self.lins(flat, [p + ".input_mix_weight_down"] + ([p + ".block_inject_weight"] if combine else []))
        mix = o.sigmoid(o.linear(o.silu(o.scale(down, 1 / hc)), self.lin(p + ".input_mix_weight_up")))
        mixed = o.reshape(o.mean(o.mul(o.reshape(mix, (m, hc, d)), hn), 1), (m, d))
        if not combine:
            return mixed
        return mixed, o.scale(o.sigmoid(o.scale(inj[0], 1 / hc)), 2)

    def hc_write(self, h4, inj, out):
        o, m = self.o, h4.shape[0]
        return o.add(h4, o.reshape(o.mul(o.reshape(inj, (m, self.hc, 1)), o.reshape(out, (m, 1, self.d))),
                                   (m, self.hc * self.d)))

    # ---- Gated DeltaNet: conv window and recurrent state stay on the card
    def deltanet(self, L, x):
        out, win, states = self.deltanet_core(L, x, self.state[L]["conv"], self.state[L]["S"])
        self.pending[L] = dict(win=win, states=states)
        return out

    def deltanet_core(self, L, x, conv, S):
        """-> (output [m, d], conv window, per-token states); no side effects (it runs inside a trace)"""
        o, c, p = self.o, self.c, f"layers.{L}.linear_attn"
        K, V, dk, dv = c.linear_num_key_heads, c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim
        m, kern = x.shape[0], c.linear_conv_kernel_dim
        qkv, z, a, b = self.lins(x, [p + ".in_proj_qkv", p + ".in_proj_z", p + ".in_proj_a", p + ".in_proj_b"])
        win = o.concat([conv, qkv], 0)                                                 # [kern - 1 + m, C], oldest first
        cw = self.const(p + ".conv", lambda: np.asarray(self.w[p + ".conv1d.weight"], np.float64)[:, 0, :].T)
        qkv = o.silu(o.concat([o.sum(o.mul(o.slice(win, 0, i, i + kern), cw), 0) for i in range(m)], 0))
        l2 = lambda t: o.mul(t, o.rsqrt(o.add(o.sum(o.mul(t, t), -1), 1e-6)))
        q = l2(o.reshape(o.slice(qkv, 1, 0, K * dk), (m, K, dk)))
        k = l2(o.reshape(o.slice(qkv, 1, K * dk, 2 * K * dk), (m, K, dk)))
        v = o.reshape(o.slice(qkv, 1, 2 * K * dk, qkv.shape[1]), (m, V, dv))
        z = o.reshape(z, (m, V, dv))
        beta = o.sigmoid(b)
        negA = self.const(p + ".negA", lambda: -np.exp(np.asarray(self.w[p + ".A_log"], np.float64))[None])
        dtb = self.const(p + ".dt_bias", lambda: np.asarray(self.w[p + ".dt_bias"], np.float64)[None])
        g = o.mul(negA, o.softplus(o.add(a, dtb)))
        out, states = o.gdr(q, k, v, g, beta, S, dk ** -0.5)
        gate = o.sigmoid(z) if self.gated_act else o.silu(z)
        out = o.mul(self.rms(out, self.norm(p + ".norm.weight", plus_one=False)), gate)
        return o.linear(o.reshape(out, (m, V * dv)), self.lin(p + ".out_proj")), win, states

    # ---- full attention with the QSA indexer; rows past the commit point are simply written again later
    def attn_proj(self, L, x):
        p = f"layers.{L}.self_attn"
        return self.lins(x, [p + ".indexer.index_qk_proj", p + ".q_proj", p + ".k_proj", p + ".v_proj"])

    def attn_prep(self, L, proj, cos, sin):
        """The m tokens' projections normed and roped together (pure: it runs inside a trace); cos, sin [m, rd] of
        their positions. -> (indexer keys, indexer queries, queries (both pre-scaled), output gate, keys, values)"""
        o, c, p = self.o, self.c, f"layers.{L}.self_attn"
        H, KV, hd, nh, ihd = c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.indexer_n_heads, c.indexer_head_dim
        qk, qg, kk, vv = proj
        m = qk.shape[0]
        cos, sin = o.reshape(cos, (m, 1, self.rd)), o.reshape(sin, (m, 1, self.rd))
        rr = lambda t, w: self.rope(self.rms(t, self.norm(p + w)), cos, sin)
        iq = o.scale(rr(o.reshape(o.slice(qk, 1, 0, nh * ihd), (m, nh, ihd)), ".indexer.q_layernorm.weight"),
                     1 / math.sqrt(ihd))
        qg = o.reshape(qg, (m, H, 2 * hd))
        q = o.scale(rr(o.slice(qg, 2, 0, hd), ".q_norm.weight"), 1 / math.sqrt(hd))
        gate = o.sigmoid(o.reshape(o.slice(qg, 2, hd, 2 * hd), (m, H * hd)))
        k = o.reshape(rr(o.reshape(kk, (m, KV, hd)), ".k_norm.weight"), (m, KV * hd))
        return o.slice(qk, 1, nh * ihd, nh * ihd + ihd), iq, q, gate, k, vv

    def attention(self, L, x, prep=None, raw=False):
        """prep: attn_prep's outputs, when a traced segment already computed them. Per token: the indexer's block
        scores (a traced segment over every block row), the choice on the host, then attention over the chosen rows
        padded to a fixed n_max (another traced segment). The caches are updated in place, so traces read them by
        address."""
        o, c, s, p = self.o, self.c, self.state[L], f"layers.{L}.self_attn"
        m, t0 = x.shape[0], self.t
        H, KV, hd = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        nh, ihd, ratio = c.indexer_n_heads, c.indexer_head_dim, c.indexer_compress_ratio
        key, iq, q, gate, k, vv = prep or self.attn_prep(L, self.attn_proj(L, x), *self.cs(t0 + self.pos0, m))
        s["idx"] = o.write_rows(s["idx"], t0, key)                                     # raw indexer keys
        s["K"] = o.write_rows(s["K"], t0, k)
        s["V"] = o.write_rows(s["V"], t0, vv)
        G, outs = H // KV, []
        n_max = c.indexer_budget // ratio * ratio + ratio - 1          # chosen blocks + the open block

        def score(iq_i):                               # the block keys are read in place (persistent, by address)
            return (o.sum(o.relu(o.matmul(o.reshape(iq_i, (nh, ihd)), o.transpose(s["blk"]))), 0),)

        def attend(q_i, ids, mask):                    # all KV heads at once over the n_max gathered rows
            Ks, Vs = o.gather_rows(s["K"], ids), o.gather_rows(s["V"], ids)            # [n_max, KV*hd]
            a = o.softmax(o.add(o.matmul(o.reshape(q_i, (KV, G, hd)), o.permute(o.reshape(Ks, (n_max, KV, hd)), (1, 2, 0))),
                                mask))                                                 # [KV, G, n_max]
            return (o.reshape(o.matmul(a, o.permute(o.reshape(Vs, (n_max, KV, hd)), (1, 0, 2))), (1, H * hd)),)
        for i in range(m):
            t = t0 + i
            if (t + 1) % ratio == 0:                   # a block completes: pool, norm and rope its key once
                b = t // ratio
                blk = o.mean(o.slice(s["idx"], 0, b * ratio, (b + 1) * ratio), 0)
                s["blk"] = o.write_rows(s["blk"], b, self.rope(self.rms(blk, self.norm(p + ".indexer.k_layernorm.weight")),
                                                               *self.cs(b * ratio + self.pos0)))
            nb = (t + 1) // ratio
            keep = list(range(nb * ratio, t + 1))
            if nb:                                     # all block scores (fixed shape) to the host; top blocks there
                sc = o.get(self.seg((L, "score"), score, o.slice(iq, 0, i, i + 1), m=m)[0])[0, :nb]
                top = np.argsort(-sc, kind="stable")[:min(c.indexer_budget // ratio, nb)]
                keep = sorted(int(j) * ratio + r for j in top for r in range(ratio)) + keep
            n = len(keep)                              # padded to n_max rows, the padding masked out
            ids = o.put_ids(keep + [0] * (n_max - n), "uint32")
            mask = o.put(np.where(np.arange(n_max) < n, 0.0, -1e9)[None, None])
            outs.append(self.seg((L, "attend", i), attend, o.slice(q, 0, i, i + 1), ids, mask, m=m)[0])
        return outs if raw else self.attn_out(L, gate, outs)

    def attn_out(self, L, gate, outs):
        return self.o.linear(self.o.mul(self.o.concat(outs, 0), gate), self.lin(f"layers.{L}.self_attn.o_proj"))

    # ---- MoE: router on the card; slot experts on the card (each read once for all m rows), the rest on the host
    def put_slot(self, slot, L, e):
        """Expert e of layer L into expert slot `slot` of this device's banks (gate and up apart, so the gate's SiLU
        rides on its matmul). ponytail: from the float weights; on the card these are tstore_expert_tt's tiles."""
        o, d, I, f = self.o, self.d, self.c.moe_intermediate_size, self.expert_fmt
        if self.banks is None:
            self.banks = (o.bank(self.n_slots, d, I, f), o.bank(self.n_slots, d, I, f), o.bank(self.n_slots, I, d, f))
        p = f"layers.{L}.mlp.experts."
        gu, dn = np.asarray(self.w[p + "gate_up_proj"][e]), np.asarray(self.w[p + "down_proj"][e])
        for bank, W in zip(self.banks, (gu[:I], gu[I:], dn)):
            o.bank_write(bank, slot, W)

    def moe(self, L, x, dense=None):
        """dense: (router probabilities, shared expert output) when a trace already computed them"""
        o = self.o
        probs, y = dense or self.dense_mlp(L, x)
        top, gate = self.route(L, probs)               # gate[i, e]: token i's weight on expert e
        es = sorted(set(top.ravel().tolist()))
        cold, hot = [], []
        for e, slot in zip(es, self.cache.lookup([(L, e) for e in es])):
            n = int((gate[:, e] > 0).sum())
            self.lookups += n
            if slot >= 0:
                self.hits += n
                hot.append((self.slots[L, e], gate[:, e]))
            else:
                cold.append(e)
        if hot:
            y = o.add(y, self.slot_experts(x, hot))
        if cold:                                          # the only vectors that cross the bus: X out, the sum back
            y = o.add(y, o.put(self.host.experts(L, o.get(x), cold, gate[:, cold].T)))
        return y

    def route(self, L, probs):
        """Router probabilities [m, E] -> (top [m, k] experts, gate [m, E] weights) on the host."""
        c, P = self.c, self.o.get(probs)               # one read of [m, E]; the choice is made on the host
        top = np.argsort(-P, axis=1, kind="stable")[:, :c.num_experts_per_tok]    # ties: lowest index, as ttnn.topk
        wts = np.take_along_axis(P, top, 1)
        wts = wts / wts.sum(1, keepdims=True) if c.norm_topk_prob else wts
        gate = np.zeros((probs.shape[0], c.num_experts))
        for i in range(probs.shape[0]):
            gate[i, top[i]] = wts[i]
            self.routes[L, self.t + i] = top[i]
        return top, gate

    def dense_mlp(self, L, x, router=True):
        """-> (router probabilities or None, the gated shared expert: this device's rows of it when the dense part
        is split); the weights that read x in one fused read."""
        o, p = self.o, f"layers.{L}.mlp"
        g, u, sg, *r = self.lins(x, [p + ".shared_expert.gate_proj", p + ".shared_expert.up_proj",
                                     p + ".shared_expert_gate"] + ([p + ".gate"] if router else []))
        y = o.mul(o.linear(o.mul(o.silu(g), u), self.lin(p + ".shared_expert.down_proj")), o.sigmoid(sg))
        return (o.softmax(r[0]) if router else None), y

    def slot_experts(self, x, items):
        """Routed experts in this device's slots on every row of x: items = [(slot, g [m] host weights, 0 where a
        token does not use it)] -> their weighted sum [m, d]. Each matrix is one gathered matmul over the banks
        (ttnn.sparse_matmul, indexed: only those slots' weights are read), traced per number of experts."""
        ids = self.o.put_ids([slot for slot, _ in items], "uint16")
        G = self.o.put(np.stack([g for _, g in items])[:, :, None])                    # [n, m, 1]
        return self.seg(("slots", len(items)), self.slot_core, x, ids, G)[0]

    def slot_core(self, x, ids, G):
        o, (m, d) = self.o, x.shape
        bg, bu, bd = self.banks                        # updated in place by promote: traces read them by address
        x3 = o.reshape(x, (1, m, d))
        a = o.mul(o.gather_matmul(x3, bg, ids, act="silu"), o.gather_matmul(x3, bu, ids))   # [n, m, I]
        return (o.reshape(o.sum(o.mul(o.gather_matmul(a, bd, ids), G), 0), (m, d)),)

    def promote(self):
        """After a step: the cache's copies into the card's slots."""
        for (L, e), ev, slot in self.cache.end_step():
            self.slots.pop(ev, None)
            self.slots[L, e] = slot
            self.put_slot(slot, L, e)

    # ---- PLE: the host hashes the N-gram ids and sends the rows; the card gates and convolves
    def ple_layer(self, L, h4, tokens):
        """-> h4 with the PLE output added; the host's N-gram rows go up, the rest is a traced segment"""
        emb = self.o.put(np.stack([self.host.ngram_rows(L, self.tokens + tokens[:i + 1]) for i in range(h4.shape[0])]))
        h4, win = self.seg(("ple", L), lambda h4, emb, st: self.ple_core(L, h4, emb, st), h4, emb, self.state[L]["ple"])
        self.pending[L, "ple"] = win
        return h4

    def ple_core(self, L, h4, emb, st):
        o, c, p, hc, d = self.o, self.c, f"layers.{L}.ple", self.hc, self.d
        m = h4.shape[0]
        key = self.rms(o.reshape(o.linear(emb, self.lin(p + ".key_proj")), (m, hc, d)), self.norm(p + ".norm_key.weight", rows=hc))
        value = o.reshape(o.linear(emb, self.lin(p + ".value_proj")), (m, 1, d))
        query = self.rms(o.reshape(h4, (m, hc, d)), self.norm(p + ".norm_query.weight", rows=hc))
        gate = o.scale(o.sum(o.mul(key, query), -1), 1 / math.sqrt(d))                 # [m, hc, 1]
        gate = o.mul(o.sqrt(o.clamp_min(o.abs(gate), 1e-6)), o.sign(gate))
        gv = o.mul(o.sigmoid(gate), value)                                             # [m, hc, d]
        gvn = o.reshape(self.rms(gv, self.norm(p + ".norm_conv.weight", rows=hc)), (m, hc * d))
        win = o.concat([st, gvn], 0)                                                   # last 9 + the m new
        cw = self.const(p + ".conv", lambda: np.asarray(self.w[p + ".conv1d.weight"], np.float64)[:, 0, :].T)
        span = (c.ple_conv_kernel_size - 1) * c.ngram_size
        conv = [o.sum(o.mul(o.concat([o.slice(win, 0, j, j + 1) for j in range(i, i + span + 1, c.ngram_size)], 0), cw), 0)
                for i in range(m)]
        return o.add(h4, o.add(o.reshape(gv, (m, hc * d)), o.silu(o.concat(conv, 0)))), win

    # ---- m tokens through the model, then keep j of them
    pos0 = 0                                           # RoPE position of cache row 0 (the MTP layer starts at 1)

    def layers(self, h4, tokens=()):
        """h4 [m, hc*d] residual rows of m consecutive positions starting at self.t -> after the last layer."""
        self.pending, carry = {}, ()
        for L in range(self.c.num_hidden_layers):
            if L in self.ple:
                h4, carry = self.settle(h4, carry), ()
                h4 = self.ple_layer(L, h4, list(tokens))
            h4, carry = (self.deltanet_layer if self.c.layer_types[L] == "linear_attention" else self.attention_layer)(L, h4, carry)
        return self.settle(h4, carry)

    def settle(self, h4, carry):
        """A layer's MoE output is written into the residual by the next layer's first traced segment (carry =
        (inject weights, MoE output)); this writes it when no segment follows."""
        return self.hc_write(h4, *carry) if carry else h4

    def seg(self, name, fn, *args, m=None):
        """fn(*args) as a traced segment when the pass has m (default: args[0]'s rows) <= self.trace tokens (fn: card
        ops only, no side effects; its outputs are overwritten by its next replay), else eagerly"""
        m = args[0].shape[0] if m is None else m
        return self.o.trace((id(self), name, m), fn, *args) if m <= self.trace else fn(*args)

    def deltanet_layer(self, L, h4, carry=()):
        def dense(h4, *rest):                          # everything up to the routing decision, on the card alone
            *carry, conv, S = rest
            h4 = self.settle(h4, carry)
            x, inj = self.hc_read(f"layers.{L}.attn_hyper_connection", h4)
            out, win, states = self.deltanet_core(L, x, conv, S)
            h4 = self.hc_write(h4, inj, out)
            x, inj = self.hc_read(f"layers.{L}.mlp_hyper_connection", h4)
            return (h4, x, inj, win, states) + self.dense_mlp(L, x)
        h4, x, inj, win, states, probs, y = self.seg((L, len(carry)), dense, h4, *carry, self.state[L]["conv"],
                                                     self.state[L]["S"])
        self.pending[L] = dict(win=win, states=states)
        return h4, (inj, self.moe(L, x, (probs, y)))

    def attention_layer(self, L, h4, carry=()):
        def pre(h4, *rest):
            *carry, cos, sin = rest
            h4 = self.settle(h4, carry)
            x, inj = self.hc_read(f"layers.{L}.attn_hyper_connection", h4)
            return (h4, x, inj) + tuple(self.attn_prep(L, self.attn_proj(L, x), cos, sin))

        def post(h4, inj, gate, *outs):
            h4 = self.hc_write(h4, inj, self.attn_out(L, gate, outs))
            x, inj = self.hc_read(f"layers.{L}.mlp_hyper_connection", h4)
            return (h4, x, inj) + self.dense_mlp(L, x)
        h4, x, inj, *prep = self.seg((L, "pre", len(carry)), pre, h4, *carry, *self.cs(self.t + self.pos0, h4.shape[0]))
        h4, x, inj, probs, y = self.seg((L, "post"), post, h4, inj, prep[3], *self.attention(L, x, prep, raw=True))
        return h4, (inj, self.moe(L, x, (probs, y)))

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
