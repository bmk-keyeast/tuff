"""The p150a and an RTX 4060 Ti decoding QFN together: the dense part split between them (tensor parallel), one expert
cache over both devices' slots, the host in between.

    card (A)   its share of every DeltaNet and attention layer and of the shared expert and LM head; everything
               replicated (hyper-connections, PLE, router) runs here, and the GPU gets the block input as a vector
    GPU (B)    the rest of those heads and rows
    host       sums the two partial results of every block (they cannot reach each other directly), the cold experts

The split is exact (tp_check.py checks it on transformers' code): DeltaNet by key-head groups (each key head with its
value heads, their conv channels, z/a/b rows, A_log, dt_bias and out_proj columns), attention by KV head (its query
heads, q/k/v rows and o_proj columns; the QSA indexer runs on both), the shared expert and the LM head by rows.
Expert slots 0 .. n_card-1 are on the card, the rest on the GPU; the cache decides, the slot number says where.

TP has Card's interface (verify, commit, lm_argmax, generate), so MTPDrafter and speculative decoding work unchanged.
device/test_tp.py checks it against a single Card.
"""
import copy

import numpy as np

import qfn


def shard(w, cfg, part):
    """Weights and config of one device. part = dict(lin=(k0, k1) key heads, att=(j0, j1) KV heads,
    shared=(s0, s1) rows, lm=(v0, v1) vocabulary rows)."""
    c = copy.deepcopy(cfg)
    K, V, dk, dv = cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
    H, KV, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    G, GA = V // K, H // KV
    (k0, k1), (j0, j1), (s0, s1), (v0, v1) = part["lin"], part["att"], part["shared"], part["lm"]
    c.linear_num_key_heads, c.linear_num_value_heads = k1 - k0, (k1 - k0) * G
    c.num_key_value_heads, c.num_attention_heads = j1 - j0, (j1 - j0) * GA
    c.shared_expert_intermediate_size = s1 - s0
    ks, vs = np.arange(k0 * dk, k1 * dk), np.arange(k0 * G * dv, k1 * G * dv)
    ch = np.concatenate([ks, K * dk + ks, 2 * K * dk + vs])
    heads = np.arange(k0 * G, k1 * G)
    qrows, kvrows, ocols = np.arange(j0 * GA * 2 * hd, j1 * GA * 2 * hd), np.arange(j0 * hd, j1 * hd), np.arange(j0 * GA * hd, j1 * GA * hd)
    out = dict(w)
    for L, kind in enumerate(cfg.layer_types[:cfg.num_hidden_layers]):
        p = f"layers.{L}."
        if kind == "linear_attention":
            a = p + "linear_attn."
            for n, idx, ax in (("in_proj_qkv.weight", ch, 0), ("conv1d.weight", ch, 0), ("in_proj_z.weight", vs, 0),
                               ("in_proj_a.weight", heads, 0), ("in_proj_b.weight", heads, 0), ("A_log", heads, 0),
                               ("dt_bias", heads, 0), ("out_proj.weight", vs, 1)):
                out[a + n] = np.take(np.asarray(w[a + n]), idx, axis=ax)
        else:
            a = p + "self_attn."
            for n, idx, ax in (("q_proj.weight", qrows, 0), ("k_proj.weight", kvrows, 0), ("v_proj.weight", kvrows, 0),
                               ("o_proj.weight", ocols, 1)):
                out[a + n] = np.take(np.asarray(w[a + n]), idx, axis=ax)
        m = p + "mlp.shared_expert."
        out[m + "gate_proj.weight"] = np.asarray(w[m + "gate_proj.weight"])[s0:s1]
        out[m + "up_proj.weight"] = np.asarray(w[m + "up_proj.weight"])[s0:s1]
        out[m + "down_proj.weight"] = np.asarray(w[m + "down_proj.weight"])[:, s0:s1]
    out["lm_head.weight"] = np.asarray(w["lm_head.weight"])[v0:v1]
    return out, c, v0


def split_parts(cfg, f):
    """The GPU's share f of every split dimension (plan.split_multi's f balances the two devices' bandwidth); the card
    keeps the first part. Attention has only 2 KV heads in QFN, so it splits 1:1 whatever f is."""
    cut = lambda n, step=1: max(step, min(n - step, int(round(n * (1 - f) / step)) * step))
    K, KV, S, V = cfg.linear_num_key_heads, cfg.num_key_value_heads, cfg.shared_expert_intermediate_size, cfg.vocab_size
    a = dict(lin=(0, cut(K)), att=(0, cut(KV)), shared=(0, cut(S, 16)), lm=(0, cut(V, 32)))
    b = dict(lin=(a["lin"][1], K), att=(a["att"][1], KV), shared=(a["shared"][1], S), lm=(a["lm"][1], V))
    return a, b


class TP:
    def __init__(self, ops_card, ops_gpu, w, cfg, host, f=0.41, n_card=10_000, n_gpu=5_000, budget=25, cache=None,
                 **fmt):
        pa, pb = split_parts(cfg, f)
        (wa, ca, _), (wb, cb, self.v_gpu) = shard(w, cfg, pa), shard(w, cfg, pb)
        self.A = qfn.Card(ops_card, wa, ca, host, slots=n_card, **fmt)
        self.B = qfn.Card(ops_gpu, wb, cb, host, slots=n_gpu, **fmt)
        self.o, self.w, self.c, self.host = ops_card, w, cfg, host
        self.cache, self.n_card = cache or qfn.PyCache(n_card + n_gpu, budget), n_card
        self.slots = {}                                  # (layer, expert) -> (device, its slot there)
        self.hits = self.lookups = 0
        self.routes = self.A.routes

    t = property(lambda self: self.A.t)
    tokens = property(lambda self: self.A.tokens)
    h4 = property(lambda self: self.A.h4)
    verified = property(lambda self: self.A.verified)

    def to(self, dst, x, src):
        return dst.o.put(src.o.get(x))                   # through host memory: no peer-to-peer between the two

    def allreduce(self, a, b):
        return self.A.o.add(a, self.to(self.A, b, self.B))

    def moe(self, L, x):
        A, B = self.A, self.B
        probs, ya = A.dense_mlp(L, x)
        top, gate = A.route(L, probs)
        xb = self.to(B, x, A)
        _, yb = B.dense_mlp(L, xb, router=False)
        es = sorted(set(top.ravel().tolist()))
        cold, hot = [], {A: [], B: []}
        for e, slot in zip(es, self.cache.lookup([(L, e) for e in es])):
            n = int((gate[:, e] > 0).sum())
            self.lookups += n
            if slot < 0:
                cold.append(e)
                continue
            self.hits += n
            dev, local = self.slots[L, e]
            hot[dev].append((local, gate[:, e]))
        if hot[A]:
            ya = A.o.add(ya, A.slot_experts(x, hot[A]))
        if hot[B]:
            yb = B.o.add(yb, B.slot_experts(xb, hot[B]))
        y = self.allreduce(ya, yb)
        if cold:
            y = A.o.add(y, A.o.put(self.host.experts(L, A.o.get(x), cold, gate[:, cold].T)))
        return y

    def layers(self, h4, tokens=()):
        A, B, c = self.A, self.B, self.c
        A.pending, B.pending = {}, {}
        B.t = A.t
        for L in range(c.num_hidden_layers):
            if L in A.ple:
                h4 = A.ple_layer(L, h4, list(tokens))
            x, inj = A.hc_read(f"layers.{L}.attn_hyper_connection", h4)
            xb = self.to(B, x, A)
            if c.layer_types[L] == "linear_attention":
                o = self.allreduce(A.deltanet(L, x), B.deltanet(L, xb))
            else:
                o = self.allreduce(A.attention(L, x), B.attention(L, xb))
            h4 = A.hc_write(h4, inj, o)
            x, inj = A.hc_read(f"layers.{L}.mlp_hyper_connection", h4)
            h4 = A.hc_write(h4, inj, self.moe(L, x))
        return h4

    def verify(self, tokens):
        tokens = [int(t) for t in tokens]
        h4 = self.A.o.put(np.stack([np.tile(self.host.embed(t), self.A.hc) for t in tokens]))
        self.A.h4 = self.layers(h4, tokens)
        self.A.verified = self.B.verified = tokens
        return self.A.hc_read("hyper_connection_mixer", self.A.h4, combine=False)

    def commit(self, j):
        self.A.commit(j)                                # states, tokens and position; A's own cache is unused
        self.B.commit(j)
        for (L, e), ev, slot in self.cache.end_step():
            self.slots.pop(ev, None)
            dev, local = (self.A, slot) if slot < self.n_card else (self.B, slot - self.n_card)
            dev.put_slot(local, L, e)
            self.slots[L, e] = (dev, local)

    def lm_argmax(self, mixed):
        """Each device takes the best of its vocabulary rows; the host keeps the better of the two."""
        A, B = self.A, self.B
        va, ia = A.o.topk(A.o.linear(mixed, A.lin("lm_head")), 1)
        vb, ib = B.o.topk(B.o.linear(self.to(B, mixed, A), B.lin("lm_head")), 1)
        va, vb = A.o.get(va)[:, 0], B.o.get(vb)[:, 0]
        return np.where(vb > va, ib[:, 0] + self.v_gpu, ia[:, 0])

    generate = qfn.Card.generate
