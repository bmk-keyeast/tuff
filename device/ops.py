"""The operations tuff's card code (device/qfn.py) is written in, with two backends:

    Torch()          float32 on the CPU; weights rounded through the card's formats (TT-exact BFP, bf16). CI runs
                     device/qfn.py on it against ref.py, so the model code is checked before there is a card.
    TTNN(device)     the same operations on a Tenstorrent card through ttnn. Not run yet (no card).
                     device/test_ops.py compares every operation with Torch() on the card; run it first.

Tensors are 2-D [rows, cols] unless an operation says otherwise. A weight W [out, in] (torch layout) is uploaded as
W^T [in, out], so linear(x, w) = x @ W^T, and a BFP block (16 consecutive outputs of one input) is a TT face row.
"""
import numpy as np


def bfp_round(w, fmt):
    """W [out, in] rounded through a TT block-float format exactly as tt-metal packs it (loader.quantize)."""
    import loader
    out, n = w.shape
    if out % 16 or n % 8:                      # not whole blocks (1-, 4- and 48-row gates): TT pads; tuff keeps bf16
        return bf16_round(w)
    return loader.dequant(np.asarray(w, np.float32), fmt).astype(np.float32)


def bf16_round(w):
    u = np.asarray(w, np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000          # round to nearest even, as torch's .bfloat16()
    return u.view(np.float32)


class Torch:
    def __init__(self):
        import torch
        self.t = torch

    # host <-> card
    def put(self, a, fmt="f32"):
        a = np.asarray(a, np.float32)
        a = bf16_round(a) if fmt == "bf16" else a
        return self.t.from_numpy(np.ascontiguousarray(a))

    def put_weight(self, w, fmt):
        """W [out, in] -> W^T [in, out] in `fmt` (f32, bf16, bfp8, bfp4)."""
        w = np.asarray(w, np.float32)
        if fmt in ("bfp8", "bfp4"):
            w = bfp_round(w, fmt)
        elif fmt == "bf16":
            w = bf16_round(w)
        return self.t.from_numpy(np.ascontiguousarray(w.T))

    def get(self, x):
        return x.numpy().astype(np.float64)

    def zeros(self, shape):
        return self.t.zeros(shape)

    # arithmetic (broadcasting like numpy)
    def linear(self, x, w): return x @ w
    def matmul(self, a, b): return a @ b
    def add(self, a, b): return a + b
    def sub(self, a, b): return a - b
    def mul(self, a, b): return a * b
    def scale(self, a, s): return a * float(s)
    def silu(self, a): return self.t.nn.functional.silu(a)
    def sigmoid(self, a): return self.t.sigmoid(a)
    def exp(self, a): return self.t.exp(a)
    def softplus(self, a): return self.t.nn.functional.softplus(a)
    def relu(self, a): return self.t.relu(a)
    def abs(self, a): return a.abs()
    def sign(self, a): return self.t.sign(a)
    def sqrt(self, a): return self.t.sqrt(a)
    def rsqrt(self, a): return self.t.rsqrt(a)
    def clamp_min(self, a, m): return a.clamp_min(m)
    def sum(self, a, dim): return a.sum(dim, keepdim=True)
    def mean(self, a, dim): return a.mean(dim, keepdim=True)
    def softmax(self, a): return self.t.softmax(a, -1)

    # shape
    def reshape(self, a, shape): return a.reshape(shape)
    def slice(self, a, dim, start, end): return a.narrow(dim, start, end - start)
    def concat(self, xs, dim): return self.t.cat(xs, dim)
    def transpose(self, a): return a.transpose(-2, -1)

    def topk(self, a, k):
        """k largest of each row of a [m, n], ties to the lowest index (as ttnn.topk(stable=True)); returns (values
        [m, k] on the card, indices [m, k] on the host)."""
        idx = np.argsort(-a.numpy(), axis=1, kind="stable")[:, :k]
        return self.t.gather(a, 1, self.t.from_numpy(idx)), idx

    def gather_rows(self, table, idx):
        return table[self.t.as_tensor(np.asarray(idx, np.int64))]

    def write_rows(self, table, i, rows):
        table[i:i + rows.shape[0]] = rows
        return table

    def gdr(self, q, k, v, g, beta, state, scale):
        """The gated delta rule over m tokens (ttnn.transformer.fused_recurrent_gated_delta_rule, T = m): q, k [m, H, K]
        (already L2-normed), v [m, HV, V], g, beta [m, HV], state [HV, K, V]; key head i serves value heads
        i*HV/H ... Returns (o [m, HV, V], the state after each token [m, HV, K, V]) -- per-token states let an MTP
        verification keep exactly the accepted tokens."""
        G = v.shape[1] // q.shape[1]
        q, k = q.repeat_interleave(G, 1) * scale, k.repeat_interleave(G, 1)
        S, outs, states = state, [], []
        for i in range(q.shape[0]):
            S = S * self.t.exp(g[i])[:, None, None]
            delta = (v[i] - self.t.einsum("hkv,hk->hv", S, k[i])) * beta[i][:, None]
            S = S + k[i][:, :, None] * delta[:, None, :]
            outs.append(self.t.einsum("hkv,hk->hv", S, q[i]))
            states.append(S)
        return self.t.stack(outs), self.t.stack(states)


class TTNN:
    """ponytail: written against tt-metal's ttnn Python API (docs + nanobind sources, 2026-11) without a card. Generic
    compositions wherever a fused op's layout rules were unclear; swap in fused ops (rms_norm, rotary, sdpa decode,
    moe) once test_ops.py passes and a profile says where the time goes."""

    def __init__(self, device):
        import torch
        import ttnn
        self.ttnn, self.t, self.dev = ttnn, torch, device
        self.dt = {"f32": ttnn.float32, "bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}

    def _to(self, a, dtype):
        return self.ttnn.from_torch(self.t.from_numpy(np.ascontiguousarray(a, np.float32)), dtype=dtype,
                                    layout=self.ttnn.TILE_LAYOUT, device=self.dev)

    def put(self, a, fmt="f32"):
        return self._to(np.asarray(a, np.float32), self.dt[fmt])

    def put_weight(self, w, fmt):
        w = np.asarray(w, np.float32)
        if fmt in ("bfp8", "bfp4") and (w.shape[0] % 16 or w.shape[1] % 8):
            fmt = "bf16"
        return self._to(w.T, self.dt[fmt])

    def get(self, x):
        return self.ttnn.to_torch(x).float().numpy().astype(np.float64)

    def zeros(self, shape):
        return self._to(np.zeros(shape, np.float32), self.ttnn.float32)

    def linear(self, x, w): return self.ttnn.linear(x, w)
    def matmul(self, a, b): return self.ttnn.matmul(a, b)
    def add(self, a, b): return self.ttnn.add(a, b)
    def sub(self, a, b): return self.ttnn.subtract(a, b)
    def mul(self, a, b): return self.ttnn.multiply(a, b)
    def scale(self, a, s): return self.ttnn.multiply(a, float(s))
    def silu(self, a): return self.ttnn.silu(a)
    def sigmoid(self, a): return self.ttnn.sigmoid_accurate(a)
    def exp(self, a): return self.ttnn.exp(a)
    def softplus(self, a): return self.ttnn.softplus(a)
    def relu(self, a): return self.ttnn.relu(a)
    def abs(self, a): return self.ttnn.abs(a)
    def sign(self, a): return self.ttnn.sign(a)
    def sqrt(self, a): return self.ttnn.sqrt(a)
    def rsqrt(self, a): return self.ttnn.rsqrt(a)
    def clamp_min(self, a, m): return self.ttnn.clamp(a, min=m)
    def sum(self, a, dim): return self.ttnn.sum(a, dim=dim, keepdim=True)
    def mean(self, a, dim): return self.ttnn.mean(a, dim=dim, keepdim=True)
    def softmax(self, a): return self.ttnn.softmax(a, dim=-1)
    def reshape(self, a, shape): return self.ttnn.reshape(a, shape)

    def slice(self, a, dim, start, end):
        lo, hi = [0] * len(a.shape), list(a.shape)
        lo[dim], hi[dim] = start, end
        return self.ttnn.slice(a, lo, hi)

    def concat(self, xs, dim): return self.ttnn.concat(xs, dim=dim)
    def transpose(self, a): return self.ttnn.transpose(a, -2, -1)

    def topk(self, a, k):
        vals, idx = self.ttnn.topk(a, k, dim=-1, largest=True, sorted=True, stable=True)
        return vals, self.ttnn.to_torch(idx).numpy().reshape(a.shape[0], -1)[:, :k].astype(np.int64)

    def gather_rows(self, table, idx):
        # ponytail: through the host; ttnn.embedding (row-major table) once the KV cache layout is settled
        return self.put(self.get(table)[np.asarray(idx, np.int64)])

    def write_rows(self, table, i, rows):
        # ponytail: through the host; ttnn.update_cache on a [1, 1, S, D] cache is the on-card version
        t = self.get(table)
        r = self.get(rows)
        t[i:i + r.shape[0]] = r
        return self.put(t)

    def gdr(self, q, k, v, g, beta, state, scale):
        tt = self.ttnn
        m, H, K = q.shape
        HV, V = v.shape[-2], v.shape[-1]
        o, s = tt.transformer.fused_recurrent_gated_delta_rule(
            tt.reshape(q, (1, m, H, K)), tt.reshape(k, (1, m, H, K)), tt.reshape(v, (1, m, HV, V)),
            tt.reshape(g, (1, m, HV)), tt.reshape(beta, (1, m, HV)), scale=scale,
            initial_state=tt.reshape(state, (1, HV, K, V)), output_per_token_state=True)
        return tt.reshape(o, (m, HV, V)), tt.reshape(s, (m, HV, K, V))
