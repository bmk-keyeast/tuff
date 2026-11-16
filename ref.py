"""Qwen3.8-Flash-Next decode, one token at a time, in plain numpy: the specification tuff's device path implements.

transformers runs QFN as a batched prefill (chunked delta rule, masks over the whole sequence). An inference engine
runs it as a recurrence: per token, per layer, a fixed amount of state is read and updated. This file writes that
recurrence out with nothing hidden in a framework, in float64, keyed by the checkpoint's own tensor names (without the
"model.language_model." prefix), so the same code runs on real weights:

    state per Gated DeltaNet layer   conv window (last 3 inputs of the 10240 q/k/v channels), recurrent state S
                                     [48 value heads][128][128]
    state per full-attention layer   K, V cache (2 KV heads), the QSA indexer's raw key per token
    state of the PLE layer           the last 2 tokens (n-gram hashing), the dilated conv's last 9 inputs
    hidden state between layers      4 hyper-connection streams of 2560

    python ref.py     # checks Ref against transformers' Qwen4ExpTextModel (tiny random model, every token)

Ref.step(token) returns the hidden state after the final hyper-connection mixer (QFN has no final norm; the LM head
multiplies it directly).
"""
import math

import numpy as np

MASK64 = (1 << 64) - 1


def silu(x):
    return x / (1 + np.exp(-x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softplus(x):
    return np.logaddexp(0, x)


def rms(x, w, eps, group=None):
    """Qwen4ExpTextRMSNorm: x / rms(x) * (1 + w), per group of `group` features (per stream / per head)."""
    shape = x.shape
    g = x.reshape(-1, group or shape[-1])
    g = g / np.sqrt((g * g).mean(-1, keepdims=True) + eps)
    return g.reshape(shape) * (1 + w)


def i64(v):
    """torch int64 arithmetic wraps: keep v as a signed 64-bit value."""
    v &= MASK64
    return v - (1 << 64) if v >> 63 else v


class Ref:
    def __init__(self, w, cfg):
        """w: name -> numpy array, names as in the checkpoint after "model.language_model." (e.g. "layers.0.mlp.gate.weight")."""
        self.w = {k: np.asarray(v, np.float64) if np.asarray(v).dtype.kind == "f" else np.asarray(v) for k, v in w.items()}
        self.c = cfg
        c = cfg
        self.d, self.hc, self.eps = c.hidden_size, c.hc_count, c.rms_norm_eps
        rp = c.rope_parameters
        self.rd = int(c.head_dim * rp.get("partial_rotary_factor", 1.0))
        self.inv_freq = 1.0 / (rp["rope_theta"] ** (np.arange(0, self.rd, 2) / self.rd))
        self.gate_act = sigmoid if (c.output_gate_type or c.hidden_act) == "sigmoid" else silu
        self.eos = c.eos_token_id[0] if isinstance(c.eos_token_id, list) else c.eos_token_id
        self.t = 0
        self.tokens = []
        self.state = {}
        for L, kind in enumerate(c.layer_types[:c.num_hidden_layers]):
            if kind == "linear_attention":
                C = 2 * c.linear_num_key_heads * c.linear_key_head_dim + c.linear_num_value_heads * c.linear_value_head_dim
                self.state[L] = dict(conv=np.zeros((C, c.linear_conv_kernel_dim - 1)),
                                     S=np.zeros((c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim)))
            else:
                self.state[L] = dict(K=[], V=[], idx=[])
        self.ple = {L - 1: i for i, L in enumerate(c.ple_layer_ids)}             # PLE runs before layer L-1
        self.ple_conv = {L: np.zeros((self.hc * self.d, (c.ple_conv_kernel_size - 1) * c.ngram_size)) for L in self.ple}

    def lin(self, name, x):
        return self.w[name + ".weight"] @ x

    def rope(self, x, pos):
        f = pos * self.inv_freq
        cos, sin = np.concatenate([np.cos(f)] * 2), np.concatenate([np.sin(f)] * 2)
        r, rest = x[..., :self.rd], x[..., self.rd:]
        half = self.rd // 2
        rot = np.concatenate([-r[..., half:], r[..., :half]], -1)
        return np.concatenate([r * cos + rot * sin, rest], -1)

    # ---- hyper-connections: 4 residual streams, mixed into one block input, the block output injected back
    def gated_residual(self, p, h4, combine=True):
        hc, d = self.hc, self.d
        hn = rms(h4, self.w[p + ".hc_norm.weight"], self.eps, d)
        mix = sigmoid(self.lin(p + ".input_mix_weight_up", silu(self.lin(p + ".input_mix_weight_down", hn) / hc)))
        mixed = (mix.reshape(hc, d) * hn.reshape(hc, d)).mean(0)
        if not combine:
            return mixed
        return mixed, 2 * sigmoid(self.lin(p + ".block_inject_weight", hn) / hc)

    # ---- Gated DeltaNet, recurrent form
    def deltanet(self, L, x):
        c, s, p = self.c, self.state[L], f"layers.{L}.linear_attn"
        K, V, dk, dv = c.linear_num_key_heads, c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim
        qkv = self.lin(p + ".in_proj_qkv", x)
        win = np.concatenate([s["conv"], qkv[:, None]], 1)                        # oldest .. current
        s["conv"] = win[:, 1:]
        qkv = silu((self.w[p + ".conv1d.weight"][:, 0, :] * win).sum(1))
        q, k, v = qkv[:K * dk].reshape(K, dk), qkv[K * dk:2 * K * dk].reshape(K, dk), qkv[2 * K * dk:].reshape(V, dv)
        z = self.lin(p + ".in_proj_z", x).reshape(V, dv)
        beta = sigmoid(self.lin(p + ".in_proj_b", x))
        g = -np.exp(self.w[p + ".A_log"]) * softplus(self.lin(p + ".in_proj_a", x) + self.w[p + ".dt_bias"])
        l2 = lambda t: t / np.sqrt((t * t).sum(-1, keepdims=True) + 1e-6)
        q, k = np.repeat(l2(q), V // K, 0) / math.sqrt(dk), np.repeat(l2(k), V // K, 0)   # key head i serves V/K value heads
        S = s["S"] * np.exp(g)[:, None, None]
        delta = (v - np.einsum("hkv,hk->hv", S, k)) * beta[:, None]
        S = S + k[:, :, None] * delta[:, None, :]
        s["S"] = S
        o = np.einsum("hkv,hk->hv", S, q)
        o = o / np.sqrt((o * o).mean(-1, keepdims=True) + self.eps) * self.w[p + ".norm.weight"] * self.gate_act(z)
        return self.lin(p + ".out_proj", o.ravel())

    # ---- full attention with the QSA indexer (sparse: the top blocks of keys plus the incomplete tail)
    def attention(self, L, x):
        c, s, p = self.c, self.state[L], f"layers.{L}.self_attn"
        t, H, KV, hd = self.t, c.num_attention_heads, c.num_key_value_heads, c.head_dim
        nh, ihd, ratio = c.indexer_n_heads, c.indexer_head_dim, c.indexer_compress_ratio
        qk = self.lin(p + ".indexer.index_qk_proj", x)
        iq, s_key = qk[:nh * ihd].reshape(nh, ihd), qk[nh * ihd:]
        s["idx"].append(s_key)                                                 # raw: no norm, no rope
        iq = self.rope(rms(iq, self.w[p + ".indexer.q_layernorm.weight"], self.eps), t)
        nb = (t + 1) // ratio
        keep = list(range(nb * ratio, t + 1))                                  # incomplete block: always kept
        if nb:
            blocks = np.array(s["idx"][:nb * ratio]).reshape(nb, ratio, ihd).mean(1)
            bk = np.stack([self.rope(b, i * ratio) for i, b in
                           enumerate(rms(blocks, self.w[p + ".indexer.k_layernorm.weight"], self.eps))])
            score = np.maximum(iq @ bk.T, 0).sum(0) / math.sqrt(ihd)
            top = np.argsort(-score, kind="stable")[:min(c.indexer_budget // ratio, nb)]
            keep = sorted(i * ratio + r for i in top for r in range(ratio)) + keep
        qg = self.lin(p + ".q_proj", x).reshape(H, 2 * hd)
        q, gate = qg[:, :hd], qg[:, hd:]
        q = self.rope(rms(q, self.w[p + ".q_norm.weight"], self.eps), t)
        k = self.rope(rms(self.lin(p + ".k_proj", x).reshape(KV, hd), self.w[p + ".k_norm.weight"], self.eps), t)
        s["K"].append(k)
        s["V"].append(self.lin(p + ".v_proj", x).reshape(KV, hd))
        Kc, Vc = np.array(s["K"])[keep], np.array(s["V"])[keep]               # [n, KV, hd]
        out = np.empty((H, hd))
        for h in range(H):
            kv = h // (H // KV)
            a = Kc[:, kv] @ q[h] / math.sqrt(hd)
            a = np.exp(a - a.max())
            out[h] = (a / a.sum()) @ Vc[:, kv]
        return self.lin(p + ".o_proj", (out * sigmoid(gate)).ravel())

    # ---- MoE: top-k of the routed experts (renormalized) + the gated shared expert
    def moe(self, L, x):
        c, p = self.c, f"layers.{L}.mlp"
        logits = self.lin(p + ".gate", x)
        pr = np.exp(logits - logits.max())
        pr /= pr.sum()
        top = np.argsort(-pr, kind="stable")[:c.num_experts_per_tok]
        wt = pr[top] / pr[top].sum() if c.norm_topk_prob else pr[top]
        y = np.zeros(self.d)
        for e, we in zip(top, wt):
            g, u = np.split(self.w[p + ".experts.gate_up_proj"][e] @ x, 2)
            y += we * (self.w[p + ".experts.down_proj"][e] @ (silu(g) * u))
        sh = self.lin(p + ".shared_expert.down_proj", silu(self.lin(p + ".shared_expert.gate_proj", x))
                      * self.lin(p + ".shared_expert.up_proj", x))
        return y + sigmoid(self.lin(p + ".shared_expert_gate", x)[0]) * sh

    # ---- PLE: hashed n-gram embeddings gate a value into every stream, plus a dilated depthwise conv
    def ngram_ids(self, L):
        c, p = self.c, f"layers.{L}.ple.ple_embedding"
        hist = [self.eos] * (c.ngram_size - 1) + self.tokens                  # no eos inside the text assumed
        sh = [hist[-1 - s] for s in range(c.ngram_size)]                       # token, previous, the one before
        mult = [int(m) for m in self.w[p + ".layer_multipliers"]]
        sizes, offs = self.w[p + ".ngram_heads_vocab_sizes"], self.w[p + ".ngram_heads_offsets"]
        ids = []
        for n in range(2, c.ngram_size + 1):
            mixed = i64(sh[0] * mult[0])
            for pos in range(1, n):
                mixed = i64(mixed ^ i64(sh[pos] * mult[pos]))
            h0 = (n - 2) * c.heads_per_ngram
            ids += [mixed % int(sizes[h]) + int(offs[h]) for h in range(h0, h0 + c.heads_per_ngram)]
        return ids

    def ple_layer(self, L, h4):
        c, p, hc, d = self.c, f"layers.{L}.ple", self.hc, self.d
        emb = self.w[p + ".ple_embedding.ngram_embedding.weight"][self.ngram_ids(L)].ravel()
        key = rms(self.lin(p + ".key_proj", emb), self.w[p + ".norm_key.weight"], self.eps, d).reshape(hc, d)
        value = self.lin(p + ".value_proj", emb)
        query = rms(h4, self.w[p + ".norm_query.weight"], self.eps, d).reshape(hc, d)
        gate = (key * query).sum(-1) / math.sqrt(d)
        gate = np.sqrt(np.maximum(np.abs(gate), 1e-6)) * np.sign(gate)
        gv = (sigmoid(gate)[:, None] * value[None, :]).ravel()
        gvn = rms(gv, self.w[p + ".norm_conv.weight"], self.eps, d)
        win = np.concatenate([self.ple_conv[L], gvn[:, None]], 1)              # last 9 inputs + current
        self.ple_conv[L] = win[:, 1:]
        taps = win[:, ::c.ngram_size]                                          # dilation = ngram_size: t-9, t-6, t-3, t
        return gv + silu((self.w[p + ".conv1d.weight"][:, 0, :] * taps).sum(1))

    def step(self, token):
        self.tokens.append(int(token))
        h4 = np.tile(self.w["embed_tokens.weight"][token], self.hc)
        for L, kind in enumerate(self.c.layer_types[:self.c.num_hidden_layers]):
            if L in self.ple:
                h4 = h4 + self.ple_layer(L, h4)
            x, inj = self.gated_residual(f"layers.{L}.attn_hyper_connection", h4)
            o = self.deltanet(L, x) if kind == "linear_attention" else self.attention(L, x)
            h4 = h4 + (inj[:, None] * o[None, :]).ravel()
            x, inj = self.gated_residual(f"layers.{L}.mlp_hyper_connection", h4)
            h4 = h4 + (inj[:, None] * self.moe(L, x)[None, :]).ravel()
        self.t += 1
        return self.gated_residual("hyper_connection_mixer", h4, combine=False)


def tiny_config():
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3.8-Flash-Next").text_config
    for k, v in dict(hidden_size=128, num_hidden_layers=8, head_dim=32, num_attention_heads=4, num_key_value_heads=2,
                     indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=16, indexer_budget=16,
                     indexer_compress_ratio=4, linear_num_key_heads=4, linear_num_value_heads=12,
                     linear_key_head_dim=16, linear_value_head_dim=16, num_experts=16, num_experts_per_tok=4,
                     moe_intermediate_size=32, shared_expert_intermediate_size=32, hc_lowrank=16, vocab_size=1000,
                     ngram_vocab_size_base=1000, ple_embed_dim=128, make_ngram_vocab_size_divisible_by=8).items():
        setattr(cfg, k, v)
    cfg.layer_types = cfg.layer_types[:8]
    cfg.rope_parameters = dict(cfg.rope_parameters, mrope_section=[2, 1, 1])
    cfg._attn_implementation = "sdpa"
    return cfg


def check(T=40, seed=0):
    import torch
    from transformers.models.qwen4_exp import modeling_qwen4_exp as M
    torch.manual_seed(seed)
    cfg = tiny_config()
    model = M.Qwen4ExpTextModel(cfg).eval()
    g = torch.Generator().manual_seed(seed)
    for n, p in model.named_parameters():                                     # beyond the zero-init norms
        if n.endswith("A_log"):
            p.data = torch.log(torch.rand(p.shape, generator=g) * 15 + 1)
        else:
            p.data = (0.3 if p.ndim == 1 else 1 / math.sqrt(p.shape[-1])) * torch.randn(p.shape, generator=g)
    ids = torch.randint(0, cfg.vocab_size, (1, T), generator=g)
    with torch.no_grad():
        want = model(input_ids=ids, use_cache=False).last_hidden_state[0].double().numpy()
    ref = Ref({k: v.numpy() for k, v in model.state_dict().items()}, cfg)
    got = np.stack([ref.step(int(t)) for t in ids[0]])
    err = np.abs(got - want).max(1) / np.abs(want).max(1)
    print(f"ref.py decode == transformers prefill: {T} tokens, {cfg.num_hidden_layers} layers "
          f"({cfg.layer_types.count('linear_attention')} DeltaNet, PLE before layer {cfg.ple_layer_ids[0] - 1}), "
          f"max relative error {err.max():.1e}")
    assert err.max() < 1e-4, err


if __name__ == "__main__":
    check()
