"""ref.py on the real Qwen3.8-Flash-Next weights, streamed from Hugging Face like trace.py (nothing written to disk).

    python ref_real.py [--tokens 32]     # -> results/ref_real.md

Runs Ref.run() on the first tokens of the wikitext text that trace.py recorded (results/routes.npz), one layer at a
time: the layer's dense weights are fetched, the experts its tokens route to are prefetched, and both are dropped
after the layer. Then it compares, layer by layer, the experts ref.py routes each token to with the ones transformers
picked in trace.py's bf16 run. Routing in layer L depends on everything before it (DeltaNet, attention + indexer, PLE,
hyper-connections, MoE), so agreement through layer 47 checks the whole recurrence on the real model; the final
hidden state through the LM head gives the next-token loss on top.
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import ref
from loader import Source

CK = "model.language_model."
ROOT = Path(__file__).parent


def as_float(data, dt, shape):
    if dt == "BF16":
        return (np.frombuffer(data, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape).astype(np.float64)
    return np.frombuffer(data, {"F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32}[dt]).reshape(shape)


class Experts:
    """One layer's fused expert tensor, by expert: kept as bf16, widened to float64 on use."""

    def __init__(self, src, name):
        self.src, self.name, self.bf = src, name, {}

    def prefetch(self, es):
        es = [e for e in es if e not in self.bf]
        with ThreadPoolExecutor(32) as ex:
            for e, a in zip(es, ex.map(lambda e: self.src.expert_bf16(self.name, e), es)):
                self.bf[e] = a

    def __getitem__(self, e):
        if e not in self.bf:
            self.prefetch([e])
        return (self.bf[e].astype(np.uint32) << 16).view(np.float32).astype(np.float64)


class Rows:
    """Rows of a 2-D BF16 table (the token embedding, or the N-gram table split over shard_i tensors)."""

    def __init__(self, src, names):
        self.src, self.parts, r = src, [], 0
        for n in names:
            f, start, t = src.info(n)
            self.parts.append((r, t["shape"][0], f, start, t["shape"][1]))
            r += t["shape"][0]

    def row(self, i):
        for r0, n, f, start, dim in self.parts:
            if r0 <= i < r0 + n:
                data = self.src._get(f, start + (i - r0) * dim * 2, start + (i - r0 + 1) * dim * 2 - 1)
                return as_float(data, "BF16", (dim,))
        raise IndexError(i)

    def __getitem__(self, ids):
        if np.ndim(ids) == 0:
            return self.row(int(ids))
        with ThreadPoolExecutor(16) as ex:
            return np.stack(list(ex.map(self.row, [int(i) for i in ids])))


class Checkpoint:
    """Mapping from ref.py's names to the checkpoint's tensors, fetched on first use."""

    def __init__(self, src, cfg, prefix=CK):
        from transformers.models.qwen4_exp import modeling_qwen4_exp as M
        self.src, self.cache, self.prefix = src, {}, prefix
        heads = (cfg.ngram_size - 1) * cfg.heads_per_ngram
        for i, L in enumerate(cfg.ple_layer_ids):                         # not stored: rebuilt as transformers does
            sizes = [M._find_nth_prime_after(cfg.ngram_vocab_size_base - 1, i * heads + h + 1) for h in range(heads)]
            p = f"layers.{L - 1}.ple.ple_embedding."
            self.cache[p + "ngram_heads_vocab_sizes"] = np.array(sizes)
            self.cache[p + "ngram_heads_offsets"] = np.concatenate([[0], np.cumsum(sizes)[:-1]])

    def __getitem__(self, name):
        if name not in self.cache:
            if name.endswith(("experts.gate_up_proj", "experts.down_proj")):
                v = Experts(self.src, self.prefix + name)
            elif name == "embed_tokens.weight":
                v = Rows(self.src, [CK + name])                               # the MTP layer shares it
            elif name.endswith("ngram_embedding.weight"):
                pre = self.prefix + name[:-len(".weight")]
                v = Rows(self.src, [f"{pre}.shard_{i}.weight" for i in range(10_000) if f"{pre}.shard_{i}.weight" in self.src.map])
            else:
                v = as_float(*self.src.raw(self.prefix + name))
            self.cache[name] = v
        return self.cache[name]

    def drop(self, L):
        for k in [k for k in self.cache if k.startswith(f"layers.{L}.") and not k.endswith(("_sizes", "_offsets"))]:
            del self.cache[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=32)
    a = ap.parse_args()
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3.8-Flash-Next").text_config
    z = np.load(ROOT / "results" / "routes.npz")
    chunks = [str(c) for c in z["chunks"]]
    c = chunks.index("wikitext")
    T = a.tokens
    ids = z["ids"][c][:T]
    want = z["routes"][:, 0, c, :T].astype(np.int64)                     # bf16 run: [L, T, k]
    src = Source("hf")
    W = Checkpoint(src, cfg)
    r = ref.Ref(W, cfg)
    t0, rows = time.time(), []

    def before_mlp(L, xs):
        gate = W[f"layers.{L}.mlp.gate.weight"]
        es = sorted({int(e) for x in xs for e in np.argsort(-(gate @ x))[:cfg.num_experts_per_tok + 2]})
        W[f"layers.{L}.mlp.experts.gate_up_proj"].prefetch(es)
        W[f"layers.{L}.mlp.experts.down_proj"].prefetch(es)

    def after_layer(L):
        agree = np.mean([len(set(r.routes[L, t].tolist()) & set(want[L, t].tolist())) / cfg.num_experts_per_tok
                         for t in range(T)])
        rows.append((L, cfg.layer_types[L], agree))
        W.drop(L)
        print(f"layer {L:2d} ({cfg.layer_types[L]}): routing agrees with transformers on {agree:.1%} "
              f"({time.time() - t0:.0f} s)", flush=True)

    H = np.stack(r.run(ids, before_mlp, after_layer))
    # next-token loss through the LM head (bf16 -> float32 in row chunks)
    data, dt, shape = src.raw("lm_head.weight")
    u = np.frombuffer(data, np.uint16).reshape(shape)
    logits = np.concatenate([H.astype(np.float32) @ (u[i:i + 16384].astype(np.uint32) << 16).view(np.float32).T
                             for i in range(0, shape[0], 16384)], 1).astype(np.float64)
    lp = logits - logits.max(1, keepdims=True)
    lp -= np.log(np.exp(lp).sum(1, keepdims=True))
    nll = -lp[np.arange(T - 1), ids[1:T]].mean()
    main_next = logits.argmax(1)                                          # greedy next token after each position

    # MTP: QFN's drafter guesses the token after next from the final residual and the next token. Its layout follows
    # Basalt's implementation (MIT); transformers has none. Not settled there: whether pre_fc_norm_hidden normalizes
    # the 4 streams together or one by one, so both run. Acceptance = the draft equals the main model's greedy pick.
    def head_argmax(X):
        best = np.full(len(X), -np.inf), np.zeros(len(X), np.int64)
        for i in range(0, shape[0], 16384):
            lg = X.astype(np.float32) @ (u[i:i + 16384].astype(np.uint32) << 16).view(np.float32).T
            j = lg.argmax(1)
            m = lg[np.arange(len(X)), j] > best[0]
            best[0][m], best[1][m] = lg[np.arange(len(X)), j][m], i + j[m]
        return best[1]

    import copy
    mcfg = copy.deepcopy(cfg)
    mcfg.num_hidden_layers, mcfg.layer_types, mcfg.ple_layer_ids = 1, ["full_attention"], []
    MW = Checkpoint(src, mcfg, prefix="mtp.")
    MW.cache["embed_tokens.weight"] = W["embed_tokens.weight"]
    d, hc = cfg.hidden_size, cfg.hc_count
    accept = {}
    for scope in ("all streams", "per stream"):
        m = ref.Ref(MW, mcfg)
        drafts = []
        for t in range(T - 1):
            Rt = r.h4[t]
            wh = MW["pre_fc_norm_hidden.weight"]
            hn = ref.rms(Rt, wh, cfg.rms_norm_eps, None if scope == "all streams" else d)
            h2 = np.stack([MW["fc_hidden.weight"] @ hn[s * d:(s + 1) * d] for s in range(hc)])
            e2 = MW["fc_embedding.weight"] @ ref.rms(W["embed_tokens.weight"][int(ids[t + 1])], MW["pre_fc_norm_embedding.weight"], cfg.rms_norm_eps)
            m.t, m.pos0, m.tokens = t, 1, list(ids[:t + 2])               # MTP cache entry t = position t + 1
            x = (h2 + e2[None]).ravel()
            x = m.mlp_block(0, m.attn_block(0, x))
            drafts.append(m.gated_residual("hyper_connection_mixer", x, combine=False))
        guess = head_argmax(np.stack(drafts))                                 # draft for position t + 2
        accept[scope] = (np.mean(guess[:T - 2] == main_next[1:T - 1]), np.mean(guess[:T - 2] == ids[2:T]))
        print(f"MTP ({scope}): draft == main model's greedy token {accept[scope][0]:.1%}, == actual text {accept[scope][1]:.1%}", flush=True)
    overall = np.mean([x for _, _, x in rows])
    out = ["## ref.py on the real Qwen3.8-Flash-Next weights\n",
           f"First {T} tokens of the wikitext text in [`results/routes.npz`](routes.npz), every layer, float64. "
           "Routing agreement = share of each token's 10 routed experts that ref.py picks the same as transformers "
           f"did in trace.py's bf16 run. Next-token loss over the {T - 1} predictions: {nll:.3f} nats "
           f"(perplexity {np.exp(nll):.2f}; short context). Overall routing agreement {overall:.1%}. "
           f"Wall time {(time.time() - t0) / 60:.0f} min.\n",
           "MTP drafter (one draft token, greedy), with pre_fc_norm_hidden over " + "; ".join(
               f"{k}: matches the main model's next token {v[0]:.1%}, the text {v[1]:.1%}" for k, v in accept.items())
           + f" (over {T - 2} positions).\n",
           "| Layer | Type | Routing agreement |", "|---:|---|---:|"]
    out += [f"| {L} | {k} | {x:.1%} |" for L, k, x in rows]
    text = "\n".join(out) + "\n"
    (ROOT / "results" / "ref_real.md").write_text(text, encoding="utf-8")
    print(text)
    # transformers ran in float32 from bf16 weights, ref.py in float64: near-ties at the 10th expert may flip
    assert overall > 0.95, overall


if __name__ == "__main__":
    main()
