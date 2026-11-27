"""DeepSeek-V4.1-Flash-1017's FP4 routed experts in the formats tuff's devices read: K200 INT4 / INT8, TT BFP4 / BFP8.

    python ds_fp4.py [--layers 0,10,20,30,39] [--experts 8]     -> results/ds_fp4.md

The checkpoint's experts are the reference (the model was trained with FP4 experts): e2m1 values {0, .5, 1, 1.5, 2, 3,
4, 6} x sign, two per int8 byte (low nibble first), times an e8m0 power-of-two scale per 32 inputs. A 4-bit integer grid
with the block scale as its step loses exactly the .5 and 1.5 levels (both are ties); a per-16 step (= TT BFP4 with the
shared exponent along the input dim) halves the step where the 16 values allow it. 8-bit formats hold FP4 exactly.
Reads only the sampled experts over HTTP range requests (~19 MB each), with loader.py's reader.
"""
import argparse

import numpy as np

import loader

loader.REPO = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/"
FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
EXP = "layers.{}.ffn.experts.{}.{}"
E, D, LIMIT = 384, 5120, 10.0          # routed experts, hidden size, swiglu_limit


def fp4(src, L, e, w):
    """(v, S): FP4 values [out, in] and the block scale broadcast to [out, in]; the weight is v * S."""
    b, dt, shape = src.raw(EXP.format(L, e, w + ".weight"))
    assert dt == "I8", dt
    b = np.frombuffer(b, np.uint8).reshape(shape)
    v = FP4[np.stack([b & 15, b >> 4], -1).reshape(shape[0], -1)]
    s, dt, shape = src.raw(EXP.format(L, e, w + ".scale"))
    assert dt == "F8_E8M0", dt
    s = np.ldexp(np.float32(1), np.frombuffer(s, np.uint8).reshape(shape).astype(np.int32) - 127)
    return v, np.repeat(s, 32, axis=1)


def int4(v, S, ties):
    """Signed 4-bit grid, step = the FP4 block scale. Ties (.5, 1.5) to even, or alternating up/down by position."""
    if ties == "even":
        return np.clip(np.round(v), -7, 7) * S
    m = np.abs(v)
    tie = m % 1 == 0.5
    alt = (np.arange(v.shape[0])[:, None] + np.arange(v.shape[1])[None, :]) % 2
    return np.sign(v) * np.where(tie, np.floor(m) + alt, m) * S


FORMATS = {   # name -> W [out, in] float32 from (v, S)
    "INT4, ties to even": lambda v, S: int4(v, S, "even"),
    "INT4, ties alternating": lambda v, S: int4(v, S, "alt"),
    "BFP4, exponent along input (= INT4, per-16 step)": lambda v, S: loader.dequant((v * S).T.copy(), "bfp4").T,
    "BFP4, exponent along output (TT default tiles)": lambda v, S: loader.dequant(v * S, "bfp4"),
    "INT8 / BFP8 along input": lambda v, S: loader.dequant((v * S).T.copy(), "bfp8").T,
}


def ffn(w1, w2, w3, x):
    gate, up = np.minimum(x @ w1.T, LIMIT), np.clip(x @ w3.T, -LIMIT, LIMIT)
    return (gate / (1 + np.exp(-gate)) * up) @ w2.T


def sqnr(ref, x):
    err = ((ref.astype(np.float64) - x) ** 2).sum()
    return np.inf if err == 0 else 10 * np.log10((ref.astype(np.float64) ** 2).sum() / err)


def measure(src, layers, n_exp, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((64, D)).astype(np.float32)   # ponytail: Gaussian inputs, as QFN's results/expert_accuracy.md
    rows = []
    for L in layers:
        for e in sorted(rng.choice(E, n_exp, replace=False)):
            vs = {w: fp4(src, L, e, w) for w in ("w1", "w2", "w3")}
            ref_w = {w: v * S for w, (v, S) in vs.items()}
            ref = ffn(ref_w["w1"], ref_w["w2"], ref_w["w3"], x)
            allv = np.concatenate([v.ravel() for v, _ in vs.values()])
            row = dict(L=L, e=int(e), lossy=float(np.isin(np.abs(allv), (.5, 1.5)).mean()))
            for name, f in FORMATS.items():
                q = {w: f(*vs[w]) for w in vs}
                row[name] = (sqnr(np.concatenate([r.ravel() for r in ref_w.values()]), np.concatenate([q[w].ravel() for w in vs])),
                             sqnr(ref, ffn(q["w1"], q["w2"], q["w3"], x)))
            print({k: (tuple(round(t, 1) for t in v) if isinstance(v, tuple) else v) for k, v in row.items()}, flush=True)
            rows.append(row)
    return rows


def report(rows, layers, n_exp):
    fmt = lambda a: "exact" if np.min(a) > 100 else f"{np.mean(a):.1f} (min {np.min(a):.1f})"   # >100 dB: fp32 matmul noise
    out = ["## DeepSeek-V4.1-Flash-1017 FP4 experts in K200 / TT formats (real weights)\n",
           f"{n_exp} random routed experts (seed 0) from each of layers {', '.join(map(str, layers))}, read from the "
           "checkpoint with HTTP range requests. The reference is the checkpoint's own FP4 expert. SQNR in dB, higher is "
           "better: weights, and the SwiGLU expert's output (with the model's swiglu_limit clamps) on 64 Gaussian inputs. "
           "For scale: QFN's BF16 experts in BFP4 gave 14.2 dB output SQNR and cost KLD 0.05-0.09 / 94-95% same top-1 "
           "(results/expert_accuracy.md, results/trace.md).\n",
           f"Share of FP4 weights at ±0.5 or ±1.5 (the levels a fixed INT4 grid loses): "
           f"{np.mean([r['lossy'] for r in rows]):.1%}.\n",
           "| Format | " + " | ".join(f"Layer {L}" for L in layers) + " | **All** |",
           "|---|" + "---:|" * (len(layers) + 1)]
    for name in FORMATS:
        for i, what in ((0, "weights"), (1, "output")):
            cells = [fmt([r[name][i] for r in rows if r["L"] == L]) for L in layers] + [fmt([r[name][i] for r in rows])]
            out.append(f"| {name}, {what} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,10,20,30,39")
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--out", default="results/ds_fp4.md")
    a = ap.parse_args()
    layers = [int(s) for s in a.layers.split(",")]
    rows = measure(loader.Source("hf"), layers, a.experts)
    text = report(rows, layers, a.experts)
    open(a.out, "w", encoding="utf-8").write(text)
    print(text)
