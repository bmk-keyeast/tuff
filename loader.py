"""Qwen3.8-Flash-Next loader: routed experts from the BF16 checkpoint to tuff's host BFP layout.

    python loader.py convert SRC OUT [--fmt bfp4] [--layers 0-47]  # SRC: local snapshot dir, or "hf" (HTTP)
    python loader.py accuracy [--layers 0,9,19,28,38,47] [--experts 8]  # real-weight accuracy -> results/
    python loader.py selftest FILE                                 # packed random matrices for kernels/test_bfp

Packing (rounding and byte layout) is bit-identical to bfp_pack in kernels/bfp.c; CI checks this with selftest.

The checkpoint fuses each layer's experts into two BF16 tensors, torch [out, in] per expert:
    model.language_model.layers.L.mlp.experts.gate_up_proj  [512, 1280, 2560]  rows 0..639 gate, 640..1279 up
    model.language_model.layers.L.mlp.experts.down_proj     [512, 2560, 640]
640 % 16 == 0, so gate and up are row-panel ranges (panels 0..39 and 40..79) of one packed matrix. Gate-before-up
follows the Hugging Face fused-experts convention; it only matters for `accuracy`'s output check, not for packing.

convert writes one file per layer, experts.LL.FMT: for e in 0..511: gate_up exp, gate_up mant, down exp, down mant.
Every piece is a multiple of 64 bytes, so expert e of a memory-mapped file is at e * expert_bytes, 64-byte aligned.
"""
import argparse
import json
import struct
import time
import urllib.request
from pathlib import Path

import numpy as np

REPO = "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/"
MBITS = {"bfp8": 7, "bfp4": 3}
E, D, I = 512, 2560, 640
EXPERT = "model.language_model.layers.{}.mlp.experts.{}"


class Source:
    """Byte ranges of the checkpoint, from a local snapshot directory or over HTTP Range requests."""

    def __init__(self, root="hf"):
        self.root = None if root == "hf" else Path(root)
        self.map = json.loads(self._get("model.safetensors.index.json"))["weight_map"]
        self.headers = {}

    def _get(self, f, start=None, end=None):   # end inclusive
        if self.root:
            with open(self.root / f, "rb") as fh:
                if start is None:
                    return fh.read()
                fh.seek(start)
                return fh.read(end - start + 1)
        req = urllib.request.Request(REPO + f, headers={} if start is None else {"Range": f"bytes={start}-{end}"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = r.read()
                if start is None or len(data) == end - start + 1:
                    return data
            except OSError:
                if attempt == 4:
                    raise
            time.sleep(2 ** attempt)
        raise IOError(f"short read: {f} {start}-{end}")

    def expert(self, name, e):
        """Expert e of a fused [E, out, in] BF16 tensor, as float32 [out, in]."""
        f = self.map[name]
        if f not in self.headers:
            n = struct.unpack("<Q", self._get(f, 0, 7))[0]
            self.headers[f] = (8 + n, json.loads(self._get(f, 8, 7 + n)))
        base, h = self.headers[f]
        t = h[name]
        assert t["dtype"] == "BF16" and len(t["shape"]) == 3, (name, t["dtype"], t["shape"])
        _, out, inn = t["shape"]
        size = out * inn * 2
        start = base + t["data_offsets"][0] + e * size
        u = np.frombuffer(self._get(f, start, start + size - 1), dtype="<u2")
        return (u.astype(np.uint32) << 16).view(np.float32).reshape(out, inn)


def quantize(w, fmt):
    """float32 [out, in] -> (exp uint8 [P, in], signed q [P, 16, in], step float32 [P, in]); bfp_pack's math."""
    mb = MBITS[fmt]
    out, n = w.shape
    assert out % 16 == 0 and n % 8 == 0, w.shape
    w = np.ascontiguousarray(w, np.float32).reshape(out // 16, 16, n)
    a = np.abs(w).max(axis=1)
    ok = (a > 0) & np.isfinite(a)
    e = np.clip(np.frexp(np.where(ok, a, 1))[1] - 1, mb - 126, 127)   # ilogbf, then clamp as in C
    step = np.ldexp(np.ones_like(a), e - (mb - 1)).astype(np.float32)
    q = np.floor(np.abs(w) / step[:, None, :] + np.float32(0.5))       # float32 arithmetic, as in C
    q = np.where(ok[:, None, :], np.minimum(q, (1 << mb) - 1), 0).astype(np.int32)
    return np.where(ok, e + 127, 0).astype(np.uint8), np.where(w < 0, -q, q), step


def dequant(w, fmt):
    _, q, step = quantize(w, fmt)
    return (q * step[:, None, :]).reshape(w.shape)


def pack(w, fmt):
    """float32 [out, in] -> (exp bytes, mant bytes) in the layout documented in kernels/bfp.h."""
    exp, q, _ = quantize(w, fmt)
    b = 8 if fmt == "bfp8" else 4
    c = 32 // b
    P, _, n = q.shape
    f = (q + (1 << (b - 1))).astype(np.uint32)                          # offset binary
    f = f.reshape(P, 2, 8, n // c, c).transpose(0, 3, 1, 2, 4)          # [p][j/c][r/8][r%8][j%c]
    dw = (f << (b * np.arange(c, dtype=np.uint32))).sum(axis=-1, dtype=np.uint32)
    return exp.tobytes(), dw.astype("<u4").tobytes()


def convert(src, out, fmt, layers):
    out.mkdir(parents=True, exist_ok=True)
    sizes = None
    for L in layers:
        t0 = time.time()
        path = out / f"experts.{L:02d}.{fmt}"
        with open(path.with_suffix(".part"), "wb") as fh:
            for e in range(E):
                parts = [p for name in ("gate_up_proj", "down_proj") for p in pack(src.expert(EXPERT.format(L, name), e), fmt)]
                sizes = sizes or [len(p) for p in parts]
                for p in parts:
                    fh.write(p)
        path.with_suffix(".part").replace(path)
        print(f"layer {L}: {path.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.0f} s", flush=True)
    (out / "manifest.json").write_text(json.dumps(dict(
        model="Qwen/Qwen3.8-Flash-Next", fmt=fmt, layers=list(layers), experts=E,
        matrices=[dict(name="gate_up", out=2 * I, inn=D), dict(name="down", out=D, inn=I)],
        piece_bytes=sizes, expert_bytes=sum(sizes), layout="kernels/bfp.h"), indent=1))


def sqnr(ref, x):
    ref, x = ref.astype(np.float64), x.astype(np.float64)
    return 10 * np.log10((ref ** 2).sum() / ((ref - x) ** 2).sum())


def ffn(gu, dn, x):
    h = x @ gu.T
    g, u = h[:, :I], h[:, I:]
    return (g / (1 + np.exp(-g)) * u) @ dn.T


def accuracy(src, layers, n_exp, seed=0):
    rng = np.random.default_rng(seed)
    # ponytail: Gaussian expert inputs; real hidden states need the full forward pass (roadmap: CPU reference)
    x = rng.standard_normal((64, D)).astype(np.float32)
    rows = []
    for L in layers:
        for e in sorted(rng.choice(E, n_exp, replace=False)):
            gu = src.expert(EXPERT.format(L, "gate_up_proj"), e)
            dn = src.expert(EXPERT.format(L, "down_proj"), e)
            ref = ffn(gu, dn, x)
            q = {f: (dequant(gu, f), dequant(dn, f)) for f in MBITS}
            rows.append(dict(
                L=L, e=int(e),
                w4=sqnr(np.concatenate([gu.ravel(), dn.ravel()]), np.concatenate([q["bfp4"][0].ravel(), q["bfp4"][1].ravel()])),
                w8=sqnr(np.concatenate([gu.ravel(), dn.ravel()]), np.concatenate([q["bfp8"][0].ravel(), q["bfp8"][1].ravel()])),
                y44=sqnr(ref, ffn(*q["bfp4"], x)),
                y48=sqnr(ref, ffn(q["bfp4"][0], q["bfp8"][1], x)),
                y88=sqnr(ref, ffn(*q["bfp8"], x)),
                # same BFP4 format with the exponent shared along the input dim instead (not what TT does)
                y44t=sqnr(ref, ffn(dequant(gu.T.copy(), "bfp4").T, dequant(dn.T.copy(), "bfp4").T, x))))
            print(rows[-1], flush=True)
    return rows


def report(rows, layers, n_exp):
    cols = ["w4", "w8", "y44", "y48", "y88", "y44t"]
    out = ["## Qwen3.8-Flash-Next routed experts in TT block-float (real weights)\n",
           f"{n_exp} random experts (seed 0) from each of layers {', '.join(map(str, layers))}, read straight from the BF16 "
           "checkpoint with HTTP range requests. SQNR in dB, higher is better. Expert output = the SwiGLU expert's output on "
           "64 Gaussian input vectors, quantized vs BF16 weights. Gaussian matrices give 19.0 dB (BFP4) and 43.1 dB (BFP8) "
           "weight SQNR (`kernels/test_bfp`).\n",
           "| Layer | Weights BFP4 | Weights BFP8 | Output, all BFP4 | Output, BFP4 gate/up + BFP8 down | Output, all BFP8 "
           "| Output, BFP4 with exponent along input dim |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for L in layers + ["all"]:
        r = [x for x in rows if L == "all" or x["L"] == L]
        mean = [np.mean([x[c] for x in r]) for c in cols]
        worst = [min(x[c] for x in r) for c in cols]
        out.append(f"| {'**all**' if L == 'all' else L} | " + " | ".join(f"{m:.1f} (min {w:.1f})" for m, w in zip(mean, worst)) + " |")
    return "\n".join(out) + "\n"


def selftest(path):
    """Random matrices + their Python packing, so kernels/test_bfp can compare against bfp_pack byte for byte."""
    rng = np.random.default_rng(1)
    with open(path, "wb") as fh:
        for fmt, out, n in [("bfp8", 32, 64), ("bfp4", 48, 40), ("bfp4", 2 * I, 64)]:
            w = (0.02 * rng.standard_normal((out, n))).astype(np.float32)
            w[:16, 3] = 0                                       # all-zero block -> exponent 0
            w[16:32, 5] *= np.float32(1e-37)                    # subnormal block: clamped exponent
            w[:16, 6] = [1, .125, .375, -.625, .875, -1, 0, .5, -.5, .25, -.125, .0625, -.375, 2 ** -9, .75, -.875]  # ties
            ex, mt = pack(w, fmt)
            fh.write(struct.pack("<iii", MBITS[fmt], out, n) + w.tobytes() + ex + mt)


def layer_list(s):
    return [int(x) for part in s.split(",") for x in
            (range(int(part.split("-")[0]), int(part.split("-")[1]) + 1) if "-" in part else [part])]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["convert", "accuracy", "selftest"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--fmt", default="bfp4", choices=list(MBITS))
    ap.add_argument("--layers", default=None)
    ap.add_argument("--experts", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest(a.args[0])
    elif a.cmd == "convert":
        convert(Source(a.args[0]), Path(a.args[1]), a.fmt, layer_list(a.layers or "0-47"))
    else:
        layers = layer_list(a.layers or "0,9,19,28,38,47")
        text = report(accuracy(Source(a.args[0] if a.args else "hf"), layers, a.experts), layers, a.experts)
        (Path(__file__).parent / "results").mkdir(exist_ok=True)
        (Path(__file__).parent / "results" / "expert_accuracy.md").write_text(text, encoding="utf-8")
        print(text)
