"""Qwen3.8-Flash-Next loader: routed experts from the BF16 checkpoint to tuff's host BFP layout.

    python loader.py convert SRC OUT [--fmt bfp4] [--layers 0-47]  # SRC: local snapshot dir, or "hf" (HTTP)
    python loader.py accuracy [--layers 0,9,19,28,38,47] [--experts 8]  # real-weight accuracy -> results/
    python loader.py selftest FILE                                 # packed random matrices for kernels/test_bfp
    python loader.py fake DIR N [--fmt bfp4]                       # N random experts as layer 0, for kernels/test_store

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
from concurrent.futures import ThreadPoolExecutor
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
        self.urls, self.headers = {}, {}
        self.map = json.loads(self._get("model.safetensors.index.json"))["weight_map"]

    def _url(self, f):
        """The signed CDN URL Hugging Face redirects to, reused for range requests (one redirect per file, not per read)."""
        u, t = self.urls.get(f, (None, 0))
        if u is None or time.time() - t > 1800:
            with urllib.request.urlopen(urllib.request.Request(REPO + f, headers={"Range": "bytes=0-0"}), timeout=60) as r:
                u = r.geturl()
            self.urls[f] = (u, time.time())
        return u

    def _get(self, f, start=None, end=None):   # end inclusive
        if self.root:
            with open(self.root / f, "rb") as fh:
                if start is None:
                    return fh.read()
                fh.seek(start)
                return fh.read(end - start + 1)
        for attempt in range(6):
            try:
                if start is None:
                    req = urllib.request.Request(REPO + f)
                else:
                    req = urllib.request.Request(self._url(f), headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = r.read()
                if start is None or len(data) == end - start + 1:
                    return data
            except OSError:
                self.urls.pop(f, None)               # expired signature or a network blip: re-resolve
                if attempt == 5:
                    raise
            time.sleep(2 ** attempt)
        raise IOError(f"short read: {f} {start}-{end}")

    def info(self, name):
        """(file, absolute byte offset, header entry) of a tensor."""
        f = self.map[name]
        if f not in self.headers:
            n = struct.unpack("<Q", self._get(f, 0, 7))[0]
            self.headers[f] = (8 + n, json.loads(self._get(f, 8, 7 + n)))
        base, h = self.headers[f]
        return f, base + h[name]["data_offsets"][0], h[name]

    def read(self, f, start, size, parts=8):
        """size bytes at start, in parallel range requests of >= 64 MB."""
        n = max(1, min(parts, size >> 26))
        cuts = [start + size * i // n for i in range(n + 1)]
        if n == 1:
            return self._get(f, start, start + size - 1)
        with ThreadPoolExecutor(n) as ex:
            return b"".join(ex.map(lambda i: self._get(f, cuts[i], cuts[i + 1] - 1), range(n)))

    def raw(self, name):
        """A whole tensor: (bytes, dtype string, shape)."""
        f, start, t = self.info(name)
        return self.read(f, start, t["data_offsets"][1] - t["data_offsets"][0]), t["dtype"], t["shape"]

    def expert_bf16(self, name, e):
        """Expert e of a fused [E, out, in] BF16 tensor, as raw little-endian uint16 [out, in]."""
        f, start, t = self.info(name)
        assert t["dtype"] == "BF16" and len(t["shape"]) == 3, (name, t["dtype"], t["shape"])
        _, out, inn = t["shape"]
        size = out * inn * 2
        return np.frombuffer(self.read(f, start + e * size, size), dtype="<u2").reshape(out, inn)

    def expert(self, name, e):
        """Expert e of a fused [E, out, in] BF16 tensor, as float32 [out, in]."""
        return (self.expert_bf16(name, e).astype(np.uint32) << 16).view(np.float32)


def tt_mag(u, shared, mb):
    """|value| in mb bits for float32 bits u in blocks with shared exponent field `shared` (broadcastable): tt-metal's
    convert_u32_to_bfp (_b formats): hidden-bit mantissa >> exponent difference, round to mb bits ties to even,
    saturate; zero/denormal -> 0. bfp_pack in kernels/bfp.c does the same per element."""
    e = (u >> 23) & 0xFF
    d = shared.astype(np.int64) - e
    mant = ((u & 0x7FFFFF) | 0x800000).astype(np.int64)
    mant = np.where(d > 31, 0, mant >> np.minimum(d, 31))
    sh = 24 - mb
    rnd, mant = mant & ((1 << sh) - 1), mant >> sh
    mant = mant + ((rnd > (1 << (sh - 1))) | ((rnd == (1 << (sh - 1))) & (mant & 1 == 1)))
    return np.where(e == 0, 0, np.minimum(mant, (1 << mb) - 1))


def quantize(w, fmt):
    """float32 [out, in] -> (exp uint8 [P, in], signed q [P, 16, in], step float32 [P, in]); bfp_pack's math, which is
    tt-metal's: shared exponent = the largest exponent field of the 16 values, mantissas rounded ties-to-even."""
    mb = MBITS[fmt]
    out, n = w.shape
    assert out % 16 == 0 and n % 8 == 0, w.shape
    u = np.ascontiguousarray(w, np.float32).view(np.uint32).reshape(out // 16, 16, n)
    shared = ((u >> 23) & 0xFF).max(axis=1)
    q = tt_mag(u, shared[:, None, :], mb)
    step = np.where(shared > mb - 1, np.ldexp(np.float32(1), shared.astype(np.int32) - 127 - (mb - 1)), 0).astype(np.float32)
    return shared.astype(np.uint8), np.where(u >> 31 == 1, -q, q).astype(np.int32), step


def tt_tiles_ref(w, fmt):
    """tt-metal's pack_as_bfp_tiles for the device tensor W^T [in][out] (what a matmul y = x W^T reads), written from
    its source rather than from tuff's panels: 32x32 tiles in row-major tile order; per tile 64 exponent bytes (faces
    0..3 = top-left, top-right, bottom-left, bottom-right; 16 face rows each), then each face's mantissas row-major,
    `32/b` per dword with element 0 in the low bits, each sign-magnitude (sign << mb | magnitude)."""
    mb = MBITS[fmt]
    b = mb + 1
    t = np.ascontiguousarray(np.asarray(w, np.float32).T)
    R, C = t.shape
    assert R % 32 == 0 and C % 32 == 0, t.shape
    u = t.view(np.uint32)
    out = bytearray()
    for ti in range(R // 32):
        for to in range(C // 32):
            faces = [u[32 * ti + 16 * tr:32 * ti + 16 * tr + 16, 32 * to + 16 * tc:32 * to + 16 * tc + 16]
                     for tr in (0, 1) for tc in (0, 1)]
            exps = [((f >> 23) & 0xFF).max(axis=1) for f in faces]                    # per face row
            out += np.concatenate(exps).astype(np.uint8).tobytes()
            for f, e in zip(faces, exps):
                mag = tt_mag(f, e[:, None], mb)
                v = (np.where((f >> 31 == 1) & (mag != 0), 1 << mb, 0) | mag).astype(np.uint32).ravel()
                c = 32 // b
                dw = (v.reshape(-1, c) << (b * np.arange(c, dtype=np.uint32))).sum(axis=1, dtype=np.uint32)
                out += dw.astype("<u4").tobytes()
    return bytes(out)


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


def convert(src, out, fmt, layers, n_experts=E):
    out.mkdir(parents=True, exist_ok=True)
    sizes = None
    for L in layers:
        t0 = time.time()
        path = out / f"experts.{L:02d}.{fmt}"
        with open(path.with_suffix(".part"), "wb") as fh:
            for e in range(n_experts):
                parts = [p for name in ("gate_up_proj", "down_proj") for p in pack(src.expert(EXPERT.format(L, name), e), fmt)]
                sizes = sizes or [len(p) for p in parts]
                for p in parts:
                    fh.write(p)
        path.with_suffix(".part").replace(path)
        print(f"layer {L}: {path.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.0f} s", flush=True)
    (out / "manifest.json").write_text(json.dumps(dict(
        model="Qwen/Qwen3.8-Flash-Next", fmt=fmt, layers=list(layers), experts=n_experts,
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
            w[16:32, 5] *= np.float32(1e-37)                    # tiny block, partly denormal
            w[:16, 6] = [1, .125, .375, -.625, .875, -1, 0, .5, -.5, .25, -.125, .0625, -.375, 2 ** -9, .75, -.875]  # ties
            w[16:32, 7] = 3 * w[16:32, 7] + np.float32(1e-3)     # assorted exponents within a block
            ex, mt = pack(w, fmt)
            tt = tt_tiles_ref(w, fmt) if out % 32 == 0 and n % 32 == 0 else b""
            fh.write(struct.pack("<iii", MBITS[fmt], out, n) + w.tobytes() + ex + mt + struct.pack("<i", len(tt)) + tt)


class FakeSource:
    """Random experts with QFN's shapes, for kernels/test_store.c: `fake` converts them like a real layer and also
    writes their float weights (weights.f32: per expert gate_up [1280][2560], then down [2560][640])."""
    def expert(self, name, e):
        rng = np.random.default_rng([e, name.endswith("down_proj")])
        shape = (D, I) if name.endswith("down_proj") else (2 * I, D)
        return (0.02 * rng.standard_normal(shape)).astype(np.float32)


def fake(out, fmt, n):
    src = FakeSource()
    convert(src, out, fmt, [0], n)
    with open(out / "weights.f32", "wb") as fh:
        for e in range(n):
            for name in ("gate_up_proj", "down_proj"):
                fh.write(src.expert(EXPERT.format(0, name), e).tobytes())


def layer_list(s):
    return [int(x) for part in s.split(",") for x in
            (range(int(part.split("-")[0]), int(part.split("-")[1]) + 1) if "-" in part else [part])]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["convert", "accuracy", "selftest", "fake"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--fmt", default="bfp4", choices=list(MBITS))
    ap.add_argument("--layers", default=None)
    ap.add_argument("--experts", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest(a.args[0])
    elif a.cmd == "fake":                                          # fake DIR N: N random experts as layer 0
        fake(Path(a.args[0]), a.fmt, int(a.args[1]))
    elif a.cmd == "convert":
        convert(Source(a.args[0]), Path(a.args[1]), a.fmt, layer_list(a.layers or "0-47"))
    else:
        layers = layer_list(a.layers or "0,9,19,28,38,47")
        text = report(accuracy(Source(a.args[0] if a.args else "hf"), layers, a.experts), layers, a.experts)
        (Path(__file__).parent / "results").mkdir(exist_ok=True)
        (Path(__file__).parent / "results" / "expert_accuracy.md").write_text(text, encoding="utf-8")
        print(text)
