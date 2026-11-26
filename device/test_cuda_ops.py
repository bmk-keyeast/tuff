"""Run this on the box with the 4060 Ti: ops.CUDA against ops.Torch, then the card + GPU split with the GPU side on
CUDA (device/tp.py) against one CPU device, on the tiny random QFN.

    make -C kernels libtuff_cuda.so && python device/test_cuda_ops.py

The CPU side stands in for the card here, so this needs no Tenstorrent device.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "device"))
import qfn  # noqa: E402
import tp  # noqa: E402
from ops import CUDA, Torch  # noqa: E402
from test_qfn import tiny, weights  # noqa: E402


def main():
    a, b = Torch(), CUDA()
    rng = np.random.default_rng(0)
    x, wt = rng.standard_normal((6, 96)), rng.standard_normal((64, 96)) / 10
    bad = 0

    def bank(o):
        bk = o.bank(4, 96, 64, "bfp4")
        for i in (0, 2):
            o.bank_write(bk, i, wt * (i + 1))
        return bk
    tests = {
        "linear, BFP8 weight (tgpu_gemv), 6 tokens": lambda o: o.linear(o.put(x), o.put_weight(wt, "bfp8")),
        "linear, BFP4 weight + silu": lambda o: o.linear(o.put(x), o.put_weight(wt, "bfp4"), act="silu"),
        "gathered matmul over a BFP4 bank": lambda o: o.gather_matmul(o.reshape(o.put(x[:2]), (1, 2, 96)), bank(o),
                                                                       o.put_ids([2, 0], "uint16")),
    }
    for name, f in tests.items():
        want, got = a.get(f(a)), b.get(f(b))
        err = np.abs(got - want).max() / np.abs(want).max()
        bad += err > 1e-4
        print(f"{'ok  ' if err <= 1e-4 else 'FAIL'} {name}: {err:.1e}")

    c = tiny()
    w = weights(c)
    ids = [int(t) for t in rng.integers(0, c.vocab_size, 24)]
    fmt = dict(dense_fmt="bfp8", expert_fmt="bfp4")
    one = qfn.Card(Torch(), w, c, qfn.Host(w, c), slots=38, budget=4, **fmt)
    want = np.stack([one.o.get(one.step(t))[0] for t in ids])
    two = tp.TP(Torch(), CUDA(), w, c, qfn.Host(w, c), f=0.5, n_card=24, n_gpu=14, budget=4, **fmt)
    got = []
    for t in ids:
        got.append(two.o.get(two.verify([t]))[0])
        two.commit(1)
    err = (np.abs(np.stack(got) - want).max(1) / np.abs(want).max(1)).max()
    print(f"card (CPU stand-in) + GPU on CUDA == one CPU device: max relative error {err:.1e}, slot hits "
          f"{two.hits}/{two.lookups} (one device {one.hits}/{one.lookups})")
    bad += err > 1e-3
    print("CUDA backend agrees" if not bad else f"{bad} checks failed")
    sys.exit(bad > 0)


if __name__ == "__main__":
    main()
