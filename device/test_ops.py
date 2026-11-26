"""Run this first on the card: every operation of ops.TTNN against ops.Torch, then the card code end to end.

    python device/test_ops.py            # needs ttnn and a Tenstorrent device

Each line says whether the ttnn version of one operation agrees with the CPU one (bf16 activations: ~1e-2 relative).
A failure points at the one place to fix (ops.py's TTNN class), since device/qfn.py is already checked on ops.Torch.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "device"))
from ops import Torch, TTNN  # noqa: E402


def main():
    import ttnn
    dev = ttnn.open_device(device_id=0, trace_region_size=1 << 20)   # ponytail: qfn's segments need more; size it from the capture error
    a, b = Torch(), TTNN(dev)
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal((4, 96)), rng.standard_normal((4, 96))
    m, wt = rng.standard_normal((96, 64)), rng.standard_normal((64, 96)) / 10
    q = rng.standard_normal((4, 16))
    qn = (q / np.linalg.norm(q, axis=1, keepdims=True)).reshape(1, 4, 16).repeat(3, 0)       # 3 tokens, 4 key heads
    kn = qn[:, ::-1].copy()
    v, g = rng.standard_normal((3, 8, 16)), -np.abs(rng.standard_normal((3, 8)))           # 8 value heads
    beta, S = 1 / (1 + np.exp(-rng.standard_normal((3, 8)))), 0.1 * rng.standard_normal((8, 16, 16))
    def bank(o, fill):
        bk = o.bank(4, 96, 64, "bfp4")
        for i in fill:
            o.bank_write(bk, i, wt * (i + 1))
        return bk

    tests = {
        "linear (weight bf16)": lambda o: o.linear(o.put(x), o.put_weight(wt, "bf16")),
        "linear (weight bfp8)": lambda o: o.linear(o.put(x), o.put_weight(wt, "bfp8")),
        "linear (weight bfp4)": lambda o: o.linear(o.put(x), o.put_weight(wt, "bfp4")),
        "linear + fused silu": lambda o: o.linear(o.put(x), o.put_weight(wt, "bf16"), act="silu"),
        "matmul": lambda o: o.matmul(o.put(x), o.put(m)),
        "rms_norm (3-D, per-row weight)": lambda o: o.rms_norm(o.reshape(o.put(x), (4, 4, 24)), o.put(y[:4, :24]), 1e-6),
        "trace: captured, replayed on new input": lambda o: [o.trace("t", lambda a: (o.silu(o.mul(a, a)),), o.put(z))
                                                             for z in (y, x)][-1][0],
        "add / mul / scale": lambda o: o.scale(o.add(o.mul(o.put(x), o.put(y)), o.put(y)), 0.5),
        "broadcast [4,1] * [1,96]": lambda o: o.mul(o.put(x[:, :1]), o.put(y[:1])),
        "silu / sigmoid / exp": lambda o: o.add(o.silu(o.put(x)), o.mul(o.sigmoid(o.put(y)), o.exp(o.scale(o.put(x), 0.1)))),
        "softplus / relu / abs / sign": lambda o: o.add(o.softplus(o.put(x)), o.mul(o.relu(o.put(y)), o.sign(o.put(x)))),
        "sqrt / rsqrt / clamp_min": lambda o: o.mul(o.sqrt(o.clamp_min(o.abs(o.put(x)), 1e-6)), o.rsqrt(o.add(o.abs(o.put(y)), 1.0))),
        "sum / mean (dim 1 and 0)": lambda o: o.add(o.sum(o.put(x), 1), o.transpose(o.mean(o.put(x[:, :4]), 0))),
        "softmax": lambda o: o.softmax(o.put(x)),
        "reshape / slice / concat / transpose": lambda o: o.transpose(o.concat(
            [o.slice(o.reshape(o.put(x), (8, 48)), 1, 0, 16), o.slice(o.reshape(o.put(y), (8, 48)), 1, 16, 32)], 0)),
        "gather_rows / write_rows": lambda o: o.gather_rows(o.write_rows(o.put(x), 1, o.put(y[:2])), o.put_ids([3, 2, 0])),
        "permute (3-D)": lambda o: o.permute(o.reshape(o.put(x), (4, 8, 12)), (1, 2, 0)),
        "expert bank: bank_write + gathered matmul (broadcast x, fused silu)": lambda o: o.gather_matmul(
            o.reshape(o.put(x[:2]), (1, 2, 96)), bank(o, (2, 0)), o.put_ids([2, 0], "uint16"), act="silu"),
        "expert bank: gathered matmul, one x per slot": lambda o: o.gather_matmul(
            o.reshape(o.put(np.concatenate([x[:2], y[:2]])), (2, 2, 96)), bank(o, (1, 2)), o.put_ids([1, 2], "uint16")),
        "topk (stable, per row)": lambda o: o.put(np.asarray(o.topk(o.put(np.round(x * 4) / 4), 8)[1], np.float64)),
        "gated delta rule, 3 tokens (output)": lambda o: o.gdr(o.put(qn), o.put(kn), o.put(v), o.put(g),
                                                               o.put(beta), o.put(S), 0.25)[0],
        "gated delta rule, 3 tokens (state after each)": lambda o: o.gdr(o.put(qn), o.put(kn), o.put(v), o.put(g),
                                                                         o.put(beta), o.put(S), 0.25)[1],
        "3-D reshape / mean / broadcast": lambda o: o.mean(o.mul(o.reshape(o.put(x), (4, 4, 24)), o.put(y[:1, :24])), 1),
    }
    bad = 0
    for name, f in tests.items():
        try:
            want, got = a.get(f(a)), b.get(f(b))
            err = np.abs(got - want).max() / max(np.abs(want).max(), 1e-9)
            ok = err < 3e-2
        except Exception as e:                                       # noqa: BLE001 -- report every op, then fail
            err, ok = repr(e)[:120], False
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {err if isinstance(err, str) else f'{err:.1e}'}")
    ttnn.close_device(dev)
    print("all ops agree" if not bad else f"{bad} ops to fix in ops.TTNN")
    sys.exit(bad > 0)


if __name__ == "__main__":
    main()
