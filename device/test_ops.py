"""Run this first on the card: every operation of ops.TTNN against ops.Torch, then the card code end to end
(device/qfn.py with traces, on the tiny random QFN) against the CPU backend.

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
    print("all ops agree" if not bad else f"{bad} ops to fix in ops.TTNN")
    if not bad:
        bad += card_end_to_end(dev)
        promotion_cost(dev)
    ttnn.close_device(dev)
    sys.exit(bad > 0)


def promotion_cost(dev, I=640, d=2560, n=8):
    """What the adaptive cache's copies cost: one QFN expert (gate, up, down) into a bank slot, through ops.TTNN's
    bank_write (from float32: ttnn packs BFP4 on the host). 25 copies per token have to fit next to a ~25 ms token."""
    import time
    o = TTNN(dev)
    rng = np.random.default_rng(0)
    banks = [o.bank(n, d, I, "bfp4"), o.bank(n, d, I, "bfp4"), o.bank(n, I, d, "bfp4")]
    ws = [rng.standard_normal((I, d)).astype(np.float32) / 50, rng.standard_normal((I, d)).astype(np.float32) / 50,
          rng.standard_normal((d, I)).astype(np.float32) / 25]
    for b, w in zip(banks, ws):                        # warm-up
        o.bank_write(b, 0, w)
    t0 = time.perf_counter()
    for s in range(1, n):
        for b, w in zip(banks, ws):
            o.bank_write(b, s, w)
    import ttnn
    ttnn.synchronize_device(dev)
    ms = (time.perf_counter() - t0) / (n - 1) * 1e3
    print(f"expert copy into a slot: {ms:.1f} ms each -> {25 * ms:.0f} ms for 25 per token "
          f"({'fine' if 25 * ms < 10 else 'too slow: lower run.py --budget, or upload tstore_expert_tt tiles as is'})")


def card_end_to_end(dev, T=24):
    """device/qfn.py on the card (ttnn, with traces) against the CPU backend on the tiny random QFN of test_qfn.py:
    a prompt pass, single steps, then speculative decoding with QFN's MTP drafter."""
    import qfn
    from test_qfn import mtp_weights, tiny, weights
    c = tiny()
    w = weights(c)
    ids = [int(t) for t in np.random.default_rng(1).integers(0, c.vocab_size, T)]
    kw = dict(dense_fmt="bf16", expert_fmt="bfp8", slots=c.num_hidden_layers * c.num_experts // 3, budget=4)
    out = {}
    for name, ops in (("cpu", Torch()), ("card", TTNN(dev))):
        card = qfn.Card(ops, w, c, qfn.Host(w, c), **kw)
        rows = [ops.get(card.verify(ids[:12]))]
        card.commit(12)
        for t in ids[12:]:
            rows.append(ops.get(card.step(t)))
        out[name] = (np.concatenate(rows), card.routes)
    (a, ra), (b, rb) = out["cpu"], out["card"]
    err = (np.abs(b - a).max(1) / np.abs(a).max(1)).max()
    same = np.mean([set(ra[k].tolist()) == set(rb[k].tolist()) for k in ra])
    print(f"card code on the card vs CPU backend: {T} tokens, max relative error {err:.1e}, same experts {same:.1%}")
    plain = qfn.Card(TTNN(dev), w, c, qfn.Host(w, c), **kw).generate(ids[:12], 16)
    spec = qfn.Card(TTNN(dev), w, c, qfn.Host(w, c), **kw).generate(
        ids[:12], 16, qfn.MTPDrafter(TTNN(dev), mtp_weights(c), c, qfn.Host(w, c), **kw), 2)
    print(f"speculative greedy on the card == plain greedy on the card: {spec == plain} (bf16 activations can split "
          f"near-ties between a 3-token verify and single steps; a few differing tokens is rounding, not a bug)")
    return int(err > 0.1)


if __name__ == "__main__":
    main()
