"""The card + GPU split (device/tp.py) == one device, on the tiny random QFN.

    python device/test_tp.py

1. TP with an even and an uneven split gives the same outputs as a single Card with as many slots, makes the same slot
   decisions, and routes the same.
2. Speculative greedy decoding on TP (oracle drafter and QFN's MTP drafter) == plain greedy decoding on one device.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "device"))
import qfn  # noqa: E402
import tp  # noqa: E402
from ops import Torch  # noqa: E402
from test_qfn import Oracle, mtp_weights, tiny, weights  # noqa: E402


def main(T=40):
    c = tiny()
    w = weights(c)
    ids = np.random.default_rng(1).integers(0, c.vocab_size, T)
    n_card, n_gpu, budget = 24, 14, 4
    fmt = dict(dense_fmt="f32", expert_fmt="f32")
    one = lambda: qfn.Card(Torch(), w, c, qfn.Host(w, c), slots=n_card + n_gpu, budget=budget, **fmt)
    ref = one()
    want = np.stack([ref.o.get(ref.step(int(t)))[0] for t in ids])
    for f in (0.5, 0.3):
        two = tp.TP(Torch(), Torch(), w, c, qfn.Host(w, c), f=f, n_card=n_card, n_gpu=n_gpu, budget=budget, **fmt)
        got = []
        for t in ids:
            got.append(two.o.get(two.verify([int(t)]))[0])
            two.commit(1)
        got = np.stack(got)
        err = (np.abs(got - want).max(1) / np.abs(want).max(1)).max()
        same = all((two.routes[k] == v).all() for k, v in ref.routes.items())
        on_gpu = sum(dev is two.B for dev, _ in two.slots.values())
        print(f"card + GPU split, GPU share {f:.0%} (DeltaNet key heads {two.A.c.linear_num_key_heads}:"
              f"{two.B.c.linear_num_key_heads}, KV heads {two.A.c.num_key_value_heads}:{two.B.c.num_key_value_heads}) "
              f"== one device: max relative error {err:.1e}, same routing {same}, slot hits {two.hits}/{two.lookups} "
              f"(one device: {ref.hits}/{ref.lookups}), {on_gpu} of {len(two.slots)} slot experts on the GPU")
        assert err < 1e-4 and same and (two.hits, two.lookups) == (ref.hits, ref.lookups)

    prompt, n = list(ids[:12]), 24
    truth = one().generate(prompt, n)
    for name, drafter, k in (("oracle drafter, 3 drafts", Oracle(truth, len(prompt), c.vocab_size), 3),
                             ("QFN MTP drafter, 2 drafts", qfn.MTPDrafter(Torch(), mtp_weights(c), c, qfn.Host(w, c), **fmt), 2)):
        two = tp.TP(Torch(), Torch(), w, c, qfn.Host(w, c), f=0.3, n_card=n_card, n_gpu=n_gpu, budget=budget, **fmt)
        out = two.generate(prompt, n, drafter, k)
        print(f"speculative greedy on card + GPU ({name}) == plain greedy on one device: {out == truth}, "
              f"{n} tokens in {two.steps} steps")
        assert out == truth


if __name__ == "__main__":
    main()
