"""The card code over the C host side (libtuff.so) == over the numpy host side, on the tiny random QFN.

    make -C kernels libtuff.so && python device/test_host_c.py      # Linux (tstore uses mmap)

1. CCache (tcache) makes the same slot decisions as qfn.PyCache: same hits, same outputs.
2. CHost (tstore + moe_forward_n on BFP4 files written like loader.convert) == the numpy host on the same BFP4 weights.
3. Speculative greedy decoding over the C host side == plain greedy decoding over it, token for token.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "device"))
import host_c  # noqa: E402
import loader  # noqa: E402
import qfn  # noqa: E402
from ops import Torch  # noqa: E402
from test_qfn import Oracle, tiny, weights  # noqa: E402


def main(T=40):
    c = tiny()
    w = weights(c)
    # the numpy host on the same BFP4 numbers the C host reads from its files
    wq = dict(w)
    for L in range(c.num_hidden_layers):
        for name in ("gate_up_proj", "down_proj"):
            k = f"layers.{L}.mlp.experts.{name}"
            wq[k] = np.stack([loader.dequant(np.asarray(m, np.float32), "bfp4") for m in w[k]])
    so = host_c.lib()
    ids = np.random.default_rng(1).integers(0, c.vocab_size, T)
    n_slots, budget = c.num_hidden_layers * c.num_experts * 3 // 10, 4
    with tempfile.TemporaryDirectory() as d:
        host_c.write_store(d, w, c, "bfp4")

        def card(host, cache):
            return qfn.Card(Torch(), w, c, host, dense_fmt="f32", expert_fmt="bfp4", slots=n_slots, budget=budget,
                            cache=cache)

        py = card(qfn.Host(wq, c), qfn.PyCache(n_slots, budget))
        cc = card(host_c.CHost(wq, c, d, so=so), host_c.CCache(n_slots, budget, c.num_hidden_layers, c.num_experts, so=so))
        a = np.stack([py.o.get(py.step(int(t)))[0] for t in ids])
        b = np.stack([cc.o.get(cc.step(int(t)))[0] for t in ids])
        err = (np.abs(a - b).max(1) / np.abs(a).max(1)).max()
        print(f"card code over libtuff (tcache + tstore/moe_forward_n) == over numpy: max relative error {err:.1e}; "
              f"slot hits {cc.hits}/{cc.lookups} (numpy cache: {py.hits}/{py.lookups})")
        assert err < 1e-4 and (cc.hits, cc.lookups) == (py.hits, py.lookups), (err, cc.hits, py.hits)

        prompt, n = list(ids[:12]), 24
        mk = lambda: card(host_c.CHost(wq, c, d, so=so), host_c.CCache(n_slots, budget, c.num_hidden_layers, c.num_experts, so=so))
        truth = mk().generate(prompt, n)
        g = mk()
        out = g.generate(prompt, n, Oracle(truth, len(prompt), c.vocab_size), 3)
        print(f"speculative greedy over libtuff == plain greedy over libtuff: {out == truth}, {n} tokens in {g.steps} steps")
        assert out == truth


if __name__ == "__main__":
    main()
