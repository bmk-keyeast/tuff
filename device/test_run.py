"""device/run.py's wiring (run.build: the card code over libtuff's store and tcache, QFN's MTP drafter) on the tiny
random QFN of test_qfn.py, on the CPU backend: speculative decoding through it == plain greedy through it.

    make -C kernels libtuff.so && python device/test_run.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "device"))
import host_c  # noqa: E402
import run  # noqa: E402
from ops import Torch  # noqa: E402
from test_qfn import mtp_weights, tiny, weights  # noqa: E402


def main():
    c = tiny()
    w = weights(c)
    mw = dict(mtp_weights(c), **{"embed_tokens.weight": w["embed_tokens.weight"]})
    so = host_c.lib()
    prompt = [int(t) for t in np.random.default_rng(3).integers(0, c.vocab_size, 20)]
    kw = dict(dense_fmt=run.DENSE, expert_fmt="bfp4", max_t=256)
    with tempfile.TemporaryDirectory() as d:
        host_c.write_store(d, w, c, "bfp4")
        out = {}
        for name, MW in (("plain", None), ("MTP", mw)):
            card, drafter = run.build(c, w, MW, host_c.CHost(w, c, d, so=so), Torch(), so, 40, 4, **kw)
            out[name] = card.generate(prompt, 24, drafter, 2)
            print(f"run.build, {name}: {len(out[name])} tokens in {card.steps} steps, card expert hits "
                  f"{card.hits}/{card.lookups}")
    print(f"speculative == plain greedy through run.build: {out['MTP'] == out['plain']}")
    assert out["MTP"] == out["plain"]


if __name__ == "__main__":
    main()
