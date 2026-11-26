"""How many card operations one decode step dispatches, and how many times it stops for the host.

    python device/opcount.py            # prints a table; --md results/opcount.md writes it too

On a Tenstorrent card every ttnn call from Python costs host time to dispatch, whatever its size; a decode step of
QFN is thousands of small ops, so their count (and the host round trips that split a step) bounds the speed as much
as DRAM bandwidth does. This counts them on the tiny random QFN (device/test_qfn.py) with ops.Torch, per layer kind,
and scales them to QFN's 36 DeltaNet + 12 attention layers with 10 routed experts per token. A traced segment
(ops.trace) counts as one replay: on the card its ops run from the trace buffer with no Python dispatch.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "device"))
import qfn  # noqa: E402
from ops import Torch  # noqa: E402
from test_qfn import tiny, weights  # noqa: E402

HOST = {"get", "put", "topk"}            # host <-> card transfers (topk: its indices come back to the host)
VIA_HOST = {"gather_rows", "write_rows"}  # one op on Torch, but through the host in ops.TTNN for now


class Counted:
    """ops wrapper counting calls per (section, op); the section is the innermost Card method being run."""

    def __init__(self, ops):
        self.o, self.n, self.sec, self.quiet = ops, Counter(), ["other"], 0

    def __getattr__(self, name):
        f = getattr(self.o, name)
        if not callable(f):
            return f

        def call(*a, **k):
            if not self.quiet:
                self.n[self.sec[-1], name] += 1
            return f(*a, **k)
        return call

    def trace(self, key, fn, *args):
        if not self.quiet:
            self.n[self.sec[-1], "trace replay"] += 1
        self.quiet += 1
        try:
            return self.o.trace(key, fn, *args)
        finally:
            self.quiet -= 1


def sections(card, ops, names):
    for name in names:
        f = getattr(card, name)

        def wrapped(*a, _f=f, _n=name, **k):
            ops.sec.append(_n)
            try:
                return _f(*a, **k)
            finally:
                ops.sec.pop()
        setattr(card, name, wrapped)


def count(m=1, T=40):
    c = tiny()
    w = weights(c)
    ops = Counted(Torch())
    n_exp = c.num_hidden_layers * c.num_experts
    card = qfn.Card(ops, w, c, qfn.Host(w, c), dense_fmt="f32", expert_fmt="f32", slots=n_exp, budget=n_exp)
    calls = Counter()
    sections(card, ops, ["ple_layer", "deltanet_layer", "attention_layer", "slot_experts", "lm_argmax"])
    ex = card.slot_experts
    card.slot_experts = lambda x, items: (calls.update(n=len(items)), ex(x, items))[1]
    ids = [int(t) for t in np.random.default_rng(1).integers(0, c.vocab_size, T)]
    card.lm_argmax(card.verify(ids[:T - m]))           # prompt; then every expert it routed to sits in a slot
    card.commit(T - m)
    ops.n.clear()
    calls.clear()
    card.lm_argmax(card.verify(ids[T - m:]))
    card.commit(m)
    kinds = Counter(c.layer_types[:c.num_hidden_layers])
    return ops.n, kinds, calls["n"]


def table(m):
    n, kinds, n_exp = count(m)
    c = tiny()
    kinds_l, k_tok = c.layer_types[:c.num_hidden_layers], c.num_experts_per_tok   # distinct experts per layer, scaled to top-10
    per = lambda sec, ops=None: sum(v for (s, op), v in n.items() if s == sec and (ops is None or op in ops))
    rows = [("DeltaNet layer", "deltanet_layer", kinds["linear_attention"], 36),
            ("attention layer", "attention_layer", kinds["full_attention"], 12),
            ("PLE (layer 1)", "ple_layer", 1, 1),
            ("routed experts in slots, per expert", "slot_experts", max(n_exp, 1),
             round(48 * 10 * n_exp / len(kinds_l) / k_tok)),
            ("LM head + argmax", "lm_argmax", 1, 1)]
    out = [f"m = {m} token(s) per pass", "",
           "| part | card ops | host transfers | through the host (TTNN, for now) | trace replays | QFN count | QFN card ops |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    tot = Counter()
    for label, sec, k, qfn_n in rows:
        a, h, v, r = (per(sec) / k, per(sec, HOST) / k, per(sec, VIA_HOST) / k, per(sec, {"trace replay"}) / k)
        out.append(f"| {label} | {a:.0f} | {h:.0f} | {v:.0f} | {r:.0f} | {qfn_n} | {a * qfn_n:,.0f} |")
        tot.update(ops=a * qfn_n, host=h * qfn_n, via=v * qfn_n, replay=r * qfn_n)
    out.append(f"| **one pass** | | | | | | **{tot['ops']:,.0f}** ops, {tot['host']:,.0f} host transfers, "
               f"{tot['via']:,.0f} through the host, {tot['replay']:,.0f} trace replays |")
    return "\n".join(out)


if __name__ == "__main__":
    text = "\n\n".join(table(m) for m in (1, 3))
    print(text)
    if "--md" in sys.argv:
        Path(sys.argv[sys.argv.index("--md") + 1]).write_text(
            "# Card ops per decode pass (device/opcount.py)\n\nEvery routed expert in a slot (the most ops a pass "
            "can take); a cold expert costs the host instead.\n\n" + text + "\n", encoding="utf-8")
