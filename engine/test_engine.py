"""The C++ engine (engine/tuff_engine, CPU reference backend) against the Python card code (device/qfn.py on ops.Torch)
on the tiny random QFN of device/test_qfn.py: the same mixer outputs step by step, the same greedy tokens, the same
expert-slot hits (tcache in C++, PyCache in Python: one policy).

    make -C engine && python engine/test_engine.py [--store]   # --store: cold experts from a BFP4 store (libtuff)

export(dir, w, cfg) writes what tuff_engine reads: config.txt, weights.bin (TUFFW1: name, dtype, shape, data).
"""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "device"))
import qfn  # noqa: E402
from ops import Torch  # noqa: E402
from test_qfn import tiny, weights  # noqa: E402

KEYS = ["hidden_size", "hc_count", "head_dim", "num_attention_heads", "num_key_value_heads", "num_hidden_layers",
        "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim",
        "linear_conv_kernel_dim", "ple_conv_kernel_size", "ngram_size", "heads_per_ngram", "indexer_n_heads",
        "indexer_head_dim", "indexer_budget", "indexer_compress_ratio", "num_experts", "num_experts_per_tok",
        "moe_intermediate_size", "shared_expert_intermediate_size", "vocab_size", "rms_norm_eps"]


def export(d, w, c):
    d = Path(d)
    eos = c.eos_token_id[0] if isinstance(c.eos_token_id, list) else c.eos_token_id
    rp = c.rope_parameters
    lines = [f"{k} {getattr(c, k)}" for k in KEYS] + [
        f"eos_token_id {eos}", f"rope_theta {rp['rope_theta']}", f"partial_rotary_factor {rp.get('partial_rotary_factor', 1.0)}",
        f"gated_act {int((c.output_gate_type or c.hidden_act) == 'sigmoid')}", f"norm_topk_prob {int(c.norm_topk_prob)}",
        "layer_types " + " ".join(c.layer_types[:c.num_hidden_layers]),
        "ple_layer_ids " + " ".join(str(x) for x in c.ple_layer_ids)]
    (d / "config.txt").write_text("\n".join(lines) + "\n")
    with open(d / "weights.bin", "wb") as f:
        f.write(b"TUFFW1" + struct.pack("<I", len(w)))
        for name, a in w.items():
            a = np.asarray(a)
            ints = a.dtype.kind in "iu"
            a = np.ascontiguousarray(a, np.int64 if ints else np.float32)
            n = name.encode()
            f.write(struct.pack("<I", len(n)) + n + struct.pack("<BI", int(ints), a.ndim))
            f.write(np.asarray(a.shape, np.int64).tobytes() + a.tobytes())


def main():
    c = tiny()
    w = weights(c)
    rng = np.random.default_rng(1)
    steps = [int(t) for t in rng.integers(0, c.vocab_size, 40)]
    prompt, n = steps[:12], 24
    slots, budget = c.num_hidden_layers * c.num_experts * 3 // 10, 4
    mk = lambda: qfn.Card(Torch(), w, c, qfn.Host(w, c), dense_fmt="f32", expert_fmt="f32", slots=slots, budget=budget,
                          max_t=256)
    py = mk()
    want = np.stack([py.o.get(py.step(t))[0] for t in steps])
    gen = mk()
    want_toks = gen.generate(prompt, n)
    exe = ROOT / "engine" / "tuff_engine"
    with tempfile.TemporaryDirectory() as d:
        export(d, w, c)
        (Path(d) / "case.txt").write_text(f"steps {' '.join(map(str, steps))}\nprompt {' '.join(map(str, prompt))}\n"
                                          f"n {n}\nslots {slots}\nbudget {budget}\n")
        args = [str(exe), d]
        if "--store" in sys.argv:
            import host_c
            host_c.write_store(Path(d) / "store", w, c, "bfp4")
            args += ["--store", str(Path(d) / "store"), "bfp4"]
        print(subprocess.run(args, check=True, capture_output=True, text=True).stdout, end="")
        got = np.fromfile(Path(d) / "out_steps.bin", np.float32).reshape(len(steps), -1)
        toks = [int(x) for x in (Path(d) / "out_tokens.txt").read_text().split()]
    errs = np.abs(got - want).max(1) / np.abs(want).max(1)
    err = errs.max()
    if "--store" in sys.argv:                       # cold experts in BFP4 (Python's are float): close, not equal
        print(f"C++ engine with cold experts from the BFP4 store vs the float Python card code: median relative "
              f"error {np.median(errs):.1e}, max {err:.1e}")
        assert np.median(errs) < 0.1, errs
    else:
        print(f"C++ engine == Python card code: {len(steps)} steps, max relative error {err:.1e} "
              f"(Python slot hits {py.hits}/{py.lookups}); greedy tokens equal: {toks == want_toks}")
        assert err < 1e-4 and toks == want_toks, (err, toks, want_toks)


if __name__ == "__main__":
    main()
