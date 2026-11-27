"""The C++ engine (engine/tuff_engine, CPU reference backend) against the Python card code (device/qfn.py, device/tp.py
on ops.Torch) on the tiny random QFN of device/test_qfn.py:

    1. single steps: the same mixer outputs, the same expert-slot hits (tcache in C++, PyCache in Python: one policy)
    2. greedy decoding: the same tokens
    3. a long pass (prefill, attention for every token at once) then steps: the same outputs
    4. speculative decoding with QFN's MTP drafter, also over a draft vocabulary: the same tokens and steps
    5. the card + GPU split at 50% and 30%: the same outputs; speculative on it == greedy

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
import tp  # noqa: E402
from ops import Torch  # noqa: E402
from test_qfn import mtp_weights, tiny, weights  # noqa: E402

KEYS = ["hidden_size", "hc_count", "head_dim", "num_attention_heads", "num_key_value_heads", "num_hidden_layers",
        "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim",
        "linear_conv_kernel_dim", "ple_conv_kernel_size", "ngram_size", "heads_per_ngram", "indexer_n_heads",
        "indexer_head_dim", "indexer_budget", "indexer_compress_ratio", "num_experts", "num_experts_per_tok",
        "moe_intermediate_size", "shared_expert_intermediate_size", "vocab_size", "rms_norm_eps"]


def export(d, w, c, name="weights.bin"):
    d = Path(d)
    eos = c.eos_token_id[0] if isinstance(c.eos_token_id, list) else c.eos_token_id
    rp = c.rope_parameters
    lines = [f"{k} {getattr(c, k)}" for k in KEYS] + [
        f"eos_token_id {eos}", f"rope_theta {rp['rope_theta']}", f"partial_rotary_factor {rp.get('partial_rotary_factor', 1.0)}",
        f"gated_act {int((c.output_gate_type or c.hidden_act) == 'sigmoid')}", f"norm_topk_prob {int(c.norm_topk_prob)}",
        "layer_types " + " ".join(c.layer_types[:c.num_hidden_layers]),
        "ple_layer_ids " + " ".join(str(x) for x in c.ple_layer_ids)]
    (d / "config.txt").write_text("\n".join(lines) + "\n")
    with open(d / name, "wb") as f:
        f.write(b"TUFFW1" + struct.pack("<I", len(w)))
        for key, a in w.items():
            a = np.asarray(a)
            ints = a.dtype.kind in "iu"
            a = np.ascontiguousarray(a, np.int64 if ints else np.float32)
            n = key.encode()
            f.write(struct.pack("<I", len(n)) + n + struct.pack("<BI", int(ints), a.ndim))
            f.write(np.asarray(a.shape, np.int64).tobytes() + a.tobytes())


def run_engine(exe, d, store):
    """tuff_engine test <d>; on Windows through WSL (the engine is a Linux build)"""
    wsl = sys.platform == "win32"
    conv = (lambda q: subprocess.run(["wsl", "-e", "wslpath", "-a", str(q)], capture_output=True, encoding="utf-8").stdout.strip())         if wsl else str
    args = [conv(exe), "test", conv(d)] + (["--store", conv(Path(d) / "store"), "bfp4"] if store else [])
    r = subprocess.run((["wsl", "-e"] if wsl else []) + args, capture_output=True, encoding="utf-8")
    if r.returncode:
        raise RuntimeError(r.stderr)
    return r.stdout


def snapshot(d, w, mw, c):
    """A Hugging Face style snapshot of the tiny model: config.json (text_config), two safetensors files with the
    checkpoint's names (model.language_model.*, mtp.*, lm_head at the top), the N-gram table in two shards, its
    per-head sizes and offsets left out (the engine rebuilds them, as transformers does)."""
    import json
    from safetensors.numpy import save_file
    d = Path(d)
    d.mkdir()
    eos = c.eos_token_id[0] if isinstance(c.eos_token_id, list) else c.eos_token_id
    tc = {k: getattr(c, k) for k in KEYS} | dict(
        eos_token_id=eos, rope_parameters=c.rope_parameters, output_gate_type=c.output_gate_type, hidden_act=c.hidden_act,
        norm_topk_prob=c.norm_topk_prob, layer_types=c.layer_types, ple_layer_ids=c.ple_layer_ids, ngram_vocab_size_base=1000)
    (d / "config.json").write_text(json.dumps({"text_config": tc}))
    t = {}
    for k, v in w.items():
        if k.endswith(("ngram_heads_vocab_sizes", "ngram_heads_offsets")):
            continue
        v = np.asarray(v)
        v = np.ascontiguousarray(v, np.int64 if v.dtype.kind in "iu" else np.float32)
        if k.endswith("ngram_embedding.weight"):
            h = len(v) // 2
            t["model.language_model." + k[:-7] + ".shard_0.weight"], t["model.language_model." + k[:-7] + ".shard_1.weight"] = v[:h], v[h:]
        else:
            t[k if k == "lm_head.weight" else "model.language_model." + k] = v
    for k, v in mw.items():
        t["mtp." + k] = np.ascontiguousarray(v, np.float32)
    names = sorted(t)
    files = {"model-00001-of-00002.safetensors": names[::2], "model-00002-of-00002.safetensors": names[1::2]}
    for f, ns in files.items():
        save_file({k: t[k] for k in ns}, str(d / f))
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: f for f, ns in files.items() for k in ns}}))


def nth_prime_after(x, n):
    """engine/weights.cpp's rebuild of the N-gram head sizes"""
    v = x
    while n:
        v += 1
        if v > 1 and all(v % q for q in range(2, int(v ** 0.5) + 1)):
            n -= 1
    return v


def rel(got, want):
    return (np.abs(got - want).max(1) / np.abs(want).max(1)).max()


def tokens(path):
    t, s = Path(path).read_text().split("\n")[:2]
    return [int(x) for x in t.split()], int(s)


def main():
    store = "--store" in sys.argv
    c = tiny()
    w = weights(c)
    mw = mtp_weights(c)
    rng = np.random.default_rng(1)
    steps = [int(t) for t in rng.integers(0, c.vocab_size, 40)]
    prompt, n, k, pre = steps[:12], 24, 2, 27
    slots, budget, n_card, n_gpu = c.num_hidden_layers * c.num_experts * 3 // 10, 4, 24, 14
    kw = dict(dense_fmt="f32", expert_fmt="f32", max_t=256)
    mk = lambda: qfn.Card(Torch(), w, c, qfn.Host(w, c), slots=slots, budget=budget, **kw)
    py = mk()
    want = np.stack([py.o.get(py.step(t))[0] for t in steps])
    greedy = mk()
    truth = greedy.generate(prompt, n)
    long = mk()
    want_pre = [long.o.get(long.verify(steps[:pre]))]
    long.commit(pre)
    want_pre += [long.o.get(long.step(t)) for t in steps[pre:]]
    want_pre = np.concatenate(want_pre)
    vocab = np.unique(np.concatenate([np.random.default_rng(2).choice(c.vocab_size, 300, replace=False), truth[:8]]))
    spec = {}
    for name, v in (("out_mtp.txt", None), ("out_mtp_vocab.txt", vocab)):
        card = mk()
        out = card.generate(prompt, n, qfn.MTPDrafter(Torch(), mw, c, qfn.Host(w, c), vocab=v, slots=slots, budget=budget, **kw), k)
        spec[name] = (out, card.steps)
    want_tp = []
    for f in (0.5, 0.3):
        two = tp.TP(Torch(), Torch(), w, c, qfn.Host(w, c), f=f, n_card=n_card, n_gpu=n_gpu, budget=budget, **kw)
        rows = []
        for t in steps:
            rows.append(two.o.get(two.verify([t]))[0])
            two.commit(1)
        want_tp.append(np.stack(rows))

    exe = ROOT / "engine" / "tuff_engine"
    with tempfile.TemporaryDirectory() as d:
        export(d, w, c)
        export(d, mw, c, "mtp.bin")
        (Path(d) / "case.txt").write_text(
            f"steps {' '.join(map(str, steps))}\nprompt {' '.join(map(str, prompt))}\nn {n}\nslots {slots}\n"
            f"budget {budget}\nprefill {pre}\nmtp {k}\nvocab {' '.join(map(str, vocab))}\ntp {n_card} {n_gpu}\n")
        hf = False
        try:
            snapshot(Path(d) / "hf", w, mw, c)
            hf = True
            with open(Path(d) / "case.txt", "a") as f:
                f.write("hf 1\n")
        except ImportError:
            print("(safetensors not installed: the snapshot reader is not checked)")
        if store:
            import host_c
            host_c.write_store(Path(d) / "store", w, c, "bfp4")
        print(run_engine(exe, d, store), end="")
        if hf:
            got_hf = np.fromfile(Path(d) / "out_hf_steps.bin", np.float32).reshape(len(steps), -1)
            hf_mtp, _ = tokens(Path(d) / "out_hf_mtp.txt")
        D = Path(d)
        got = np.fromfile(D / "out_steps.bin", np.float32).reshape(len(steps), -1)
        got_pre = np.fromfile(D / "out_prefill.bin", np.float32).reshape(len(steps), -1)
        toks, _ = tokens(D / "out_tokens.txt")
        got_spec = {name: tokens(D / name) for name in spec}
        got_tp = [np.fromfile(D / f"out_tp{i}.bin", np.float32).reshape(len(steps), -1) for i in range(2)]
        tp_mtp, _ = tokens(D / "out_tp_mtp.txt")

    if store:                                       # cold experts in BFP4 (Python's are float): close, not equal
        errs = np.abs(got - want).max(1) / np.abs(want).max(1)
        print(f"C++ engine, cold experts from the BFP4 store, vs the float Python card code: median relative error "
              f"{np.median(errs):.1e}, max {errs.max():.1e}")
        assert np.median(errs) < 0.1, errs
        return
    checks = [
        (f"1. steps: max relative error {rel(got, want):.1e} (Python slot hits {py.hits}/{py.lookups})", rel(got, want) < 1e-4),
        (f"2. greedy tokens equal: {toks == truth}", toks == truth),
        (f"3. prefill ({pre} tokens in one pass) then steps: max relative error {rel(got_pre, want_pre):.1e}",
         rel(got_pre, want_pre) < 1e-4),
    ]
    for name, (out, st) in spec.items():
        g, gs = got_spec[name]
        checks.append((f"4. MTP{' (draft vocabulary)' if 'vocab' in name else ''}: tokens equal {g == out}, "
                       f"== greedy {g == truth}, steps C++ {gs} / Python {st}", g == out == truth and gs == st))
    for i, f in enumerate((0.5, 0.3)):
        e = rel(got_tp[i], want_tp[i])
        checks.append((f"5. card + GPU split {f:.0%}: max relative error vs Python's split {e:.1e}, vs one device "
                       f"{rel(got_tp[i], want):.1e}", e < 1e-4))
    checks.append((f"5. speculative (MTP) on the split == greedy: {tp_mtp == truth}", tp_mtp == truth))
    if hf:
        e = rel(got_hf, want)
        checks.append((f"6. from a Hugging Face snapshot (safetensors, sharded table, rebuilt N-gram sizes): max "
                       f"relative error {e:.1e}; MTP == greedy {hf_mtp == truth}", e < 1e-4 and hf_mtp == truth))
    try:
        from transformers.models.qwen4_exp import modeling_qwen4_exp as M
        same = all(M._find_nth_prime_after(x, n) == nth_prime_after(x, n) for x in (999, 19_999_999) for n in (1, 5, 32))
        checks.append((f"6. the N-gram head sizes as transformers rebuilds them: {same}", same))
    except ImportError:
        pass
    for line, ok in checks:
        print(("ok   " if ok else "FAIL ") + line)
    assert all(ok for _, ok in checks)


if __name__ == "__main__":
    main()
