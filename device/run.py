"""Decode Qwen3.8-Flash-Next on the p150a: the real checkpoint, ttnn, the C host side, MTP drafts.

    make -C kernels libtuff.so
    python loader.py convert <snapshot> <store> --fmt bfp4      # the experts as BFP4 files, once
    python device/test_ops.py                                     # first: every ttnn op against the CPU backend
    python device/run.py --model <snapshot> --store <store> --prompt "..." [--n 128] [--mtp 2] [--gpu] [--cpu]

<snapshot>: a local Hugging Face snapshot of Qwen/Qwen3.8-Flash-Next (config, tokenizer, safetensors). --cpu runs the
same code on ops.Torch (very slow on the full model; for checking the setup). Prints the text, tokens/s, steps and
the card's expert hit rate.
ponytail: written without a card or the full checkpoint at hand; not run yet.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "device"))
import plan  # noqa: E402
import qfn  # noqa: E402
import ref_real  # noqa: E402
from loader import Source  # noqa: E402

DENSE = {"lm_head": "bfp4", "lin_z": "bfp4"}           # tuff's default: the rest BFP8 (plan.fmt_of)


class Weights(ref_real.Checkpoint):
    """The checkpoint for the card code: dense weights as float32 on use and not kept (the card keeps its copy),
    experts fetched per expert and not kept (the slots are filled once each; cold experts come from the store)."""

    def __getitem__(self, name):
        if name in self.cache:
            return self.cache[name]
        if name == "lm_head.weight":
            return ref_real.as_float(*self.src.raw(name)).astype(np.float32)
        if name.endswith(("experts.gate_up_proj", "experts.down_proj")):
            return Expert(self.src, self.prefix + name)
        if name == "embed_tokens.weight" or name.endswith("ngram_embedding.weight"):
            return super().__getitem__(name)                # row tables, read per row
        return np.asarray(ref_real.as_float(*self.src.raw(self.prefix + name)), np.float32)


class Expert:
    def __init__(self, src, name):
        self.src, self.name = src, name

    def __getitem__(self, e):
        return (self.src.expert_bf16(self.name, e).astype(np.uint32) << 16).view(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--prompt", default="The history of the transistor begins")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--mtp", type=int, default=2, help="drafts per step (0: off)")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--trace-mb", type=int, default=256, help="trace region (raise it if capture says so)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--gpu", action="store_true", help="split the dense part with an NVIDIA GPU (device/tp.py)")
    a = ap.parse_args()

    from transformers import AutoConfig, AutoTokenizer
    cfg = AutoConfig.from_pretrained(a.model).text_config
    tok = AutoTokenizer.from_pretrained(a.model)
    src = Source(a.model)
    W = Weights(src, cfg)
    import host_c
    so = host_c.lib()
    host = host_c.CHost(W, cfg, a.store, "bfp4", so=so)
    mm = plan.memory_map(DENSE, "bfp4", a.ctx, plan.HOSTS["E5-2690v4, 4x16GB"])
    n_slots, L, E = mm["n_dev"], cfg.num_hidden_layers, cfg.num_experts
    if a.cpu:
        from ops import Torch
        ops = Torch()
    else:
        import ttnn
        from ops import TTNN
        ops = TTNN(ttnn.open_device(device_id=0, trace_region_size=a.trace_mb << 20))
    kw = dict(dense_fmt=DENSE, expert_fmt="bfp4", max_t=a.ctx)
    if a.gpu:                                          # the 4060 Ti: its share of every head and row, its own slots
        import tp
        from ops import CUDA
        sp = plan.split_multi(plan.GPUS["RTX 4060 Ti 16GB"], DENSE, "bfp4", a.ctx)
        cache = host_c.CCache(sp["n_card"], 25, L, E, so=so, n_other=sp["n_gpu"])
        card = tp.TP(ops, CUDA(), W, cfg, host, f=sp["f"], n_card=sp["n_card"], n_gpu=sp["n_gpu"], cache=cache, **kw)
    else:
        card = qfn.Card(ops, W, cfg, host, slots=n_slots, cache=host_c.CCache(n_slots, 25, L, E, so=so), **kw)
    drafter = None
    if a.mtp:
        import copy
        mcfg = copy.deepcopy(cfg)
        mcfg.num_hidden_layers, mcfg.layer_types, mcfg.ple_layer_ids = 1, ["full_attention"], []
        MW = Weights(src, mcfg, prefix="mtp.")
        MW.cache["embed_tokens.weight"] = W["embed_tokens.weight"]
        # ponytail: the MTP layer's experts from the checkpoint into its own slots (256); store them too if it matters
        drafter = qfn.MTPDrafter(ops, MW, cfg, qfn.Host(MW, mcfg), slots=256, budget=10, **kw)

    ids = tok(a.prompt)["input_ids"]
    t0 = time.perf_counter()
    out = card.generate(ids, a.n, drafter, a.mtp)
    t1 = time.perf_counter()
    print(tok.decode(ids + out))
    print(f"\nprompt {len(ids)} tokens: {card.t_prompt - t0:.1f} s (includes weight uploads); {a.n} tokens: "
          f"{t1 - card.t_prompt:.1f} s = {a.n / max(t1 - card.t_prompt, 1e-9):.1f} t/s (includes the first trace "
          f"captures), {card.steps} steps, card expert hits {card.hits / max(card.lookups, 1):.1%}")


if __name__ == "__main__":
    main()
