"""Run the real Qwen3.8-Flash-Next forward pass on a machine with 16 GB of RAM, streaming the 360 GB checkpoint.

    python trace.py [--tokens 1024] [--layers 48]       # -> results/trace.md, results/routes.npz

Uses transformers' own Qwen4Exp modules, built on the meta device. Each decoder layer's weights are fetched with
HTTP range requests just before it runs and dropped right after, routed experts only for the experts the router
picked, the N-gram table and the token embeddings only for the rows the text needs. Nothing is written to disk.

The batch holds every (expert format, text) pair, so one pass over the weights measures all of them:
  bf16    reference (checkpoint precision; dense weights run in fp32)
  bfp4    every routed expert in TT bfp4_b
  bfp4+8  gate/up in bfp4_b, down in bfp8_b
and, from the bf16 rows, the routing trace: which experts each token uses at each layer.

Outputs: perplexity of each format (the bf16 one checks this harness against the published model), KL divergence and
top-1 agreement against bf16, routing statistics, and how often a token's experts would be on the card for a given
placement (results/trace.md via analyze()).
"""
import argparse
import inspect
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from loader import Source

REPO_ID = "Qwen/Qwen3.8-Flash-Next"
CK = "model.language_model."                  # checkpoint prefix of the text model
FORMATS = {"bf16": (None, None), "bfp4": (3, 3), "bfp4+8": (3, 7)}   # mantissa bits for (gate_up, down)
DT = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16, "I64": torch.int64, "I32": torch.int32}
ROOT = Path(__file__).parent


def bfp(w, mb):
    """TT block-float round trip of a [out, in] float32 matrix: the math of loader.quantize / bfp_pack, in torch."""
    out, n = w.shape
    x = w.reshape(out // 16, 16, n)
    a = x.abs().amax(1)
    ok = (a > 0) & torch.isfinite(a)
    e = (torch.frexp(torch.where(ok, a, torch.ones_like(a)))[1] - 1).clamp(mb - 126, 127)
    step = torch.ldexp(torch.ones_like(a), e - (mb - 1))[:, None]
    q = torch.floor(x.abs() / step + 0.5).clamp_max((1 << mb) - 1) * ok[:, None]
    return (torch.copysign(q, x) * step).reshape(out, n)


def tensor(src, name, dtype=torch.float32):
    data, dt, shape = src.raw(name)
    t = torch.frombuffer(bytearray(data), dtype=DT[dt]).reshape(shape)
    return t.to(dtype) if dtype is not None and t.is_floating_point() else t


def rows(src, name, ids, threads=64):
    """Rows `ids` (1-D int64) of a 2-D BF16 tensor, as float32 [len(ids), dim]; one range request per run of rows."""
    f, start, t = src.info(name)
    dim = t["shape"][1]
    u = np.unique(ids)
    runs = np.split(u, np.flatnonzero(np.diff(u) != 1) + 1)
    def get(run):
        return src._get(f, start + int(run[0]) * dim * 2, start + (int(run[-1]) + 1) * dim * 2 - 1)
    with ThreadPoolExecutor(threads) as ex:
        data = b"".join(ex.map(get, runs))
    table = torch.frombuffer(bytearray(data), dtype=torch.bfloat16).reshape(-1, dim).float()
    return table[torch.from_numpy(np.searchsorted(u, ids))]


class NGramRows(torch.nn.Module):
    """Stands in for the 51B-parameter N-gram nn.Embedding: fetches only the rows a forward pass asks for."""

    def __init__(self, src, prefix):
        super().__init__()
        self.src, self.parts, r = src, [], 0
        while f"{prefix}.shard_{len(self.parts)}.weight" in src.map:
            name = f"{prefix}.shard_{len(self.parts)}.weight"
            n = src.info(name)[2]["shape"][0]
            self.parts.append((r, n, name))
            r += n
        self.weight = torch.empty(0)            # transformers reads .weight.device
        self.rows_fetched = 0

    def forward(self, ids):
        flat = ids.reshape(-1).numpy()
        out = None
        for r0, n, name in self.parts:
            m = (flat >= r0) & (flat < r0 + n)
            if m.any():
                v = rows(self.src, name, flat[m] - r0)
                out = out if out is not None else torch.zeros(len(flat), v.shape[1])
                out[torch.from_numpy(m)] = v
        self.rows_fetched += len(np.unique(flat))
        return out.reshape(*ids.shape, -1)


def put(module, dotted, t):
    *path, leaf = dotted.split(".")
    for p in path:
        module = getattr(module, p)
    if leaf in module._parameters:
        module._parameters[leaf] = torch.nn.Parameter(t, requires_grad=False)
    else:
        module._buffers[leaf] = t


def materialize(src, module, ck_prefix, skip=()):
    names = [n for n, t in list(module.named_parameters()) + list(module.named_buffers()) if t.device.type == "meta"]
    names = [n for n in names if not n.startswith(skip)]
    with ThreadPoolExecutor(8) as ex:
        for n, t in zip(names, ex.map(lambda n: tensor(src, ck_prefix + n), names)):
            put(module, n, t)


def release(module):
    for n, t in list(module.named_parameters()) + list(module.named_buffers()):
        if t.device.type != "meta":
            put(module, n, torch.empty(t.shape, dtype=t.dtype, device="meta"))


def make_experts_forward(src, layer, fmt_of_row, fmt_bits, stats):
    """Replaces Qwen4ExpTextExperts.forward: fetches the routed experts, applies each batch row's expert format."""
    gu_name, dn_name = f"{CK}layers.{layer}.mlp.experts.gate_up_proj", f"{CK}layers.{layer}.mlp.experts.down_proj"

    def forward(hidden_states, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden_states)
        hit = torch.unique(top_k_index).tolist()
        t0 = time.time()
        with ThreadPoolExecutor(32) as ex:
            gus = list(ex.map(lambda e: src.expert_bf16(gu_name, e), hit))
            dns = list(ex.map(lambda e: src.expert_bf16(dn_name, e), hit))
        stats["fetch_s"] += time.time() - t0
        stats["experts"] += len(hit)
        for e, gu16, dn16 in zip(hit, gus, dns):
            tok, pos = torch.where(top_k_index == e)
            gu = torch.from_numpy((gu16.astype(np.uint32) << 16).view(np.float32))
            dn = torch.from_numpy((dn16.astype(np.uint32) << 16).view(np.float32))
            cache = {}
            for fi, (bg, bd) in enumerate(fmt_bits):
                sel = fmt_of_row[tok] == fi
                if not sel.any():
                    continue
                t, p = tok[sel], pos[sel]
                if ("g", bg) not in cache:
                    cache["g", bg] = gu if bg is None else bfp(gu, bg)
                if ("d", bd) not in cache:
                    cache["d", bd] = dn if bd is None else bfp(dn, bd)
                g, u = F.linear(hidden_states[t], cache["g", bg]).chunk(2, dim=-1)
                y = F.linear(F.silu(g) * u, cache["d", bd]) * top_k_weights[t, p, None]
                out.index_add_(0, t, y)
        return out
    return forward


def texts(tok, T):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from transformers.models.qwen4_exp import modeling_qwen4_exp as M
    p = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset")
    wiki = "".join(pq.read_table(p).column("text").to_pylist())
    code = inspect.getsource(M)               # Python source: a second, very different domain
    out = {}
    for name, text in [("wikitext", wiki[20000:]), ("code", code[5000:])]:
        ids = tok(text, add_special_tokens=False)["input_ids"][:T]
        assert len(ids) == T, (name, len(ids))
        out[name] = ids
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=48, help="run only the first N layers (smoke test; no logits)")
    a = ap.parse_args()
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen4_exp import modeling_qwen4_exp as M

    t_start = time.time()
    cfg = AutoConfig.from_pretrained(REPO_ID).text_config
    cfg._attn_implementation = "sdpa"
    tok = AutoTokenizer.from_pretrained(REPO_ID)
    src = Source("hf")
    data = texts(tok, a.tokens)
    chunks, fmts = list(data), list(FORMATS)
    C, Fn, T = len(chunks), len(fmts), a.tokens
    ids = torch.tensor([data[c] for c in chunks])                       # [C, T]
    batch_ids = ids.repeat(Fn, 1)                                       # row b = f * C + c
    fmt_of_row = (torch.arange(Fn * C * T) // T) // C

    with torch.device("meta"):
        model = M.Qwen4ExpForCausalLM(cfg)
    tm = model.model
    tm.rotary_emb = M.Qwen4ExpTextRotaryEmbedding(cfg)                  # real inv_freq buffers
    full = a.layers == cfg.num_hidden_layers
    tm.layers = tm.layers[: a.layers]
    tm.config.num_hidden_layers = a.layers                              # same object as cfg
    materialize(src, tm.hyper_connection_mixer, CK + "hyper_connection_mixer.")
    embeds = rows(src, CK + "embed_tokens.weight", batch_ids.reshape(-1).numpy()).reshape(Fn * C, T, -1)

    routes = np.zeros((a.layers, Fn * C * T, cfg.num_experts_per_tok), np.int16)
    stats = dict(fetch_s=0.0, experts=0)
    timing = []
    for L, layer in enumerate(tm.layers):
        if layer.ple is not None:
            emb = layer.ple.ple_embedding
            emb.ngram_embedding = NGramRows(src, f"{CK}layers.{L}.ple.ple_embedding.ngram_embedding")
        layer.mlp.experts.forward = make_experts_forward(src, L, fmt_of_row, list(FORMATS.values()), stats)

        def pre(mod, args, L=L):
            t0 = time.time()
            materialize(src, mod, f"{CK}layers.{L}.", skip=("mlp.experts.", "ple.ple_embedding.ngram_embedding."))
            mod._t = (time.time(), time.time() - t0, stats["fetch_s"], stats["experts"])

        def post(mod, args, out, L=L):
            t0, load_s, f0, n0 = mod._t
            release(mod)
            timing.append((L, load_s, stats["fetch_s"] - f0, stats["experts"] - n0, time.time() - t0))
            print(f"layer {L:2d} ({mod.layer_type}): dense load {load_s:.0f} s, {stats['experts'] - n0} experts "
                  f"fetched in {stats['fetch_s'] - f0:.0f} s, total {time.time() - t0:.0f} s", flush=True)

        layer.register_forward_pre_hook(pre)
        layer.register_forward_hook(post)
        layer.mlp.gate.register_forward_hook(lambda m, i, o, L=L: routes[L].__setitem__(..., o[2].numpy()))

    with torch.no_grad():
        h = tm(inputs_embeds=embeds, ple_input_ids=batch_ids, use_cache=False).last_hidden_state   # [Fn*C, T, d]
    del embeds

    out = dict(routes=routes.reshape(a.layers, Fn, C, T, -1), chunks=np.array(chunks), fmts=np.array(fmts),
               ids=ids.numpy(), timing=np.array(timing))
    if full:
        W = tensor(src, "lm_head.weight")
        metrics = np.zeros((Fn, C, 3))                                   # nll, kld, top-1 agreement (sums)
        for c in range(C):
            for t0 in range(0, T - 1, 128):
                t1 = min(t0 + 128, T - 1)
                lp = torch.log_softmax(h[[f * C + c for f in range(Fn)], t0:t1] @ W.T, dim=-1)   # [Fn, t, V]
                tgt = ids[c, t0 + 1:t1 + 1]
                for f in range(Fn):
                    metrics[f, c, 0] -= lp[f, torch.arange(t1 - t0), tgt].sum().item()
                    metrics[f, c, 1] += (lp[0].exp() * (lp[0] - lp[f])).sum().item()
                    metrics[f, c, 2] += (lp[f].argmax(-1) == lp[0].argmax(-1)).sum().item()
        out["metrics"] = metrics / (T - 1)
        print("ppl", np.exp(out["metrics"][:, :, 0]).round(3).tolist())
    out["wall_s"] = time.time() - t_start
    (ROOT / "results").mkdir(exist_ok=True)
    np.savez_compressed(ROOT / "results" / "routes.npz", **out)
    print(f"done in {out['wall_s'] / 60:.1f} min; {stats['experts']} expert fetches")


def basalt_order(path, E):
    """Basalt's shipped expert profile (data/profiles/qwen4exp-48x512.prof, MIT): u32 version, layers, experts, n,
    slots; then n (u16 layer, u16 expert) pairs, best first; then a rank table. Returns flat layer*E+expert, best first."""
    import struct
    d = Path(path).read_bytes()
    _, _, _, n, _ = struct.unpack("<5I", d[:20])
    p = np.frombuffer(d[20:20 + 4 * n], "<u2").reshape(n, 2).astype(np.int64)
    return p[:, 0] * E + p[:, 1]


def analyze(prof=None):
    import plan
    z = np.load(ROOT / "results" / "routes.npz")
    R = z["routes"].astype(np.int64)                                    # [L, F, C, T, k]
    L, Fn, C, T, k = R.shape
    E = plan.QFN["E"]
    chunks, fmts = [str(c) for c in z["chunks"]], [str(f) for f in z["fmts"]]
    ref = R[:, 0]                                                       # bf16 routing, [L, C, T, k]
    flat = ref + (np.arange(L) * E)[:, None, None, None]                # global expert id layer*E+e
    counts = np.stack([np.bincount(flat[:, c].ravel(), minlength=L * E) for c in range(C)])
    host = plan.HOSTS["E5-2690v4, 4x16GB"]
    caps = {d: plan.memory_map(d, "bfp4", 32768, host)["n_dev"] * L // plan.QFN["L"] for d in ("bfp8", "bfp4")}

    def hit(order, n, c):
        on = np.zeros(L * E, bool)
        on[order[:n]] = True
        return on[flat[:, c]].mean()

    orders = {"in-sample (same text, upper bound)": lambda c: np.argsort(-counts[c], kind="stable")}
    if C > 1:
        orders["the other text's counts"] = lambda c: np.argsort(-counts[1 - c], kind="stable")
    if prof:
        b = basalt_order(prof, E)
        b = b[b < L * E]                                                 # partial runs: the layers that ran
        orders["Basalt's shipped profile"] = lambda c: b

    out = ["## Qwen3.8-Flash-Next forward pass, streamed from the checkpoint\n",
           f"{T} tokens of each text ({', '.join(chunks)}), {L} layers, batch = every (expert format, text) pair. "
           f"Wall time {float(z['wall_s']) / 60:.0f} min on a GitHub runner.\n"]
    if "metrics" in z:
        m = z["metrics"]
        out += ["### Accuracy of BFP experts (dense weights unquantized)\n",
                "KLD and top-1 agreement are against the bf16 run on the same tokens.\n",
                "| Text | Experts | PPL | KLD mean | Same top-1 |", "|---|---|---:|---:|---:|"]
        for c, name in enumerate(chunks):
            for f, fm in enumerate(fmts):
                out.append(f"| {name} | {fm} | {np.exp(m[f, c, 0]):.3f} | "
                           f"{'–' if f == 0 else f'{m[f, c, 1]:.4f}'} | {'–' if f == 0 else f'{100 * m[f, c, 2]:.2f}%'} |")
    out += ["\n### Routing\n",
            "Hit rate = share of the 10 routed experts per token and layer that a placement keeps on the card. Card capacity "
            f"from plan.py: {caps['bfp8']:,} experts with dense weights in BFP8, {caps['bfp4']:,} with dense in BFP4 "
            f"(of {L * E:,}).\n",
            "| Placement | Text | " + " | ".join(f"hit, {n:,} on card" for n in caps.values()) + " |",
            "|---|---|" + "---:|" * len(caps)]
    out.append("| uniform | any | " + " | ".join(f"{n / (L * E):.1%}" for n in caps.values()) + " |")
    hits = {}
    for pname, fn in orders.items():
        for c, name in enumerate(chunks):
            hits[pname, name] = [hit(fn(c), n, c) for n in caps.values()]
            out.append(f"| {pname} | {name} | " + " | ".join(f"{h:.1%}" for h in hits[pname, name]) + " |")
    reuse = [np.mean([len(set(ref[l, c, t]) & set(ref[l, c, t - 1])) / k for l in range(L) for t in range(1, T)])
             for c in range(C)]
    out.append("\nShare of a token's experts that the previous token also used, same layer: " +
               ", ".join(f"{n} {r:.1%}" for n, r in zip(chunks, reuse)) + ".")
    for f in range(1, Fn):
        same = np.mean([len(set(R[l, 0, c, t]) & set(R[l, f, c, t])) / k
                        for l in range(L) for c in range(C) for t in range(T)])
        out.append(f"With {fmts[f]} experts, {same:.1%} of the routed experts are the same as with bf16.")
    if prof and "metrics" in z:
        key = ("Basalt's shipped profile", chunks[0])
        out += ["\n### Predicted decode with the measured hit rate\n",
                f"plan.py's model with the hit rate measured for Basalt's profile on {chunks[0]} (E5-2690 v4, 64 GB, "
                "batch 1, no MTP).\n", "| Dense on card | Hit rate | t/s |", "|---|---:|---:|"]
        for (d, n), h in zip(caps.items(), hits[key]):
            out.append(f"| {d} | {h:.1%} | {plan.decode(host, h, d)[0]:.1f} |")
    text = "\n".join(out) + "\n"
    (ROOT / "results" / "trace.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        analyze(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        main()
