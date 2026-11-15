"""Run the real Qwen3.8-Flash-Next forward pass on a machine with 16 GB of RAM, streaming the 360 GB checkpoint.

    python trace.py [--tokens 1024] [--layers 48]       # -> results/trace.md, results/routes.npz

Uses transformers' own Qwen4Exp modules, built on the meta device. Each decoder layer's weights are fetched with
HTTP range requests just before it runs and dropped right after, routed experts only for the experts the router
picked, the N-gram table and the token embeddings only for the rows the text needs. Nothing is written to disk.

The batch holds every (format, text) pair, so one pass over the weights measures all of them:
  bf16       reference (checkpoint precision; dense weights run in fp32)
  bfp4       routed experts in TT bfp4_b, dense weights as in the checkpoint
  tuff       tuff's default card: dense projections and LM head in bfp8_b, routed experts in bfp4_b
  tuff-fast  dense projections in bfp4_b, LM head bfp8_b, routed experts bfp4_b
(router, norms, convolutions and the 1- and 4-row gates stay unquantized), and, from the bf16 rows, the routing
trace: which experts each token uses at each layer.

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
FORMATS = {"bf16": (None, None, None), "bfp4": (None, 3, 3), "tuff": (7, 3, 3), "tuff-fast": (3, 3, 3)}
# mantissa bits for (dense projections, expert gate_up, expert down); None = unquantized. LM head: bfp8 when dense is.
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


def split_linears(module, row_bits):
    """Every nn.Linear in `module` (dense projections; the routed experts are not nn.Linear) gets one weight per
    format: batch row b multiplies by its weight quantized to row_bits[b] (None = as loaded). Layers whose shape is
    not whole 16-row blocks (the 1- and 4-output gates) stay unquantized."""
    n_rows = len(row_bits)
    for lin in module.modules():
        if not isinstance(lin, torch.nn.Linear) or lin.out_features % 16 or lin.in_features % 8:
            continue
        assert lin.bias is None
        groups = [(torch.tensor([b for b in range(n_rows) if row_bits[b] == bits]),
                   lin.weight if bits is None else bfp(lin.weight, bits)) for bits in set(row_bits)]

        def forward(x, groups=groups, out=lin.out_features):
            xb = x.reshape(n_rows, -1, x.shape[-1])
            y = x.new_empty(n_rows, xb.shape[1], out)
            for rows, w in groups:
                y[rows] = F.linear(xb[rows], w)
            return y.reshape(*x.shape[:-1], out)
        lin.forward = forward


def release(module):
    for lin in module.modules():
        lin.__dict__.pop("forward", None)               # split_linears' quantized copies
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
        layer.mlp.experts.forward = make_experts_forward(src, L, fmt_of_row, [(g, d) for _, g, d in FORMATS.values()],
                                                         stats)

        def pre(mod, args, L=L):
            t0 = time.time()
            materialize(src, mod, f"{CK}layers.{L}.", skip=("mlp.experts.", "ple.ple_embedding.ngram_embedding."))
            split_linears(mod, [FORMATS[fmts[b // C]][0] for b in range(Fn * C)])
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
        W8 = torch.cat([bfp(W[i:i + 16384], 7) for i in range(0, W.shape[0], 16384)])   # chunks: bounded temporaries
        Wf = [W if FORMATS[f][0] is None else W8 for f in fmts]         # LM head bfp8 wherever dense is quantized
        metrics = np.zeros((Fn, C, 3))                                   # nll, kld, top-1 agreement (sums)
        for c in range(C):
            for t0 in range(0, T - 1, 128):
                t1 = min(t0 + 128, T - 1)
                lp = torch.log_softmax(torch.stack([h[f * C + c, t0:t1] @ Wf[f].T for f in range(Fn)]), dim=-1)
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


def simulate_cache(seq, n, budget, init, LE):
    """Hit rate of an adaptive card cache on a routing trace. seq: [T, L, k] global expert ids in decode order.
    The card starts with init[:n]. A routed expert that is not on the card runs on the CPU for that token and is
    queued; after each token up to `budget` of its misses are copied to the card (PCIe DMA overlapping the next
    token), evicting the least recently used experts. Returns (hit rate, mean copies per token)."""
    res = np.zeros(LE, bool)
    res[init[:n]] = True
    last = np.full(LE, -np.inf)
    last[init[:n]] = -1.0 - np.arange(n)                # better-ranked counts as more recently used
    hits = copies = 0
    for t, r in enumerate(seq):
        ids = r.ravel()
        on = res[ids]
        hits += on.sum()
        last[ids] = t
        miss = np.unique(ids[~on])[:budget]             # ponytail: any `budget` of the misses; all were just used
        if len(miss):
            cand = np.flatnonzero(res)
            victims = cand[np.argpartition(last[cand], len(miss))[:len(miss)]]
            res[victims], res[miss] = False, True
            copies += len(miss)
    return hits / seq.size, copies / len(seq)


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
    if prof:
        n8 = caps["bfp8"]
        b = orders["Basalt's shipped profile"](0)
        seqs = {name: flat[:, c].transpose(1, 0, 2) for c, name in enumerate(chunks)}       # [T, L, k]
        if C > 1:
            seqs[" then ".join(chunks)] = np.concatenate(list(seqs.values()))               # a change of topic
        budgets = (0, 10, 25, 50, 100)
        out += ["\n### Adaptive card cache (simulated on the trace)\n",
                f"{n8:,} experts on the card (dense BFP8), starting from Basalt's profile. A miss runs on the CPU and the "
                "expert is copied to the card for the next token, evicting the least recently used; 'copies' caps the "
                "copies per token (0 = static placement). t/s from plan.py, which charges the copies to host DRAM "
                "bandwidth and to the PCIe 3.0 link (E5-2690 v4, dense BFP8, no MTP).\n",
                "| Text | " + " | ".join(f"{x} copies" for x in budgets) + " |", "|---|" + "---:|" * len(budgets)]
        for name, s in seqs.items():
            cells = []
            for bud in budgets:
                h, cp = simulate_cache(s, n8, bud, b, L * E)
                cells.append(f"{h:.1%} ({plan.decode(host, h, 'bfp8', promote=cp)[0]:.1f} t/s)")
            out.append(f"| {name} | " + " | ".join(cells) + " |")
    # MTP verification runs m consecutive tokens through each layer at once: weights are read once per step, so what
    # matters is the union of their experts (card-resident and cold), measured here on the real routing.
    if prof:
        on = np.zeros(L * E, bool)
        on[orders["Basalt's shipped profile"](0)[:caps["bfp8"]]] = True
        out += ["\n### Experts per layer when m consecutive tokens are verified together (MTP)\n",
                f"Mean over layers, texts and windows; placement = Basalt's profile with {caps['bfp8']:,} experts on the card.\n",
                "| m tokens | unique experts | on card | cold (CPU) |", "|---:|---:|---:|---:|"]
        win = {}
        for m in (1, 2, 3, 4, 6, 8):
            u = [np.unique(flat[l, c, t:t + m]) for l in range(L) for c in range(C) for t in range(0, T - m + 1, m)]
            win[m] = (np.mean([len(x) for x in u]), np.mean([on[x].sum() for x in u]), np.mean([(~on[x]).sum() for x in u]))
            out.append(f"| {m} | {win[m][0]:.1f} | {win[m][1]:.1f} | {win[m][2]:.1f} |")
        ms = np.array(list(win))
        # Basalt community medians (docs/community/benchmarks.json, 20 runs): draft length, acceptance per draft
        scen = {"prose": (1.8, 0.6675), "chat": (2.35, 0.7285), "structured": (6.9, 0.9835)}
        out += ["\nPredicted decode with MTP (E5-2690 v4, 64 GB), assuming Basalt's measured draft lengths and acceptance; "
                "the experts per step are interpolated from the table above.\n",
                "| Text kind | drafts / step | tokens / step | dense BFP8 | dense BFP4 |", "|---|---:|---:|---:|---:|"]
        for kind, (dl, acc) in scen.items():
            m = 1 + dl
            hot, cold = (np.interp(m, ms, [win[x][i] for x in ms]) for i in (1, 2))
            r = [plan.decode_mtp(host, hot, cold, m, 1 + dl * acc, d) for d in ("bfp8", "bfp4")]
            out.append(f"| {kind} | {dl} | {1 + dl * acc:.2f} | {r[0]:.1f} t/s | {r[1]:.1f} t/s |")
        # everything together: adaptive cache (25 copies per token) + MTP, on the topic-change sequence
        mix = seqs[" then ".join(chunks)] if C > 1 else seqs[chunks[0]]
        out += ["\n### Predicted decode: adaptive cache + MTP\n",
                f"Adaptive cache with up to 25 copies per token, hit rate measured on '{' then '.join(chunks)}'; MTP "
                "as above, with each step's union of experts split by that hit rate. Batch 1, 4k context.\n",
                "| Host | Dense on card | Hit rate | no MTP | prose | chat | structured |", "|---|---|---:|---:|---:|---:|---:|"]
        for hn in ("E5-2690v4, 4x16GB", "8259CL, 6x32GB"):
            hh = plan.HOSTS[hn]
            for d in ("bfp8", "bfp4"):
                hr, cp = simulate_cache(mix, caps[d], 25, b, L * E)
                cells = [f"{plan.decode(hh, hr, d, promote=cp)[0]:.1f}"]
                for kind, (dl, acc) in scen.items():
                    m = 1 + dl
                    u = np.interp(m, ms, [win[x][0] for x in ms])
                    cells.append(f"{plan.decode_mtp(hh, u * hr, u * (1 - hr), m, 1 + dl * acc, d, promote=cp * (1 + dl * acc)):.1f}")
                out.append(f"| {hn} | {d} | {hr:.1%} | " + " | ".join(cells) + " |")
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
