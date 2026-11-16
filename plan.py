"""Memory map and decode-time model for Qwen3.8-Flash-Next (QFN) on one Blackhole p150a + a DDR4 Xeon host.

Placement (batch-1 decode):
    p150a GDDR6:  attention (36 Gated-DeltaNet + 12 full-attention layers), router, shared expert,
                  hyper-connection mixers, LM head, KV cache / recurrent state, and as many routed
                  experts as fit ("hot")
    host DRAM:    every other routed expert, computed in place by the CPU ("cold"), token embeddings,
                  the MTP drafter's experts; the 51B N-gram embedding table too when RAM allows
    NVMe:         the N-gram table otherwise (memory-mapped, as Strata does); a few rows per token
    PCIe 3.0:     hidden-state vectors only, never weights

    t_token = t_dense + max(t_hot, t_cold + t_pcie)

    python plan.py        -> results/plan.md
"""
from pathlib import Path

GB, GiB = 1e9, 2**30
B = {"bf16": 2.0, "bfp8": 1088 / 1024, "bfp4": 576 / 1024}   # bytes/param, TT block-float formats

# Qwen/Qwen3.8-Flash-Next config.json (text_config, model_type qwen4_exp_text)
QFN = dict(d=2560, L=48, full_every=4,                       # layer_types: 3 linear_attention : 1 full_attention
           H=24, KV=2, hd=256,                               # full attention, sigmoid output gate
           idx_heads=4, idx_hd=128, idx_kv=1,                # sparse-attention indexer (budget 2048)
           lin_k_heads=16, lin_v_heads=48, lin_hd=128, conv=4,  # Gated DeltaNet
           E=512, k=10, I=640, S=640,                        # routed experts, top-10, shared expert
           hc=4, hc_rank=320,                                # hyper-connections: 4 residual streams
           V=248320, ngram_rows=20_000_000)                  # vocab, ngram_vocab_size_base


def params(m=QFN):
    """Parameter counts per placement class. Full/linear attention follow the Qwen3-Next layout;
    hyper-connection mixers are approximated as two low-rank 4-stream mixers per layer (<1% of the model)."""
    d, L = m["d"], m["L"]
    n_full = L // m["full_every"]
    n_lin = L - n_full
    full = (d * m["H"] * m["hd"] * 2                      # q + output gate
            + 2 * d * m["KV"] * m["hd"] + m["H"] * m["hd"] * d
            + d * (m["idx_heads"] + m["idx_kv"]) * m["idx_hd"])
    kd, vd = m["lin_k_heads"] * m["lin_hd"], m["lin_v_heads"] * m["lin_hd"]
    lin = d * (2 * kd + 2 * vd) + d * 2 * m["lin_v_heads"] + vd * d + (2 * kd + vd) * m["conv"]
    hc = 2 * 2 * (m["hc"] * d * m["hc_rank"])
    expert = 3 * d * m["I"]
    return dict(attn=n_full * full + n_lin * lin, attn_full=full, shared=L * 3 * d * m["S"], router=L * d * m["E"], hc=L * hc,
                lm_head=m["V"] * d, embed=m["V"] * d, ngram=m["ngram_rows"] * d,
                expert=expert, n_experts=L * m["E"], mtp_experts=m["E"] * expert,
                n_full=n_full, n_lin=n_lin,
                # dense groups that can take different formats (trace.py measures each one's accuracy)
                lin_qkv=n_lin * d * (2 * kd + vd), lin_z=n_lin * d * vd, lin_out=n_lin * vd * d,
                lin_misc=n_lin * (d * 2 * m["lin_v_heads"] + (2 * kd + vd) * m["conv"]), full=n_full * full)


DENSE = ("lin_qkv", "lin_z", "lin_out", "lin_misc", "full", "shared", "router", "hc", "lm_head")


def fmt_of(dense_fmt, group):
    """dense_fmt: one format for every dense group, or {group: format} overrides on top of bfp8."""
    return dense_fmt.get(group, "bfp8") if isinstance(dense_fmt, dict) else dense_fmt


def dense_bytes(dense_fmt="bfp8"):
    p = params()
    return sum(p[g] * B[fmt_of(dense_fmt, g)] for g in DENSE)


DEVICE = dict(mem=32 * GiB, reserve=2 * GiB, bw=512 * GB,
              eff=0.43)   # calibrated in bmk-keyeast/blackhole-llm (published Llama 3.1 8B p150 decode)
HOSTS = {
    # Xeon E5-2690 v4: 14C/28T Broadwell-EP, AVX2+FMA, 4 x DDR4-2133 ECC RDIMM (4 channels), PCIe 3.0 x16
    "E5-2690v4, 4x16GB": dict(ram=64 * GiB, bw=4 * 2133e6 * 8),
    "E5-2690v4, 128GB": dict(ram=128 * GiB, bw=4 * 2133e6 * 8),
    # Xeon Platinum 8259CL (LGA3647): 24C Cascade Lake, AVX-512 VNNI, 6 x DDR4-2933, still PCIe 3.0.
    # Capacity is an assumption (6 x 32 GB).
    "8259CL, 6x32GB": dict(ram=192 * GiB, bw=6 * 2933e6 * 8),
}
PCIE = 15.75 * GB
# ponytail: one efficiency knob per link, replaced by bench/ numbers on the real box
CPU_EFF, PCIE_EFF, PCIE_RT_US = 0.70, 0.80, 30.0
HOST_RESERVE = 6 * GiB
NGRAM_BYTES_PER_PARAM = 28.8e9 / 51.2e9   # Strata's quantized N-gram shard: 28.8 GB for 51.2B params


def memory_map(dense_fmt="bfp8", expert_fmt="bfp4", ctx=32768, host=None):
    p, m = params(), QFN
    dense = dense_bytes(dense_fmt)
    kv = 2 * p["n_full"] * m["KV"] * m["hd"] * B["bfp8"] * ctx
    state = p["n_lin"] * m["lin_v_heads"] * m["lin_hd"] * m["lin_hd"] * 4   # fp32 recurrent state
    eb = p["expert"] * B[expert_fmt]
    room = DEVICE["mem"] - DEVICE["reserve"] - dense - kv - state
    n_dev = max(0, min(p["n_experts"], int(room // eb)))
    host_need = (p["n_experts"] - n_dev) * eb + p["embed"] * B["bf16"] + p["mtp_experts"] * B[expert_fmt]
    ngram = p["ngram"] * NGRAM_BYTES_PER_PARAM
    free = host["ram"] - HOST_RESERVE - host_need if host else 0
    return dict(dense=dense, kv=kv, state=state, eb=eb, n_dev=n_dev, f=n_dev / p["n_experts"], host_need=host_need,
                ngram=ngram, fits=host_need <= (host["ram"] - HOST_RESERVE) if host else None,
                ngram_in_ram=free >= ngram if host else None)


def decode(host, hit=None, dense_fmt="bfp8", expert_fmt="bfp4", ctx=4096, promote=0):
    """Tokens/s. hit = share of activated routed experts that are on the card (None = uniform routing).
    promote = experts copied host -> card per token by an adaptive cache; the DMA overlaps compute, but its reads
    share host DRAM bandwidth with the cold experts, and the link must finish them within the token."""
    p, m = params(), QFN
    mm = memory_map(dense_fmt, expert_fmt, ctx, host)
    h = mm["f"] if hit is None else hit
    dev_bw, cpu_bw = DEVICE["bw"] * DEVICE["eff"], host["bw"] * CPU_EFF
    tok_experts = m["L"] * m["k"] * mm["eb"]
    t_dense = (mm["dense"] + mm["kv"] + mm["state"]) / dev_bw
    t_hot = h * tok_experts / dev_bw
    t_cold = ((1 - h) * tok_experts + promote * mm["eb"]) / cpu_bw
    t_pcie = m["L"] * (PCIE_RT_US * 1e-6 + 2 * m["d"] * 2 / (PCIE * PCIE_EFF))
    t = max(t_dense + max(t_hot, t_cold + t_pcie), t_pcie + promote * mm["eb"] / (PCIE * PCIE_EFF))
    return 1 / t, dict(dense=t_dense, hot=t_hot, cold=t_cold, pcie=t_pcie)


DRAFT_VOCAB = 106_299   # ponytail: Basalt's convert/draft_vocab.bin read as int32 ids; replace with the real count


def decode_mtp(host, hot, cold, m, accepted, dense_fmt="bfp8", expert_fmt="bfp4", ctx=4096, promote=0):
    """Tokens/s with MTP speculative decoding. Each step drafts m-1 tokens with the MTP layer, then verifies m tokens
    in one pass, so the card reads its dense weights once per step instead of once per token.
    hot, cold: routed experts per layer that the m tokens need together, on the card and on the CPU (measured by
    trace.py). accepted: tokens emitted per step (1 + accepted drafts)."""
    p, q = params(), QFN
    mm = memory_map(dense_fmt, expert_fmt, ctx, host)
    dev_bw, cpu_bw = DEVICE["bw"] * DEVICE["eff"], host["bw"] * CPU_EFF
    b = {g: B[fmt_of(dense_fmt, g)] for g in DENSE}
    # ponytail: card compute ignored; m <= 8 tokens stays bandwidth-bound on 120 Tensix cores
    t_dense = (mm["dense"] + mm["kv"] * m + mm["state"]) / dev_bw
    t_hot = q["L"] * hot * mm["eb"] / dev_bw
    t_cold = (q["L"] * cold + promote) * mm["eb"] / cpu_bw         # promote: adaptive-cache copies per step
    t_pcie = q["L"] * (PCIE_RT_US * 1e-6 + 2 * m * q["d"] * 2 / (PCIE * PCIE_EFF))
    # one draft token: the MTP layer (full attention, shared expert, router, hyper-connections, two d x d input
    # projections, draft head over the reduced vocabulary) on the card, its 10 routed experts on the CPU
    mtp_dense = (p["attn_full"] * b["full"] + 3 * q["d"] * q["S"] * b["shared"] + q["d"] * q["E"] * b["router"]
                 + p["hc"] / q["L"] * b["hc"] + (2 * q["d"] ** 2 + DRAFT_VOCAB * q["d"]) * b["lm_head"])
    t_draft = mtp_dense / dev_bw + q["k"] * mm["eb"] / cpu_bw + PCIE_RT_US * 1e-6
    t = t_dense + max(t_hot, t_cold + t_pcie) + (m - 1) * t_draft
    return accepted / max(t, t_pcie + promote * mm["eb"] / (PCIE * PCIE_EFF))


def report():
    p = params()
    main = p["attn"] + p["shared"] + p["router"] + p["hc"] + p["lm_head"] + p["embed"] + p["n_experts"] * p["expert"]
    active = (p["attn"] + p["shared"] + p["router"] + p["hc"] + p["lm_head"] + p["embed"]
              + QFN["L"] * QFN["k"] * p["expert"])
    out = [f"## Qwen3.8-Flash-Next on p150a + Xeon\n",
           f"Parameters (this count): main model {main / 1e9:.1f}B (published: ~125B), active per token {active / 1e9:.1f}B "
           f"(published: ~6B; ours includes embedding and LM head), N-gram embeddings {p['ngram'] / 1e9:.1f}B (published: ~51B), "
           f"routed experts {p['n_experts']:,} x {p['expert'] / 1e6:.2f}M params.\n",
           "### Memory map (32k context)\n",
           "| Host | Dense on card | Routed experts on card | Cold experts + embeddings + MTP experts in host RAM | Fits | N-gram table (28.8 GB) |",
           "|---|---|---:|---:|:-:|---|"]
    for hn, h in HOSTS.items():
        for df in ["bfp8", "bfp4"]:
            mm = memory_map(df, "bfp4", 32768, h)
            out.append(f"| {hn} | {df}, {mm['dense'] / GB:.1f} GB | {mm['n_dev']:,} / {p['n_experts']:,} ({mm['f']:.0%}) | "
                       f"{mm['host_need'] / GB:.1f} GB | {'yes' if mm['fits'] else 'no'} | "
                       f"{'host RAM' if mm['ngram_in_ram'] else 'NVMe (mmap)'} |")
    out += ["\n### Decode speed, batch 1, 4k context, experts BFP4\n",
            f"Efficiencies: card {DEVICE['eff']:.0%} of 512 GB/s, DRAM {CPU_EFF:.0%} of peak, PCIe {PCIE_EFF:.0%}, "
            f"{PCIE_RT_US:.0f} us round trip per layer. 'Card hit rate' = share of the 10 activated experts per layer found "
            "on the card. Uniform routing gives the share of experts that fit; a measured popularity profile (Basalt ships "
            "one) raises it. MTP speculative decoding is not included.\n",
            "| Host | Dense format | uniform routing | 60% hits | 80% hits | time split at uniform: dense / hot / cold+PCIe (ms) |",
            "|---|---|---:|---:|---:|---|"]
    for hn, h in HOSTS.items():
        for df in ["bfp8", "bfp4"]:
            r0, t = decode(h, None, df)
            r6, _ = decode(h, 0.6, df)
            r8, _ = decode(h, 0.8, df)
            out.append(f"| {hn} | {df} | {r0:.1f} t/s | {r6:.1f} | {r8:.1f} | "
                       f"{1e3 * t['dense']:.1f} / {1e3 * t['hot']:.1f} / {1e3 * (t['cold'] + t['pcie']):.1f} |")
    return "\n".join(out)


def _check():
    p = params()
    main = p["attn"] + p["shared"] + p["router"] + p["hc"] + p["lm_head"] + p["embed"] + p["n_experts"] * p["expert"]
    assert 123e9 < main < 128e9, main                       # published: ~125B main model
    active = (p["attn"] + p["shared"] + p["router"] + p["hc"] + p["lm_head"] + p["embed"]
              + QFN["L"] * QFN["k"] * p["expert"])
    assert 5.5e9 < active < 7.5e9, active   # published "~6B"; ours also counts embedding + LM head (1.3B)
    assert sum(p[g] for g in DENSE) == p["attn"] + p["shared"] + p["router"] + p["hc"] + p["lm_head"]
    assert dense_bytes({g: "bfp4" for g in DENSE}) == dense_bytes("bfp4")
    assert abs(p["ngram"] / 1e9 - 51.2) < 0.1               # published: ~51B N-gram embeddings
    mm = memory_map(host=HOSTS["E5-2690v4, 4x16GB"])
    assert mm["fits"] and 0 < mm["f"] < 1
    r_uni, _ = decode(HOSTS["E5-2690v4, 4x16GB"])
    assert decode(HOSTS["E5-2690v4, 4x16GB"], 0.8)[0] > r_uni > 0
    # MTP with m = 1 and nothing accepted beyond the token itself is plain decode (no draft cost)
    h = memory_map(ctx=4096, host=HOSTS["E5-2690v4, 4x16GB"])["f"]    # decode()'s context
    assert abs(decode_mtp(HOSTS["E5-2690v4, 4x16GB"], 10 * h, 10 * (1 - h), 1, 1) - r_uni) < 1e-6 * r_uni


if __name__ == "__main__":
    _check()
    root = Path(__file__).parent
    (root / "results").mkdir(exist_ok=True)
    text = report()
    (root / "results" / "plan.md").write_text(text + "\n", encoding="utf-8")
    print(text)
