"""Expert-placement planner for MoE decode on one Blackhole p150a + a DDR4 Xeon host.

Split per decoder layer (batch-1 decode):
    device (p150a): attention, router, shared/dense MLPs, LM head, KV cache, and as many routed
                    experts as fit in the remaining GDDR6 ("hot" experts)
    host CPU:       every other routed expert, computed in place from DRAM ("cold" experts)
    PCIe:           only the hidden-state vector crosses the bus (to the host and back), never weights

    t_token = t_dense_dev + max(t_hot_dev, t_cold_cpu + t_pcie)

The alternative of streaming cold experts over PCIe to the card is modeled as well, because it shows
why the CPU path wins on this host: PCIe 3.0 x16 (~13 GB/s) is far slower than 4-6 channel DDR4.

    python plan.py        -> results/plan.md
"""
from pathlib import Path

GB = 1e9
GiB = 2**30
B = {"bf16": 2.0, "bfp8": 1088 / 1024, "bfp4": 576 / 1024}   # bytes/param, TT block-float formats

DEVICE = dict(name="p150a", mem=32 * GiB, reserve=2 * GiB, bw=512 * GB,
              # calibrated in bmk-keyeast/blackhole-llm: published Llama 3.1 8B p150 decode = 43% of peak
              eff=0.43)

HOSTS = {
    # E5-2690 v4 (Broadwell-EP): 14C/28T, AVX2+FMA, 4ch DDR4-2133 RDIMM, PCIe 3.0 x16 to the card
    "E5-2690v4 64GB": dict(ram=64 * GiB, bw=4 * 2133e6 * 8, pcie=15.75 * GB),
    "E5-2690v4 128GB": dict(ram=128 * GiB, bw=4 * 2133e6 * 8, pcie=15.75 * GB),
    # Xeon Platinum 8259CL (Cascade Lake, LGA3647): 24C, AVX-512 VNNI, 6ch DDR4-2933, still PCIe 3.0.
    # Capacity is an assumption (6 x 32 GB); change it to the real build.
    "8259CL 192GB": dict(ram=192 * GiB, bw=6 * 2933e6 * 8, pcie=15.75 * GB),
}
# ponytail: one efficiency knob per link. STREAM on 2-socket Broadwell/Cascade Lake typically reaches
# ~70-80% of peak per socket; PCIe 3.0 x16 DMA ~80%. Replace with numbers from bench/ on the real box.
CPU_EFF, PCIE_EFF = 0.70, 0.80
PCIE_RT_US = 30.0     # ponytail: host<->device round trip per layer, unknown until measured on the card
HOST_RESERVE = 6 * GiB  # OS, runtime, page cache, KV for prefill staging

# From each model's HF config.json. d=hidden, L=layers, H/KV/hd=attention, E=routed experts,
# k=experts per token, I=expert intermediate, S=shared-expert intermediate total, Dk/DI=leading dense
# layers and their MLP width, V=vocab.
MODELS = {
    "Qwen3-30B-A3B":     dict(d=2048, L=48, H=32, KV=4, hd=128, E=128, k=8,  I=768,  S=0,    Dk=0, DI=0,     V=151936),
    "Qwen3-Next-80B-A3B": dict(d=2048, L=48, H=16, KV=2, hd=256, E=512, k=10, I=512,  S=512,  Dk=0, DI=0,     V=151936),
    "gpt-oss-20b":       dict(d=2880, L=24, H=64, KV=8, hd=64,  E=32,  k=4,  I=2880, S=0,    Dk=0, DI=0,     V=201088),
    "gpt-oss-120b":      dict(d=2880, L=36, H=64, KV=8, hd=64,  E=128, k=4,  I=2880, S=0,    Dk=0, DI=0,     V=201088),
    "GLM-4.5-Air":       dict(d=4096, L=46, H=96, KV=8, hd=128, E=128, k=8,  I=1408, S=1408, Dk=1, DI=10944, V=151552),
    "DeepSeek-V2-Lite":  dict(d=2048, L=27, H=16, KV=16, hd=128, E=64, k=6,  I=1408, S=2816, Dk=1, DI=10944, V=102400),
    "Mixtral-8x7B":      dict(d=4096, L=32, H=32, KV=8, hd=128, E=8,   k=2,  I=14336, S=0,   Dk=0, DI=0,     V=32000),
    "Qwen3-235B-A22B":   dict(d=4096, L=94, H=64, KV=4, hd=128, E=128, k=8,  I=1536, S=0,    Dk=0, DI=0,     V=151936),
}
# Not modeled exactly (documented upper bounds): DeepSeek MLA and Qwen3-Next linear-attention layers are
# counted as plain GQA; gpt-oss expert biases and sliding-window KV savings are ignored.

EXPERT_FMT, DENSE_FMT, KV_FMT = "bfp4", "bfp8", "bfp8"


def parts(m):
    """Parameter counts by placement class."""
    moe_layers = m["L"] - m["Dk"]
    attn = m["L"] * (m["d"] * (m["H"] + 2 * m["KV"]) * m["hd"] + m["H"] * m["hd"] * m["d"])
    dense = moe_layers * m["d"] * m["E"] + moe_layers * 3 * m["d"] * m["S"] + m["Dk"] * 3 * m["d"] * m["DI"]
    expert = 3 * m["d"] * m["I"]
    return dict(attn=attn, dense=dense, lm_head=m["V"] * m["d"], embed=m["V"] * m["d"],
                expert=expert, n_experts=moe_layers * m["E"], moe_layers=moe_layers)


def plan(m, host, ctx=4096):
    p = parts(m)
    kv = 2 * m["L"] * m["KV"] * m["hd"] * B[KV_FMT] * ctx
    dev_fixed = (p["attn"] + p["dense"] + p["lm_head"]) * B[DENSE_FMT] + kv
    eb = p["expert"] * B[EXPERT_FMT]
    room = DEVICE["mem"] - DEVICE["reserve"] - dev_fixed
    n_dev = max(0, min(p["n_experts"], int(room // eb)))
    f = n_dev / p["n_experts"]               # uniform routing: P(an activated expert is on the card)
    host_bytes = (p["n_experts"] - n_dev) * eb + p["embed"] * B["bf16"]
    fits = room > 0 and host_bytes <= host["ram"] - HOST_RESERVE

    tok_expert_bytes = p["moe_layers"] * m["k"] * eb
    kv_read = kv  # full context read once per token
    dev_bw, cpu_bw, pcie_bw = DEVICE["bw"] * DEVICE["eff"], host["bw"] * CPU_EFF, host["pcie"] * PCIE_EFF
    t_dense = ((p["attn"] + p["dense"] + p["lm_head"]) * B[DENSE_FMT] + kv_read) / dev_bw
    t_hot = f * tok_expert_bytes / dev_bw
    t_cold = (1 - f) * tok_expert_bytes / cpu_bw
    t_pcie = p["moe_layers"] * (PCIE_RT_US * 1e-6 + 2 * m["d"] * 2 / pcie_bw)
    hybrid = t_dense + max(t_hot, t_cold + t_pcie)
    stream = t_dense + t_hot + (1 - f) * tok_expert_bytes / pcie_bw          # fetch cold experts to the card
    active = (p["attn"] + p["dense"] + p["lm_head"]) * B[DENSE_FMT] + kv_read + tok_expert_bytes
    cpu_only = active / cpu_bw
    total = (p["attn"] + p["dense"] + p["lm_head"] + p["n_experts"] * p["expert"] + p["embed"])
    return dict(total_params=total, dev_fixed=dev_fixed, n_dev=n_dev, n_exp=p["n_experts"], f=f,
                host_bytes=host_bytes, fits=fits, hybrid=1 / hybrid, stream=1 / stream, cpu_only=1 / cpu_only,
                t_dense=t_dense, t_hot=t_hot, t_cold=t_cold, t_pcie=t_pcie)


def table(ctx=4096):
    out = [f"Batch-1 decode, {ctx}-token context. Experts {EXPERT_FMT}, everything else {DENSE_FMT} on the card. "
           f"Uniform routing (no expert-popularity skew). Efficiencies: card {DEVICE['eff']:.0%}, DRAM {CPU_EFF:.0%}, "
           f"PCIe {PCIE_EFF:.0%}, {PCIE_RT_US:.0f} us PCIe round trip per MoE layer.\n"]
    for hname, h in HOSTS.items():
        out += [f"\n### {hname} (DRAM {h['bw'] / GB:.0f} GB/s peak, PCIe 3.0 x16)\n",
                "| Model | Params (B) | Experts on card | Host RAM for experts (GB) | Fits | CPU-only t/s | Stream-to-card t/s | **Hybrid t/s** | Hybrid time split: dense / hot / cold+PCIe (ms) |",
                "|---|---:|---:|---:|:-:|---:|---:|---:|---|"]
        for name, m in MODELS.items():
            r = plan(m, h, ctx)
            out.append(f"| {name} | {r['total_params'] / 1e9:.1f} | {r['n_dev']}/{r['n_exp']} ({r['f']:.0%}) | "
                       f"{r['host_bytes'] / GB:.1f} | {'yes' if r['fits'] else 'no'} | {r['cpu_only']:.1f} | "
                       f"{r['stream']:.1f} | **{r['hybrid']:.1f}** | {1e3 * r['t_dense']:.1f} / {1e3 * r['t_hot']:.1f} / "
                       f"{1e3 * (r['t_cold'] + r['t_pcie']):.1f} |")
    return "\n".join(out)


def _check():
    m = MODELS["Qwen3-30B-A3B"]
    p = parts(m)
    total = p["attn"] + p["dense"] + p["lm_head"] + p["embed"] + p["n_experts"] * p["expert"]
    assert abs(total / 1e9 - 30.5) < 0.2, total
    active = p["attn"] + p["dense"] + p["lm_head"] + p["embed"] + p["moe_layers"] * m["k"] * p["expert"]
    assert abs(active / 1e9 - 3.3) < 0.15, active
    g = MODELS["gpt-oss-120b"]
    pg = parts(g)
    assert abs((pg["attn"] + pg["lm_head"] + pg["embed"] + pg["n_experts"] * pg["expert"]) / 1e9 - 116.8) < 1.0
    assert plan(m, HOSTS["E5-2690v4 64GB"])["f"] == 1.0          # 30B-A3B experts fit on the card entirely
    r = plan(g, HOSTS["E5-2690v4 64GB"])
    assert r["fits"] and 0 < r["f"] < 1 and r["hybrid"] > r["stream"] > 0


if __name__ == "__main__":
    _check()
    root = Path(__file__).parent
    (root / "results").mkdir(exist_ok=True)
    text = table()
    (root / "results" / "plan.md").write_text(text + "\n", encoding="utf-8")
    print(text)
