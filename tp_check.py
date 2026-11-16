"""The tensor-parallel split of plan.split_multi is exact: run QFN's own layer code (transformers' Qwen4Exp modules,
small random weights) whole and as two halves, and check that the halves' outputs add up to the whole.

    python tp_check.py

Splits (device 0 = p150a, device 1 = GPU; each keeps a copy of what is not split):
  Gated DeltaNet   by key-head groups: device h gets key heads h*K/2.., their value heads (each key head serves
                   V/K consecutive value heads), the matching conv channels, z/a/b rows, A_log, dt_bias, and
                   out_proj columns. Conv, delta rule and gated norm are all per head, so nothing crosses devices
                   until out_proj's partial sums.
  Full attention   by KV head: device h gets KV head h and the query heads that share it (q rows incl. the output
                   gate, k/v rows, o_proj columns). The QSA indexer picks the same tokens for every head, so both
                   devices run a copy of it (1.6M params per layer).
  Shared expert    by intermediate rows (gate/up rows, down columns); the scalar shared_expert_gate is copied.
Router, hyper-connection mixers and norms are copied (plan.REPLICATED); routed experts are whole on one device.
"""
import copy

import torch

REPO_ID = "Qwen/Qwen3.8-Flash-Next"


def small_config():
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(REPO_ID).text_config
    for k, v in dict(hidden_size=128, head_dim=32, num_attention_heads=4, num_key_value_heads=2, indexer_n_heads=2,
                     indexer_kv_heads=1, indexer_head_dim=16, indexer_budget=16, indexer_compress_ratio=4,
                     linear_num_key_heads=4, linear_num_value_heads=12, linear_key_head_dim=16,
                     linear_value_head_dim=16, shared_expert_intermediate_size=64).items():
        setattr(cfg, k, v)
    cfg.rope_parameters = dict(cfg.rope_parameters, mrope_section=[2, 1, 1])
    cfg._attn_implementation = "sdpa"
    return cfg


def randomize(module, seed):
    g = torch.Generator().manual_seed(seed)
    for p in module.parameters():
        p.data = 0.2 * torch.randn(p.shape, generator=g)


def check(name, full, halves):
    err = (sum(halves) - full).abs().max().item() / full.abs().max().item()
    print(f"{name}: two halves == whole, max relative error {err:.1e}")
    assert err < 1e-5, err


def main():
    from transformers.models.qwen4_exp import modeling_qwen4_exp as M
    torch.manual_seed(0)
    cfg = small_config()
    T, d = 48, cfg.hidden_size
    x = torch.randn(1, T, d)

    # Gated DeltaNet
    lin = next(i for i, t in enumerate(cfg.layer_types) if t == "linear_attention")
    full = M.Qwen4ExpTextGatedDeltaNet(cfg, lin)
    randomize(full, 1)
    K, V, dk, dv = cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
    hc = copy.deepcopy(cfg)
    hc.linear_num_key_heads, hc.linear_num_value_heads = K // 2, V // 2
    outs = []
    for h in range(2):
        part = M.Qwen4ExpTextGatedDeltaNet(hc, lin)
        ks, vs = slice(h * K // 2 * dk, (h + 1) * K // 2 * dk), slice(h * V // 2 * dv, (h + 1) * V // 2 * dv)
        ch = torch.cat([torch.arange(K * dk)[ks], K * dk + torch.arange(K * dk)[ks], 2 * K * dk + torch.arange(V * dv)[vs]])
        heads = slice(h * V // 2, (h + 1) * V // 2)
        sd = full.state_dict()
        part.load_state_dict({
            "in_proj_qkv.weight": sd["in_proj_qkv.weight"][ch], "conv1d.weight": sd["conv1d.weight"][ch],
            "in_proj_z.weight": sd["in_proj_z.weight"][vs], "in_proj_a.weight": sd["in_proj_a.weight"][heads],
            "in_proj_b.weight": sd["in_proj_b.weight"][heads], "A_log": sd["A_log"][heads],
            "dt_bias": sd["dt_bias"][heads], "norm.weight": sd["norm.weight"], "out_proj.weight": sd["out_proj.weight"][:, vs]})
        outs.append(part(x))
    check("Gated DeltaNet, split by key-head groups", full(x), outs)

    # Full attention with the QSA indexer
    att = next(i for i, t in enumerate(cfg.layer_types) if t != "linear_attention")   # "indexed_attention"
    full = M.Qwen4ExpTextAttention(cfg, att)
    randomize(full, 2)
    H, KV, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    rot = M.Qwen4ExpTextRotaryEmbedding(cfg)
    pe = rot(x, torch.arange(T)[None])
    mask = torch.ones(T, T, dtype=torch.bool).tril()[None, None]
    hc = copy.deepcopy(cfg)
    hc.num_attention_heads, hc.num_key_value_heads = H // 2, KV // 2
    outs = []
    for h in range(2):
        part = M.Qwen4ExpTextAttention(hc, att)
        sd = full.state_dict()
        qs, kvs, os_ = (slice(h * H // 2 * 2 * hd, (h + 1) * H // 2 * 2 * hd), slice(h * KV // 2 * hd, (h + 1) * KV // 2 * hd),
                        slice(h * H // 2 * hd, (h + 1) * H // 2 * hd))
        new = {k: v for k, v in sd.items() if k.startswith(("indexer.", "q_norm.", "k_norm."))}
        new.update({"q_proj.weight": sd["q_proj.weight"][qs], "k_proj.weight": sd["k_proj.weight"][kvs],
                    "v_proj.weight": sd["v_proj.weight"][kvs], "o_proj.weight": sd["o_proj.weight"][:, os_]})
        part.load_state_dict(new)
        outs.append(part(x, pe, mask)[0])
    check("full attention + indexer, split by KV head", full(x, pe, mask)[0], outs)

    # Shared expert
    I = cfg.shared_expert_intermediate_size
    full = M.Qwen4ExpTextMLP(cfg, intermediate_size=I)
    randomize(full, 3)
    outs = []
    for h in range(2):
        part = M.Qwen4ExpTextMLP(cfg, intermediate_size=I // 2)
        r = slice(h * I // 2, (h + 1) * I // 2)
        sd = full.state_dict()
        part.load_state_dict({"gate_proj.weight": sd["gate_proj.weight"][r], "up_proj.weight": sd["up_proj.weight"][r],
                              "down_proj.weight": sd["down_proj.weight"][:, r]})
        outs.append(part(x))
    check("shared expert, split by intermediate rows", full(x), outs)
    print("tensor-parallel split is exact")


if __name__ == "__main__":
    with torch.no_grad():
        main()
