"""device/qfn.py on ops.Torch() == ref.py, token by token, on a tiny random QFN (no transformers needed).

    python device/test_qfn.py

1. formats f32: the card code must reproduce ref.py (float64) to float32 rounding, with experts split between card slots
   and the host and promoted between tokens.
2. tuff's formats (dense: plan's default mix; experts BFP4, TT-exact): how far the card's rounding moves the output.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "device"))
import ref  # noqa: E402
import qfn  # noqa: E402
from ops import Torch  # noqa: E402


def tiny():
    c = SimpleNamespace(
        hidden_size=128, hc_count=4, hc_lowrank=16, rms_norm_eps=1e-6, head_dim=32, num_attention_heads=4,
        num_key_value_heads=2, rope_parameters=dict(rope_theta=1e7, partial_rotary_factor=0.25),
        output_gate_type="sigmoid", hidden_act="silu", eos_token_id=999,
        layer_types=["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"],
        num_hidden_layers=8, linear_num_key_heads=4, linear_num_value_heads=12, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_conv_kernel_dim=4, ple_layer_ids=[2], ple_conv_kernel_size=4, ngram_size=3,
        heads_per_ngram=8, ple_embed_dim=128, indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=16,
        indexer_budget=16, indexer_compress_ratio=4, num_experts=16, num_experts_per_tok=4, norm_topk_prob=True,
        moe_intermediate_size=32, shared_expert_intermediate_size=32, vocab_size=1000)
    return c


def weights(c, seed=0):
    rng = np.random.default_rng(seed)
    d, hc, r = c.hidden_size, c.hc_count, c.hc_lowrank
    w = {}

    def mat(name, out, inn):
        w[name] = rng.standard_normal((out, inn)) / np.sqrt(inn)

    def vec(name, n, s=0.3):
        w[name] = s * rng.standard_normal(n)

    def hyper(p, combine=True):
        vec(p + ".hc_norm.weight", hc * d)
        mat(p + ".input_mix_weight_down.weight", r, hc * d)
        mat(p + ".input_mix_weight_up.weight", hc * d, r)
        if combine:
            mat(p + ".block_inject_weight.weight", hc, hc * d)

    w["embed_tokens.weight"] = rng.standard_normal((c.vocab_size, d)) / np.sqrt(d)
    K, V, dk, dv = c.linear_num_key_heads, c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim
    H, KV, hd, nh, ihd = c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.indexer_n_heads, c.indexer_head_dim
    E, I, S = c.num_experts, c.moe_intermediate_size, c.shared_expert_intermediate_size
    for L, kind in enumerate(c.layer_types):
        p = f"layers.{L}."
        hyper(p + "attn_hyper_connection")
        hyper(p + "mlp_hyper_connection")
        if kind == "linear_attention":
            C = 2 * K * dk + V * dv
            mat(p + "linear_attn.in_proj_qkv.weight", C, d)
            mat(p + "linear_attn.in_proj_z.weight", V * dv, d)
            mat(p + "linear_attn.in_proj_b.weight", V, d)
            mat(p + "linear_attn.in_proj_a.weight", V, d)
            w[p + "linear_attn.conv1d.weight"] = 0.5 * rng.standard_normal((C, 1, c.linear_conv_kernel_dim))
            w[p + "linear_attn.A_log"] = np.log(rng.uniform(1, 16, V))
            vec(p + "linear_attn.dt_bias", V)
            vec(p + "linear_attn.norm.weight", dv)
            mat(p + "linear_attn.out_proj.weight", d, V * dv)
        else:
            a = p + "self_attn."
            mat(a + "q_proj.weight", H * 2 * hd, d)
            mat(a + "k_proj.weight", KV * hd, d)
            mat(a + "v_proj.weight", KV * hd, d)
            mat(a + "o_proj.weight", d, H * hd)
            vec(a + "q_norm.weight", hd)
            vec(a + "k_norm.weight", hd)
            mat(a + "indexer.index_qk_proj.weight", (nh + c.indexer_kv_heads) * ihd, d)
            vec(a + "indexer.q_layernorm.weight", ihd)
            vec(a + "indexer.k_layernorm.weight", ihd)
        m = p + "mlp."
        mat(m + "gate.weight", E, d)
        w[m + "experts.gate_up_proj"] = rng.standard_normal((E, 2 * I, d)) / np.sqrt(d)
        w[m + "experts.down_proj"] = rng.standard_normal((E, d, I)) / np.sqrt(I)
        mat(m + "shared_expert.gate_proj.weight", S, d)
        mat(m + "shared_expert.up_proj.weight", S, d)
        mat(m + "shared_expert.down_proj.weight", d, S)
        mat(m + "shared_expert_gate.weight", 1, d)
    for i, Lp in enumerate(c.ple_layer_ids):
        p = f"layers.{Lp - 1}.ple."
        heads = (c.ngram_size - 1) * c.heads_per_ngram
        mat(p + "key_proj.weight", hc * d, c.ple_embed_dim)
        mat(p + "value_proj.weight", d, c.ple_embed_dim)
        for n in ("norm_key", "norm_query", "norm_conv"):
            vec(p + n + ".weight", hc * d)
        w[p + "conv1d.weight"] = 0.5 * rng.standard_normal((hc * d, 1, c.ple_conv_kernel_size))
        primes = [q for q in range(1000, 2000) if all(q % f for f in range(2, int(q ** 0.5) + 1))][:heads]
        w[p + "ple_embedding.layer_multipliers"] = 2 * rng.integers(1, 1 << 60, c.ngram_size) + 1
        w[p + "ple_embedding.ngram_heads_vocab_sizes"] = np.array(primes)
        w[p + "ple_embedding.ngram_heads_offsets"] = np.concatenate([[0], np.cumsum(primes)[:-1]])
        w[p + "ple_embedding.ngram_embedding.weight"] = rng.standard_normal((sum(primes), c.ple_embed_dim // heads))
    hyper("hyper_connection_mixer", combine=False)
    return w


def run(card, T, ids):
    return np.stack([card.o.get(card.step(int(t)))[0] for t in ids[:T]])


def main(T=40):
    import plan  # noqa: F401
    c = tiny()
    w = weights(c)
    ids = np.random.default_rng(1).integers(0, c.vocab_size, T)
    r = ref.Ref(w, c)
    want = np.stack([r.step(int(t)) for t in ids])
    n_slots = c.num_hidden_layers * c.num_experts * 3 // 10                     # 30% of the experts fit on the card
    card = qfn.Card(Torch(), w, c, qfn.Host(w, c), dense_fmt="f32", expert_fmt="f32", slots=n_slots, budget=4)
    got = run(card, T, ids)
    err = (np.abs(got - want).max(1) / np.abs(want).max(1)).max()
    same = np.mean([set(card.routes[k].tolist()) == set(r.routes[k].tolist()) for k in r.routes])
    print(f"card code (ops.Torch, f32) == ref.py: {T} tokens, max relative error {err:.1e}, same experts {same:.1%}, "
          f"card slot hits {card.hits / card.lookups:.1%}")
    assert err < 1e-4 and same == 1.0, (err, same)

    tuff = qfn.Card(Torch(), w, c, qfn.Host(w, c), dense_fmt={"lm_head": "bfp4", "lin_z": "bfp4"}, expert_fmt="bfp4",
                    slots=n_slots, budget=4)
    got = run(tuff, T, ids)
    err = np.median(np.abs(got - want).max(1) / np.abs(want).max(1))
    same = np.mean([len(set(tuff.routes[k].tolist()) & set(r.routes[k].tolist())) / c.num_experts_per_tok for k in r.routes])
    print(f"card code with tuff's formats (dense BFP8, LM head + DeltaNet z BFP4, card experts BFP4): median relative "
          f"error {err:.1e}, routing agreement {same:.1%}")
    assert np.isfinite(got).all()


if __name__ == "__main__":
    main()
