// Qwen3.8-Flash-Next decode in C++: device/qfn.py (Card, MTPDrafter) and device/tp.py (TP) over ops.h, with
// libtuff's cache (tcache) and expert store (tstore + moe_forward_n) linked in. engine/test_engine.py checks every
// piece against the Python card code.
#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ops.h"

extern "C" {
#include "../kernels/cache.h"
}

namespace tuff {

struct Config {
    int d, hc, head_dim, H, KV, layers, lin_k_heads, lin_v_heads, lin_k_dim, lin_v_dim, conv_k, ple_conv_k, ngram,
        heads_per_ngram, ix_heads, ix_dim, ix_budget, ix_ratio, E, top_k, I, S, vocab, eos, ngram_base = 0;
    float eps, rope_theta, partial_rotary;
    bool gated_act, norm_topk;
    std::vector<std::string> layer_types;              // "linear_attention" / "full_attention"
    std::vector<int> ple_layer_ids;                    // as in the HF config: the PLE runs before layer id - 1
    static Config load(const std::string& path);       // "key value ..." lines (engine/test_engine.py writes them)
    static Config load_hf(const std::string& config_json);   // a Hugging Face snapshot's config.json (text_config)
    Config mtp() const;                                // QFN's MTP layer: one full-attention MoE layer, no PLE
};

struct Array {
    std::vector<int64_t> shape;
    std::vector<float> f;                              // float tensors
    std::vector<int64_t> i;                            // integer tensors (N-gram multipliers, sizes, offsets)
    size_t row_size() const;
};

// Weights by name: from a TUFFW1 file (tests), or lazily from a Hugging Face snapshot's safetensors (mmap'ed;
// bf16 widened on use). Dense tensors are cached until drop(); experts and table rows are read per use.
struct Weights {
    mutable std::map<std::string, Array> t;
    struct Src;
    std::shared_ptr<Src> src;
    std::string prefix;                                // the snapshot's name prefix ("model.language_model.", "mtp.")

    bool has(const std::string& name) const;
    const Array& at(const std::string& name) const;
    void drop(const std::string& name) const { t.erase(name); }
    std::vector<float> expert(const std::string& name, int e) const;   // [E, ...] -> expert e
    std::vector<float> row(const std::string& name, int64_t r) const;  // a table's row (sharded tables too)
    static Weights load(const std::string& path);
    static Weights open_hf(const std::string& dir, const std::string& prefix, const Config& c);
    Weights shard() const;                             // a copy sharing the source (device/tp.py's shard() edits it)
};

// The host side: embeddings, N-gram rows, the experts not in a slot.
struct Host {
    const Config& c;
    const Weights& w;
    Host(const Config& c, const Weights& w) : c(c), w(w) {}
    virtual ~Host() = default;
    std::vector<float> embed(int tok) const;
    std::vector<float> ngram_rows(int L, const std::vector<int>& tokens) const;     // tokens: all so far, current last
    // a layer's cold experts for m tokens: X [m][d], es: k experts, G [k][m] gate weights -> Y [m][d] (added to)
    virtual void experts(int L, const float* X, int m, const std::vector<int>& es, const std::vector<float>& G,
                         float* Y) const;
    virtual const void* expert_tiles(int, int, size_t&) const { return nullptr; }    // TT tiles of an expert, if any
    struct Bfp { const uint8_t *exp, *mant; int out, in, mbits; };
    virtual bool expert_bfp(int, int, Bfp*) const { return false; }  // gate, up, down in the host layout, if any
};

// Cold experts from the converted BFP store (loader.py convert), one moe_forward_n call per layer and step; slot
// fills as TT tiles straight from it.
struct StoreHost : Host {
    void* store = nullptr;                             // tstore
    mutable std::vector<uint8_t> tiles;
    StoreHost(const Config& c, const Weights& w, const std::string& dir, const std::string& fmt);
    ~StoreHost() override;
    void experts(int L, const float* X, int m, const std::vector<int>& es, const std::vector<float>& G,
                 float* Y) const override;
    const void* expert_tiles(int L, int e, size_t& bytes) const override;
    bool expert_bfp(int L, int e, Bfp* out) const override;
};

struct Decoder;

struct Drafter {                                       // speculative decoding
    virtual ~Drafter() = default;
    virtual void catch_up(Decoder& card, int start, int j) = 0;
    virtual std::vector<int> propose(Decoder& card, int nxt, int k) = 0;
};

// What generate() and a drafter need: one device (Card) or the card + GPU split (TP).
struct Decoder {
    Ops& o;                                            // where h4 lives
    const Weights& w;                                  // the main model's (the draft head takes LM head rows)
    int t = 0, steps = 0, prefill_chunk = 512;
    long hits = 0, lookups = 0;
    double t_prompt = 0;
    std::vector<int> tokens, verified;
    Tensor h4;
    Decoder(Ops& o, const Weights& w) : o(o), w(w) {}
    virtual ~Decoder() = default;
    virtual Tensor verify(const std::vector<int>& toks) = 0;   // m tokens in one pass -> mixer outputs [m, d]
    virtual void commit(int j) = 0;                             // keep the first j
    virtual std::vector<int> lm_argmax(const Tensor& mixed) = 0;
    Tensor step(int tok);
    std::vector<int> generate(const std::vector<int>& prompt, int n, Drafter* drafter = nullptr, int k = 2);
};

struct Card : Decoder {
    const Config& c;
    Host& host;
    std::string dense_fmt, expert_fmt;
    int max_t, n_slots, trace = 8, pos0 = 0;
    tcache cache{};

    Card(Ops& o, const Config& c, const Weights& w, Host& host, const std::string& dense_fmt,
         const std::string& expert_fmt, int n_slots, int budget, int max_t = 4096);
    ~Card() override;
    Tensor verify(const std::vector<int>& toks) override;
    void commit(int j) override;
    std::vector<int> lm_argmax(const Tensor& mixed) override;

    // the pieces, public for TP and the drafter
    struct State { Tensor conv, S, K, V, idx, blk, ple; };
    struct Pending { Tensor win, states, ple; bool lin = false, has_ple = false; };
    using Carry = std::vector<Tensor>;                 // (inject weights, MoE output) of the previous layer, or empty
    std::vector<State> st;
    std::vector<Pending> pend;
    std::map<std::string, Tensor> dev;
    std::map<std::string, std::vector<std::pair<Tensor, std::vector<std::pair<std::string, int>>>>> fused;
    Tensor banks[3];
    bool have_banks = false;
    int rd;
    std::vector<double> inv_freq;

    std::string fmt(const std::string& name) const;
    Tensor lin(const std::string& name);
    std::vector<Tensor> lins(const Tensor& x, const std::vector<std::string>& names);
    Tensor norm(const std::string& name, bool plus_one = true, int rows = 1);
    Tensor constant(const std::string& key, const std::vector<float>& v, Shape s);
    Tensor scalar(float v);
    Tensor rms(const Tensor& x, const Tensor& wt) { return o.rms_norm(x, wt, c.eps); }
    std::vector<Tensor> seg(const std::string& name, const Ops::Fn& fn, const std::vector<Tensor>& args, int m);
    std::pair<Tensor, Tensor> cs(int pos, int n = 1);
    Tensor rope(const Tensor& x, const Tensor& cos, const Tensor& sin);
    std::pair<Tensor, Tensor> hc_read(const std::string& p, const Tensor& h4, bool combine = true);
    Tensor hc_write(const Tensor& h4, const Tensor& inj, const Tensor& out);
    Tensor settle(const Tensor& h4, const Carry& carry) { return carry.empty() ? h4 : hc_write(h4, carry[0], carry[1]); }
    Tensor conv(const Tensor& win, const Tensor& cw, int m, int step);
    std::vector<Tensor> deltanet_core(int L, const Tensor& x, const Tensor& conv, const Tensor& S);
    Tensor deltanet(int L, const Tensor& x);           // with its pending state (TP)
    std::vector<Tensor> attn_prep(int L, const Tensor& x, const Tensor& cos, const Tensor& sin);
    std::vector<Tensor> attention_core(int L, const std::vector<Tensor>& prep, int m);   // per-token outputs
    Tensor attention_all(int L, const Tensor& iq, const Tensor& q, int m);
    Tensor attn_out(int L, const Tensor& gate, const std::vector<Tensor>& outs);
    Tensor attention(int L, const Tensor& x);          // eager, whole (TP)
    std::pair<Tensor, Tensor> dense_mlp(int L, const Tensor& x, bool router = true);
    void route(int L, const Tensor& probs, int m, std::vector<float>& gate, std::vector<int>& es);
    Tensor moe(int L, const Tensor& x, const Tensor& probs, Tensor y);
    Tensor slot_experts(const Tensor& x, const std::vector<std::pair<int, std::vector<float>>>& hot);
    void put_slot(int slot, int L, int e);
    void promote();
    Tensor ple_layer(int L, const Tensor& h4, const std::vector<int>& toks);
    std::pair<Tensor, Carry> deltanet_layer(int L, const Tensor& h4, const Carry& carry);
    std::pair<Tensor, Carry> attention_layer(int L, const Tensor& h4, const Carry& carry);
    Tensor layers(Tensor h4, const std::vector<int>& toks);
};

// QFN's MTP layer (device/qfn.py's MTPDrafter): its own one-layer Card, the main model's LM head or a draft vocabulary.
struct MTPDrafter : Drafter {
    Config mc;
    std::unique_ptr<Card> m;
    std::vector<int> vocab;                            // empty: the main model's whole vocabulary
    Tensor R, head;
    bool have_head = false;
    Decoder* lm = nullptr;
    MTPDrafter(Ops& o, const Config& c, const Weights& w_mtp, Host& host, const std::string& dense_fmt,
               const std::string& expert_fmt, int n_slots, int budget, int max_t, std::vector<int> vocab = {});
    std::pair<Tensor, int> run(const Tensor& R, const std::vector<int>& toks, int row, bool guess);
    void catch_up(Decoder& card, int start, int j) override;
    std::vector<int> propose(Decoder& card, int nxt, int k) override;
};

// The p150a and an NVIDIA GPU together (device/tp.py): the dense part split by heads and rows, one expert cache over
// both devices' slots, partial sums through the host.
struct TP : Decoder {
    const Config& c;
    Host& host;
    Config ca, cb;                                     // each device's share of the heads
    Weights wa, wb;
    std::unique_ptr<Card> A, B;
    int n_card, v_gpu;
    tcache cache{};
    TP(Ops& card, Ops& gpu, const Config& c, const Weights& w, Host& host, float f, int n_card, int n_gpu, int budget,
       const std::string& dense_fmt, const std::string& expert_fmt, int max_t = 4096);
    ~TP() override;
    Tensor verify(const std::vector<int>& toks) override;
    void commit(int j) override;
    std::vector<int> lm_argmax(const Tensor& mixed) override;
    Tensor to(Card& dst, const Tensor& x, Card& src) { return dst.o.put(src.o.get(x), x.shape); }
    Tensor allreduce(const Tensor& a, const Tensor& b) { return A->o.add(a, to(*A, b, *B)); }
    Tensor moe(int L, const Tensor& x);
};

}  // namespace tuff
