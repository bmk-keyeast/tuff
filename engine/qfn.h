// Qwen3.8-Flash-Next decode in C++: device/qfn.py's Card, line for line, over ops.h, with libtuff's cache (tcache) and
// expert store (tstore + moe_forward_n) linked in directly. engine/test_engine.py checks it against the Python card code.
#pragma once
#include <map>
#include <string>
#include <vector>

#include "ops.h"

extern "C" {
#include "../kernels/cache.h"
}

namespace tuff {

struct Config {
    int d, hc, head_dim, H, KV, layers, lin_k_heads, lin_v_heads, lin_k_dim, lin_v_dim, conv_k, ple_conv_k, ngram,
        heads_per_ngram, ix_heads, ix_dim, ix_budget, ix_ratio, E, top_k, I, S, vocab, eos;
    float eps, rope_theta, partial_rotary;
    bool gated_act, norm_topk;
    std::vector<std::string> layer_types;              // "linear_attention" / "full_attention"
    std::vector<int> ple_layer_ids;                    // as in the HF config: the PLE runs before layer id - 1
    static Config load(const std::string& path);       // "key value ..." lines (engine/export.py writes them)
};

struct Array {
    std::vector<int64_t> shape;
    std::vector<float> f;                              // float tensors
    std::vector<int64_t> i;                            // integer tensors (N-gram multipliers, sizes, offsets)
    size_t row_size() const;
};

struct Weights {
    std::map<std::string, Array> t;
    const Array& at(const std::string& name) const;
    static Weights load(const std::string& path);      // engine/export.py's binary format
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
};

// Cold experts from the converted BFP store (loader.py convert), one moe_forward_n call per layer and step.
struct StoreHost : Host {
    void* store = nullptr;                             // tstore
    StoreHost(const Config& c, const Weights& w, const std::string& dir, const std::string& fmt);
    ~StoreHost() override;
    void experts(int L, const float* X, int m, const std::vector<int>& es, const std::vector<float>& G,
                 float* Y) const override;
};

struct Drafter {                                       // speculative decoding (QFN's MTP layer goes here)
    virtual ~Drafter() = default;
    virtual void catch_up(struct Card& card, int start, int j) = 0;
    virtual std::vector<int> propose(struct Card& card, int nxt, int k) = 0;
};

struct Card {
    Ops& o;
    const Config& c;
    const Weights& w;
    Host& host;
    std::string dense_fmt, expert_fmt;
    int max_t, n_slots, trace = 8, prefill_chunk = 512, pos0 = 0;
    tcache cache{};
    int t = 0, steps = 0;
    long hits = 0, lookups = 0;
    std::vector<int> tokens, verified;
    Tensor h4;

    Card(Ops& o, const Config& c, const Weights& w, Host& host, const std::string& dense_fmt,
         const std::string& expert_fmt, int n_slots, int budget, int max_t = 4096);
    ~Card();
    Tensor verify(const std::vector<int>& toks);       // m tokens in one pass -> mixer outputs [m, d]
    void commit(int j);                                // keep the first j
    Tensor step(int tok);
    std::vector<int> lm_argmax(const Tensor& mixed);
    std::vector<int> generate(const std::vector<int>& prompt, int n, Drafter* drafter = nullptr, int k = 2);

  private:
    struct State { Tensor conv, S, K, V, idx, blk, ple; };
    struct Pending { Tensor win, states, ple; bool lin = false, has_ple = false; };
    std::vector<State> st;
    std::vector<Pending> pend;
    std::map<std::string, Tensor> dev;                 // weights and constants on the device
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
    std::pair<Tensor, Tensor> cs(int pos, int n = 1);
    Tensor rope(const Tensor& x, const Tensor& cos, const Tensor& sin);
    std::pair<Tensor, Tensor> hc_read(const std::string& p, const Tensor& h4, bool combine = true);
    Tensor hc_write(const Tensor& h4, const Tensor& inj, const Tensor& out);
    Tensor conv(const Tensor& win, const Tensor& cw, int m, int step);
    std::vector<Tensor> deltanet_core(int L, const Tensor& x, const Tensor& conv, const Tensor& S);
    Tensor attention(int L, const Tensor& x);
    std::pair<Tensor, Tensor> dense_mlp(int L, const Tensor& x);
    Tensor moe(int L, const Tensor& x, const Tensor& probs, Tensor y);
    Tensor slot_experts(const Tensor& x, const std::vector<std::pair<int, std::vector<float>>>& hot);
    void put_slot(int slot, int L, int e);
    void promote();
    Tensor ple_layer(int L, const Tensor& h4, const std::vector<int>& toks);
    Tensor layers(Tensor h4, const std::vector<int>& toks);
};

}  // namespace tuff
