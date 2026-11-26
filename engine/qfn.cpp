// See qfn.h. Each method follows the device/qfn.py method of the same name; comments there explain the why.
#include "qfn.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <set>
#include <sstream>

extern "C" {
#include "../kernels/bfp.h"
#include "../kernels/store.h"
}

namespace tuff {

// ---------------------------------------------------------------- config and weights
Config Config::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::map<std::string, std::vector<std::string>> kv;
    for (std::string line; std::getline(f, line);) {
        std::istringstream ss(line);
        std::string k, v;
        ss >> k;
        while (ss >> v) kv[k].push_back(v);
    }
    auto I = [&](const char* k) { return std::stoi(kv.at(k).at(0)); };
    auto F = [&](const char* k) { return std::stof(kv.at(k).at(0)); };
    Config c{};
    c.d = I("hidden_size"); c.hc = I("hc_count"); c.head_dim = I("head_dim"); c.H = I("num_attention_heads");
    c.KV = I("num_key_value_heads"); c.layers = I("num_hidden_layers"); c.lin_k_heads = I("linear_num_key_heads");
    c.lin_v_heads = I("linear_num_value_heads"); c.lin_k_dim = I("linear_key_head_dim");
    c.lin_v_dim = I("linear_value_head_dim"); c.conv_k = I("linear_conv_kernel_dim");
    c.ple_conv_k = I("ple_conv_kernel_size"); c.ngram = I("ngram_size"); c.heads_per_ngram = I("heads_per_ngram");
    c.ix_heads = I("indexer_n_heads"); c.ix_dim = I("indexer_head_dim"); c.ix_budget = I("indexer_budget");
    c.ix_ratio = I("indexer_compress_ratio"); c.E = I("num_experts"); c.top_k = I("num_experts_per_tok");
    c.I = I("moe_intermediate_size"); c.S = I("shared_expert_intermediate_size"); c.vocab = I("vocab_size");
    c.eos = I("eos_token_id"); c.eps = F("rms_norm_eps"); c.rope_theta = F("rope_theta");
    c.partial_rotary = F("partial_rotary_factor"); c.gated_act = I("gated_act"); c.norm_topk = I("norm_topk_prob");
    c.layer_types = kv.at("layer_types");
    for (auto& s : kv["ple_layer_ids"]) c.ple_layer_ids.push_back(std::stoi(s));
    return c;
}

size_t Array::row_size() const {
    size_t n = 1;
    for (size_t i = 1; i < shape.size(); i++) n *= shape[i];
    return n;
}

const Array& Weights::at(const std::string& name) const {
    auto it = t.find(name);
    if (it == t.end()) throw std::runtime_error("no weight " + name);
    return it->second;
}

Weights Weights::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    char magic[6];
    f.read(magic, 6);
    if (!f || std::memcmp(magic, "TUFFW1", 6)) throw std::runtime_error("not a tuff weights file: " + path);
    uint32_t n;
    f.read((char*)&n, 4);
    Weights W;
    for (uint32_t k = 0; k < n; k++) {
        uint32_t len, nd;
        uint8_t dt;
        f.read((char*)&len, 4);
        std::string name(len, '\0');
        f.read(&name[0], len);
        f.read((char*)&dt, 1);
        f.read((char*)&nd, 4);
        Array a;
        a.shape.resize(nd);
        f.read((char*)a.shape.data(), 8 * nd);
        size_t cnt = 1;
        for (auto s : a.shape) cnt *= s;
        if (dt == 0) { a.f.resize(cnt); f.read((char*)a.f.data(), 4 * cnt); }
        else { a.i.resize(cnt); f.read((char*)a.i.data(), 8 * cnt); }
        W.t[name] = std::move(a);
    }
    if (!f) throw std::runtime_error("truncated " + path);
    return W;
}

// ---------------------------------------------------------------- host
std::vector<float> Host::embed(int tok) const {
    const Array& e = w.at("embed_tokens.weight");
    size_t d = e.row_size();
    return std::vector<float>(e.f.begin() + tok * d, e.f.begin() + (tok + 1) * d);
}

std::vector<float> Host::ngram_rows(int L, const std::vector<int>& tokens) const {
    std::string p = "layers." + std::to_string(L) + ".ple.ple_embedding.";
    const Array &mult = w.at(p + "layer_multipliers"), &sizes = w.at(p + "ngram_heads_vocab_sizes"),
                &offs = w.at(p + "ngram_heads_offsets"), &tab = w.at(p + "ngram_embedding.weight");
    std::vector<int64_t> hist(c.ngram - 1, c.eos);
    hist.insert(hist.end(), tokens.begin(), tokens.end());
    std::vector<int64_t> sh(c.ngram);
    for (int s = 0; s < c.ngram; s++) sh[s] = hist[hist.size() - 1 - s];
    std::vector<float> out;
    size_t D = tab.row_size();
    for (int n = 2; n <= c.ngram; n++) {
        uint64_t mixed = (uint64_t)sh[0] * (uint64_t)mult.i[0];                // int64 wraps, as torch
        for (int pos = 1; pos < n; pos++) mixed ^= (uint64_t)sh[pos] * (uint64_t)mult.i[pos];
        int64_t v = (int64_t)mixed;
        int h0 = (n - 2) * c.heads_per_ngram;
        for (int h = h0; h < h0 + c.heads_per_ngram; h++) {
            int64_t s = sizes.i[h], r = ((v % s) + s) % s + offs.i[h];        // Python's non-negative %
            out.insert(out.end(), tab.f.begin() + r * D, tab.f.begin() + (r + 1) * D);
        }
    }
    return out;
}

void Host::experts(int L, const float* X, int m, const std::vector<int>& es, const std::vector<float>& G,
                   float* Y) const {
    std::string p = "layers." + std::to_string(L) + ".mlp.experts.";
    const Array &gu = w.at(p + "gate_up_proj"), &dn = w.at(p + "down_proj");
    int d = c.d, I = c.I;
    std::vector<float> h(2 * I);
    for (size_t k = 0; k < es.size(); k++) {
        const float *GU = &gu.f[(size_t)es[k] * 2 * I * d], *DN = &dn.f[(size_t)es[k] * d * I];
        for (int i = 0; i < m; i++) {
            float g = G[k * m + i];
            if (g == 0) continue;
            for (int r = 0; r < 2 * I; r++) {
                double s = 0;
                for (int j = 0; j < d; j++) s += (double)GU[(size_t)r * d + j] * X[(size_t)i * d + j];
                h[r] = (float)s;
            }
            for (int r = 0; r < I; r++) h[r] = h[r] / (1 + std::exp(-h[r])) * h[I + r];
            for (int o = 0; o < d; o++) {
                double s = 0;
                for (int r = 0; r < I; r++) s += (double)DN[(size_t)o * I + r] * h[r];
                Y[(size_t)i * d + o] += g * (float)s;
            }
        }
    }
}

StoreHost::StoreHost(const Config& c, const Weights& w, const std::string& dir, const std::string& fmt) : Host(c, w) {
    auto* s = new tstore{};
    if (tstore_open_dims(s, dir.c_str(), fmt == "bfp8" ? BFP8 : BFP4, c.layers, 0, c.d, c.I)) {
        delete s;
        throw std::runtime_error("tstore_open " + dir);
    }
    store = s;
}

StoreHost::~StoreHost() {
    tstore_close((tstore*)store);
    delete (tstore*)store;
}

void StoreHost::experts(int L, const float* X, int m, const std::vector<int>& es, const std::vector<float>& G,
                        float* Y) const {
    // as capi.c's tuff_moe: the layer's experts read once per 4 tokens
    int k = (int)es.size();
    std::vector<bfp_mat> g(k), u(k), dn(k);
    std::vector<const bfp_mat*> pg(k), pu(k), pd(k);
    for (int i = 0; i < k; i++) {
        tstore_expert((tstore*)store, L, es[i], &g[i], &u[i], &dn[i]);
        pg[i] = &g[i]; pu[i] = &u[i]; pd[i] = &dn[i];
    }
    std::vector<float> y((size_t)m * c.d, 0.0f), tmp((size_t)k * std::min(m, 4) * (2 * c.I + c.d));
    moe_forward_n(pg.data(), pu.data(), pd.data(), G.data(), k, m, X, y.data(), tmp.data());
    for (size_t i = 0; i < y.size(); i++) Y[i] += y[i];
}

// ---------------------------------------------------------------- card
static const std::vector<std::pair<std::string, std::string>> PATTERN = {
    {"lin_qkv", "linear_attn.in_proj_qkv"}, {"lin_z", "linear_attn.in_proj_z"}, {"lin_out", "linear_attn.out_proj"},
    {"lin_misc", "linear_attn."}, {"full", "self_attn."}, {"shared", "shared_expert"}, {"router", "mlp.gate"},
    {"hc", "hyper_connection"}, {"ple", "ple."}, {"lm_head", "lm_head"}};

Card::Card(Ops& o, const Config& c, const Weights& w, Host& host, const std::string& dense_fmt,
           const std::string& expert_fmt, int n_slots, int budget, int max_t)
    : o(o), c(c), w(w), host(host), dense_fmt(dense_fmt), expert_fmt(expert_fmt), max_t(max_t), n_slots(n_slots) {
    if (tcache_init(&cache, c.layers * c.E, n_slots, 0, budget, nullptr, 0)) throw std::runtime_error("tcache_init");
    rd = (int)(c.head_dim * c.partial_rotary);
    for (int i = 0; i < rd; i += 2) inv_freq.push_back(1.0 / std::pow((double)c.rope_theta, (double)i / rd));
    st.resize(c.layers);
    pend.resize(c.layers);
    std::set<int> ple;
    for (int id : c.ple_layer_ids) ple.insert(id - 1);
    for (int L = 0; L < c.layers; L++) {
        State& s = st[L];
        if (c.layer_types[L] == "linear_attention") {
            int C = 2 * c.lin_k_heads * c.lin_k_dim + c.lin_v_heads * c.lin_v_dim;
            s.conv = o.zeros({c.conv_k - 1, C});
            s.S = o.zeros({c.lin_v_heads, c.lin_k_dim, c.lin_v_dim});
        } else {
            s.K = o.zeros({max_t, c.KV * c.head_dim});
            s.V = o.zeros({max_t, c.KV * c.head_dim});
            s.idx = o.zeros({max_t, c.ix_dim});
            s.blk = o.zeros({max_t / c.ix_ratio, c.ix_dim});
        }
        if (ple.count(L)) s.ple = o.zeros({(c.ple_conv_k - 1) * c.ngram, c.hc * c.d});
    }
}

Card::~Card() { tcache_free(&cache); }

// dense_fmt: "bfp8" for all, or "bfp8,lm_head=bfp4,lin_z=bfp4" (the default, then per plan.DENSE group)
std::string Card::fmt(const std::string& name) const {
    std::string g = "lin_misc";
    for (auto& [grp, s] : PATTERN)
        if (name.find(s) != std::string::npos) { g = grp; break; }
    std::istringstream ss(dense_fmt);
    std::string item, def;
    std::getline(ss, def, ',');
    while (std::getline(ss, item, ','))
        if (item.rfind(g + "=", 0) == 0) return item.substr(g.size() + 1);
    return def;
}

Tensor Card::lin(const std::string& name) {
    auto it = dev.find(name);
    if (it != dev.end()) return it->second;
    const Array& a = w.at(name + ".weight");
    return dev[name] = o.put_weight(a.f.data(), (int)a.shape[0], (int)a.row_size(), fmt(name));
}

std::vector<Tensor> Card::lins(const Tensor& x, const std::vector<std::string>& names) {
    std::string key;
    for (auto& n : names) key += n + "|";
    if (!fused.count(key)) {
        std::vector<std::string> order;
        std::map<std::string, std::vector<std::string>> groups;
        for (auto& n : names) {
            const Array& a = w.at(n + ".weight");
            std::string f = fmt(n);
            if ((f == "bfp8" || f == "bfp4") && (a.shape[0] % 16 || a.row_size() % 8)) f = "bf16";
            if (!groups.count(f)) order.push_back(f);
            groups[f].push_back(n);
        }
        auto& out = fused[key];
        for (auto& f : order) {
            std::vector<float> cat;
            std::vector<std::pair<std::string, int>> parts;
            int in = 0;
            for (auto& n : groups[f]) {
                const Array& a = w.at(n + ".weight");
                cat.insert(cat.end(), a.f.begin(), a.f.end());
                parts.push_back({n, (int)a.shape[0]});
                in = (int)a.row_size();
            }
            out.push_back({o.put_weight(cat.data(), (int)(cat.size() / in), in, f), parts});
        }
    }
    std::map<std::string, Tensor> got;
    for (auto& [wt, parts] : fused[key]) {
        Tensor y = o.linear(x, wt);
        int i = 0;
        for (auto& [n, r] : parts) {
            got[n] = parts.size() == 1 ? y : o.slice(y, 1, i, i + r);
            i += r;
        }
    }
    std::vector<Tensor> res;
    for (auto& n : names) res.push_back(got[n]);
    return res;
}

Tensor Card::norm(const std::string& name, bool plus_one, int rows) {
    std::string key = name + "#" + std::to_string(rows);
    auto it = dev.find(key);
    if (it != dev.end()) return it->second;
    std::vector<float> v = w.at(name).f;
    if (plus_one)
        for (float& x : v) x += 1;
    return dev[key] = o.put(v, {rows, (int)v.size() / rows});
}

Tensor Card::constant(const std::string& key, const std::vector<float>& v, Shape s) {
    auto it = dev.find(key);
    if (it != dev.end()) return it->second;
    return dev[key] = o.put(v, std::move(s));
}

Tensor Card::scalar(float v) { return constant("scalar" + std::to_string(v), {v}, {1}); }

std::pair<Tensor, Tensor> Card::cs(int pos, int n) {
    if (!dev.count("cos")) {
        int half = rd / 2, T = max_t + 1;
        std::vector<float> cs((size_t)T * rd), sn((size_t)T * rd);
        for (int p = 0; p < T; p++)
            for (int i = 0; i < half; i++) {
                double f = p * inv_freq[i];
                cs[(size_t)p * rd + i] = cs[(size_t)p * rd + half + i] = (float)std::cos(f);
                sn[(size_t)p * rd + i] = sn[(size_t)p * rd + half + i] = (float)std::sin(f);
            }
        dev["cos"] = o.put(cs, {T, rd});
        dev["sin"] = o.put(sn, {T, rd});
    }
    return {o.slice(dev["cos"], 0, pos, pos + n), o.slice(dev["sin"], 0, pos, pos + n)};
}

Tensor Card::rope(const Tensor& x, const Tensor& cos, const Tensor& sin) {
    int half = rd / 2, dim = x.rank() - 1, D = x.dim(-1);
    if (!dev.count("rot")) {
        std::vector<float> P((size_t)rd * rd, 0.0f);                  // r @ P = [-r2, r1]
        for (int i = 0; i < half; i++) {
            P[(size_t)i * rd + half + i] = 1;
            P[(size_t)(half + i) * rd + i] = -1;
        }
        dev["rot"] = o.put(P, {rd, rd});
    }
    Tensor r = D > rd ? o.slice(x, dim, 0, rd) : x;
    Tensor y = o.add(o.mul(r, cos), o.mul(o.matmul(r, dev["rot"]), sin));
    return D > rd ? o.concat({y, o.slice(x, dim, rd, D)}, dim) : y;
}

std::pair<Tensor, Tensor> Card::hc_read(const std::string& p, const Tensor& h4, bool combine) {
    int m = h4.dim(0), hc = c.hc, d = c.d;
    Tensor hn = rms(o.reshape(h4, {m, hc, d}), norm(p + ".hc_norm.weight", true, hc));
    Tensor flat = o.reshape(hn, {m, hc * d});
    std::vector<std::string> names = {p + ".input_mix_weight_down"};
    if (combine) names.push_back(p + ".block_inject_weight");
    auto ys = lins(flat, names);
    Tensor mix = o.unary(o.linear(o.unary(o.scale(ys[0], 1.0f / hc), Unary::silu), lin(p + ".input_mix_weight_up")),
                         Unary::sigmoid);
    Tensor mixed = o.reshape(o.mean(o.mul(o.reshape(mix, {m, hc, d}), hn), 1), {m, d});
    if (!combine) return {mixed, Tensor{}};
    return {mixed, o.scale(o.unary(o.scale(ys[1], 1.0f / hc), Unary::sigmoid), 2)};
}

Tensor Card::hc_write(const Tensor& h4, const Tensor& inj, const Tensor& out) {
    int m = h4.dim(0);
    return o.add(h4, o.reshape(o.mul(o.reshape(inj, {m, c.hc, 1}), o.reshape(out, {m, 1, c.d})), {m, c.hc * c.d}));
}

Tensor Card::conv(const Tensor& win, const Tensor& cw, int m, int step) {
    Tensor y;
    for (int j = 0; j < cw.dim(0); j++) {
        Tensor t = o.mul(o.slice(win, 0, j * step, j * step + m), o.slice(cw, 0, j, j + 1));
        y = j ? o.add(y, t) : t;
    }
    return y;
}

// conv1d weight [C, 1, k] -> [k, C] on the device
static std::vector<float> conv_taps(const Array& a) {
    size_t C = a.shape[0], k = a.shape[2];
    std::vector<float> v(C * k);
    for (size_t ch = 0; ch < C; ch++)
        for (size_t j = 0; j < k; j++) v[j * C + ch] = a.f[ch * k + j];
    return v;
}

std::vector<Tensor> Card::deltanet_core(int L, const Tensor& x, const Tensor& conv_st, const Tensor& S) {
    std::string p = "layers." + std::to_string(L) + ".linear_attn";
    int K = c.lin_k_heads, V = c.lin_v_heads, dk = c.lin_k_dim, dv = c.lin_v_dim, m = x.dim(0);
    auto ys = lins(x, {p + ".in_proj_qkv", p + ".in_proj_z", p + ".in_proj_a", p + ".in_proj_b"});
    Tensor win = o.concat({conv_st, ys[0]}, 0);
    const Array& cwa = w.at(p + ".conv1d.weight");
    Tensor cw = constant(p + ".conv", conv_taps(cwa), {(int)cwa.shape[2], (int)cwa.shape[0]});
    Tensor qkv = o.unary(conv(win, cw, m, 1), Unary::silu);
    auto l2 = [&](const Tensor& t) {
        return o.mul(t, o.unary(o.add(o.sum(o.mul(t, t), -1), scalar(1e-6f)), Unary::rsqrt));
    };
    int C = qkv.dim(1);
    Tensor q = l2(o.reshape(o.slice(qkv, 1, 0, K * dk), {m, K, dk}));
    Tensor k = l2(o.reshape(o.slice(qkv, 1, K * dk, 2 * K * dk), {m, K, dk}));
    Tensor v = o.reshape(o.slice(qkv, 1, 2 * K * dk, C), {m, V, dv});
    Tensor z = o.reshape(ys[1], {m, V, dv});
    Tensor beta = o.unary(ys[3], Unary::sigmoid);
    std::vector<float> negA = w.at(p + ".A_log").f;
    for (float& a : negA) a = -std::exp(a);
    Tensor g = o.mul(constant(p + ".negA", negA, {1, V}),
                     o.unary(o.add(ys[2], constant(p + ".dt_bias", w.at(p + ".dt_bias").f, {1, V})), Unary::softplus));
    auto [out, states] = o.gdr(q, k, v, g, beta, S, 1.0f / std::sqrt((float)dk), m <= trace);
    Tensor gate = o.unary(z, c.gated_act ? Unary::sigmoid : Unary::silu);
    out = o.mul(rms(out, norm(p + ".norm.weight", false)), gate);
    return {o.linear(o.reshape(out, {m, V * dv}), lin(p + ".out_proj")), win, states};
}

Tensor Card::attention(int L, const Tensor& x) {
    std::string p = "layers." + std::to_string(L) + ".self_attn";
    State& s = st[L];
    int m = x.dim(0), t0 = t, H = c.H, KV = c.KV, hd = c.head_dim, nh = c.ix_heads, ihd = c.ix_dim, ratio = c.ix_ratio;
    int G = H / KV;
    auto [cs0, sn0] = cs(t0 + pos0, m);
    Tensor cos = o.reshape(cs0, {m, 1, rd}), sin = o.reshape(sn0, {m, 1, rd});
    auto pr = lins(x, {p + ".indexer.index_qk_proj", p + ".q_proj", p + ".k_proj", p + ".v_proj"});
    auto rr = [&](const Tensor& t, const std::string& wn) { return rope(rms(t, norm(p + wn)), cos, sin); };
    Tensor iq = o.scale(rr(o.reshape(o.slice(pr[0], 1, 0, nh * ihd), {m, nh, ihd}), ".indexer.q_layernorm.weight"),
                        1.0f / std::sqrt((float)ihd));
    Tensor qg = o.reshape(pr[1], {m, H, 2 * hd});
    Tensor q = o.scale(rr(o.slice(qg, 2, 0, hd), ".q_norm.weight"), 1.0f / std::sqrt((float)hd));
    Tensor gate = o.unary(o.reshape(o.slice(qg, 2, hd, 2 * hd), {m, H * hd}), Unary::sigmoid);
    Tensor k = o.reshape(rr(o.reshape(pr[2], {m, KV, hd}), ".k_norm.weight"), {m, KV * hd});
    o.write_rows(s.idx, t0, o.slice(pr[0], 1, nh * ihd, nh * ihd + ihd));            // raw indexer keys
    o.write_rows(s.K, t0, k);
    o.write_rows(s.V, t0, pr[3]);
    std::vector<Tensor> outs;
    for (int i = 0; i < m; i++) {
        int tt = t0 + i;
        if ((tt + 1) % ratio == 0) {                       // a block completes: pool, norm and rope its key once
            int b = tt / ratio;
            Tensor blk = o.mean(o.slice(s.idx, 0, b * ratio, (b + 1) * ratio), 0);
            auto [bc, bs] = cs(b * ratio + pos0);
            o.write_rows(s.blk, b, rope(rms(blk, norm(p + ".indexer.k_layernorm.weight")), bc, bs));
        }
        int nb = (tt + 1) / ratio;
        std::vector<int> keep;
        if (nb) {                                          // block scores to the host; top blocks, ties to the lowest
            std::vector<float> sc = o.get(o.sum(o.unary(o.matmul(o.reshape(o.slice(iq, 0, i, i + 1), {nh, ihd}),
                                                                 o.transpose(o.slice(s.blk, 0, 0, nb))),
                                                        Unary::relu), 0));
            std::vector<int> top(nb);
            std::iota(top.begin(), top.end(), 0);
            std::stable_sort(top.begin(), top.end(), [&](int a, int b2) { return sc[a] > sc[b2]; });
            top.resize(std::min(c.ix_budget / ratio, nb));
            std::sort(top.begin(), top.end());
            for (int j : top)
                for (int r = 0; r < ratio; r++) keep.push_back(j * ratio + r);
        }
        for (int r = nb * ratio; r <= tt; r++) keep.push_back(r);
        int n = (int)keep.size();
        Tensor ids = o.put_ids(keep);
        Tensor Ks = o.gather_rows(s.K, ids), Vs = o.gather_rows(s.V, ids);
        Tensor a = o.softmax(o.matmul(o.reshape(o.slice(q, 0, i, i + 1), {KV, G, hd}),
                                      o.permute(o.reshape(Ks, {n, KV, hd}), {1, 2, 0})));      // [KV, G, n]
        outs.push_back(o.reshape(o.matmul(a, o.permute(o.reshape(Vs, {n, KV, hd}), {1, 0, 2})), {1, H * hd}));
    }
    return o.linear(o.mul(o.concat(outs, 0), gate), lin(p + ".o_proj"));
}

std::pair<Tensor, Tensor> Card::dense_mlp(int L, const Tensor& x) {
    std::string p = "layers." + std::to_string(L) + ".mlp";
    auto ys = lins(x, {p + ".shared_expert.gate_proj", p + ".shared_expert.up_proj", p + ".shared_expert_gate", p + ".gate"});
    Tensor y = o.mul(o.linear(o.mul(o.unary(ys[0], Unary::silu), ys[1]), lin(p + ".shared_expert.down_proj")),
                     o.unary(ys[2], Unary::sigmoid));
    return {o.softmax(ys[3]), y};
}

void Card::put_slot(int slot, int L, int e) {
    int d = c.d, I = c.I;
    if (!have_banks) {
        banks[0] = o.bank(n_slots, d, I, expert_fmt);
        banks[1] = o.bank(n_slots, d, I, expert_fmt);
        banks[2] = o.bank(n_slots, I, d, expert_fmt);
        have_banks = true;
    }
    // ponytail: from the float weights; on the card bank_write_tiles(tstore_expert_tt's tiles), no repacking
    std::string p = "layers." + std::to_string(L) + ".mlp.experts.";
    const float* gu = &w.at(p + "gate_up_proj").f[(size_t)e * 2 * I * d];
    o.bank_write(banks[0], slot, gu, I, d);
    o.bank_write(banks[1], slot, gu + (size_t)I * d, I, d);
    o.bank_write(banks[2], slot, &w.at(p + "down_proj").f[(size_t)e * d * I], d, I);
}

Tensor Card::slot_experts(const Tensor& x, const std::vector<std::pair<int, std::vector<float>>>& hot) {
    int m = x.dim(0), d = x.dim(1), n = (int)hot.size();
    std::vector<int> slots;
    std::vector<float> G;
    for (auto& [s, g] : hot) {
        slots.push_back(s);
        G.insert(G.end(), g.begin(), g.end());
    }
    Tensor ids = o.put_ids(slots), Gt = o.put(G, {n, m, 1});
    Tensor x3 = o.reshape(x, {1, m, d});
    Tensor a = o.mul(o.gather_matmul(x3, banks[0], ids, Act::silu), o.gather_matmul(x3, banks[1], ids));
    return o.reshape(o.sum(o.mul(o.gather_matmul(a, banks[2], ids), Gt), 0), {m, d});
}

Tensor Card::moe(int L, const Tensor& x, const Tensor& probs, Tensor y) {
    int m = x.dim(0), E = c.E;
    std::vector<float> P = o.get(probs);                    // one read of [m, E]; the choice is made on the host
    std::vector<float> gate((size_t)m * E, 0.0f);
    std::set<int> es;
    for (int i = 0; i < m; i++) {
        std::vector<int> idx(E);
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return P[i * E + a] > P[i * E + b]; });
        float s = 0;
        for (int j = 0; j < c.top_k; j++) s += P[i * E + idx[j]];
        for (int j = 0; j < c.top_k; j++) {
            gate[(size_t)i * E + idx[j]] = c.norm_topk ? P[i * E + idx[j]] / s : P[i * E + idx[j]];
            es.insert(idx[j]);
        }
    }
    std::vector<int32_t> ids, slot(es.size());
    for (int e : es) ids.push_back(L * E + e);
    tcache_lookup(&cache, ids.data(), (int)ids.size(), slot.data());
    std::vector<std::pair<int, std::vector<float>>> hot;
    std::vector<int> cold;
    std::vector<float> Gc;
    int k = 0;
    for (int e : es) {
        std::vector<float> g(m);
        int n = 0;
        for (int i = 0; i < m; i++) n += (g[i] = gate[(size_t)i * E + e]) > 0;
        lookups += n;
        if (slot[k] >= 0) {
            hits += n;
            hot.push_back({slot[k], g});
        } else {
            cold.push_back(e);
            Gc.insert(Gc.end(), g.begin(), g.end());
        }
        k++;
    }
    std::vector<float> xh;
    if (!cold.empty()) xh = o.get(x);                       // read before queueing the slot experts: overlap
    if (!hot.empty()) y = o.add(y, slot_experts(x, hot));
    if (!cold.empty()) {
        std::vector<float> yc((size_t)m * c.d, 0.0f);
        host.experts(L, xh.data(), m, cold, Gc, yc.data());
        y = o.add(y, o.put(yc, {m, c.d}));
    }
    return y;
}

void Card::promote() {
    std::vector<tcache_copy> cp(cache.budget + 1);
    int n = tcache_end_token(&cache, cp.data());
    for (int i = 0; i < n; i++) put_slot(cp[i].slot, cp[i].expert / c.E, cp[i].expert % c.E);
}

Tensor Card::ple_layer(int L, const Tensor& h4, const std::vector<int>& toks) {
    std::string p = "layers." + std::to_string(L) + ".ple";
    int m = h4.dim(0), hc = c.hc, d = c.d;
    std::vector<float> rows;
    std::vector<int> hist = tokens;
    for (int i = 0; i < m; i++) {
        hist.push_back(toks[i]);
        auto r = host.ngram_rows(L, hist);
        rows.insert(rows.end(), r.begin(), r.end());
    }
    Tensor emb = o.put(rows, {m, (int)(rows.size() / m)});
    Tensor key = rms(o.reshape(o.linear(emb, lin(p + ".key_proj")), {m, hc, d}), norm(p + ".norm_key.weight", true, hc));
    Tensor value = o.reshape(o.linear(emb, lin(p + ".value_proj")), {m, 1, d});
    Tensor query = rms(o.reshape(h4, {m, hc, d}), norm(p + ".norm_query.weight", true, hc));
    Tensor gate = o.scale(o.sum(o.mul(key, query), -1), 1.0f / std::sqrt((float)d));
    gate = o.mul(o.unary(o.clamp_min(o.unary(gate, Unary::abs), 1e-6f), Unary::sqrt), o.unary(gate, Unary::sign));
    Tensor gv = o.mul(o.unary(gate, Unary::sigmoid), value);
    Tensor gvn = o.reshape(rms(gv, norm(p + ".norm_conv.weight", true, hc)), {m, hc * d});
    Tensor win = o.concat({st[L].ple, gvn}, 0);
    pend[L].ple = win;
    pend[L].has_ple = true;
    const Array& cwa = w.at(p + ".conv1d.weight");
    Tensor cw = constant(p + ".conv", conv_taps(cwa), {(int)cwa.shape[2], (int)cwa.shape[0]});
    return o.add(h4, o.add(o.reshape(gv, {m, hc * d}), o.unary(conv(win, cw, m, c.ngram), Unary::silu)));
}

Tensor Card::layers(Tensor h, const std::vector<int>& toks) {
    std::set<int> ple;
    for (int id : c.ple_layer_ids) ple.insert(id - 1);
    for (auto& p : pend) p = Pending{};
    for (int L = 0; L < c.layers; L++) {
        std::string pre = "layers." + std::to_string(L);
        if (ple.count(L)) h = ple_layer(L, h, toks);
        if (c.layer_types[L] == "linear_attention") {
            // everything up to the routing decision is one traced segment on the card
            auto seg = o.trace(pre + "#" + std::to_string(h.dim(0)), [&](const std::vector<Tensor>& a) {
                auto [x, inj] = hc_read(pre + ".attn_hyper_connection", a[0]);
                auto dn = deltanet_core(L, x, a[1], a[2]);
                Tensor h2 = hc_write(a[0], inj, dn[0]);
                auto [x2, inj2] = hc_read(pre + ".mlp_hyper_connection", h2);
                auto [probs, y] = dense_mlp(L, x2);
                return std::vector<Tensor>{h2, x2, inj2, dn[1], dn[2], probs, y};
            }, {h, st[L].conv, st[L].S});
            pend[L].win = seg[3];
            pend[L].states = seg[4];
            pend[L].lin = true;
            h = hc_write(seg[0], seg[2], moe(L, seg[1], seg[5], seg[6]));
        } else {
            auto [x, inj] = hc_read(pre + ".attn_hyper_connection", h);
            h = hc_write(h, inj, attention(L, x));
            auto [x2, inj2] = hc_read(pre + ".mlp_hyper_connection", h);
            auto [probs, y] = dense_mlp(L, x2);
            h = hc_write(h, inj2, moe(L, x2, probs, y));
        }
    }
    return h;
}

Tensor Card::verify(const std::vector<int>& toks) {
    int m = (int)toks.size(), d = c.d;
    std::vector<float> rows;
    for (int tok : toks) {
        auto e = host.embed(tok);
        for (int s = 0; s < c.hc; s++) rows.insert(rows.end(), e.begin(), e.end());
    }
    h4 = layers(o.put(rows, {m, c.hc * d}), toks);
    verified = toks;
    return hc_read("hyper_connection_mixer", h4, false).first;
}

void Card::commit(int j) {
    for (int L = 0; L < c.layers; L++) {
        Pending& p = pend[L];
        if (p.lin) {
            int n = p.states.dim(0);
            st[L].conv = o.slice(p.win, 0, j, j + c.conv_k - 1);
            st[L].S = o.reshape(o.slice(p.states, 0, std::min(j, n) - 1, std::min(j, n)),
                                {p.states.dim(1), p.states.dim(2), p.states.dim(3)});
        }
        if (p.has_ple) st[L].ple = o.slice(p.ple, 0, j, j + (c.ple_conv_k - 1) * c.ngram);
    }
    tokens.insert(tokens.end(), verified.begin(), verified.begin() + j);
    t += j;
    promote();
}

Tensor Card::step(int tok) {
    Tensor out = verify({tok});
    commit(1);
    return out;
}

std::vector<int> Card::lm_argmax(const Tensor& mixed) {
    std::vector<float> l = o.get(o.linear(mixed, lin("lm_head")));
    int V = c.vocab, m = (int)(l.size() / V);
    std::vector<int> out;
    for (int i = 0; i < m; i++) out.push_back((int)(std::max_element(&l[(size_t)i * V], &l[(size_t)(i + 1) * V]) - &l[(size_t)i * V]));
    return out;
}

std::vector<int> Card::generate(const std::vector<int>& prompt, int n, Drafter* drafter, int k) {
    Tensor mixed;
    for (size_t i = 0; i < prompt.size(); i += prefill_chunk) {
        std::vector<int> chunk(prompt.begin() + i, prompt.begin() + std::min(prompt.size(), i + prefill_chunk));
        mixed = verify(chunk);
        commit((int)chunk.size());
        if (drafter) drafter->catch_up(*this, (int)i, (int)chunk.size());
    }
    int nxt = lm_argmax(mixed).back();
    std::vector<int> out;
    steps = 0;
    while ((int)out.size() < n) {
        std::vector<int> drafts = drafter ? drafter->propose(*this, nxt, k) : std::vector<int>{};
        int start = t;
        std::vector<int> toks = {nxt};
        toks.insert(toks.end(), drafts.begin(), drafts.end());
        std::vector<int> pred = lm_argmax(verify(toks));
        int j = 1;
        while (j <= (int)drafts.size() && drafts[j - 1] == pred[j - 1]) j++;
        commit(j);
        out.insert(out.end(), tokens.end() - j, tokens.end());
        nxt = pred[j - 1];
        steps++;
        if (drafter) drafter->catch_up(*this, start, j);
    }
    out.resize(n);
    return out;
}

}  // namespace tuff
