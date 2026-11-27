// Config, weights (TUFFW1 test files and Hugging Face snapshots), and the host side (embeddings, N-gram rows, cold
// experts). The snapshot reader replaces ref_real.Checkpoint / device/run.Weights: safetensors mmap'ed, bf16 widened
// to float32 per use, experts and table rows read without loading the whole tensor.
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>

#include "json.h"
#include "qfn.h"

extern "C" {
#include "../kernels/bfp.h"
#include "../kernels/store.h"
}

namespace tuff {

// ---------------------------------------------------------------- config
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

Config Config::load_hf(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    Json all = Json::parse(ss.str());
    const Json& j = all.has("text_config") ? all["text_config"] : all;
    auto I = [&](const char* k) { return j[k].i(); };
    Config c{};
    c.d = I("hidden_size"); c.hc = I("hc_count"); c.head_dim = I("head_dim"); c.H = I("num_attention_heads");
    c.KV = I("num_key_value_heads"); c.layers = I("num_hidden_layers"); c.lin_k_heads = I("linear_num_key_heads");
    c.lin_v_heads = I("linear_num_value_heads"); c.lin_k_dim = I("linear_key_head_dim");
    c.lin_v_dim = I("linear_value_head_dim"); c.conv_k = I("linear_conv_kernel_dim");
    c.ple_conv_k = I("ple_conv_kernel_size"); c.ngram = I("ngram_size"); c.heads_per_ngram = I("heads_per_ngram");
    c.ix_heads = I("indexer_n_heads"); c.ix_dim = I("indexer_head_dim"); c.ix_budget = I("indexer_budget");
    c.ix_ratio = I("indexer_compress_ratio"); c.E = I("num_experts"); c.top_k = I("num_experts_per_tok");
    c.I = I("moe_intermediate_size"); c.S = I("shared_expert_intermediate_size"); c.vocab = I("vocab_size");
    c.ngram_base = I("ngram_vocab_size_base");
    const Json& eos = j["eos_token_id"];
    c.eos = eos.kind == Json::array ? eos[0].i() : eos.i();
    c.eps = (float)j["rms_norm_eps"].f();
    const Json& rp = j["rope_parameters"];
    c.rope_theta = (float)rp["rope_theta"].f();
    c.partial_rotary = rp.has("partial_rotary_factor") ? (float)rp["partial_rotary_factor"].f() : 1.0f;
    std::string gate = j.has("output_gate_type") && j["output_gate_type"].kind == Json::string ? j["output_gate_type"].str
                                                                                                : j["hidden_act"].str;
    c.gated_act = gate == "sigmoid";
    c.norm_topk = j["norm_topk_prob"].i();
    for (auto& x : j["layer_types"].arr) c.layer_types.push_back(x.str);
    c.layer_types.resize(c.layers);
    for (auto& x : j["ple_layer_ids"].arr) c.ple_layer_ids.push_back(x.i());
    return c;
}

Config Config::mtp() const {
    Config m = *this;
    m.layers = 1;
    m.layer_types = {"full_attention"};
    m.ple_layer_ids.clear();
    return m;
}

size_t Array::row_size() const {
    size_t n = 1;
    for (size_t i = 1; i < shape.size(); i++) n *= shape[i];
    return n;
}

// ---------------------------------------------------------------- weights
struct Weights::Src {
    struct Entry { const uint8_t* data; std::string dtype; std::vector<int64_t> shape; };
    std::map<std::string, Entry> map;
    std::vector<std::pair<void*, size_t>> maps;
    ~Src() {
        for (auto& [p, n] : maps) munmap(p, n);
    }
};

static size_t dsize(const std::string& dt) {
    if (dt == "F32" || dt == "I32") return 4;
    if (dt == "BF16" || dt == "F16") return 2;
    if (dt == "I64") return 8;
    throw std::runtime_error("dtype " + dt);
}

static float half(uint16_t h) {                       // IEEE fp16 -> float
    uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, u;
    if (e == 0) {
        float v = std::ldexp((float)m, -24);
        return s ? -v : v;
    }
    u = s | ((e == 31 ? 255 : e + 112) << 23) | (m << 13);
    float v;
    std::memcpy(&v, &u, 4);
    return v;
}

static void widen(const uint8_t* p, const std::string& dt, size_t n, Array& a) {
    if (dt == "I64" || dt == "I32") {
        a.i.resize(n);
        for (size_t k = 0; k < n; k++) {
            if (dt == "I64") std::memcpy(&a.i[k], p + 8 * k, 8);
            else { int32_t v; std::memcpy(&v, p + 4 * k, 4); a.i[k] = v; }
        }
        return;
    }
    a.f.resize(n);
    for (size_t k = 0; k < n; k++) {
        if (dt == "F32") { std::memcpy(&a.f[k], p + 4 * k, 4); continue; }
        uint16_t h;
        std::memcpy(&h, p + 2 * k, 2);
        if (dt == "F16") { a.f[k] = half(h); continue; }
        uint32_t u = (uint32_t)h << 16;
        std::memcpy(&a.f[k], &u, 4);
    }
}

bool Weights::has(const std::string& name) const {
    return t.count(name) || (src && (src->map.count(prefix + name) || src->map.count(name)));
}

const Array& Weights::at(const std::string& name) const {
    auto it = t.find(name);
    if (it != t.end()) return it->second;
    if (!src) throw std::runtime_error("no weight " + name);
    auto e = src->map.find(prefix + name);
    if (e == src->map.end()) e = src->map.find(name);                 // the LM head has no prefix
    if (e == src->map.end()) throw std::runtime_error("no weight " + prefix + name);
    Array a;
    a.shape = e->second.shape;
    size_t n = 1;
    for (auto s : a.shape) n *= s;
    widen(e->second.data, e->second.dtype, n, a);
    return t[name] = std::move(a);
}

std::vector<float> Weights::expert(const std::string& name, int ex) const {
    auto it = t.find(name);
    if (it != t.end()) {
        size_t n = it->second.row_size();
        return std::vector<float>(it->second.f.begin() + ex * n, it->second.f.begin() + (ex + 1) * n);
    }
    auto& en = src->map.at(prefix + name);
    size_t n = 1;
    for (size_t i = 1; i < en.shape.size(); i++) n *= en.shape[i];
    Array a;
    widen(en.data + ex * n * dsize(en.dtype), en.dtype, n, a);
    return a.f;
}

std::vector<float> Weights::row(const std::string& name, int64_t r) const {
    auto it = t.find(name);
    if (it != t.end()) {
        size_t n = it->second.row_size();
        return std::vector<float>(it->second.f.begin() + r * n, it->second.f.begin() + (r + 1) * n);
    }
    auto get = [&](const Src::Entry& en, int64_t i) {
        size_t n = en.shape[1];
        Array a;
        widen(en.data + i * n * dsize(en.dtype), en.dtype, n, a);
        return a.f;
    };
    std::string full = prefix + name;
    if (!src->map.count(full)) full = name;
    if (src->map.count(full)) return get(src->map.at(full), r);
    std::string pre = prefix + name.substr(0, name.size() - 7);       // "<table>.shard_<i>.weight"
    for (int i = 0;; i++) {
        auto e = src->map.find(pre + ".shard_" + std::to_string(i) + ".weight");
        if (e == src->map.end()) throw std::runtime_error("row " + std::to_string(r) + " not in " + name);
        if (r < e->second.shape[0]) return get(e->second, r);
        r -= e->second.shape[0];
    }
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

// the n-th prime after x (transformers' _find_nth_prime_after; test_engine.py checks it against transformers)
static int64_t nth_prime_after(int64_t x, int n) {
    auto prime = [](int64_t v) {
        if (v < 2) return false;
        for (int64_t q = 2; q * q <= v; q++)
            if (v % q == 0) return false;
        return true;
    };
    for (int64_t v = x + 1;; v++)
        if (prime(v) && --n == 0) return v;
}

Weights Weights::open_hf(const std::string& dir, const std::string& prefix, const Config& c) {
    Weights W;
    W.prefix = prefix;
    W.src = std::make_shared<Src>();
    std::ifstream fi(dir + "/model.safetensors.index.json");
    std::stringstream ss;
    ss << fi.rdbuf();
    std::set<std::string> files;
    Json index = Json::parse(ss.str());                // kept: a temporary's member would dangle in the loop
    for (auto& [name, file] : index["weight_map"].obj) files.insert(file.str);
    for (auto& file : files) {
        std::string path = dir + "/" + file;
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open " + path);
        struct stat sb;
        fstat(fd, &sb);
        void* p = mmap(nullptr, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (p == MAP_FAILED) throw std::runtime_error("mmap " + path);
        W.src->maps.push_back({p, (size_t)sb.st_size});
        const uint8_t* base = (const uint8_t*)p;
        uint64_t hlen;
        std::memcpy(&hlen, base, 8);
        Json h = Json::parse(std::string((const char*)base + 8, hlen));
        for (auto& [name, e] : h.obj) {
            if (name == "__metadata__") continue;
            Src::Entry en{base + 8 + hlen + (size_t)e["data_offsets"][0].num, e["dtype"].str, {}};
            for (auto& s : e["shape"].arr) en.shape.push_back((int64_t)s.num);
            W.src->map[name] = en;
        }
    }
    // the N-gram tables' per-head sizes and offsets are not stored: rebuilt as transformers does
    int heads = (c.ngram - 1) * c.heads_per_ngram;
    for (size_t i = 0; i < c.ple_layer_ids.size(); i++) {
        std::string p = "layers." + std::to_string(c.ple_layer_ids[i] - 1) + ".ple.ple_embedding.";
        Array sizes, offs;
        sizes.shape = offs.shape = {heads};
        int64_t o = 0;
        for (int h = 0; h < heads; h++) {
            sizes.i.push_back(nth_prime_after(c.ngram_base - 1, (int)i * heads + h + 1));
            offs.i.push_back(o);
            o += sizes.i.back();
        }
        W.t[p + "ngram_heads_vocab_sizes"] = sizes;
        W.t[p + "ngram_heads_offsets"] = offs;
    }
    return W;
}

Weights Weights::shard() const {
    Weights W;
    W.src = src;
    W.prefix = prefix;
    W.t = t;
    return W;
}

// ---------------------------------------------------------------- host
std::vector<float> Host::embed(int tok) const { return w.row("embed_tokens.weight", tok); }

std::vector<float> Host::ngram_rows(int L, const std::vector<int>& tokens) const {
    std::string p = "layers." + std::to_string(L) + ".ple.ple_embedding.";
    const Array &mult = w.at(p + "layer_multipliers"), &sizes = w.at(p + "ngram_heads_vocab_sizes"),
                &offs = w.at(p + "ngram_heads_offsets");
    std::vector<int64_t> hist(c.ngram - 1, c.eos);
    hist.insert(hist.end(), tokens.begin(), tokens.end());
    std::vector<int64_t> sh(c.ngram);
    for (int s = 0; s < c.ngram; s++) sh[s] = hist[hist.size() - 1 - s];
    std::vector<float> out;
    for (int n = 2; n <= c.ngram; n++) {
        uint64_t mixed = (uint64_t)sh[0] * (uint64_t)mult.i[0];                // int64 wraps, as torch
        for (int pos = 1; pos < n; pos++) mixed ^= (uint64_t)sh[pos] * (uint64_t)mult.i[pos];
        int64_t v = (int64_t)mixed;
        int h0 = (n - 2) * c.heads_per_ngram;
        for (int h = h0; h < h0 + c.heads_per_ngram; h++) {
            int64_t s = sizes.i[h], r = ((v % s) + s) % s + offs.i[h];        // Python's non-negative %
            auto row = w.row(p + "ngram_embedding.weight", r);
            out.insert(out.end(), row.begin(), row.end());
        }
    }
    return out;
}

void Host::experts(int L, const float* X, int m, const std::vector<int>& es, const std::vector<float>& G,
                   float* Y) const {
    std::string p = "layers." + std::to_string(L) + ".mlp.experts.";
    int d = c.d, I = c.I;
    std::vector<float> h(2 * I);
    for (size_t k = 0; k < es.size(); k++) {
        std::vector<float> GU = w.expert(p + "gate_up_proj", es[k]), DN = w.expert(p + "down_proj", es[k]);
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
    int k = (int)es.size();                            // as capi.c's tuff_moe: each expert read once per 4 tokens
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

const void* StoreHost::expert_tiles(int L, int e, size_t& bytes) const {
    const tstore* s = (const tstore*)store;
    bytes = bfp_tt_bytes(s->fmt, 2 * s->I, s->D) + bfp_tt_bytes(s->fmt, s->D, s->I);
    tiles.resize(bytes);
    tstore_expert_tt(s, L, e, tiles.data());
    return tiles.data();
}

}  // namespace tuff

namespace tuff {
bool StoreHost::expert_bfp(int L, int e, Bfp* out) const {
    bfp_mat g, u, d;
    tstore_expert((const tstore*)store, L, e, &g, &u, &d);
    int mb = g.fmt;
    for (auto [m, o] : {std::pair{&g, 0}, {&u, 1}, {&d, 2}}) out[o] = Bfp{m->exp, m->mant, m->out, m->in, mb};
    return true;
}
}  // namespace tuff
