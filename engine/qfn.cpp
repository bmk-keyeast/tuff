// See qfn.h. Each method follows the device/qfn.py / device/tp.py method of the same name; comments there explain why.
#include "qfn.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <set>
#include <sstream>

extern "C" {
#include "../kernels/bfp.h"
}

namespace tuff {

static std::string num(int v) { return std::to_string(v); }

// ---------------------------------------------------------------- decoding loop (one device or the split)
Tensor Decoder::step(int tok) {
    Tensor out = verify({tok});
    commit(1);
    return out;
}

std::vector<int> Decoder::generate(const std::vector<int>& prompt, int n, Drafter* drafter, int k) {
    Tensor mixed;
    for (size_t i = 0; i < prompt.size(); i += prefill_chunk) {       // the prompt in passes of prefill_chunk
        std::vector<int> chunk(prompt.begin() + i, prompt.begin() + std::min(prompt.size(), i + prefill_chunk));
        mixed = verify(chunk);
        commit((int)chunk.size());
        if (drafter) drafter->catch_up(*this, (int)i, (int)chunk.size());
    }
    int nxt = lm_argmax(mixed).back();
    t_prompt = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
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

// ---------------------------------------------------------------- card
static const std::vector<std::pair<std::string, std::string>> PATTERN = {
    {"lin_qkv", "linear_attn.in_proj_qkv"}, {"lin_z", "linear_attn.in_proj_z"}, {"lin_out", "linear_attn.out_proj"},
    {"lin_misc", "linear_attn."}, {"full", "self_attn."}, {"shared", "shared_expert"}, {"router", "mlp.gate"},
    {"hc", "hyper_connection"}, {"ple", "ple."}, {"lm_head", "lm_head"}};

Card::Card(Ops& o, const Config& c, const Weights& w, Host& host, const std::string& dense_fmt,
           const std::string& expert_fmt, int n_slots, int budget, int max_t)
    : Decoder(o, w), c(c), host(host), dense_fmt(dense_fmt), expert_fmt(expert_fmt), max_t(max_t), n_slots(n_slots) {
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
    Tensor t = o.put_weight(a.f.data(), (int)a.shape[0], (int)a.row_size(), fmt(name));
    return dev[name] = t;
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
    std::string key = name + "#" + num(rows);
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

std::vector<Tensor> Card::seg(const std::string& name, const Ops::Fn& fn, const std::vector<Tensor>& args, int m) {
    if (m > trace) return fn(args);
    return o.trace(std::to_string((uintptr_t)this) + "/" + name + "/" + num(m), fn, args);
}

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

static std::vector<float> conv_taps(const Array& a) {    // conv1d weight [C, 1, k] -> [k, C]
    size_t C = a.shape[0], k = a.shape[2];
    std::vector<float> v(C * k);
    for (size_t ch = 0; ch < C; ch++)
        for (size_t j = 0; j < k; j++) v[j * C + ch] = a.f[ch * k + j];
    return v;
}

// ---- Gated DeltaNet
std::vector<Tensor> Card::deltanet_core(int L, const Tensor& x, const Tensor& conv_st, const Tensor& S) {
    std::string p = "layers." + num(L) + ".linear_attn";
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

Tensor Card::deltanet(int L, const Tensor& x) {
    auto r = deltanet_core(L, x, st[L].conv, st[L].S);
    pend[L].win = r[1];
    pend[L].states = r[2];
    pend[L].lin = true;
    return r[0];
}

// ---- full attention with the QSA indexer
std::vector<Tensor> Card::attn_prep(int L, const Tensor& x, const Tensor& cs0, const Tensor& sn0) {
    std::string p = "layers." + num(L) + ".self_attn";
    int m = x.dim(0), H = c.H, KV = c.KV, hd = c.head_dim, nh = c.ix_heads, ihd = c.ix_dim;
    Tensor cos = o.reshape(cs0, {m, 1, rd}), sin = o.reshape(sn0, {m, 1, rd});
    auto pr = lins(x, {p + ".indexer.index_qk_proj", p + ".q_proj", p + ".k_proj", p + ".v_proj"});
    auto rr = [&](const Tensor& t, const std::string& wn) { return rope(rms(t, norm(p + wn)), cos, sin); };
    Tensor iq = o.scale(rr(o.reshape(o.slice(pr[0], 1, 0, nh * ihd), {m, nh, ihd}), ".indexer.q_layernorm.weight"),
                        1.0f / std::sqrt((float)ihd));
    Tensor qg = o.reshape(pr[1], {m, H, 2 * hd});
    Tensor q = o.scale(rr(o.slice(qg, 2, 0, hd), ".q_norm.weight"), 1.0f / std::sqrt((float)hd));
    Tensor gate = o.unary(o.reshape(o.slice(qg, 2, hd, 2 * hd), {m, H * hd}), Unary::sigmoid);
    Tensor k = o.reshape(rr(o.reshape(pr[2], {m, KV, hd}), ".k_norm.weight"), {m, KV * hd});
    return {o.slice(pr[0], 1, nh * ihd, nh * ihd + ihd), iq, q, gate, k, pr[3]};
}

std::vector<Tensor> Card::attention_core(int L, const std::vector<Tensor>& prep, int m) {
    std::string p = "layers." + num(L) + ".self_attn";
    State& s = st[L];
    int t0 = t, H = c.H, KV = c.KV, hd = c.head_dim, nh = c.ix_heads, ihd = c.ix_dim, ratio = c.ix_ratio, G = H / KV;
    const Tensor &iq = prep[1], &q = prep[2];
    o.write_rows(s.idx, t0, prep[0]);                  // raw indexer keys
    o.write_rows(s.K, t0, prep[4]);
    o.write_rows(s.V, t0, prep[5]);
    if (m > trace) return {attention_all(L, iq, q, m)};
    int n_max = c.ix_budget / ratio * ratio + ratio - 1;                 // chosen blocks + the open block
    Ops::Fn score = [&](const std::vector<Tensor>& a) {                   // block keys read in place, by address
        return std::vector<Tensor>{o.sum(o.unary(o.matmul(o.reshape(a[0], {nh, ihd}), o.transpose(s.blk)), Unary::relu), 0)};
    };
    Ops::Fn attend = [&](const std::vector<Tensor>& a) {                  // all KV heads over the n_max gathered rows
        Tensor Ks = o.gather_rows(s.K, a[1]), Vs = o.gather_rows(s.V, a[1]);
        Tensor att = o.softmax(o.add(o.matmul(o.reshape(a[0], {KV, G, hd}), o.permute(o.reshape(Ks, {n_max, KV, hd}), {1, 2, 0})),
                                     a[2]));
        return std::vector<Tensor>{o.reshape(o.matmul(att, o.permute(o.reshape(Vs, {n_max, KV, hd}), {1, 0, 2})), {1, H * hd})};
    };
    std::vector<Tensor> outs;
    for (int i = 0; i < m; i++) {
        int tt = t0 + i;
        if ((tt + 1) % ratio == 0) {                   // a block completes: pool, norm and rope its key once
            int b = tt / ratio;
            Tensor blk = o.mean(o.slice(s.idx, 0, b * ratio, (b + 1) * ratio), 0);
            auto [bc, bs] = cs(b * ratio + pos0);
            o.write_rows(s.blk, b, rope(rms(blk, norm(p + ".indexer.k_layernorm.weight")), bc, bs));
        }
        int nb = (tt + 1) / ratio;
        std::vector<int> keep;
        if (nb) {                                      // all block scores to the host; top blocks, ties to the lowest
            std::vector<float> sc = o.get(seg("L" + num(L) + "score", score, {o.slice(iq, 0, i, i + 1)}, m)[0]);
            std::vector<int> top(nb);
            std::iota(top.begin(), top.end(), 0);
            std::stable_sort(top.begin(), top.end(), [&](int a, int b2) { return sc[a] > sc[b2]; });
            top.resize(std::min(c.ix_budget / ratio, nb));
            std::sort(top.begin(), top.end());
            for (int j : top)
                for (int r = 0; r < ratio; r++) keep.push_back(j * ratio + r);
        }
        for (int r = nb * ratio; r <= tt; r++) keep.push_back(r);
        std::vector<float> mask(n_max, -1e9f);
        std::fill(mask.begin(), mask.begin() + keep.size(), 0.0f);
        keep.resize(n_max, 0);                         // padded to n_max rows, the padding masked out
        outs.push_back(seg("L" + num(L) + "attend" + num(i), attend,
                           {o.slice(q, 0, i, i + 1), o.put_ids(keep), o.put(mask, {1, 1, n_max})}, m)[0]);
    }
    return outs;
}

Tensor Card::attention_all(int L, const Tensor& iq, const Tensor& q, int m) {
    std::string p = "layers." + num(L) + ".self_attn";
    State& s = st[L];
    int t0 = t, H = c.H, KV = c.KV, hd = c.head_dim, ihd = c.ix_dim, ratio = c.ix_ratio, G = H / KV, T = t0 + m;
    int b0 = (t0 + ratio) / ratio - 1, b1 = T / ratio;                  // blocks completing in this pass
    if (b1 > b0) {
        Tensor blk = o.reshape(o.mean(o.reshape(o.slice(s.idx, 0, b0 * ratio, b1 * ratio), {b1 - b0, ratio, ihd}), 1),
                               {b1 - b0, ihd});
        cs(0);
        std::vector<int> pos;
        for (int b = b0; b < b1; b++) pos.push_back(b * ratio + pos0);
        Tensor ids = o.put_ids(pos);
        o.write_rows(s.blk, b0, rope(rms(blk, norm(p + ".indexer.k_layernorm.weight")), o.gather_rows(dev["cos"], ids),
                                     o.gather_rows(dev["sin"], ids)));
    }
    std::vector<float> mask((size_t)m * T, -1e9f), sc;
    if (b1) sc = o.get(o.sum(o.unary(o.matmul(iq, o.transpose(o.slice(s.blk, 0, 0, b1))), Unary::relu), 1));   // [m, b1]
    for (int i = 0; i < m; i++) {
        int tt = t0 + i, nb = (tt + 1) / ratio;
        for (int r = nb * ratio; r <= tt; r++) mask[(size_t)i * T + r] = 0;
        if (nb) {
            std::vector<int> top(nb);
            std::iota(top.begin(), top.end(), 0);
            std::stable_sort(top.begin(), top.end(), [&](int a, int b2) { return sc[i * b1 + a] > sc[i * b1 + b2]; });
            top.resize(std::min(c.ix_budget / ratio, nb));
            for (int j : top)
                for (int r = 0; r < ratio; r++) mask[(size_t)i * T + j * ratio + r] = 0;
        }
    }
    Tensor Q = o.reshape(o.permute(o.reshape(q, {m, KV, G, hd}), {1, 2, 0, 3}), {KV, G * m, hd});
    Tensor K = o.permute(o.reshape(o.slice(s.K, 0, 0, T), {T, KV, hd}), {1, 2, 0});
    Tensor V = o.permute(o.reshape(o.slice(s.V, 0, 0, T), {T, KV, hd}), {1, 0, 2});
    Tensor a = o.softmax(o.add(o.reshape(o.matmul(Q, K), {KV, G, m, T}), o.put(mask, {m, T})));
    Tensor y = o.matmul(o.reshape(a, {KV, G * m, T}), V);
    return o.reshape(o.permute(o.reshape(y, {KV, G, m, hd}), {2, 0, 1, 3}), {m, H * hd});
}

Tensor Card::attn_out(int L, const Tensor& gate, const std::vector<Tensor>& outs) {
    return o.linear(o.mul(o.concat(outs, 0), gate), lin("layers." + num(L) + ".self_attn.o_proj"));
}

Tensor Card::attention(int L, const Tensor& x) {
    int m = x.dim(0);
    auto [cs0, sn0] = cs(t + pos0, m);
    auto prep = attn_prep(L, x, cs0, sn0);
    return attn_out(L, prep[3], attention_core(L, prep, m));
}

// ---- MoE
std::pair<Tensor, Tensor> Card::dense_mlp(int L, const Tensor& x, bool router) {
    std::string p = "layers." + num(L) + ".mlp";
    std::vector<std::string> names = {p + ".shared_expert.gate_proj", p + ".shared_expert.up_proj", p + ".shared_expert_gate"};
    if (router) names.push_back(p + ".gate");
    auto ys = lins(x, names);
    Tensor y = o.mul(o.linear(o.mul(o.unary(ys[0], Unary::silu), ys[1]), lin(p + ".shared_expert.down_proj")),
                     o.unary(ys[2], Unary::sigmoid));
    return {router ? o.softmax(ys[3]) : Tensor{}, y};
}

void Card::route(int, const Tensor& probs, int m, std::vector<float>& gate, std::vector<int>& es) {
    int E = c.E;
    std::vector<float> P = o.get(probs);               // one read of [m, E]; the choice is made on the host
    gate.assign((size_t)m * E, 0.0f);
    std::set<int> used;
    for (int i = 0; i < m; i++) {
        std::vector<int> idx(E);
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return P[i * E + a] > P[i * E + b]; });
        float s = 0;
        for (int j = 0; j < c.top_k; j++) s += P[i * E + idx[j]];
        for (int j = 0; j < c.top_k; j++) {
            gate[(size_t)i * E + idx[j]] = c.norm_topk ? P[i * E + idx[j]] / s : P[i * E + idx[j]];
            used.insert(idx[j]);
        }
    }
    es.assign(used.begin(), used.end());
}

void Card::put_slot(int slot, int L, int e) {
    int d = c.d, I = c.I;
    if (!have_banks) {
        banks[0] = o.bank(n_slots, d, I, expert_fmt);
        banks[1] = o.bank(n_slots, d, I, expert_fmt);
        banks[2] = o.bank(n_slots, I, d, expert_fmt);
        have_banks = true;
    }
    size_t bytes = 0;
    const uint8_t* tl = o.takes_tiles() ? (const uint8_t*)host.expert_tiles(L, e, bytes) : nullptr;
    if (tl) {                                          // TT tiles from the store as they are: no packing on the host
        enum bfp_fmt f = expert_fmt == "bfp8" ? BFP8 : BFP4;
        size_t tb = bfp_tt_bytes(f, 32, 32), rows = d / 32, cols = 2 * I / 32, half = cols / 2;
        std::vector<uint8_t> g(rows * half * tb), u(rows * half * tb);   // W^T [d, 2I]: gate | up tile columns
        for (size_t r = 0; r < rows; r++) {
            std::copy(tl + (r * cols) * tb, tl + (r * cols + half) * tb, g.begin() + r * half * tb);
            std::copy(tl + (r * cols + half) * tb, tl + (r * cols + cols) * tb, u.begin() + r * half * tb);
        }
        size_t gu = bfp_tt_bytes(f, 2 * I, d);
        o.bank_write_tiles(banks[0], slot, g.data(), g.size());
        o.bank_write_tiles(banks[1], slot, u.data(), u.size());
        o.bank_write_tiles(banks[2], slot, tl + gu, bytes - gu);
        return;
    }
    Host::Bfp m[3];
    if (o.takes_bfp() && host.expert_bfp(L, e, m)) {   // the store's BFP as it is (the GPU's kernels read it)
        for (int i = 0; i < 3; i++) o.bank_write_bfp(banks[i], slot, m[i].exp, m[i].mant, m[i].out, m[i].in, m[i].mbits);
        return;
    }
    std::string p = "layers." + num(L) + ".mlp.experts.";
    std::vector<float> gu = w.expert(p + "gate_up_proj", e), dn = w.expert(p + "down_proj", e);
    o.bank_write(banks[0], slot, gu.data(), I, d);
    o.bank_write(banks[1], slot, gu.data() + (size_t)I * d, I, d);
    o.bank_write(banks[2], slot, dn.data(), d, I);
}

Tensor Card::slot_experts(const Tensor& x, const std::vector<std::pair<int, std::vector<float>>>& hot) {
    int m = x.dim(0), d = x.dim(1), n = (int)hot.size();
    std::vector<int> slots;
    std::vector<float> G;
    for (auto& [s, g] : hot) {
        slots.push_back(s);
        G.insert(G.end(), g.begin(), g.end());
    }
    Ops::Fn core = [&](const std::vector<Tensor>& a) {   // banks updated in place: traces read them by address
        Tensor x3 = o.reshape(a[0], {1, m, d});
        Tensor h = o.mul(o.gather_matmul(x3, banks[0], a[1], Act::silu), o.gather_matmul(x3, banks[1], a[1]));
        return std::vector<Tensor>{o.reshape(o.sum(o.mul(o.gather_matmul(h, banks[2], a[1]), a[2]), 0), {m, d})};
    };
    return seg("slots" + num(n), core, {x, o.put_ids(slots), o.put(G, {n, m, 1})}, m)[0];
}

Tensor Card::moe(int L, const Tensor& x, const Tensor& probs, Tensor y) {
    int m = x.dim(0), E = c.E;
    std::vector<float> gate;
    std::vector<int> es;
    route(L, probs, m, gate, es);
    std::vector<int32_t> ids, slot(es.size());
    for (int e : es) ids.push_back(L * E + e);
    tcache_lookup(&cache, ids.data(), (int)ids.size(), slot.data());
    std::vector<std::pair<int, std::vector<float>>> hot;
    std::vector<int> cold;
    std::vector<float> Gc;
    for (size_t k = 0; k < es.size(); k++) {
        std::vector<float> g(m);
        int n = 0;
        for (int i = 0; i < m; i++) n += (g[i] = gate[(size_t)i * E + es[k]]) > 0;
        lookups += n;
        if (slot[k] >= 0) {
            hits += n;
            hot.push_back({slot[k], g});
        } else {
            cold.push_back(es[k]);
            Gc.insert(Gc.end(), g.begin(), g.end());
        }
    }
    std::vector<float> xh;
    if (!cold.empty()) xh = o.get(x);                  // read before queueing the slot experts: they overlap
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

// ---- PLE
Tensor Card::ple_layer(int L, const Tensor& h4, const std::vector<int>& toks) {
    std::string p = "layers." + num(L) + ".ple";
    int m = h4.dim(0), hc = c.hc, d = c.d;
    std::vector<float> rows;
    std::vector<int> hist = tokens;
    for (int i = 0; i < m; i++) {
        hist.push_back(toks[i]);
        auto r = host.ngram_rows(L, hist);
        rows.insert(rows.end(), r.begin(), r.end());
    }
    Ops::Fn core = [&](const std::vector<Tensor>& a) {
        const Tensor &h = a[0], &emb = a[1];
        Tensor key = rms(o.reshape(o.linear(emb, lin(p + ".key_proj")), {m, hc, d}), norm(p + ".norm_key.weight", true, hc));
        Tensor value = o.reshape(o.linear(emb, lin(p + ".value_proj")), {m, 1, d});
        Tensor query = rms(o.reshape(h, {m, hc, d}), norm(p + ".norm_query.weight", true, hc));
        Tensor g = o.scale(o.sum(o.mul(key, query), -1), 1.0f / std::sqrt((float)d));
        g = o.mul(o.unary(o.clamp_min(o.unary(g, Unary::abs), 1e-6f), Unary::sqrt), o.unary(g, Unary::sign));
        Tensor gv = o.mul(o.unary(g, Unary::sigmoid), value);
        Tensor gvn = o.reshape(rms(gv, norm(p + ".norm_conv.weight", true, hc)), {m, hc * d});
        Tensor win = o.concat({a[2], gvn}, 0);
        const Array& cwa = w.at(p + ".conv1d.weight");
        Tensor cw = constant(p + ".conv", conv_taps(cwa), {(int)cwa.shape[2], (int)cwa.shape[0]});
        return std::vector<Tensor>{o.add(h, o.add(o.reshape(gv, {m, hc * d}), o.unary(conv(win, cw, m, c.ngram), Unary::silu))),
                                   win};
    };
    auto r = seg("ple" + num(L), core, {h4, o.put(rows, {m, (int)(rows.size() / m)}), st[L].ple}, m);
    pend[L].ple = r[1];
    pend[L].has_ple = true;
    return r[0];
}

// ---- a layer = one or a few traced segments, broken where the host decides
std::pair<Tensor, Card::Carry> Card::deltanet_layer(int L, const Tensor& h4, const Carry& carry) {
    std::string pre = "layers." + num(L);
    size_t nc = carry.size();
    Ops::Fn dense = [&](const std::vector<Tensor>& a) {          // everything up to the routing decision
        Tensor h = settle(a[0], Carry(a.begin() + 1, a.begin() + 1 + nc));
        auto [x, inj] = hc_read(pre + ".attn_hyper_connection", h);
        auto dn = deltanet_core(L, x, a[1 + nc], a[2 + nc]);
        h = hc_write(h, inj, dn[0]);
        auto [x2, inj2] = hc_read(pre + ".mlp_hyper_connection", h);
        auto [probs, y] = dense_mlp(L, x2);
        return std::vector<Tensor>{h, x2, inj2, dn[1], dn[2], probs, y};
    };
    std::vector<Tensor> args = {h4};
    args.insert(args.end(), carry.begin(), carry.end());
    args.push_back(st[L].conv);
    args.push_back(st[L].S);
    auto r = seg("L" + num(L) + "dn" + num((int)nc), dense, args, h4.dim(0));
    pend[L].win = r[3];
    pend[L].states = r[4];
    pend[L].lin = true;
    return {r[0], {r[2], moe(L, r[1], r[5], r[6])}};
}

std::pair<Tensor, Card::Carry> Card::attention_layer(int L, const Tensor& h4, const Carry& carry) {
    std::string pre = "layers." + num(L);
    int m = h4.dim(0);
    size_t nc = carry.size();
    Ops::Fn prep = [&](const std::vector<Tensor>& a) {
        Tensor h = settle(a[0], Carry(a.begin() + 1, a.begin() + 1 + nc));
        auto [x, inj] = hc_read(pre + ".attn_hyper_connection", h);
        std::vector<Tensor> out = {h, inj};
        auto pr = attn_prep(L, x, a[1 + nc], a[2 + nc]);
        out.insert(out.end(), pr.begin(), pr.end());
        return out;
    };
    Ops::Fn post = [&](const std::vector<Tensor>& a) {
        Tensor h = hc_write(a[0], a[1], attn_out(L, a[2], std::vector<Tensor>(a.begin() + 3, a.end())));
        auto [x2, inj2] = hc_read(pre + ".mlp_hyper_connection", h);
        auto [probs, y] = dense_mlp(L, x2);
        return std::vector<Tensor>{h, x2, inj2, probs, y};
    };
    auto [cs0, sn0] = cs(t + pos0, m);
    std::vector<Tensor> args = {h4};
    args.insert(args.end(), carry.begin(), carry.end());
    args.push_back(cs0);
    args.push_back(sn0);
    auto a = seg("L" + num(L) + "pre" + num((int)nc), prep, args, m);
    auto outs = attention_core(L, std::vector<Tensor>(a.begin() + 2, a.end()), m);
    std::vector<Tensor> pargs = {a[0], a[1], a[5]};                  // h, inject weights, output gate
    pargs.insert(pargs.end(), outs.begin(), outs.end());
    auto r = seg("L" + num(L) + "post", post, pargs, m);
    return {r[0], {r[2], moe(L, r[1], r[3], r[4])}};
}

Tensor Card::layers(Tensor h, const std::vector<int>& toks) {
    std::set<int> ple;
    for (int id : c.ple_layer_ids) ple.insert(id - 1);
    for (auto& p : pend) p = Pending{};
    Carry carry;
    for (int L = 0; L < c.layers; L++) {
        if (ple.count(L)) {
            h = settle(h, carry);
            carry.clear();
            h = ple_layer(L, h, toks);
        }
        std::tie(h, carry) = c.layer_types[L] == "linear_attention" ? deltanet_layer(L, h, carry) : attention_layer(L, h, carry);
    }
    return settle(h, carry);
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
            int n = p.states.dim(0), mm = p.win.dim(0) - (c.conv_k - 1);
            if (n < mm && j != mm) throw std::runtime_error("a long pass keeps only its last state");
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

std::vector<int> Card::lm_argmax(const Tensor& mixed) {
    std::vector<float> l = o.get(o.linear(mixed, lin("lm_head")));
    int V = (int)(l.size() / mixed.dim(0)), m = mixed.dim(0);
    std::vector<int> out;
    for (int i = 0; i < m; i++) out.push_back((int)(std::max_element(&l[(size_t)i * V], &l[(size_t)(i + 1) * V]) - &l[(size_t)i * V]));
    return out;
}

// ---------------------------------------------------------------- MTP drafter
MTPDrafter::MTPDrafter(Ops& o, const Config& c, const Weights& w_mtp, Host& host, const std::string& dense_fmt,
                       const std::string& expert_fmt, int n_slots, int budget, int max_t, std::vector<int> vocab)
    : mc(c.mtp()), vocab(std::move(vocab)) {
    m = std::make_unique<Card>(o, mc, w_mtp, host, dense_fmt, expert_fmt, n_slots, budget, max_t);
    m->pos0 = 1;                                       // cache row r holds position r + 1
}

std::pair<Tensor, int> MTPDrafter::run(const Tensor& Rin, const std::vector<int>& toks, int row, bool guess) {
    Card& M = *m;
    Ops& o = M.o;
    int hc = mc.hc, d = mc.d, n = (int)toks.size();
    Tensor hn = M.rms(o.reshape(Rin, {n, hc, d}), M.norm("pre_fc_norm_hidden.weight", true, hc));
    Tensor h2 = o.linear(hn, M.lin("fc_hidden"));                         // [n, hc, d], per stream
    std::vector<float> e;
    for (int tk : toks) {
        auto r = M.host.embed(tk);
        e.insert(e.end(), r.begin(), r.end());
    }
    Tensor e2 = o.linear(M.rms(o.put(e, {n, d}), M.norm("pre_fc_norm_embedding.weight")), M.lin("fc_embedding"));
    M.t = row;
    Tensor x = M.layers(o.reshape(o.add(h2, o.reshape(e2, {n, 1, d})), {n, hc * d}), {});
    for (auto& p : M.pend) p = Card::Pending{};
    M.promote();
    if (!guess) return {x, -1};
    Tensor mixed = M.hc_read("hyper_connection_mixer", o.slice(x, 0, n - 1, n), false).first;
    if (vocab.empty()) return {x, lm->lm_argmax(mixed)[0]};
    if (!have_head) {
        const Array& lh = lm->w.at("lm_head.weight");
        size_t D = lh.row_size();
        std::vector<float> rows;
        for (int v : vocab) rows.insert(rows.end(), lh.f.begin() + v * D, lh.f.begin() + (v + 1) * D);
        head = o.put_weight(rows.data(), (int)vocab.size(), (int)D, M.fmt("lm_head"));
        have_head = true;
    }
    std::vector<float> l = o.get(o.linear(mixed, head));
    return {x, vocab[std::max_element(l.begin(), l.end()) - l.begin()]};
}

void MTPDrafter::catch_up(Decoder& card, int start, int j) {
    lm = &card;
    Ops& o = card.o;
    int p0 = std::max(start - 1, 0), p1 = start + j - 1;                 // cache rows p0 .. p1-1
    if (p1 > p0) {
        std::vector<Tensor> rows;
        if (p0 == start - 1) rows.push_back(R);
        if (p1 - start > std::max(p0 - start, 0)) rows.push_back(o.slice(card.h4, 0, std::max(p0 - start, 0), p1 - start));
        Tensor Rr = rows.size() == 1 ? rows[0] : o.concat(rows, 0);
        run(Rr, std::vector<int>(card.tokens.begin() + p0 + 1, card.tokens.begin() + p1 + 1), p0, false);
    }
    R = o.slice(card.h4, 0, j - 1, j);                 // residual of the last accepted
}

std::vector<int> MTPDrafter::propose(Decoder& card, int nxt, int k) {
    std::vector<int> drafts;
    Tensor Rr = R;
    int tok = nxt;
    for (int i = 0; i < k; i++) {
        auto [x, g] = run(Rr, {tok}, card.t - 1 + i, true);
        drafts.push_back(g);
        Rr = x;
        tok = g;
    }
    return drafts;
}

// ---------------------------------------------------------------- card + GPU (tensor parallel)
static Array take_rows(const Array& a, const std::vector<int64_t>& idx) {
    Array b;
    size_t n = a.row_size();
    b.shape = a.shape;
    b.shape[0] = idx.size();
    for (auto r : idx) b.f.insert(b.f.end(), a.f.begin() + r * n, a.f.begin() + (r + 1) * n);
    return b;
}

static Array take_cols(const Array& a, const std::vector<int64_t>& idx) {
    Array b;
    size_t n = a.shape[1];
    b.shape = {a.shape[0], (int64_t)idx.size()};
    for (int64_t r = 0; r < a.shape[0]; r++)
        for (auto j : idx) b.f.push_back(a.f[r * n + j]);
    return b;
}

static std::vector<int64_t> range(int64_t a, int64_t b) {
    std::vector<int64_t> v(b - a);
    std::iota(v.begin(), v.end(), a);
    return v;
}

TP::TP(Ops& card_ops, Ops& gpu_ops, const Config& c, const Weights& w, Host& host, float f, int n_card, int n_gpu,
       int budget, const std::string& dense_fmt, const std::string& expert_fmt, int max_t)
    : Decoder(card_ops, w), c(c), host(host), ca(c), cb(c), wa(w.shard()), wb(w.shard()), n_card(n_card) {
    // split_parts: the GPU's share f of every split dimension; the card keeps the first part
    auto cut = [&](int n, int step) {
        return std::max(step, std::min(n - step, (int)std::nearbyint(n * (1 - f) / step) * step));
    };
    int K = c.lin_k_heads, V = c.lin_v_heads, dk = c.lin_k_dim, dv = c.lin_v_dim, H = c.H, KV = c.KV, hd = c.head_dim;
    int G = V / K, GA = H / KV;
    int ck = cut(K, 1), cj = cut(KV, 1), csh = cut(c.S, 16), cv = cut(c.vocab, 32);
    v_gpu = cv;
    for (int side = 0; side < 2; side++) {
        Config& cc = side ? cb : ca;
        Weights& ww = side ? wb : wa;
        int k0 = side ? ck : 0, k1 = side ? K : ck, j0 = side ? cj : 0, j1 = side ? KV : cj;
        int s0 = side ? csh : 0, s1 = side ? c.S : csh, v0 = side ? cv : 0, v1 = side ? c.vocab : cv;
        cc.lin_k_heads = k1 - k0;
        cc.lin_v_heads = (k1 - k0) * G;
        cc.KV = j1 - j0;
        cc.H = (j1 - j0) * GA;
        cc.S = s1 - s0;
        std::vector<int64_t> ks = range(k0 * dk, k1 * dk), vs = range((int64_t)k0 * G * dv, (int64_t)k1 * G * dv), ch;
        for (auto x : ks) ch.push_back(x);
        for (auto x : ks) ch.push_back(K * dk + x);
        for (auto x : vs) ch.push_back(2 * K * dk + x);
        auto heads = range(k0 * G, k1 * G);
        auto qrows = range((int64_t)j0 * GA * 2 * hd, (int64_t)j1 * GA * 2 * hd), kvrows = range(j0 * hd, j1 * hd),
             ocols = range((int64_t)j0 * GA * hd, (int64_t)j1 * GA * hd);
        for (int L = 0; L < c.layers; L++) {
            std::string p = "layers." + num(L) + ".";
            if (c.layer_types[L] == "linear_attention") {
                std::string a = p + "linear_attn.";
                for (auto [n, idx] : std::vector<std::pair<std::string, std::vector<int64_t>*>>{
                         {"in_proj_qkv.weight", &ch}, {"conv1d.weight", &ch}, {"in_proj_z.weight", &vs},
                         {"in_proj_a.weight", &heads}, {"in_proj_b.weight", &heads}, {"A_log", &heads}, {"dt_bias", &heads}})
                    ww.t[a + n] = take_rows(w.at(a + n), *idx);
                ww.t[a + "out_proj.weight"] = take_cols(w.at(a + "out_proj.weight"), vs);
            } else {
                std::string a = p + "self_attn.";
                ww.t[a + "q_proj.weight"] = take_rows(w.at(a + "q_proj.weight"), qrows);
                ww.t[a + "k_proj.weight"] = take_rows(w.at(a + "k_proj.weight"), kvrows);
                ww.t[a + "v_proj.weight"] = take_rows(w.at(a + "v_proj.weight"), kvrows);
                ww.t[a + "o_proj.weight"] = take_cols(w.at(a + "o_proj.weight"), ocols);
            }
            std::string m = p + "mlp.shared_expert.";
            ww.t[m + "gate_proj.weight"] = take_rows(w.at(m + "gate_proj.weight"), range(s0, s1));
            ww.t[m + "up_proj.weight"] = take_rows(w.at(m + "up_proj.weight"), range(s0, s1));
            ww.t[m + "down_proj.weight"] = take_cols(w.at(m + "down_proj.weight"), range(s0, s1));
        }
        ww.t["lm_head.weight"] = take_rows(w.at("lm_head.weight"), range(v0, v1));
    }
    A = std::make_unique<Card>(card_ops, ca, wa, host, dense_fmt, expert_fmt, n_card, budget, max_t);
    B = std::make_unique<Card>(gpu_ops, cb, wb, host, dense_fmt, expert_fmt, n_gpu, budget, max_t);
    if (tcache_init(&cache, c.layers * c.E, n_card, n_gpu, budget, nullptr, 0)) throw std::runtime_error("tcache_init");
}

TP::~TP() { tcache_free(&cache); }

Tensor TP::moe(int L, const Tensor& x) {
    Card &a = *A, &b = *B;
    int m = x.dim(0), E = c.E;
    auto [probs, ya] = a.dense_mlp(L, x);
    std::vector<float> gate;
    std::vector<int> es;
    a.route(L, probs, m, gate, es);
    std::vector<float> xh = a.o.get(x);
    Tensor xb = b.o.put(xh, x.shape);                  // through host memory: no peer-to-peer between the two
    Tensor yb = b.dense_mlp(L, xb, false).second;
    std::vector<int32_t> ids, slot(es.size());
    for (int e : es) ids.push_back(L * E + e);
    tcache_lookup(&cache, ids.data(), (int)ids.size(), slot.data());
    std::vector<std::pair<int, std::vector<float>>> hot_a, hot_b;
    std::vector<int> cold;
    std::vector<float> Gc;
    for (size_t k = 0; k < es.size(); k++) {
        std::vector<float> g(m);
        int n = 0;
        for (int i = 0; i < m; i++) n += (g[i] = gate[(size_t)i * E + es[k]]) > 0;
        lookups += n;
        if (slot[k] < 0) {
            cold.push_back(es[k]);
            Gc.insert(Gc.end(), g.begin(), g.end());
            continue;
        }
        hits += n;
        if (slot[k] < n_card) hot_a.push_back({slot[k], g});
        else hot_b.push_back({slot[k] - n_card, g});
    }
    if (!hot_a.empty()) ya = a.o.add(ya, a.slot_experts(x, hot_a));     // both devices queue their experts;
    if (!hot_b.empty()) yb = b.o.add(yb, b.slot_experts(xb, hot_b));    // meanwhile the host runs the cold ones
    std::vector<float> yc;
    if (!cold.empty()) {
        yc.assign((size_t)m * c.d, 0.0f);
        host.experts(L, xh.data(), m, cold, Gc, yc.data());
    }
    Tensor y = allreduce(ya, yb);
    return cold.empty() ? y : a.o.add(y, a.o.put(yc, {m, c.d}));
}

Tensor TP::verify(const std::vector<int>& toks) {
    Card &a = *A, &b = *B;
    int m = (int)toks.size();
    std::vector<float> rows;
    for (int tok : toks) {
        auto e = host.embed(tok);
        for (int s = 0; s < c.hc; s++) rows.insert(rows.end(), e.begin(), e.end());
    }
    Tensor h = a.o.put(rows, {m, c.hc * c.d});
    for (auto& p : a.pend) p = Card::Pending{};
    for (auto& p : b.pend) p = Card::Pending{};
    b.t = a.t;
    std::set<int> ple;
    for (int id : c.ple_layer_ids) ple.insert(id - 1);
    for (int L = 0; L < c.layers; L++) {
        std::string pre = "layers." + num(L);
        if (ple.count(L)) h = a.ple_layer(L, h, toks);
        auto [x, inj] = a.hc_read(pre + ".attn_hyper_connection", h);
        Tensor xb = to(b, x, a);
        Tensor out = c.layer_types[L] == "linear_attention" ? allreduce(a.deltanet(L, x), b.deltanet(L, xb))
                                                            : allreduce(a.attention(L, x), b.attention(L, xb));
        h = a.hc_write(h, inj, out);
        auto [x2, inj2] = a.hc_read(pre + ".mlp_hyper_connection", h);
        h = a.hc_write(h, inj2, moe(L, x2));
    }
    a.h4 = h4 = h;
    a.verified = b.verified = verified = toks;
    return a.hc_read("hyper_connection_mixer", h, false).first;
}

void TP::commit(int j) {
    A->commit(j);                                      // states, tokens, position; their own caches are unused
    B->commit(j);
    tokens = A->tokens;
    t = A->t;
    std::vector<tcache_copy> cp(cache.budget + 1);
    int n = tcache_end_token(&cache, cp.data());
    for (int i = 0; i < n; i++) {
        int s = cp[i].slot, L = cp[i].expert / c.E, e = cp[i].expert % c.E;
        if (s < n_card) A->put_slot(s, L, e);
        else B->put_slot(s - n_card, L, e);
    }
}

std::vector<int> TP::lm_argmax(const Tensor& mixed) {
    // each device the best of its vocabulary rows; the host keeps the better of the two (ties: the card's)
    int m = mixed.dim(0);
    std::vector<float> la = A->o.get(A->o.linear(mixed, A->lin("lm_head")));
    std::vector<float> lb = B->o.get(B->o.linear(to(*B, mixed, *A), B->lin("lm_head")));
    int Va = (int)(la.size() / m), Vb = (int)(lb.size() / m);
    std::vector<int> out;
    for (int i = 0; i < m; i++) {
        auto ia = std::max_element(&la[(size_t)i * Va], &la[(size_t)(i + 1) * Va]);
        auto ib = std::max_element(&lb[(size_t)i * Vb], &lb[(size_t)(i + 1) * Vb]);
        out.push_back(*ib > *ia ? v_gpu + (int)(ib - &lb[(size_t)i * Vb]) : (int)(ia - &la[(size_t)i * Va]));
    }
    return out;
}

}  // namespace tuff
