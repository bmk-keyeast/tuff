// ops.h on the host in float32: the reference the engine is checked on (as ops.Torch is for device/qfn.py).
// ponytail: plain loops, no BFP rounding of weights (fmt ignored), no traces; it is a reference, not a backend to run.
#include <algorithm>
#include <cmath>
#include <numeric>

#include "ops.h"

namespace tuff {
namespace {

using Buf = std::vector<float>;

struct Bank {
    int n, k, N;
    std::vector<Buf> w;                                  // per slot W^T [k][N]
};

size_t numel(const Shape& s) { return std::accumulate(s.begin(), s.end(), (size_t)1, [](size_t a, int b) { return a * b; }); }
Buf& buf(const Tensor& t) { return *std::static_pointer_cast<Buf>(t.impl); }
Tensor make(Shape s, Buf b) { return Tensor{std::move(s), std::make_shared<Buf>(std::move(b))}; }

// output shape and per-input strides (0 where broadcast) for numpy broadcasting
struct Bcast {
    Shape out;
    std::vector<size_t> sa, sb;
    Bcast(const Shape& a, const Shape& b) {
        int r = (int)std::max(a.size(), b.size());
        Shape A(r, 1), B(r, 1);
        std::copy(a.begin(), a.end(), A.begin() + (r - a.size()));
        std::copy(b.begin(), b.end(), B.begin() + (r - b.size()));
        out.resize(r);
        sa.assign(r, 0);
        sb.assign(r, 0);
        size_t ka = 1, kb = 1;
        for (int i = r - 1; i >= 0; i--) {
            if (A[i] != B[i] && A[i] != 1 && B[i] != 1) throw std::runtime_error("broadcast: shapes differ");
            out[i] = std::max(A[i], B[i]);
            sa[i] = A[i] == 1 ? 0 : ka;
            sb[i] = B[i] == 1 ? 0 : kb;
            ka *= A[i];
            kb *= B[i];
        }
    }
};

template <class F>
Tensor binary(const Tensor& a, const Tensor& b, F f) {
    Bcast bc(a.shape, b.shape);
    const Buf &A = buf(a), &B = buf(b);
    Buf y(numel(bc.out));
    int r = (int)bc.out.size();
    std::vector<int> idx(r, 0);
    size_t ia = 0, ib = 0;
    for (size_t o = 0; o < y.size(); o++) {
        y[o] = f(A[ia], B[ib]);
        for (int d = r - 1; d >= 0; d--) {                  // odometer
            idx[d]++;
            ia += bc.sa[d];
            ib += bc.sb[d];
            if (idx[d] < bc.out[d]) break;
            ia -= bc.sa[d] * idx[d];
            ib -= bc.sb[d] * idx[d];
            idx[d] = 0;
        }
    }
    return make(bc.out, std::move(y));
}

float act(float v, Unary u) {
    switch (u) {
        case Unary::silu: return v / (1.0f + std::exp(-v));
        case Unary::sigmoid: return 1.0f / (1.0f + std::exp(-v));
        case Unary::softplus: return v > 20.0f ? v : std::log1p(std::exp(v));
        case Unary::relu: return v > 0 ? v : 0.0f;
        case Unary::exp: return std::exp(v);
        case Unary::sqrt: return std::sqrt(v);
        case Unary::rsqrt: return 1.0f / std::sqrt(v);
        case Unary::abs: return std::fabs(v);
        case Unary::sign: return (float)((v > 0) - (v < 0));
    }
    return v;
}

// [outer, n, inner] view of a dim
void split(const Shape& s, int dim, size_t& outer, size_t& n, size_t& inner) {
    if (dim < 0) dim += (int)s.size();
    outer = inner = 1;
    for (int i = 0; i < dim; i++) outer *= s[i];
    n = s[dim];
    for (size_t i = dim + 1; i < s.size(); i++) inner *= s[i];
}

struct CPU : Ops {
    Tensor put(const std::vector<float>& a, Shape s) override {
        if (a.size() != numel(s)) throw std::runtime_error("put: size");
        return make(std::move(s), a);
    }
    Tensor put_weight(const float* w, int out, int in, const std::string&) override {
        Buf t((size_t)in * out);
        for (int o = 0; o < out; o++)
            for (int i = 0; i < in; i++) t[(size_t)i * out + o] = w[(size_t)o * in + i];
        return make({in, out}, std::move(t));
    }
    std::vector<float> get(const Tensor& x) override { return buf(x); }
    Tensor zeros(Shape s) override { size_t n = numel(s); return make(std::move(s), Buf(n, 0.0f)); }
    Tensor put_ids(const std::vector<int>& ids) override {
        return make({(int)ids.size()}, Buf(ids.begin(), ids.end()));
    }

    Tensor matmul(const Tensor& a, const Tensor& b) override {
        int M = a.dim(-2), K = a.dim(-1), N = b.dim(-1);
        if (b.dim(-2) != K) throw std::runtime_error("matmul: inner dims");
        Shape ba(a.shape.begin(), a.shape.end() - 2), bb(b.shape.begin(), b.shape.end() - 2);
        Bcast bc(ba, bb);                                       // batch dims
        size_t nb = numel(bc.out);
        Buf y(nb * M * N, 0.0f);
        const Buf &A = buf(a), &B = buf(b);
        int r = (int)bc.out.size();
        std::vector<int> idx(r, 0);
        size_t ia = 0, ib = 0;
        for (size_t z = 0; z < nb; z++) {
            const float *pa = &A[ia * M * K], *pb = &B[ib * K * N];
            float* py = &y[z * M * N];
            for (int m = 0; m < M; m++)
                for (int k = 0; k < K; k++) {
                    float v = pa[(size_t)m * K + k];
                    for (int n = 0; n < N; n++) py[(size_t)m * N + n] += v * pb[(size_t)k * N + n];
                }
            for (int d = r - 1; d >= 0; d--) {
                idx[d]++;
                ia += bc.sa[d];
                ib += bc.sb[d];
                if (idx[d] < bc.out[d]) break;
                ia -= bc.sa[d] * idx[d];
                ib -= bc.sb[d] * idx[d];
                idx[d] = 0;
            }
        }
        Shape s = bc.out;
        s.push_back(M);
        s.push_back(N);
        return make(s, std::move(y));
    }
    Tensor linear(const Tensor& x, const Tensor& w, Act a) override {
        Tensor y = matmul(x, w);
        return a == Act::none ? y : unary(y, a == Act::silu ? Unary::silu : Unary::sigmoid);
    }
    Tensor add(const Tensor& a, const Tensor& b) override { return binary(a, b, [](float x, float y) { return x + y; }); }
    Tensor mul(const Tensor& a, const Tensor& b) override { return binary(a, b, [](float x, float y) { return x * y; }); }
    Tensor scale(const Tensor& a, float s) override {
        Buf y = buf(a);
        for (float& v : y) v *= s;
        return make(a.shape, std::move(y));
    }
    Tensor unary(const Tensor& a, Unary u) override {
        Buf y = buf(a);
        for (float& v : y) v = act(v, u);
        return make(a.shape, std::move(y));
    }
    Tensor clamp_min(const Tensor& a, float m) override {
        Buf y = buf(a);
        for (float& v : y) v = std::max(v, m);
        return make(a.shape, std::move(y));
    }
    Tensor reduce(const Tensor& a, int dim, bool avg) {
        size_t outer, n, inner;
        split(a.shape, dim, outer, n, inner);
        const Buf& A = buf(a);
        Buf y(outer * inner, 0.0f);
        for (size_t o = 0; o < outer; o++)
            for (size_t i = 0; i < n; i++)
                for (size_t j = 0; j < inner; j++) y[o * inner + j] += A[(o * n + i) * inner + j];
        if (avg)
            for (float& v : y) v /= (float)n;
        Shape s = a.shape;
        s[dim < 0 ? dim + s.size() : dim] = 1;
        return make(s, std::move(y));
    }
    Tensor sum(const Tensor& a, int dim) override { return reduce(a, dim, false); }
    Tensor mean(const Tensor& a, int dim) override { return reduce(a, dim, true); }
    Tensor softmax(const Tensor& a) override {
        Buf y = buf(a);
        size_t n = a.dim(-1);
        for (size_t r = 0; r < y.size() / n; r++) {
            float* p = &y[r * n];
            float mx = *std::max_element(p, p + n), s = 0;
            for (size_t i = 0; i < n; i++) s += (p[i] = std::exp(p[i] - mx));
            for (size_t i = 0; i < n; i++) p[i] /= s;
        }
        return make(a.shape, std::move(y));
    }
    Tensor rms_norm(const Tensor& x, const Tensor& w, float eps) override {
        Buf y = buf(x);
        size_t n = x.dim(-1);
        for (size_t r = 0; r < y.size() / n; r++) {
            float* p = &y[r * n];
            float ss = 0;
            for (size_t i = 0; i < n; i++) ss += p[i] * p[i];
            float k = 1.0f / std::sqrt(ss / n + eps);
            for (size_t i = 0; i < n; i++) p[i] *= k;
        }
        return mul(make(x.shape, std::move(y)), w);
    }

    Tensor reshape(const Tensor& a, Shape s) override {
        if (numel(s) != numel(a.shape)) throw std::runtime_error("reshape: size");
        return Tensor{std::move(s), a.impl};                    // shares the data, as torch's view
    }
    Tensor slice(const Tensor& a, int dim, int start, int end) override {
        size_t outer, n, inner;
        split(a.shape, dim, outer, n, inner);
        const Buf& A = buf(a);
        Buf y(outer * (end - start) * inner);
        for (size_t o = 0; o < outer; o++)
            std::copy(&A[(o * n + start) * inner], &A[(o * n + end) * inner], &y[o * (end - start) * inner]);
        Shape s = a.shape;
        s[dim < 0 ? dim + s.size() : dim] = end - start;
        return make(s, std::move(y));
    }
    Tensor concat(const std::vector<Tensor>& xs, int dim) override {
        Shape s = xs[0].shape;
        int d = dim < 0 ? dim + (int)s.size() : dim, tot = 0;
        for (auto& x : xs) tot += x.shape[d];
        s[d] = tot;
        size_t outer, n, inner;
        split(s, d, outer, n, inner);
        Buf y(numel(s));
        size_t off = 0;
        for (auto& x : xs) {
            const Buf& X = buf(x);
            size_t k = x.shape[d];
            for (size_t o = 0; o < outer; o++)
                std::copy(&X[o * k * inner], &X[(o + 1) * k * inner], &y[(o * n + off) * inner]);
            off += k;
        }
        return make(s, std::move(y));
    }
    Tensor permute(const Tensor& a, const std::vector<int>& dims) override {
        int r = a.rank();
        Shape s(r);
        std::vector<size_t> st(r), sin(r);
        size_t k = 1;
        for (int i = r - 1; i >= 0; i--) { sin[i] = k; k *= a.shape[i]; }
        for (int i = 0; i < r; i++) { s[i] = a.shape[dims[i]]; st[i] = sin[dims[i]]; }
        const Buf& A = buf(a);
        Buf y(A.size());
        std::vector<int> idx(r, 0);
        size_t src = 0;
        for (size_t o = 0; o < y.size(); o++) {
            y[o] = A[src];
            for (int d = r - 1; d >= 0; d--) {
                idx[d]++;
                src += st[d];
                if (idx[d] < s[d]) break;
                src -= st[d] * idx[d];
                idx[d] = 0;
            }
        }
        return make(s, std::move(y));
    }
    Tensor transpose(const Tensor& a) override {
        std::vector<int> d(a.rank());
        std::iota(d.begin(), d.end(), 0);
        std::swap(d[a.rank() - 1], d[a.rank() - 2]);
        return permute(a, d);
    }

    Tensor gather_rows(const Tensor& table, const Tensor& ids) override {
        const Buf &T = buf(table), &I = buf(ids);
        size_t D = table.dim(-1);
        Buf y(I.size() * D);
        for (size_t r = 0; r < I.size(); r++) std::copy(&T[(size_t)I[r] * D], &T[((size_t)I[r] + 1) * D], &y[r * D]);
        return make({(int)I.size(), (int)D}, std::move(y));
    }
    void write_rows(Tensor& table, int row, const Tensor& rows) override {
        Buf& T = buf(table);
        const Buf& R = buf(rows);
        std::copy(R.begin(), R.end(), &T[(size_t)row * table.dim(-1)]);
    }

    std::pair<Tensor, Tensor> gdr(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g, const Tensor& beta,
                                  const Tensor& state, float scale, bool per_token) override {
        int m = q.dim(0), H = q.dim(1), K = q.dim(2), HV = v.dim(1), V = v.dim(2), G = HV / H;
        const Buf &Q = buf(q), &Kb = buf(k), &Vb = buf(v), &Gb = buf(g), &Bb = buf(beta);
        Buf S = buf(state), o((size_t)m * HV * V), states;
        std::vector<float> kv(V);
        for (int t = 0; t < m; t++) {
            for (int h = 0; h < HV; h++) {
                const float *qh = &Q[((size_t)t * H + h / G) * K], *kh = &Kb[((size_t)t * H + h / G) * K];
                float* Sh = &S[(size_t)h * K * V];
                float decay = std::exp(Gb[(size_t)t * HV + h]), b = Bb[(size_t)t * HV + h];
                for (int i = 0; i < K * V; i++) Sh[i] *= decay;
                std::fill(kv.begin(), kv.end(), 0.0f);
                for (int i = 0; i < K; i++)
                    for (int j = 0; j < V; j++) kv[j] += Sh[i * V + j] * kh[i];
                for (int j = 0; j < V; j++) kv[j] = (Vb[((size_t)t * HV + h) * V + j] - kv[j]) * b;   // delta
                for (int i = 0; i < K; i++)
                    for (int j = 0; j < V; j++) Sh[i * V + j] += kh[i] * kv[j];
                float* oh = &o[((size_t)t * HV + h) * V];
                for (int j = 0; j < V; j++) {
                    float s = 0;
                    for (int i = 0; i < K; i++) s += Sh[i * V + j] * qh[i] * scale;
                    oh[j] = s;
                }
            }
            if (per_token || t == m - 1) states.insert(states.end(), S.begin(), S.end());
        }
        int n = per_token ? m : 1;
        return {make({m, HV, V}, std::move(o)), make({n, HV, K, V}, std::move(states))};
    }

    Tensor bank(int n, int k, int N, const std::string&) override {
        auto b = std::make_shared<Bank>(Bank{n, k, N, std::vector<Buf>(n)});
        return Tensor{{n, k, N}, b};
    }
    void bank_write(Tensor& bank, int slot, const float* w, int N, int k) override {
        Bank& b = *std::static_pointer_cast<Bank>(bank.impl);
        Buf t((size_t)k * N);
        for (int o = 0; o < N; o++)
            for (int i = 0; i < k; i++) t[(size_t)i * N + o] = w[(size_t)o * k + i];
        b.w[slot] = std::move(t);
    }
    Tensor gather_matmul(const Tensor& x, const Tensor& bank, const Tensor& ids, Act a) override {
        const Bank& b = *std::static_pointer_cast<Bank>(bank.impl);
        const Buf& I = buf(ids);
        int g = x.dim(0), m = x.dim(1);
        std::vector<Tensor> ys;
        for (size_t i = 0; i < I.size(); i++) {
            Tensor xi = slice(x, 0, g > 1 ? (int)i : 0, (g > 1 ? (int)i : 0) + 1);
            Tensor w = make({b.k, b.N}, b.w[(size_t)I[i]]);
            ys.push_back(reshape(linear(reshape(xi, {m, b.k}), w, a), {1, m, b.N}));
        }
        return concat(ys, 0);
    }
};

}  // namespace

std::unique_ptr<Ops> cpu_ops() { return std::make_unique<CPU>(); }

}  // namespace tuff
