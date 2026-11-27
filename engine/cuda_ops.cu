// ops.h on an NVIDIA GPU (the 4060 Ti next to the card): BFP weights stay BFP in GPU memory, in the host layout of
// kernels/bfp.h, and multiply through kernels/bfp_cuda.cu (tgpu_gemv); the rest are plain CUDA kernels below.
// ponytail: simple one-thread-per-output kernels: the GPU's share is bandwidth-bound BFP matmuls (tgpu_gemv); the rest
// is small. Compiled in CI (no GPU there); run tuff_engine test against the CPU reference on the box first.
#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <numeric>

#include "ops.h"

extern "C" {
#include "../kernels/bfp.h"
#include "../kernels/bfp_cuda.h"
}

namespace tuff {
namespace {

#define CK(x)                                                                                \
    do {                                                                                     \
        cudaError_t e_ = (x);                                                                \
        if (e_ != cudaSuccess) throw std::runtime_error(std::string("cuda: ") + cudaGetErrorString(e_)); \
    } while (0)

struct Dev {                                           // a float buffer, or a BFP weight (exp + mant)
    float* p = nullptr;
    size_t n = 0;
    uint8_t *exp = nullptr, *mant = nullptr;
    int out = 0, in = 0, mb = 0;
    ~Dev() {
        cudaFree(p);
        cudaFree(exp);
        cudaFree(mant);
    }
};

struct Bank { int n, k, N; std::vector<std::shared_ptr<Dev>> w; };

constexpr int R = 6;                                   // max rank
struct Strides { int r; int shape[R]; long long sa[R], sb[R]; };

size_t numel(const Shape& s) { return std::accumulate(s.begin(), s.end(), (size_t)1, [](size_t a, int b) { return a * b; }); }
Dev& dv(const Tensor& t) { return *std::static_pointer_cast<Dev>(t.impl); }
float* P(const Tensor& t) { return dv(t).p; }

std::shared_ptr<Dev> alloc(size_t n) {
    auto d = std::make_shared<Dev>();
    d->n = n;
    CK(cudaMalloc(&d->p, std::max<size_t>(n, 1) * 4));
    return d;
}
Tensor make(Shape s) { size_t n = numel(s); return Tensor{std::move(s), alloc(n)}; }

Strides bcast(const Shape& a, const Shape& b, Shape& out) {
    Strides st{};
    int r = (int)std::max(a.size(), b.size());
    st.r = r;
    Shape A(r, 1), B(r, 1);
    std::copy(a.begin(), a.end(), A.begin() + (r - a.size()));
    std::copy(b.begin(), b.end(), B.begin() + (r - b.size()));
    out.assign(r, 0);
    long long ka = 1, kb = 1;
    for (int i = r - 1; i >= 0; i--) {
        if (A[i] != B[i] && A[i] != 1 && B[i] != 1) throw std::runtime_error("broadcast: shapes differ");
        out[i] = st.shape[i] = std::max(A[i], B[i]);
        st.sa[i] = A[i] == 1 ? 0 : ka;
        st.sb[i] = B[i] == 1 ? 0 : kb;
        ka *= A[i];
        kb *= B[i];
    }
    return st;
}

__device__ void offsets(const Strides& s, size_t o, long long& ia, long long& ib) {
    ia = ib = 0;
    for (int d = s.r - 1; d >= 0; d--) {
        int i = o % s.shape[d];
        o /= s.shape[d];
        ia += i * s.sa[d];
        ib += i * s.sb[d];
    }
}

__global__ void k_binary(const float* a, const float* b, float* y, size_t n, Strides s, int op) {
    size_t o = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (o >= n) return;
    long long ia, ib;
    offsets(s, o, ia, ib);
    y[o] = op == 0 ? a[ia] + b[ib] : a[ia] * b[ib];
}

__global__ void k_unary(const float* a, float* y, size_t n, int u, float s) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = a[i];
    switch (u) {
        case 0: v = v / (1 + expf(-v)); break;                         // silu
        case 1: v = 1 / (1 + expf(-v)); break;                         // sigmoid
        case 2: v = v > 20 ? v : log1pf(expf(v)); break;               // softplus
        case 3: v = v > 0 ? v : 0; break;
        case 4: v = expf(v); break;
        case 5: v = sqrtf(v); break;
        case 6: v = rsqrtf(v); break;
        case 7: v = fabsf(v); break;
        case 8: v = (float)((v > 0) - (v < 0)); break;
        case 100: v *= s; break;                                       // scale
        case 101: v = fmaxf(v, s); break;                              // clamp_min
    }
    y[i] = v;
}

__global__ void k_reduce(const float* a, float* y, size_t outer, size_t n, size_t inner, int avg) {
    size_t o = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (o >= outer * inner) return;
    size_t b = o / inner, j = o % inner;
    float s = 0;
    for (size_t i = 0; i < n; i++) s += a[(b * n + i) * inner + j];
    y[o] = avg ? s / n : s;
}

__global__ void k_rows(const float* a, float* y, size_t rows, size_t n, int kind, float eps) {   // softmax, rms
    size_t r = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* p = a + r * n;
    float* q = y + r * n;
    if (kind == 0) {
        float mx = p[0], s = 0;
        for (size_t i = 1; i < n; i++) mx = fmaxf(mx, p[i]);
        for (size_t i = 0; i < n; i++) s += (q[i] = expf(p[i] - mx));
        for (size_t i = 0; i < n; i++) q[i] /= s;
    } else {
        float ss = 0;
        for (size_t i = 0; i < n; i++) ss += p[i] * p[i];
        float k = rsqrtf(ss / n + eps);
        for (size_t i = 0; i < n; i++) q[i] = p[i] * k;
    }
}

__global__ void k_matmul(const float* a, const float* b, float* y, int M, int K, int N, size_t nb, Strides s) {
    size_t o = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (o >= nb * M * N) return;
    size_t z = o / ((size_t)M * N);
    int m = (o / N) % M, n = o % N;
    long long ia, ib;
    offsets(s, z, ia, ib);
    const float *pa = a + ia * M * K + (size_t)m * K, *pb = b + ib * K * N + n;
    float acc = 0;
    for (int k = 0; k < K; k++) acc += pa[k] * pb[(size_t)k * N];
    y[o] = acc;
}

__global__ void k_permute(const float* a, float* y, size_t n, int r, Strides s) {   // shape: out, sa: input strides
    size_t o = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (o >= n) return;
    size_t rem = o;
    long long src = 0;
    for (int d = r - 1; d >= 0; d--) {
        src += (rem % s.shape[d]) * s.sa[d];
        rem /= s.shape[d];
    }
    y[o] = a[src];
}

__global__ void k_slice(const float* a, float* y, size_t outer, size_t n, size_t inner, size_t start, size_t len, size_t yn,
                        size_t yoff) {
    size_t o = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (o >= outer * len * inner) return;
    size_t b = o / (len * inner), i = (o / inner) % len, j = o % inner;
    y[(b * yn + yoff + i) * inner + j] = a[(b * n + start + i) * inner + j];
}

__global__ void k_gather(const float* t, const float* ids, float* y, size_t n, size_t D) {
    size_t o = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (o >= n * D) return;
    y[o] = t[(size_t)ids[o / D] * D + o % D];
}

// gated delta rule: one thread per (value head, value column); a column of the state depends only on itself
__global__ void k_gdr(const float* q, const float* k, const float* v, const float* g, const float* beta, float* S,
                      float* o, float* states, int m, int H, int K, int HV, int V, float scale, int per_token) {
    int h = blockIdx.x, j = threadIdx.x;
    if (j >= V) return;
    int kh = h / (HV / H);
    float* Sh = S + (size_t)h * K * V;
    for (int t = 0; t < m; t++) {
        const float *qt = q + ((size_t)t * H + kh) * K, *kt = k + ((size_t)t * H + kh) * K;
        float decay = expf(g[t * HV + h]), kv = 0, qs = 0;
        for (int i = 0; i < K; i++) kv += (Sh[i * V + j] *= decay) * kt[i];
        float delta = (v[((size_t)t * HV + h) * V + j] - kv) * beta[t * HV + h];
        for (int i = 0; i < K; i++) qs += (Sh[i * V + j] += kt[i] * delta) * qt[i];
        o[((size_t)t * HV + h) * V + j] = qs * scale;
        if (per_token || t == m - 1) {
            float* dst = states + ((size_t)(per_token ? t : 0) * HV + h) * K * V;
            for (int i = 0; i < K; i++) dst[i * V + j] = Sh[i * V + j];
        }
    }
}

unsigned blocks(size_t n) { return (unsigned)((n + 255) / 256); }

struct CUDA : Ops {
    Tensor put(const std::vector<float>& a, Shape s) override {
        Tensor t = make(std::move(s));
        CK(cudaMemcpy(P(t), a.data(), a.size() * 4, cudaMemcpyHostToDevice));
        return t;
    }
    std::shared_ptr<Dev> pack(const float* w, int out, int in, const std::string& fmt) {
        bfp_mat m;
        if (bfp_alloc(&m, fmt == "bfp8" ? BFP8 : BFP4, out, in)) throw std::runtime_error("bfp_alloc");
        bfp_pack(&m, w);
        auto d = std::make_shared<Dev>();
        size_t eb = (size_t)out / 16 * in, mb = bfp_mant_bytes(m.fmt, out, in);
        CK(cudaMalloc(&d->exp, eb));
        CK(cudaMalloc(&d->mant, mb));
        CK(cudaMemcpy(d->exp, m.exp, eb, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d->mant, m.mant, mb, cudaMemcpyHostToDevice));
        d->out = out, d->in = in, d->mb = m.fmt;
        bfp_free(&m);
        return d;
    }
    Tensor put_weight(const float* w, int out, int in, const std::string& fmt) override {
        if ((fmt == "bfp8" || fmt == "bfp4") && !(out % 16) && !(in % 8)) return Tensor{{in, out}, pack(w, out, in, fmt)};
        std::vector<float> t((size_t)in * out);                     // W^T [in, out] in float32
        for (int o = 0; o < out; o++)
            for (int i = 0; i < in; i++) t[(size_t)i * out + o] = w[(size_t)o * in + i];
        return put(t, {in, out});
    }
    std::vector<float> get(const Tensor& x) override {
        std::vector<float> h(numel(x.shape));
        CK(cudaMemcpy(h.data(), P(x), h.size() * 4, cudaMemcpyDeviceToHost));
        return h;
    }
    Tensor zeros(Shape s) override {
        Tensor t = make(std::move(s));
        CK(cudaMemset(P(t), 0, numel(t.shape) * 4));
        return t;
    }
    Tensor put_ids(const std::vector<int>& ids) override {
        return put(std::vector<float>(ids.begin(), ids.end()), {(int)ids.size()});
    }

    Tensor bfp_linear(const Tensor& x, const Dev& w) {
        int m = (int)(numel(x.shape) / x.dim(-1)), n = w.in;
        Shape s = x.shape;
        s.back() = w.out;
        Tensor y = make(s);
        for (int t0 = 0; t0 < m; t0 += 4) {                // tgpu_gemv: up to 4 tokens per launch
            tgpu_job j{w.exp, w.mant, P(x) + (size_t)t0 * n, P(y) + (size_t)t0 * w.out, w.out, n, n, w.out};
            int e = tgpu_gemv(w.mb, &j, 1, std::min(4, m - t0), nullptr);
            if (e) throw std::runtime_error("tgpu_gemv: " + std::string(cudaGetErrorString((cudaError_t)e)));
        }
        return y;
    }
    Tensor linear(const Tensor& x, const Tensor& w, Act a) override {
        Tensor y = dv(w).exp ? bfp_linear(x, dv(w)) : matmul(x, w);
        return a == Act::none ? y : unary(y, a == Act::silu ? Unary::silu : Unary::sigmoid);
    }
    Tensor matmul(const Tensor& a, const Tensor& b) override {
        int M = a.dim(-2), K = a.dim(-1), N = b.dim(-1);
        Shape bs;
        Strides st = bcast(Shape(a.shape.begin(), a.shape.end() - 2), Shape(b.shape.begin(), b.shape.end() - 2), bs);
        if (st.r == 0) { st.r = 1; st.shape[0] = 1; st.sa[0] = st.sb[0] = 0; }
        size_t nb = numel(bs);
        Shape s = bs;
        s.push_back(M);
        s.push_back(N);
        Tensor y = make(s);
        size_t n = nb * M * N;
        k_matmul<<<blocks(n), 256>>>(P(a), P(b), P(y), M, K, N, nb, st);
        CK(cudaGetLastError());
        return y;
    }
    Tensor binary(const Tensor& a, const Tensor& b, int op) {
        Shape s;
        Strides st = bcast(a.shape, b.shape, s);
        Tensor y = make(s);
        size_t n = numel(s);
        k_binary<<<blocks(n), 256>>>(P(a), P(b), P(y), n, st, op);
        CK(cudaGetLastError());
        return y;
    }
    Tensor add(const Tensor& a, const Tensor& b) override { return binary(a, b, 0); }
    Tensor mul(const Tensor& a, const Tensor& b) override { return binary(a, b, 1); }
    Tensor elementwise(const Tensor& a, int u, float s) {
        Tensor y = make(a.shape);
        size_t n = numel(a.shape);
        k_unary<<<blocks(n), 256>>>(P(a), P(y), n, u, s);
        CK(cudaGetLastError());
        return y;
    }
    Tensor scale(const Tensor& a, float s) override { return elementwise(a, 100, s); }
    Tensor unary(const Tensor& a, Unary u) override { return elementwise(a, (int)u, 0); }
    Tensor clamp_min(const Tensor& a, float m) override { return elementwise(a, 101, m); }
    Tensor reduce(const Tensor& a, int dim, int avg) {
        int d = dim < 0 ? dim + a.rank() : dim;
        size_t outer = 1, inner = 1, n = a.shape[d];
        for (int i = 0; i < d; i++) outer *= a.shape[i];
        for (int i = d + 1; i < a.rank(); i++) inner *= a.shape[i];
        Shape s = a.shape;
        s[d] = 1;
        Tensor y = make(s);
        k_reduce<<<blocks(outer * inner), 256>>>(P(a), P(y), outer, n, inner, avg);
        CK(cudaGetLastError());
        return y;
    }
    Tensor sum(const Tensor& a, int dim) override { return reduce(a, dim, 0); }
    Tensor mean(const Tensor& a, int dim) override { return reduce(a, dim, 1); }
    Tensor rows(const Tensor& a, int kind, float eps) {
        Tensor y = make(a.shape);
        size_t n = a.dim(-1), r = numel(a.shape) / n;
        k_rows<<<blocks(r), 256>>>(P(a), P(y), r, n, kind, eps);
        CK(cudaGetLastError());
        return y;
    }
    Tensor softmax(const Tensor& a) override { return rows(a, 0, 0); }
    Tensor rms_norm(const Tensor& x, const Tensor& w, float eps) override { return mul(rows(x, 1, eps), w); }

    Tensor reshape(const Tensor& a, Shape s) override { return Tensor{std::move(s), a.impl}; }
    Tensor slice(const Tensor& a, int dim, int start, int end) override {
        int d = dim < 0 ? dim + a.rank() : dim;
        size_t outer = 1, inner = 1, n = a.shape[d], len = end - start;
        for (int i = 0; i < d; i++) outer *= a.shape[i];
        for (int i = d + 1; i < a.rank(); i++) inner *= a.shape[i];
        Shape s = a.shape;
        s[d] = (int)len;
        Tensor y = make(s);
        k_slice<<<blocks(outer * len * inner), 256>>>(P(a), P(y), outer, n, inner, start, len, len, 0);
        CK(cudaGetLastError());
        return y;
    }
    Tensor concat(const std::vector<Tensor>& xs, int dim) override {
        Shape s = xs[0].shape;
        int d = dim < 0 ? dim + (int)s.size() : dim;
        s[d] = 0;
        for (auto& x : xs) s[d] += x.shape[d];
        Tensor y = make(s);
        size_t outer = 1, inner = 1, off = 0;
        for (int i = 0; i < d; i++) outer *= s[i];
        for (size_t i = d + 1; i < s.size(); i++) inner *= s[i];
        for (auto& x : xs) {
            size_t len = x.shape[d];
            if (len) k_slice<<<blocks(outer * len * inner), 256>>>(P(x), P(y), outer, len, inner, 0, len, s[d], off);
            off += len;
        }
        CK(cudaGetLastError());
        return y;
    }
    Tensor permute(const Tensor& a, const std::vector<int>& dims) override {
        int r = a.rank();
        Strides st{};
        st.r = r;
        long long in[R], k = 1;
        for (int i = r - 1; i >= 0; i--) { in[i] = k; k *= a.shape[i]; }
        Shape s(r);
        for (int i = 0; i < r; i++) { s[i] = st.shape[i] = a.shape[dims[i]]; st.sa[i] = in[dims[i]]; }
        Tensor y = make(s);
        size_t n = numel(s);
        k_permute<<<blocks(n), 256>>>(P(a), P(y), n, r, st);
        CK(cudaGetLastError());
        return y;
    }
    Tensor transpose(const Tensor& a) override {
        std::vector<int> d(a.rank());
        std::iota(d.begin(), d.end(), 0);
        std::swap(d[a.rank() - 1], d[a.rank() - 2]);
        return permute(a, d);
    }

    Tensor gather_rows(const Tensor& table, const Tensor& ids) override {
        int n = ids.dim(0), D = table.dim(-1);
        Tensor y = make({n, D});
        k_gather<<<blocks((size_t)n * D), 256>>>(P(table), P(ids), P(y), n, D);
        CK(cudaGetLastError());
        return y;
    }
    void write_rows(Tensor& table, int row, const Tensor& rows) override {
        CK(cudaMemcpy(P(table) + (size_t)row * table.dim(-1), P(rows), numel(rows.shape) * 4, cudaMemcpyDeviceToDevice));
    }

    std::pair<Tensor, Tensor> gdr(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g, const Tensor& beta,
                                  const Tensor& state, float scale, bool per_token) override {
        int m = q.dim(0), H = q.dim(1), K = q.dim(2), HV = v.dim(1), V = v.dim(2);
        Tensor S = make(state.shape), o = make({m, HV, V}), states = make({per_token ? m : 1, HV, K, V});
        CK(cudaMemcpy(P(S), P(state), numel(state.shape) * 4, cudaMemcpyDeviceToDevice));
        k_gdr<<<HV, V>>>(P(q), P(k), P(v), P(g), P(beta), P(S), P(o), P(states), m, H, K, HV, V, scale, per_token);
        CK(cudaGetLastError());
        return {o, states};
    }

    Tensor bank(int n, int k, int N, const std::string&) override {
        auto b = std::make_shared<Bank>(Bank{n, k, N, std::vector<std::shared_ptr<Dev>>(n)});
        return Tensor{{n, k, N}, b};
    }
    void bank_write(Tensor& bank, int slot, const float* w, int N, int k) override {
        auto& b = *std::static_pointer_cast<Bank>(bank.impl);
        b.w[slot] = pack(w, N, k, "bfp4");             // ponytail: experts are BFP4 on the GPU
    }
    bool takes_bfp() const override { return true; }
    void bank_write_bfp(Tensor& bank, int slot, const uint8_t* exp, const uint8_t* mant, int out, int in, int mbits) override {
        auto& b = *std::static_pointer_cast<Bank>(bank.impl);
        auto d = std::make_shared<Dev>();
        size_t eb = (size_t)out / 16 * in, mb = bfp_mant_bytes((enum bfp_fmt)mbits, out, in);
        CK(cudaMalloc(&d->exp, eb));
        CK(cudaMalloc(&d->mant, mb));
        CK(cudaMemcpy(d->exp, exp, eb, cudaMemcpyHostToDevice));   // ponytail: pinned staging + a copy stream next
        CK(cudaMemcpy(d->mant, mant, mb, cudaMemcpyHostToDevice));
        d->out = out, d->in = in, d->mb = mbits;
        b.w[slot] = d;
    }
    Tensor gather_matmul(const Tensor& x, const Tensor& bank, const Tensor& ids, Act a) override {
        auto& b = *std::static_pointer_cast<Bank>(bank.impl);
        std::vector<float> I = get(ids);
        int g = x.dim(0), m = x.dim(1), n = (int)I.size();
        Tensor y = make({n, m, b.N});
        for (int i = 0; i < n; i++) {   // ponytail: tgpu_moe takes a layer's experts in one launch; a gemv per slot here
            const Dev& w = *b.w[(size_t)I[i]];
            for (int t0 = 0; t0 < m; t0 += 4) {
                tgpu_job j{w.exp, w.mant, P(x) + ((size_t)(g > 1 ? i : 0) * m + t0) * b.k,
                           P(y) + ((size_t)i * m + t0) * b.N, w.out, w.in, w.in, w.out};
                int e = tgpu_gemv(w.mb, &j, 1, std::min(4, m - t0), nullptr);
                if (e) throw std::runtime_error("tgpu_gemv: " + std::string(cudaGetErrorString((cudaError_t)e)));
            }
        }
        return a == Act::none ? y : unary(y, a == Act::silu ? Unary::silu : Unary::sigmoid);
    }
};

}  // namespace

std::unique_ptr<Ops> cuda_ops() { return std::make_unique<CUDA>(); }

}  // namespace tuff
