// GPU kernels == CPU reference, then the bandwidth they reach. Needs an NVIDIA GPU: make test_cuda && ./test_cuda
#include "bfp_cuda.h"
extern "C" {
#include "bfp.h"
}
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static float randn() {
    float u = (rand() + 1.f) / (RAND_MAX + 2.f), v = (rand() + 1.f) / (RAND_MAX + 2.f);
    return sqrtf(-2 * logf(u)) * cosf(6.2831853f * v);
}

struct Dev { uint8_t *exp, *mant; };
static Dev upload(const bfp_mat &m) {
    Dev d;
    size_t ne = (size_t)(m.out / 16) * m.in, nm = bfp_mant_bytes(m.fmt, m.out, m.in);
    CK(cudaMalloc(&d.exp, ne)); CK(cudaMalloc(&d.mant, nm));
    CK(cudaMemcpy(d.exp, m.exp, ne, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.mant, m.mant, nm, cudaMemcpyHostToDevice));
    return d;
}
static bfp_mat make(enum bfp_fmt f, int out, int in, float s) {
    bfp_mat m;
    std::vector<float> w((size_t)out * in);
    for (auto &v : w) v = s * randn();
    assert(bfp_alloc(&m, f, out, in) == 0);
    bfp_pack(&m, w.data());
    return m;
}

static void gemv(enum bfp_fmt f, int out, int in, int nt) {
    bfp_mat m = make(f, out, in, 0.02f);
    Dev d = upload(m);
    std::vector<float> x((size_t)nt * in), y((size_t)nt * out), ref(out);
    for (auto &v : x) v = randn();
    float *dx, *dy;
    CK(cudaMalloc(&dx, x.size() * 4)); CK(cudaMalloc(&dy, y.size() * 4));
    CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    tgpu_job j = {d.exp, d.mant, dx, dy, out, in, in, out};
    CK((cudaError_t)tgpu_gemv(f, &j, 1, nt, 0));
    CK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
    std::vector<float> wq((size_t)out * in);
    bfp_unpack(&m, wq.data());
    for (int t = 0; t < nt; t++) {
        bfp_gemv_ref(&m, x.data() + (size_t)t * in, ref.data());
        for (int o = 0; o < out; o++) {
            double l1 = 0;
            for (int i = 0; i < in; i++) l1 += fabs((double)wq[(size_t)o * in + i] * x[(size_t)t * in + i]);
            assert(fabs(y[(size_t)t * out + o] - ref[o]) <= 1e-5 * l1 + 1e-30);
        }
    }
    printf("tgpu_gemv %s %dx%d, %d token(s) == bfp_gemv_ref\n", f == BFP8 ? "bfp8" : "bfp4", out, in, nt);
    CK(cudaFree(dx)); CK(cudaFree(dy)); CK(cudaFree(d.exp)); CK(cudaFree(d.mant));
    bfp_free(&m);
}

int main() {
    int dev;
    cudaDeviceProp prop;
    CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
    printf("%s, compute capability %d.%d\n", prop.name, prop.major, prop.minor);
    for (int nt = 1; nt <= 4; nt++) {
        gemv(BFP4, 640, 2560, nt); gemv(BFP4, 2560, 640, nt); gemv(BFP8, 640, 2560, nt);
    }
    gemv(BFP8, 12288, 2560, 1);                    // full-attention q projection
    gemv(BFP4, 2560, 6144, 2);                     // DeltaNet output projection

    // MoE layer, n tokens, k experts: GPU == moe_forward_n
    enum { K = 12, D = 2560, I = 640, N = 3 };
    std::vector<bfp_mat> a(K), c(K), dn(K);
    std::vector<tgpu_mat> ga(K), gc(K), gd(K);
    std::vector<const bfp_mat *> pa(K), pc(K), pd(K);
    for (int e = 0; e < K; e++) {
        a[e] = make(BFP4, I, D, 0.02f); c[e] = make(BFP4, I, D, 0.02f); dn[e] = make(BFP4, D, I, 0.02f);
        Dev x = upload(a[e]), y = upload(c[e]), z = upload(dn[e]);
        ga[e] = {x.exp, x.mant}; gc[e] = {y.exp, y.mant}; gd[e] = {z.exp, z.mant};
        pa[e] = &a[e]; pc[e] = &c[e]; pd[e] = &dn[e];
    }
    std::vector<float> x(N * D), y(N * D), ref(N * D), g(K * N), tmp((size_t)K * 4 * (2 * I + D));
    for (auto &v : x) v = randn();
    for (int i = 0; i < K * N; i++) g[i] = (i % 4) ? 0.1f : 0.f;
    float *dx, *dy, *dg, *dt;
    CK(cudaMalloc(&dx, x.size() * 4)); CK(cudaMalloc(&dy, y.size() * 4)); CK(cudaMalloc(&dg, g.size() * 4));
    CK(cudaMalloc(&dt, tmp.size() * 4));
    CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dg, g.data(), g.size() * 4, cudaMemcpyHostToDevice));
    CK((cudaError_t)tgpu_moe(BFP4, ga.data(), gc.data(), gd.data(), I, D, dg, K, N, dx, dy, dt, 0));
    CK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
    moe_forward_n(pa.data(), pc.data(), pd.data(), g.data(), K, N, x.data(), ref.data(), tmp.data());
    for (int i = 0; i < N * D; i++) assert(fabsf(y[i] - ref[i]) <= 1e-4f * (fabsf(ref[i]) + 5e-2f));
    puts("tgpu_moe == moe_forward_n");

    // bandwidth: the K experts, 1 token, many times
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    const int iters = 200;
    CK(cudaEventRecord(t0));
    for (int it = 0; it < iters; it++) CK((cudaError_t)tgpu_moe(BFP4, ga.data(), gc.data(), gd.data(), I, D, dg, K, 1, dx, dy, dt, 0));
    CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
    float ms;
    CK(cudaEventElapsedTime(&ms, t0, t1));
    double bytes = (double)K * 3 * (bfp_mant_bytes(BFP4, I, D) + (size_t)(I / 16) * D) * iters;
    printf("tgpu_moe, %d BFP4 experts, 1 token: %.1f us per layer, %.0f GB/s (this is GPU_EFF x peak in plan.py)\n",
           K, 1e3 * ms / iters, bytes / (ms * 1e-3) / 1e9);
    puts("all GPU tests passed");
    return 0;
}
