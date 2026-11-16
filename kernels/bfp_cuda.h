/* NVIDIA GPU (RTX 4060 Ti) kernels for the host BFP layout of bfp.h, unchanged: an expert copied from host RAM (or
 * from a tstore mapping) to the GPU is used as is, so the host keeps one copy of every expert for CPU and GPU.
 *
 * All pointers below are device pointers. Calls are asynchronous on `stream` (a cudaStream_t, 0 = default stream).
 * Not run in CI (no GPU there; CI only compiles): `make test_cuda && ./test_cuda` on the real box checks them
 * against the CPU reference and prints the bandwidth they reach. */
#ifndef TUFF_BFP_CUDA_H
#define TUFF_BFP_CUDA_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* one matrix of a launch: y[t][o] = sum_j W[o][j] x[t][j] for t < nt; x at x + t*ldx, y at y + t*ldy */
typedef struct {
    const uint8_t *exp, *mant;
    const float *x;
    float *y;
    int out, in, ldx, ldy;
} tgpu_job;

/* fmt: 7 (bfp8) or 3 (bfp4), as enum bfp_fmt; nt: tokens, 1..4. Returns 0, or a cudaError_t. */
int tgpu_gemv(int fmt, const tgpu_job *jobs, int njobs, int nt, void *stream);

/* One MoE layer for n tokens (any n; groups of 4) through k experts, like moe_forward_n:
 * gate[i], up[i], down[i]: expert i's matrices (exp, mant; out/in as in tstore: gate/up [640][2560], down [2560][640]),
 * g: [k][n] gate weights (0 = token not routed there), x, y: [n][d]. tmp: k * min(n,4) * (2*I + d) floats. */
typedef struct { const uint8_t *exp, *mant; } tgpu_mat;
int tgpu_moe(int fmt, const tgpu_mat *gate, const tgpu_mat *up, const tgpu_mat *down, int I, int d,
             const float *g, int k, int n, const float *x, float *y, float *tmp, void *stream);

#ifdef __cplusplus
}
#endif
#endif
