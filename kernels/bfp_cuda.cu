// GPU kernels for the host BFP layout (bfp.h). See bfp_cuda.h.
//
// One 128-thread block per (matrix, 16-row panel). Thread t takes row r = t % 16 of column groups g = t / 16, +8,
// +16, ...: for each step the block reads 8 groups x 16 dwords = 512 contiguous bytes, fully coalesced. Each field
// decodes to (n - 2^(b-1)) * 2^(exp - 127 - (mbits - 1)) with shifts, an int->float and the exponent's bits; x is read
// through L1 (the 16 threads of a group read the same x[j]). The 8 partial sums per row meet in shared memory.
#include "bfp_cuda.h"
#include <cuda_runtime.h>

namespace {

constexpr int MAXJOBS = 64;   // tgpu_job is 48 bytes: 64 of them stay under the 4 KB kernel parameter limit
struct Jobs { tgpu_job j[MAXJOBS]; };

__device__ __forceinline__ float scale_of(unsigned e, int mb) {
    return e > (unsigned)(mb - 1) ? __int_as_float((int)(e - (mb - 1)) << 23) : 0.f;   // exp 0 -> 0, as on the CPU
}

template <int B, int NT>
__global__ void __launch_bounds__(128) k_gemv(const Jobs jobs, int mb) {
    constexpr int C = 32 / B;                       // columns per dword
    constexpr unsigned M = (1u << B) - 1, OFF = 1u << (B - 1);
    const tgpu_job &jb = jobs.j[blockIdx.y];
    const int p = blockIdx.x;
    if (p >= jb.out / 16) return;                   // whole block leaves together: no __syncthreads below is skipped
    const int tid = threadIdx.x, r = tid & 15, gl = tid >> 4, in = jb.in;
    const uint8_t *e = jb.exp + (size_t)p * in;
    const uint32_t *mt = reinterpret_cast<const uint32_t *>(jb.mant + (size_t)p * in * 2 * B);
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; t++) acc[t] = 0.f;
    for (int g = gl; g < in / C; g += 8) {
        const uint32_t d = __ldg(mt + g * 16 + r);  // dword order [p][j/c][r/8][r%8]: index r within the group
#pragma unroll
        for (int k = 0; k < C; k++) {
            const int j = g * C + k;
            const float w = scale_of(__ldg(e + j), mb) * (float)((int)((d >> (B * k)) & M) - (int)OFF);
#pragma unroll
            for (int t = 0; t < NT; t++) acc[t] = fmaf(__ldg(jb.x + (size_t)t * jb.ldx + j), w, acc[t]);
        }
    }
    __shared__ float red[8][16][NT];
#pragma unroll
    for (int t = 0; t < NT; t++) red[gl][r][t] = acc[t];
    __syncthreads();
    if (tid < 16 * NT) {
        const int rr = tid & 15, t = tid >> 4;
        float s = 0.f;
#pragma unroll
        for (int q = 0; q < 8; q++) s += red[q][rr][t];
        jb.y[(size_t)t * jb.ldy + 16 * p + rr] = s;
    }
}

template <int B>
cudaError_t launch(const Jobs &js, int nj, int max_out, int nt, int mb, cudaStream_t s) {
    dim3 grid((max_out + 15) / 16, nj);
    switch (nt) {
    case 1: k_gemv<B, 1><<<grid, 128, 0, s>>>(js, mb); break;
    case 2: k_gemv<B, 2><<<grid, 128, 0, s>>>(js, mb); break;
    case 3: k_gemv<B, 3><<<grid, 128, 0, s>>>(js, mb); break;
    default: k_gemv<B, 4><<<grid, 128, 0, s>>>(js, mb); break;
    }
    return cudaGetLastError();
}

// h[e][t][0:I] = silu(h[e][t][0:I]) * h[e][t][I:2I]
__global__ void k_swiglu(float *h, int I, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * I) return;
    float *q = h + (size_t)(i / I) * 2 * I + i % I, u = q[I], v = *q;
    *q = v / (1.f + __expf(-v)) * u;
}

// y[t0+t][o] = sum_e g[e][t0+t] * part[e][t][o]
__global__ void k_combine(const float *part, const float *g, int k, int n, int t0, int gn, int d, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= gn * d) return;
    int t = i / d, o = i % d;
    float s = 0.f;
    for (int e = 0; e < k; e++) s = fmaf(g[(size_t)e * n + t0 + t], part[((size_t)e * gn + t) * d + o], s);
    y[(size_t)(t0 + t) * d + o] = s;
}

}  // namespace

extern "C" int tgpu_gemv(int fmt, const tgpu_job *jobs, int njobs, int nt, void *stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    for (int j0 = 0; j0 < njobs; j0 += MAXJOBS) {
        Jobs js;
        int nj = njobs - j0 < MAXJOBS ? njobs - j0 : MAXJOBS, max_out = 0;
        for (int i = 0; i < nj; i++) {
            js.j[i] = jobs[j0 + i];
            if (js.j[i].out > max_out) max_out = js.j[i].out;
        }
        cudaError_t err = fmt == 7 ? launch<8>(js, nj, max_out, nt, fmt, s) : launch<4>(js, nj, max_out, nt, fmt, s);
        if (err != cudaSuccess) return err;
    }
    return 0;
}

extern "C" int tgpu_moe(int fmt, const tgpu_mat *gate, const tgpu_mat *up, const tgpu_mat *down, int I, int d,
                        const float *g, int k, int n, const float *x, float *y, float *tmp, void *stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    tgpu_job jobs[2 * MAXJOBS];
    for (int t0 = 0; t0 < n; t0 += 4) {
        const int gn = n - t0 < 4 ? n - t0 : 4;
        float *h = tmp, *part = tmp + (size_t)k * gn * 2 * I;           // h[e][t][2I], part[e][t][d]
        for (int e0 = 0; e0 < k; e0 += MAXJOBS) {                       // gate and up: 2 jobs per expert
            int ke = k - e0 < MAXJOBS ? k - e0 : MAXJOBS, nj = 0;
            for (int e = e0; e < e0 + ke; e++) {
                float *he = h + (size_t)e * gn * 2 * I;
                jobs[nj++] = {gate[e].exp, gate[e].mant, x + (size_t)t0 * d, he, I, d, d, 2 * I};
                jobs[nj++] = {up[e].exp, up[e].mant, x + (size_t)t0 * d, he + I, I, d, d, 2 * I};
            }
            if (int err = tgpu_gemv(fmt, jobs, nj, gn, stream)) return err;
        }
        k_swiglu<<<(k * gn * I + 255) / 256, 256, 0, s>>>(h, I, k * gn);
        for (int e0 = 0; e0 < k; e0 += MAXJOBS) {
            int ke = k - e0 < MAXJOBS ? k - e0 : MAXJOBS;
            for (int i = 0; i < ke; i++) {
                int e = e0 + i;
                jobs[i] = {down[e].exp, down[e].mant, h + (size_t)e * gn * 2 * I, part + (size_t)e * gn * d, d, I, 2 * I, d};
            }
            if (int err = tgpu_gemv(fmt, jobs, ke, gn, stream)) return err;
        }
        k_combine<<<(gn * d + 255) / 256, 256, 0, s>>>(part, g, k, n, t0, gn, d, y);
        if (cudaError_t err = cudaGetLastError()) return err;
    }
    return 0;
}
