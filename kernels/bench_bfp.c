/* Cold-expert throughput for Qwen3.8-Flash-Next: one MoE layer computed on the CPU from a DRAM pool.
 *   ./bench_bfp [cold_experts_per_layer=10] [bfp4|bfp8] [pool_MB=2048] [iters=400] [tokens=1]
 * tokens > 1: that many tokens verified together (MTP), each expert read once per 4 tokens (moe_forward_n).
 * QFN experts are SwiGLU 2560 -> 640 -> 2560, top-10 of 512 per layer, 48 layers. Experts are picked at random
 * from a pool larger than the caches, so every layer streams its experts from DRAM, as cold experts will in
 * the engine. 10 cold experts per layer = nothing on the card (worst case); 6 = uniform routing with ~40% on the card.
 */
#include "bfp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

enum { D = 2560, I = 640, LAYERS = 48 };   /* Qwen3.8-Flash-Next text_config */

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

static void copy_mat(bfp_mat *dst, const bfp_mat *src) {
    memcpy(dst->exp, src->exp, (size_t)(src->out / 16) * src->in);
    memcpy(dst->mant, src->mant, bfp_mant_bytes(src->fmt, src->out, src->in));
}

int main(int argc, char **argv) {
    int k = argc > 1 ? atoi(argv[1]) : 10;
    enum bfp_fmt fmt = (argc > 2 && !strcmp(argv[2], "bfp8")) ? BFP8 : BFP4;
    double pool_mb = argc > 3 ? atof(argv[3]) : 2048;
    int iters = argc > 4 ? atoi(argv[4]) : 400, n = argc > 5 ? atoi(argv[5]) : 1;
    if (k < 1 || k > 64 || iters < 1 || n < 1 || n > 8) {
        fprintf(stderr, "usage: bench_bfp [1..64] [bfp4|bfp8] [pool_MB] [iters] [tokens 1..8]\n");
        return 1;
    }

    size_t eb = 2 * (bfp_mant_bytes(fmt, I, D) + (size_t)(I / 16) * D) + bfp_mant_bytes(fmt, D, I) + (size_t)(D / 16) * I;
    int ne = (int)(pool_mb * 1e6 / eb);
    if (ne < 4 * k) ne = 4 * k;
    bfp_mat *w1 = malloc(sizeof *w1 * ne), *w3 = malloc(sizeof *w3 * ne), *w2 = malloc(sizeof *w2 * ne);
    float *rnd = malloc(sizeof(float) * (size_t)I * D);
    for (size_t i = 0; i < (size_t)I * D; i++) rnd[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int e = 0; e < ne; e++) {   /* pack once, copy blocks: packing every expert would dominate setup */
        if (bfp_alloc(&w1[e], fmt, I, D) || bfp_alloc(&w3[e], fmt, I, D) || bfp_alloc(&w2[e], fmt, D, I)) {
            fprintf(stderr, "out of memory at expert %d\n", e);
            return 1;
        }
        if (e == 0) { bfp_pack(&w1[0], rnd); bfp_pack(&w3[0], rnd); bfp_pack(&w2[0], rnd); continue; }
        copy_mat(&w1[e], &w1[0]); copy_mat(&w3[e], &w3[0]); copy_mat(&w2[e], &w2[0]);
    }
    float x[8 * D], y[8 * D], gate[64 * 8], *tmp = malloc(sizeof(float) * k * 4 * (2 * I + D));
    for (int i = 0; i < 8 * D; i++) x[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < 64 * 8; i++) gate[i] = 1.f / k;
    const bfp_mat *a[64], *b[64], *c[64];

    double t = 0;
    for (int it = -20; it < iters; it++) {     /* 20 warm-up layers */
        for (int i = 0; i < k; i++) { int e = rand() % ne; a[i] = &w1[e]; b[i] = &w3[e]; c[i] = &w2[e]; }
        double t0 = now();
        if (n == 1) moe_forward(a, b, c, gate, k, x, y, tmp);
        else moe_forward_n(a, b, c, gate, k, n, x, y, tmp);
        if (it >= 0) t += now() - t0;
        x[(it + 20) % D] += y[0] * 1e-9f;      /* keep the result live */
    }
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    double per_layer = t / iters;
    printf("qfn %s cold=%d/layer tokens=%d threads=%d pool=%d experts (%.0f MB), expert %.2f MB | layer %.1f us, %.1f GB/s | "
           "CPU expert time per token %.1f ms (%d layers)\n",
           fmt == BFP8 ? "bfp8" : "bfp4", k, n, threads, ne, ne * eb / 1e6, eb / 1e6,
           per_layer * 1e6, k * eb / per_layer / 1e9, per_layer * LAYERS * 1e3, LAYERS);
    return 0;
}
