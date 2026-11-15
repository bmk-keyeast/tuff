/* Cold-expert throughput: MoE layers computed on the CPU from a DRAM-resident expert pool.
 *   ./bench_bfp [preset] [bfp4|bfp8] [pool_MB] [iters]
 * Experts are picked at random from a pool larger than the caches, so every layer streams its
 * k experts from DRAM, as cold experts will in the engine. Reports achieved GB/s and the decode
 * rate if *all* experts lived on the CPU (the engine's worst case).
 */
#include "bfp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static const struct { const char *name; int d, I, k, L; } P[] = {
    {"qwen3-30b-a3b", 2048, 768, 8, 48},
    {"qwen3-next-80b", 2048, 512, 10, 48},
    {"gpt-oss-120b", 2880, 2880, 4, 36},
    {"glm-4.5-air", 4096, 1408, 8, 45},
};

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

int main(int argc, char **argv) {
    int pi = 0;
    for (int i = 0; i < (int)(sizeof P / sizeof *P); i++) if (argc > 1 && !strcmp(argv[1], P[i].name)) pi = i;
    enum bfp_fmt fmt = (argc > 2 && !strcmp(argv[2], "bfp8")) ? BFP8 : BFP4;
    double pool_mb = argc > 3 ? atof(argv[3]) : 2048;
    int iters = argc > 4 ? atoi(argv[4]) : 400;
    const int d = P[pi].d, I = P[pi].I, k = P[pi].k;

    size_t eb = 2 * (bfp_mant_bytes(fmt, I, d) + (size_t)(I / 16) * d) + bfp_mant_bytes(fmt, d, I) + (size_t)(d / 16) * I;
    int ne = (int)(pool_mb * 1e6 / eb);
    if (ne < 4 * k) ne = 4 * k;
    bfp_mat *w1 = malloc(sizeof *w1 * ne), *w3 = malloc(sizeof *w3 * ne), *w2 = malloc(sizeof *w2 * ne);
    float *rnd = malloc(sizeof(float) * (size_t)I * d);
    for (size_t i = 0; i < (size_t)I * d; i++) rnd[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int e = 0; e < ne; e++) {   /* pack once, copy blocks: packing every expert would dominate setup */
        if (bfp_alloc(&w1[e], fmt, I, d) || bfp_alloc(&w3[e], fmt, I, d) || bfp_alloc(&w2[e], fmt, d, I)) {
            fprintf(stderr, "out of memory at expert %d\n", e); return 1;
        }
        if (e == 0) { bfp_pack(&w1[0], rnd); bfp_pack(&w3[0], rnd); bfp_pack(&w2[0], rnd); continue; }
        memcpy(w1[e].exp, w1[0].exp, (size_t)(I / 16) * d); memcpy(w1[e].mant, w1[0].mant, bfp_mant_bytes(fmt, I, d));
        memcpy(w3[e].exp, w3[0].exp, (size_t)(I / 16) * d); memcpy(w3[e].mant, w3[0].mant, bfp_mant_bytes(fmt, I, d));
        memcpy(w2[e].exp, w2[0].exp, (size_t)(d / 16) * I); memcpy(w2[e].mant, w2[0].mant, bfp_mant_bytes(fmt, d, I));
    }
    float *x = malloc(sizeof(float) * d), *y = malloc(sizeof(float) * d);
    float *tmp = malloc(sizeof(float) * k * (2 * I + d)), gate[16];
    for (int i = 0; i < d; i++) x[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < k; i++) gate[i] = 1.f / k;
    const bfp_mat *a[16], *b[16], *c[16];

    double t = 0;
    for (int it = -20; it < iters; it++) {     /* 20 warm-up layers */
        for (int i = 0; i < k; i++) { int e = rand() % ne; a[i] = &w1[e]; b[i] = &w3[e]; c[i] = &w2[e]; }
        double t0 = now();
        moe_forward(a, b, c, gate, k, x, y, tmp);
        if (it >= 0) t += now() - t0;
        x[it & (d - 1) % d] += y[0] * 1e-9f;   /* keep the result live */
    }
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    double per_layer = t / iters, gbs = k * eb / per_layer / 1e9;
    printf("%s %s threads=%d pool=%d experts (%.0f MB) expert=%.2f MB | MoE layer %.1f us, %.1f GB/s | "
           "all-CPU experts: %.1f t/s (%d layers)\n",
           P[pi].name, fmt == BFP8 ? "bfp8" : "bfp4", threads, ne, ne * eb / 1e6, eb / 1e6,
           per_layer * 1e6, gbs, 1 / (per_layer * P[pi].L), P[pi].L);
    return 0;
}
