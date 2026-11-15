/* Correctness checks: make test */
#include "bfp.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static float randn(void) {   /* Box-Muller */
    float u = (rand() + 1.f) / (RAND_MAX + 2.f), v = (rand() + 1.f) / (RAND_MAX + 2.f);
    return sqrtf(-2 * logf(u)) * cosf(6.2831853f * v);
}

static double sqnr_db(const float *a, const float *b, size_t n) {
    double s = 0, e = 0;
    for (size_t i = 0; i < n; i++) { s += (double)a[i] * a[i]; e += (double)(a[i] - b[i]) * (a[i] - b[i]); }
    return 10 * log10(s / e);
}

static void check(enum bfp_fmt fmt, int out, int in, double min_sqnr) {
    float *w = malloc(sizeof(float) * out * in), *wq = malloc(sizeof(float) * out * in);
    float *x = malloc(sizeof(float) * in), *y = malloc(sizeof(float) * out), *yr = malloc(sizeof(float) * out);
    for (int i = 0; i < out * in; i++) w[i] = 0.02f * randn();
    for (int i = 0; i < in; i++) x[i] = randn();
    bfp_mat m;
    assert(bfp_alloc(&m, fmt, out, in) == 0);
    bfp_pack(&m, w);
    bfp_unpack(&m, wq);
    double q = sqnr_db(w, wq, (size_t)out * in);

    /* reference GEMV on the dequantized matrix, in double */
    bfp_gemv_ref(&m, x, yr);
    bfp_gemv(&m, x, y);
    for (int o = 0; o < out; o++) {
        /* float32 sums of `in` terms can cancel to ~0, so bound the error by the L1 mass of the terms */
        double d = 0, l1 = 0;
        for (int j = 0; j < in; j++) {
            double t = (double)wq[(size_t)o * in + j] * x[j];
            d += t; l1 += fabs(t);
        }
        assert(fabs(d - yr[o]) <= 1e-5 * l1);           /* scalar == fp64 math on dequantized weights */
        assert(fabs(y[o] - yr[o]) <= 1e-5 * l1);        /* SIMD == scalar up to FMA rounding */
    }

    printf("%s %dx%d: weight SQNR %.1f dB (need > %.0f), SIMD/scalar/fp64 agree\n",
           fmt == BFP8 ? "bfp8" : "bfp4", out, in, q, min_sqnr);
    assert(q > min_sqnr);
    bfp_free(&m); free(w); free(wq); free(x); free(y); free(yr);
}

int main(void) {
    /* exactly representable block: must round-trip bit-exactly (incl. negative and zero) */
    float w[16] = {1, -0.5f, 0.25f, 0.75f, 0, -1, 0.5f, -0.25f};   /* multiples of 1/4: exact in bfp4 too */
    for (int fmt = 0; fmt < 2; fmt++) {
        bfp_mat m; float back[128];
        assert(bfp_alloc(&m, fmt ? BFP4 : BFP8, 16, 8) == 0);
        float v[128] = {0};
        for (int r = 0; r < 16; r++) v[r * 8] = w[r];    /* column 0 holds the block */
        bfp_pack(&m, v); bfp_unpack(&m, back);
        for (int i = 0; i < 128; i++) assert(back[i] == v[i]);
        assert(m.exp[1] == 0);                           /* all-zero column -> exponent 0 */
        bfp_free(&m);
    }
    /* SQNR bounds match the Gaussian results in blackhole-llm (bfp8 ~43 dB, bfp4 ~19 dB) */
    check(BFP8, 640, 2560, 40);    /* QFN expert gate/up: 2560 -> 640 */
    check(BFP4, 640, 2560, 17);
    check(BFP4, 2560, 640, 17);    /* QFN expert down: 640 -> 2560 */
    /* moe_forward (one fused parallel region) == gate-weighted sum of expert_forward */
    {
        enum { K = 3, D = 256, I = 96 };
        bfp_mat a[K], c[K], dn[K];
        const bfp_mat *pa[K], *pc[K], *pd[K];
        float *w1 = malloc(sizeof(float) * I * D), *w2 = malloc(sizeof(float) * D * I);
        float x[D], y[D], ye[D], ref[D] = {0}, tmp[K * (2 * I + D)], gate[K] = {0.5f, 0.3f, 0.2f};
        for (int e = 0; e < K; e++) {
            bfp_alloc(&a[e], BFP4, I, D); bfp_alloc(&c[e], BFP4, I, D); bfp_alloc(&dn[e], BFP8, D, I);
            for (int i = 0; i < I * D; i++) w1[i] = 0.05f * randn(), w2[i] = 0.05f * randn();
            bfp_pack(&a[e], w1); bfp_pack(&dn[e], w2);
            for (int i = 0; i < I * D; i++) w1[i] = 0.05f * randn();
            bfp_pack(&c[e], w1);
            pa[e] = &a[e]; pc[e] = &c[e]; pd[e] = &dn[e];
        }
        for (int i = 0; i < D; i++) x[i] = randn();
        for (int e = 0; e < K; e++) {
            expert_forward(&a[e], &c[e], &dn[e], x, ye, tmp);
            for (int i = 0; i < D; i++) ref[i] += gate[e] * ye[i];
        }
        moe_forward(pa, pc, pd, gate, K, x, y, tmp);
        for (int i = 0; i < D; i++) assert(fabsf(y[i] - ref[i]) <= 1e-5f * (fabsf(ref[i]) + 1e-2f));
        puts("moe_forward == sum of expert_forward");
        for (int e = 0; e < K; e++) { bfp_free(&a[e]); bfp_free(&c[e]); bfp_free(&dn[e]); }
        free(w1); free(w2);
    }
    puts("all tests passed");
    return 0;
}
