/* Correctness checks: make test */
#include "bfp.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* `test_bfp FILE`: FILE from `python loader.py selftest`, records of {int32 mbits, out, in; float w[out][in];
 * exp; mant} packed by the Python loader. The C packer must produce the same bytes. */
static int check_loader(const char *path) {
    FILE *f = fopen(path, "rb");
    int hdr[3], n = 0;
    assert(f);
    while (fread(hdr, sizeof hdr, 1, f) == 1) {
        bfp_mat m;
        assert(bfp_alloc(&m, (enum bfp_fmt)hdr[0], hdr[1], hdr[2]) == 0);
        size_t ne = (size_t)(hdr[1] / 16) * hdr[2], nm = bfp_mant_bytes(m.fmt, hdr[1], hdr[2]);
        float *w = malloc(sizeof(float) * hdr[1] * hdr[2]);
        uint8_t *ex = malloc(ne), *mt = malloc(nm);
        assert(fread(w, sizeof(float), (size_t)hdr[1] * hdr[2], f) == (size_t)hdr[1] * hdr[2]);
        assert(fread(ex, 1, ne, f) == ne && fread(mt, 1, nm, f) == nm);
        bfp_pack(&m, w);
        assert(!memcmp(ex, m.exp, ne) && !memcmp(mt, m.mant, nm));
        bfp_free(&m); free(w); free(ex); free(mt);
        n++;
    }
    fclose(f);
    assert(n > 0);
    printf("loader.py packing == bfp_pack, byte for byte (%d matrices)\n", n);
    return 0;
}

int main(int argc, char **argv) {
    if (argc > 1) return check_loader(argv[1]);
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
        /* moe_forward_n (n tokens, groups of 4) == moe_forward once per token with that token's gates */
        enum { N = 7 };
        float xs[N][D], ys[N][D], g[K][N], gt[K], tmpn[K * 4 * (2 * I + D)];
        for (int t = 0; t < N; t++)
            for (int i = 0; i < D; i++) xs[t][i] = randn();
        for (int e = 0; e < K; e++)
            for (int t = 0; t < N; t++) g[e][t] = (e + t) % 3 ? 0.f : 0.2f + 0.1f * e;   /* some tokens skip some experts */
        moe_forward_n(pa, pc, pd, &g[0][0], K, N, &xs[0][0], &ys[0][0], tmpn);
        for (int t = 0; t < N; t++) {
            for (int e = 0; e < K; e++) gt[e] = g[e][t];
            moe_forward(pa, pc, pd, gt, K, xs[t], ref, tmp);
            /* different accumulation order than the 1-token kernel: float rounding ~1e-6; a wrong token or gate is ~1e-2 */
            for (int i = 0; i < D; i++) assert(fabsf(ys[t][i] - ref[i]) <= 1e-4f * (fabsf(ref[i]) + 5e-2f));
        }
        puts("moe_forward_n == moe_forward per token");
        for (int e = 0; e < K; e++) { bfp_free(&a[e]); bfp_free(&c[e]); bfp_free(&dn[e]); }
        free(w1); free(w2);
    }
    puts("all tests passed");
    return 0;
}
