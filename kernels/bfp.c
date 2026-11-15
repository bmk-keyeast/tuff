#include "bfp.h"
#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int bpb(enum bfp_fmt f) { return f == BFP8 ? 16 : 8; }   /* mantissa bytes per block */

size_t bfp_mant_bytes(enum bfp_fmt fmt, int out, int in) { return (size_t)(out / 16) * in * bpb(fmt); }

int bfp_alloc(bfp_mat *m, enum bfp_fmt fmt, int out, int in) {
    if (out % 16) return -1;
    m->fmt = fmt; m->out = out; m->in = in;
    /* 64-byte aligned so panels stream cleanly; sizes rounded up for aligned_alloc */
    size_t ne = (size_t)(out / 16) * in, nm = bfp_mant_bytes(fmt, out, in);
    m->exp = aligned_alloc(64, (ne + 63) & ~(size_t)63);
    m->mant = aligned_alloc(64, (nm + 63) & ~(size_t)63);
    return (m->exp && m->mant) ? 0 : -1;
}

void bfp_free(bfp_mat *m) { free(m->exp); free(m->mant); m->exp = m->mant = NULL; }

/* 2^(e) as float for e in the normal range, built from bits (exact, no libm) */
static inline float pow2i(int e) {
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)(e + 127) << 23;
    return v.f;
}

void bfp_pack(bfp_mat *m, const float *w) {
    const int mb = m->fmt, qmax = (1 << mb) - 1, P = m->out / 16;
    for (int p = 0; p < P; p++)
        for (int j = 0; j < m->in; j++) {
            float a = 0;
            for (int r = 0; r < 16; r++) a = fmaxf(a, fabsf(w[(size_t)(16 * p + r) * m->in + j]));
            size_t b = (size_t)p * m->in + j;
            uint8_t *dst = m->mant + b * bpb(m->fmt);
            if (a == 0 || !isfinite(a)) { m->exp[b] = 0; memset(dst, 0, bpb(m->fmt)); continue; }
            int e = ilogbf(a);                       /* floor(log2(a)) */
            if (e < mb - 126) e = mb - 126;          /* keep the scale a normal float (tiny blocks flush) */
            if (e > 127) e = 127;
            m->exp[b] = (uint8_t)(e + 127);
            float step = pow2i(e - (mb - 1));
            if (m->fmt == BFP4) memset(dst, 0, 8);
            for (int r = 0; r < 16; r++) {
                float v = w[(size_t)(16 * p + r) * m->in + j];
                int q = (int)floorf(fabsf(v) / step + 0.5f);
                if (q > qmax) q = qmax;
                int s = v < 0 && q;                  /* no negative zero */
                if (m->fmt == BFP8) dst[r] = (uint8_t)(s << 7 | q);
                else dst[r >> 1] |= (uint8_t)((s << 3 | q) << (4 * (r & 1)));
            }
        }
}

static inline int elem(const bfp_mat *m, size_t b, int r) {
    const uint8_t *src = m->mant + b * bpb(m->fmt);
    int v, s;
    if (m->fmt == BFP8) { v = src[r] & 0x7f; s = src[r] >> 7; }
    else { int n = (src[r >> 1] >> (4 * (r & 1))) & 0xf; v = n & 7; s = n >> 3; }
    return s ? -v : v;
}

static inline float scale_of(const bfp_mat *m, size_t b) {
    return m->exp[b] ? pow2i(m->exp[b] - 127 - (m->fmt - 1)) : 0.f;
}

void bfp_unpack(const bfp_mat *m, float *w) {
    for (int p = 0; p < m->out / 16; p++)
        for (int j = 0; j < m->in; j++) {
            size_t b = (size_t)p * m->in + j;
            float sc = scale_of(m, b);
            for (int r = 0; r < 16; r++) w[(size_t)(16 * p + r) * m->in + j] = sc * elem(m, b, r);
        }
}

void bfp_gemv_ref(const bfp_mat *m, const float *x, float *y) {
    for (int p = 0; p < m->out / 16; p++) {
        float acc[16] = {0};
        for (int j = 0; j < m->in; j++) {
            size_t b = (size_t)p * m->in + j;
            float s = x[j] * scale_of(m, b);
            for (int r = 0; r < 16; r++) acc[r] += s * elem(m, b, r);
        }
        memcpy(y + 16 * p, acc, sizeof acc);
    }
}

/* 16 sign-magnitude int8 -> two __m256 floats */
static inline void sm8_to_ps(__m128i b, __m256 *lo, __m256 *hi) {
    __m128i sign = _mm_cmpgt_epi8(_mm_setzero_si128(), b);            /* 0xff where bit7 set */
    __m128i mag = _mm_and_si128(b, _mm_set1_epi8(0x7f));
    __m128i v = _mm_sub_epi8(_mm_xor_si128(mag, sign), sign);          /* conditional negate */
    *lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v));
    *hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v, 8)));
}

/* block j of a panel -> 16 signed values as two __m256 */
static inline void load16(enum bfp_fmt fmt, const uint8_t *mt, int j, __m256 *lo, __m256 *hi) {
    if (fmt == BFP8) { sm8_to_ps(_mm_loadu_si128((const __m128i *)(mt + 16 * (size_t)j)), lo, hi); return; }
    /* nibble (sign<<3 | mag) -> signed int8 via pshufb */
    const __m128i lut = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 0, -1, -2, -3, -4, -5, -6, -7);
    const __m128i lo4 = _mm_set1_epi8(0x0f);
    __m128i b = _mm_loadl_epi64((const __m128i *)(mt + 8 * (size_t)j));
    __m128i v = _mm_shuffle_epi8(lut, _mm_unpacklo_epi8(_mm_and_si128(b, lo4), _mm_and_si128(_mm_srli_epi16(b, 4), lo4)));
    *lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v));
    *hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v, 8)));
}

/* exponent byte -> block scale, per format; scale[0] = 0 handles all-zero blocks without a branch */
static float scale8[256], scale4[256];
__attribute__((constructor)) static void init_scales(void) {
    for (int e = 1; e < 256; e++) {
        scale8[e] = e - 127 - 6 >= -126 ? pow2i(e - 127 - 6) : 0.f;
        scale4[e] = e - 127 - 2 >= -126 ? pow2i(e - 127 - 2) : 0.f;
    }
}

/* one 16-row panel: y16 = W[16p:16p+16] x. Two blocks per iteration into four accumulators,
 * so consecutive FMAs are independent (a single chain is bound by FMA latency, 5 cycles on Broadwell). */
static inline void panel(const bfp_mat *m, int p, const float *x, float *y16) {
    const int in = m->in;
    const float *sc = m->fmt == BFP8 ? scale8 : scale4;
    const uint8_t *e = m->exp + (size_t)p * in;
    const uint8_t *mt = m->mant + (size_t)p * in * bpb(m->fmt);
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, lo, hi;
    int j = 0;
    for (; j + 1 < in; j += 2) {
        load16(m->fmt, mt, j, &lo, &hi);
        __m256 s = _mm256_set1_ps(x[j] * sc[e[j]]);
        a0 = _mm256_fmadd_ps(s, lo, a0);
        a1 = _mm256_fmadd_ps(s, hi, a1);
        load16(m->fmt, mt, j + 1, &lo, &hi);
        s = _mm256_set1_ps(x[j + 1] * sc[e[j + 1]]);
        a2 = _mm256_fmadd_ps(s, lo, a2);
        a3 = _mm256_fmadd_ps(s, hi, a3);
    }
    if (j < in) {
        load16(m->fmt, mt, j, &lo, &hi);
        __m256 s = _mm256_set1_ps(x[j] * sc[e[j]]);
        a0 = _mm256_fmadd_ps(s, lo, a0);
        a1 = _mm256_fmadd_ps(s, hi, a1);
    }
    _mm256_storeu_ps(y16, _mm256_add_ps(a0, a2));
    _mm256_storeu_ps(y16 + 8, _mm256_add_ps(a1, a3));
}

void bfp_gemv(const bfp_mat *m, const float *x, float *y) {
#pragma omp parallel for schedule(static)
    for (int p = 0; p < m->out / 16; p++) panel(m, p, x, y + 16 * p);
}

/* One MoE layer for one token: y = sum_i gate[i] * expert_i(x) over k selected experts.
 * A single parallel region over (expert, panel) pairs: per-GEMV fork/join would cost more than the
 * GEMV itself at expert sizes (a 768x2048 bfp4 matrix is ~0.9 MB, ~15 us of DRAM time).
 * tmp: k * (2*I + d) floats. */
void moe_forward(const bfp_mat *const *w1, const bfp_mat *const *w3, const bfp_mat *const *w2,
                 const float *gate, int k, const float *x, float *y, float *tmp) {
    const int I = w1[0]->out, d = w2[0]->out, PI = I / 16, PD = d / 16;
    float *h = tmp, *part = tmp + (size_t)k * 2 * I;
#pragma omp parallel
    {
#pragma omp for collapse(2) schedule(static)
        for (int e = 0; e < k; e++)
            for (int p = 0; p < 2 * PI; p++)
                panel(p < PI ? w1[e] : w3[e], p % PI, x, h + (size_t)e * 2 * I + 16 * p);
#pragma omp for schedule(static)
        for (int i = 0; i < k * I; i++) {
            float *g = h + (size_t)(i / I) * 2 * I + i % I, u = g[I];
            *g = *g / (1.f + expf(-*g)) * u;
        }
#pragma omp for collapse(2) schedule(static)
        for (int e = 0; e < k; e++)
            for (int p = 0; p < PD; p++)
                panel(w2[e], p, h + (size_t)e * 2 * I, part + (size_t)e * d + 16 * p);
#pragma omp for schedule(static)
        for (int o = 0; o < d; o++) {
            float s = 0;
            for (int e = 0; e < k; e++) s += gate[e] * part[(size_t)e * d + o];
            y[o] = s;
        }
    }
}

void expert_forward(const bfp_mat *w1, const bfp_mat *w3, const bfp_mat *w2,
                    const float *x, float *y, float *tmp) {
    const int I = w1->out;
    float *g = tmp, *u = tmp + I;
    bfp_gemv(w1, x, g);
    bfp_gemv(w3, x, u);
    for (int i = 0; i < I; i++) g[i] = g[i] / (1.f + expf(-g[i])) * u[i];
    bfp_gemv(w2, g, y);
}
