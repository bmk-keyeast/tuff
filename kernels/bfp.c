#include "bfp.h"
#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int bits(enum bfp_fmt f) { return f == BFP8 ? 8 : 4; }   /* stored bits per element */

size_t bfp_mant_bytes(enum bfp_fmt fmt, int out, int in) { return (size_t)(out / 16) * in * 2 * bits(fmt); }

int bfp_alloc(bfp_mat *m, enum bfp_fmt fmt, int out, int in) {
    if (out % 16 || in % 8) return -1;
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

/* the dword holding W[16p+r][j], and the field's bit offset in it (layout in bfp.h) */
static inline uint32_t *slot(const bfp_mat *m, int p, int j, int r, int *sh) {
    int b = bits(m->fmt), c = 32 / b;
    *sh = b * (j % c);
    return (uint32_t *)m->mant + (((size_t)p * (m->in / c) + j / c) * 2 + r / 8) * 8 + r % 8;
}

void bfp_pack(bfp_mat *m, const float *w) {
    const int mb = m->fmt, qmax = (1 << mb) - 1, off = qmax + 1, P = m->out / 16;
    memset(m->mant, 0, bfp_mant_bytes(m->fmt, m->out, m->in));
    for (int p = 0; p < P; p++)
        for (int j = 0; j < m->in; j++) {
            float a = 0, step = 0;
            for (int r = 0; r < 16; r++) a = fmaxf(a, fabsf(w[(size_t)(16 * p + r) * m->in + j]));
            size_t b = (size_t)p * m->in + j;
            m->exp[b] = 0;                           /* all-zero (or non-finite) block */
            if (a != 0 && isfinite(a)) {
                int e = ilogbf(a);                   /* floor(log2(a)) */
                if (e < mb - 126) e = mb - 126;      /* keep the scale a normal float (tiny blocks flush) */
                if (e > 127) e = 127;
                m->exp[b] = (uint8_t)(e + 127);
                step = pow2i(e - (mb - 1));
            }
            for (int r = 0; r < 16; r++) {
                float v = w[(size_t)(16 * p + r) * m->in + j];
                int q = step ? (int)floorf(fabsf(v) / step + 0.5f) : 0, sh;
                if (q > qmax) q = qmax;
                uint32_t *d = slot(m, p, j, r, &sh);     /* sets sh: must happen before the shift */
                *d |= (uint32_t)(off + (v < 0 ? -q : q)) << sh;
            }
        }
}

static inline int elem(const bfp_mat *m, int p, int j, int r) {
    int sh, b = bits(m->fmt);
    uint32_t d = *slot(m, p, j, r, &sh);
    return (int)((d >> sh) & ((1u << b) - 1)) - (1 << (b - 1));
}

static inline float scale_of(const bfp_mat *m, size_t b) {
    return m->exp[b] ? pow2i(m->exp[b] - 127 - (m->fmt - 1)) : 0.f;
}

void bfp_unpack(const bfp_mat *m, float *w) {
    for (int p = 0; p < m->out / 16; p++)
        for (int j = 0; j < m->in; j++) {
            float sc = scale_of(m, (size_t)p * m->in + j);
            for (int r = 0; r < 16; r++) w[(size_t)(16 * p + r) * m->in + j] = sc * elem(m, p, j, r);
        }
}

void bfp_gemv_ref(const bfp_mat *m, const float *x, float *y) {
    for (int p = 0; p < m->out / 16; p++) {
        float acc[16] = {0};
        for (int j = 0; j < m->in; j++) {
            float s = x[j] * scale_of(m, (size_t)p * m->in + j);
            for (int r = 0; r < 16; r++) acc[r] += s * elem(m, p, j, r);
        }
        memcpy(y + 16 * p, acc, sizeof acc);
    }
}

static inline float hsum(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)));
}

/* field k (b bits wide) of every dword in v, as int32; the first and last fields need one op, not two */
#define FIELD(v, k, b, mask) ((k) == 0 ? _mm256_and_si256(v, mask)                          \
                              : (k) == 32 / (b) - 1 ? _mm256_srli_epi32(v, 32 - (b))        \
                              : _mm256_and_si256(_mm256_srli_epi32(v, (b) * (k)), mask))
/* column k of the group: 8 rows (lo) + 8 rows (hi) += a[j] * n */
#define STEP(k, b, L, H) {                                                                  \
    __m256 s = _mm256_broadcast_ss(ag + (k));                                               \
    L = _mm256_fmadd_ps(s, _mm256_cvtepi32_ps(FIELD(lo, k, b, mask)), L);                   \
    H = _mm256_fmadd_ps(s, _mm256_cvtepi32_ps(FIELD(hi, k, b, mask)), H); }

/* One 16-row panel: y16 = W[16p:16p+16] x.
 * Pass 1, 8 columns per step: a[j] = x[j] * 2^(block exponent), built from the exponent bits; A = sum a[j].
 * Pass 2: a 256-bit load is 8 rows x one column group; each field is an unsigned n = value + off, so it
 * decodes with shift + and + int->float (no shuffles, which all queue on port 5 on Broadwell) and the
 * offset is removed once per row at the end: y = sum a[j] n - off * A. */
static inline void panel(const bfp_mat *m, int p, const float *x, float *y16) {
    const int in = m->in, mb = m->fmt;
    float a[in];
    const uint8_t *e = m->exp + (size_t)p * in;
    const __m256i bias = _mm256_set1_epi32(mb - 1);
    __m256 As = _mm256_setzero_ps();
    for (int j = 0; j < in; j += 8) {
        __m256i ev = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(e + j)));
        __m256i sb = _mm256_and_si256(_mm256_cmpgt_epi32(ev, bias),          /* exp 0 -> scale 0 */
                                      _mm256_slli_epi32(_mm256_sub_epi32(ev, bias), 23));
        __m256 v = _mm256_mul_ps(_mm256_loadu_ps(x + j), _mm256_castsi256_ps(sb));
        _mm256_storeu_ps(a + j, v);
        As = _mm256_add_ps(As, v);
    }
    const __m256i *mt = (const __m256i *)(m->mant + (size_t)p * in * 2 * bits(m->fmt));
    __m256 l0 = _mm256_setzero_ps(), l1 = l0, l2 = l0, l3 = l0, h0 = l0, h1 = l0, h2 = l0, h3 = l0;
    if (m->fmt == BFP4) {
        const __m256i mask = _mm256_set1_epi32(0xf);
        for (const float *ag = a; ag < a + in; ag += 8, mt += 2) {
            __m256i lo = _mm256_loadu_si256(mt), hi = _mm256_loadu_si256(mt + 1);
            STEP(0, 4, l0, h0) STEP(1, 4, l1, h1) STEP(2, 4, l2, h2) STEP(3, 4, l3, h3)
            STEP(4, 4, l0, h0) STEP(5, 4, l1, h1) STEP(6, 4, l2, h2) STEP(7, 4, l3, h3)
        }
    } else {
        const __m256i mask = _mm256_set1_epi32(0xff);
        for (const float *ag = a; ag < a + in; ag += 4, mt += 2) {
            __m256i lo = _mm256_loadu_si256(mt), hi = _mm256_loadu_si256(mt + 1);
            STEP(0, 8, l0, h0) STEP(1, 8, l1, h1) STEP(2, 8, l2, h2) STEP(3, 8, l3, h3)
        }
    }
    __m256 corr = _mm256_set1_ps(-(float)(1 << mb) * hsum(As));
    _mm256_storeu_ps(y16, _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(l0, l1), _mm256_add_ps(l2, l3)), corr));
    _mm256_storeu_ps(y16 + 8, _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(h0, h1), _mm256_add_ps(h2, h3)), corr));
}

void bfp_gemv(const bfp_mat *m, const float *x, float *y) {
#pragma omp parallel for schedule(static)
    for (int p = 0; p < m->out / 16; p++) panel(m, p, x, y + 16 * p);
}

/* One MoE layer for one token: y = sum_i gate[i] * expert_i(x) over k selected experts.
 * A single parallel region over (expert, panel) pairs: per-GEMV fork/join would cost more than the
 * GEMV itself at expert sizes (a 640x2560 bfp4 matrix is ~0.9 MB, ~15 us of DRAM time).
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
