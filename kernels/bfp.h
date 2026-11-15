/* CPU kernels for MoE experts stored in Tenstorrent block-float formats (bfp8_b / bfp4_b).
 *
 * Why TT formats on the host: a cold expert computed here and a hot expert resident on the p150a
 * use the *same blocks*, so promoting/demoting an expert between host RAM and the card is a block
 * permutation (re-tiling), never a re-quantization.
 *
 * Layout ("panel" format) for a torch-style weight W[out][in]:
 *   panel p = output rows 16p .. 16p+15. For each input column j, the 16 values W[16p+r][j] form one
 *   block with one shared 8-bit exponent -- the same grouping as a TT tile face row of W^T[in][out].
 *   exp[p][j]          uint8, biased like float32 (0 = all-zero block)
 *   mant bfp8: [p][j][16]  uint8 = sign<<7 | 7-bit magnitude
 *   mant bfp4: [p][j][8]   uint8, two elements per byte: low nibble = row 2i, high = row 2i+1,
 *                          nibble = sign<<3 | 3-bit magnitude
 *   value = (-1)^sign * magnitude * 2^(exp - 127 - (mbits - 1)),  mbits = 7 (bfp8) or 3 (bfp4)
 * Rounding when packing is round-to-nearest with saturation, as in blackhole-llm/bfp.py.
 */
#ifndef TUFF_BFP_H
#define TUFF_BFP_H
#include <stddef.h>
#include <stdint.h>

enum bfp_fmt { BFP8 = 7, BFP4 = 3 };   /* value = magnitude bits */

typedef struct {
    enum bfp_fmt fmt;
    int out, in;          /* out % 16 == 0 */
    uint8_t *exp;         /* out/16 * in */
    uint8_t *mant;        /* out/16 * in * (fmt == BFP8 ? 16 : 8) */
} bfp_mat;

size_t bfp_mant_bytes(enum bfp_fmt fmt, int out, int in);
int bfp_alloc(bfp_mat *m, enum bfp_fmt fmt, int out, int in);   /* 0 on success */
void bfp_free(bfp_mat *m);
void bfp_pack(bfp_mat *m, const float *w);                      /* w: row-major [out][in] */
void bfp_unpack(const bfp_mat *m, float *w);                    /* dequantize back to [out][in] */

/* y[out] = W x[in] */
void bfp_gemv_ref(const bfp_mat *m, const float *x, float *y);  /* scalar reference */
void bfp_gemv(const bfp_mat *m, const float *x, float *y);      /* AVX2+FMA, OpenMP over panels */

/* SwiGLU expert: y = W2 (silu(W1 x) * (W3 x)); W1,W3: [I][d], W2: [d][I]; tmp: 2*I floats */
void expert_forward(const bfp_mat *w1, const bfp_mat *w3, const bfp_mat *w2,
                    const float *x, float *y, float *tmp);
/* One MoE layer, one token: y = sum_i gate[i] * expert_i(x). tmp: k*(2*I + d) floats */
void moe_forward(const bfp_mat *const *w1, const bfp_mat *const *w3, const bfp_mat *const *w2,
                 const float *gate, int k, const float *x, float *y, float *tmp);
#endif
