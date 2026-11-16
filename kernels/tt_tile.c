/* Host panel layout (bfp.h) -> Tenstorrent bfp8_b / bfp4_b tiles, the bytes a promoted expert needs on the p150a.
 *
 * The card multiplies by W^T [in][out] in 32x32 tiles (row-major tile order). A tile is 4 faces of 16x16 (top-left,
 * top-right, bottom-left, bottom-right); each face row (one input column j, 16 consecutive outputs) shares one
 * exponent, which is exactly a host block W[16p..16p+15][j]. Per tile: the 64 face-row exponents (face 0 rows 0..15,
 * face 1, ...), then each face's values row-major, 32/b per dword with element 0 in the low bits, sign-magnitude
 * (sign << mbits | magnitude). bfp_pack rounds like tt-metal, so the conversion only moves and recodes fields.
 * Checked byte for byte against a port of tt-metal's pack_as_bfp_tiles (loader.tt_tiles_ref) in test_bfp.
 *
 * The host keeps a block's 8 (bfp4) or 4 (bfp8) columns of one row per dword; the card keeps 8 or 4 rows of one
 * column per dword. So each 8x8 nibble (4x4 byte) square is transposed in registers with three (two) masked swaps,
 * and each byte is recoded offset binary -> sign-magnitude through a 256-entry table.
 * Speed: 1.1 GB/s per core (laptop Zen 3, one thread). 25 promotions per token at ~40 t/s is ~2.8 GB/s, so the copy
 * engine needs ~3 cores for it; ponytail: scalar, split across threads by panel or move the recode into a Tensix
 * kernel on the card if those cores are missed by the cold experts. */
#include "bfp.h"
#include <string.h>

size_t bfp_tt_bytes(enum bfp_fmt fmt, int out, int in) {
    return (size_t)(out / 32) * (in / 32) * (64 + 1024 * (fmt + 1) / 8);
}

/* swap the `s`-bit fields selected by mask between a (high) and b (low) */
#define SWAP(a, b, s, mask) { uint32_t t_ = (((a) >> (s)) ^ (b)) & (mask); (b) ^= t_; (a) ^= t_ << (s); }

static uint8_t lut4[256], lut8[256];

static void init_luts(void) {
    for (int x = 0; x < 256; x++) {
        int v[2];
        for (int h = 0; h < 2; h++) {                              /* two 4-bit fields: n - 8 -> sign-magnitude */
            int q = ((x >> (4 * h)) & 15) - 8;
            v[h] = q < 0 ? 8 | -q : q;
        }
        lut4[x] = (uint8_t)(v[0] | v[1] << 4);
        int q = x - 128;                                           /* one 8-bit field */
        lut8[x] = (uint8_t)(q < 0 ? 128 | -q : q);
    }
}

static inline uint32_t recode(uint32_t w, const uint8_t *lut) {
    return lut[w & 255] | (uint32_t)lut[(w >> 8) & 255] << 8 | (uint32_t)lut[(w >> 16) & 255] << 16 |
           (uint32_t)lut[w >> 24] << 24;
}

/* b is a literal at both call sites (always_inline), so every / and % below is a shift or a mask */
static inline __attribute__((always_inline)) void convert(const bfp_mat *m, uint8_t *dst, const int b) {
    const int c = 32 / b, in = m->in, TO = m->out / 32;
    const size_t tile = 64 + 1024 * b / 8;
    const uint32_t *mt = (const uint32_t *)m->mant;
    const uint8_t *lut = b == 4 ? lut4 : lut8;
    for (int p = 0; p < m->out / 16; p++) {
        const int to = p / 2, tc = p % 2;
        for (int j = 0; j < in; j++) {                             /* exponents */
            const int ti = j / 32, f = 2 * ((j % 32) / 16) + tc;
            dst[(size_t)(ti * TO + to) * tile + f * 16 + j % 16] = m->exp[(size_t)p * in + j];
        }
        for (int g = 0; g < in / c; g++) {                         /* one host column group: 16 dwords (rows) */
            const uint32_t *h = mt + ((size_t)p * (in / c) + g) * 16;
            for (int half = 0; half < 16 / c; half++) {           /* bfp4: rows 0-7, 8-15; bfp8: four row quads */
                uint32_t x[8];
                for (int r = 0; r < c; r++) x[r] = recode(h[half * c + r], lut);
                if (b == 4) {
                    SWAP(x[0], x[4], 16, 0xffffu) SWAP(x[1], x[5], 16, 0xffffu)
                    SWAP(x[2], x[6], 16, 0xffffu) SWAP(x[3], x[7], 16, 0xffffu)
                    SWAP(x[0], x[2], 8, 0xff00ffu) SWAP(x[1], x[3], 8, 0xff00ffu)
                    SWAP(x[4], x[6], 8, 0xff00ffu) SWAP(x[5], x[7], 8, 0xff00ffu)
                    SWAP(x[0], x[1], 4, 0x0f0f0f0fu) SWAP(x[2], x[3], 4, 0x0f0f0f0fu)
                    SWAP(x[4], x[5], 4, 0x0f0f0f0fu) SWAP(x[6], x[7], 4, 0x0f0f0f0fu)
                } else {
                    SWAP(x[0], x[2], 16, 0xffffu) SWAP(x[1], x[3], 16, 0xffffu)
                    SWAP(x[0], x[1], 8, 0xff00ffu) SWAP(x[2], x[3], 8, 0xff00ffu)
                }
                for (int k = 0; k < c; k++) {                      /* x[k] = rows of column j = g*c + k */
                    const int j = g * c + k, ti = j / 32, f = 2 * ((j % 32) / 16) + tc, i = j % 16;
                    uint32_t *d = (uint32_t *)(dst + (size_t)(ti * TO + to) * tile + 64);
                    d[f * (256 / c) + i * (16 / c) + half] = x[k];
                }
            }
        }
    }
}

int bfp_to_tt_tiles(const bfp_mat *m, uint8_t *dst) {
    if (m->out % 32 || m->in % 32) return -1;
    if (!lut8[0]) init_luts();          /* lut8[0] = 128 once built; ponytail: racy first call is benign (same bytes) */
    if (m->fmt == BFP4) convert(m, dst, 4); else convert(m, dst, 8);
    return 0;
}
