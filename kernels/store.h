/* Routed experts on the host: the per-layer files from `loader.py convert`, memory-mapped, used in place.
 *
 * experts.LL.FMT holds, for e = 0 .. n-1: gate_up exp, gate_up mant, down exp, down mant (gate_up = [1280][2560],
 * gate rows 0..639 then up rows 640..1279; down = [2560][640]), every piece a multiple of 64 bytes. Gate and up are
 * row-panel ranges of one packed matrix, so all three bfp_mat views point into the mapping: no copy, no repacking.
 * An expert is one contiguous run of expert_bytes, which is also what a copy to the card or the GPU moves.
 *
 * Memory: all 48 layers are 68 GB at BFP4. The adaptive cache needs every expert on the host (a miss can be any of
 * them), so 128 GB keeps them all in RAM; with 64 GB the least-used ones stay on NVMe and fault in through the page
 * cache. tstore_open(..., lock=1) asks the kernel to keep the mapping resident (mlock), which needs enough RAM and
 * `ulimit -l`. */
#ifndef TUFF_STORE_H
#define TUFF_STORE_H
#include "bfp.h"

enum { TSTORE_MAX_LAYERS = 64, TSTORE_D = 2560, TSTORE_I = 640 };

typedef struct {
    enum bfp_fmt fmt;
    int layers, experts;                      /* experts per layer (512 for QFN; fewer in tests) */
    size_t piece[4], expert_bytes;            /* gate_up exp, gate_up mant, down exp, down mant */
    uint8_t *base[TSTORE_MAX_LAYERS];
    size_t len[TSTORE_MAX_LAYERS];
} tstore;

/* opens DIR/experts.00.FMT .. experts.(layers-1).FMT; 0 on success, -1 (and a message on stderr) otherwise */
int tstore_open(tstore *s, const char *dir, enum bfp_fmt fmt, int layers, int lock);
void tstore_close(tstore *s);
/* views of expert e of layer L: gate [640][2560], up [640][2560], down [2560][640] */
void tstore_expert(const tstore *s, int L, int e, bfp_mat *gate, bfp_mat *up, bfp_mat *down);
static inline const uint8_t *tstore_bytes(const tstore *s, int L, int e) { return s->base[L] + (size_t)e * s->expert_bytes; }
#endif
