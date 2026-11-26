/* libtuff.so: the host side of the engine for the Python card code (device/host_c.py, ctypes).
 *
 *   store   tstore: the converted expert files, memory-mapped; tuff_moe() computes a layer's cold experts for n tokens
 *           at once with moe_forward_n (each expert read once per 4 tokens), straight from the mapping
 *   cache   tcache: which experts sit in the card's slots, and which to copy there after each step
 *   card    tuff_expert_tt(): an expert's card copy (TT tiles) for a promotion
 */
#include "bfp.h"
#include "cache.h"
#include "store.h"
#include <stdlib.h>

tstore *tuff_store_open(const char *dir, int fmt, int layers, int D, int I) {
    tstore *s = malloc(sizeof *s);
    if (s && tstore_open_dims(s, dir, (enum bfp_fmt)fmt, layers, 0, D, I) == 0) return s;
    free(s);
    return NULL;
}

void tuff_store_close(tstore *s) { if (s) { tstore_close(s); free(s); } }
int tuff_store_experts(const tstore *s) { return s->experts; }
size_t tuff_expert_bytes(const tstore *s) { return s->expert_bytes; }
void tuff_expert_tt(const tstore *s, int L, int e, uint8_t *dst) { tstore_expert_tt(s, L, e, dst); }

/* y[n][D] = sum over i < k of gate[i][t] * expert es[i] of layer L (x[t]); gate[i][t] = 0: token t skips expert i.
 * Returns 0, or -1 if out of memory. */
int tuff_moe(const tstore *s, int L, const int32_t *es, int k, const float *gate, int n, const float *x, float *y) {
    if (k <= 0) return 0;
    bfp_mat *m = malloc(sizeof *m * 3 * k);
    const bfp_mat **p = malloc(sizeof *p * 3 * k);
    float *tmp = malloc(sizeof(float) * (size_t)k * (n < 4 ? n : 4) * (2 * s->I + s->D));
    if (!m || !p || !tmp) { free(m); free(p); free(tmp); return -1; }
    for (int i = 0; i < k; i++) {
        tstore_expert(s, L, es[i], &m[i], &m[k + i], &m[2 * k + i]);
        p[i] = &m[i]; p[k + i] = &m[k + i]; p[2 * k + i] = &m[2 * k + i];
    }
    moe_forward_n(p, p + k, p + 2 * k, gate, k, n, x, y, tmp);
    free(m); free(p); free(tmp);
    return 0;
}

tcache *tuff_cache_new(int n_experts, int n_card, int n_other, int budget) {
    tcache *c = malloc(sizeof *c);
    if (c && tcache_init(c, n_experts, n_card, n_other, budget, NULL, 0) == 0) return c;
    if (c) tcache_free(c);
    free(c);
    return NULL;
}

void tuff_cache_free(tcache *c) { if (c) { tcache_free(c); free(c); } }
void tuff_cache_lookup(tcache *c, const int32_t *ids, int n, int32_t *slot) { tcache_lookup(c, ids, n, slot); }

/* copies after a step, as {expert, slot, evicted} triples in out[3 * budget]; returns how many */
int tuff_cache_end(tcache *c, int32_t *out) {
    tcache_copy *cp = malloc(sizeof *cp * (c->budget + 1));
    if (!cp) return -1;
    int n = tcache_end_token(c, cp);
    for (int i = 0; i < n; i++) { out[3 * i] = cp[i].expert; out[3 * i + 1] = cp[i].slot; out[3 * i + 2] = cp[i].evicted; }
    free(cp);
    return n;
}
