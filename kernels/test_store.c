/* test_store DIR FMT: DIR from `python loader.py fake DIR N --fmt FMT`. The views tstore_expert() hands out must be
 * byte for byte what bfp_pack makes of the same weights: gate = gate_up rows 0..639, up = rows 640..1279, down. */
#include "store.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void same(const bfp_mat *view, const float *w) {
    bfp_mat m;
    assert(bfp_alloc(&m, view->fmt, view->out, view->in) == 0);
    bfp_pack(&m, w);
    assert(!memcmp(m.exp, view->exp, (size_t)(m.out / 16) * m.in));
    assert(!memcmp(m.mant, view->mant, bfp_mant_bytes(m.fmt, m.out, m.in)));
    bfp_free(&m);
}

int main(int argc, char **argv) {
    assert(argc == 3);
    enum bfp_fmt fmt = strcmp(argv[2], "bfp8") ? BFP4 : BFP8;
    tstore s;
    assert(tstore_open(&s, argv[1], fmt, 1, 0) == 0);
    const int D = TSTORE_D, I = TSTORE_I;
    char path[4096];
    snprintf(path, sizeof path, "%s/weights.f32", argv[1]);
    FILE *f = fopen(path, "rb");
    float *gu = malloc(sizeof(float) * 2 * I * D), *dn = malloc(sizeof(float) * D * I);
    assert(f && gu && dn);
    for (int e = 0; e < s.experts; e++) {
        assert(fread(gu, sizeof(float), (size_t)2 * I * D, f) == (size_t)2 * I * D);
        assert(fread(dn, sizeof(float), (size_t)D * I, f) == (size_t)D * I);
        bfp_mat g, u, d;
        tstore_expert(&s, 0, e, &g, &u, &d);
        assert(tstore_bytes(&s, 0, e) == g.exp && ((uintptr_t)g.exp & 63) == 0);
        same(&g, gu); same(&u, gu + (size_t)I * D); same(&d, dn);
        /* card copy: tiles of the whole gate_up matrix, then of down, the same size as the host copy */
        bfp_mat m1, m2;
        uint8_t *tt = malloc(s.expert_bytes), *ref = malloc(s.expert_bytes);
        size_t n1 = bfp_tt_bytes(fmt, 2 * I, D);
        assert(n1 + bfp_tt_bytes(fmt, D, I) == s.expert_bytes);
        assert(bfp_alloc(&m1, fmt, 2 * I, D) == 0 && bfp_alloc(&m2, fmt, D, I) == 0);
        bfp_pack(&m1, gu); bfp_pack(&m2, dn);
        assert(bfp_to_tt_tiles(&m1, ref) == 0 && bfp_to_tt_tiles(&m2, ref + n1) == 0);
        tstore_expert_tt(&s, 0, e, tt);
        assert(!memcmp(tt, ref, s.expert_bytes));
        bfp_free(&m1); bfp_free(&m2); free(tt); free(ref);
    }
    fclose(f);
    printf("tstore (%s): %d experts of %zu bytes, views == bfp_pack, card copies == bfp_to_tt_tiles\n", argv[2],
           s.experts, s.expert_bytes);
    tstore_close(&s);
    free(gu); free(dn);
    return 0;
}
