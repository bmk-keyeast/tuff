/* tcache tests: a small hand-checked case, and (`test_cache FILE`) a replay of the real routing trace against the
 * hit rates trace.py's simulate_cache computed for the same trace (FILE from `python trace.py cache-check`):
 * int32 T, K (ids per token), n_experts, n_init, n_cases; n_cases x {n_slots, budget, python hit * 1e6};
 * init[n_init]; ids[T][K]. */
#include "cache.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void small(void) {
    tcache c;
    tcache_copy cp[4];
    int32_t init[] = {5, 6, 7}, slot[4];
    assert(tcache_init(&c, 10, 2, 1, 2, init, 3) == 0);    /* 2 card slots + 1 GPU slot */
    assert(c.slot_of[5] == 0 && c.slot_of[6] == 1 && c.slot_of[7] == 2);
    int32_t t0[] = {6, 1, 2, 1};                            /* 6 hits; 1, 2 miss (1 twice) */
    tcache_lookup(&c, t0, 4, slot);
    assert(slot[0] == 1 && slot[1] == -1 && slot[2] == -1 && slot[3] == -1);
    assert(tcache_end_token(&c, cp) == 2);
    /* recency before the copies: 6 (used), 5 (best init), 7: 7 then 5 are evicted */
    assert(cp[0].expert == 1 && cp[0].evicted == 7 && cp[0].slot == 2);
    assert(cp[1].expert == 2 && cp[1].evicted == 5 && cp[1].slot == 0);
    assert(c.slot_of[5] == -1 && c.slot_of[7] == -1 && c.slot_of[1] == 2 && c.slot_of[2] == 0);
    int32_t t1[] = {1, 2, 6};
    tcache_lookup(&c, t1, 3, slot);
    assert(slot[0] == 2 && slot[1] == 0 && slot[2] == 1);
    assert(tcache_end_token(&c, cp) == 0);
    assert(c.hits == 4 && c.lookups == 7 && c.copies == 2);
    tcache_free(&c);
    puts("tcache: hand-checked LRU case ok");
}

static int replay(const char *path) {
    FILE *f = fopen(path, "rb");
    int32_t h[5];
    assert(f && fread(h, sizeof h, 1, f) == 1);
    int T = h[0], K = h[1], NE = h[2], NI = h[3], NC = h[4];
    int32_t *cases = malloc(sizeof(int32_t) * 3 * NC), *init = malloc(sizeof(int32_t) * NI);
    int32_t *ids = malloc(sizeof(int32_t) * (size_t)T * K), *slot = malloc(sizeof(int32_t) * K);
    assert(fread(cases, sizeof(int32_t), 3 * NC, f) == (size_t)3 * NC);
    assert(fread(init, sizeof(int32_t), NI, f) == (size_t)NI);
    assert(fread(ids, sizeof(int32_t), (size_t)T * K, f) == (size_t)T * K);
    fclose(f);
    for (int i = 0; i < NC; i++) {
        int n = cases[3 * i], budget = cases[3 * i + 1];
        double want = cases[3 * i + 2] / 1e6;
        tcache c;
        tcache_copy *cp = malloc(sizeof(tcache_copy) * (budget + 1));
        assert(tcache_init(&c, NE, n, 0, budget, init, NI) == 0);
        for (int t = 0; t < T; t++) {
            tcache_lookup(&c, ids + (size_t)t * K, K, slot);
            tcache_end_token(&c, cp);
        }
        double got = (double)c.hits / c.lookups;
        printf("replay: %d slots, %3d copies/token: hit %.4f (python %.4f), %.1f copies/token\n", n, budget, got, want,
               (double)c.copies / T);
        /* LRU ties (experts used by the same token) break differently from numpy's argpartition */
        assert(fabs(got - want) < 0.005);
        tcache_free(&c);
        free(cp);
    }
    free(cases); free(init); free(ids); free(slot);
    puts("tcache == simulate_cache on the real trace");
    return 0;
}

int main(int argc, char **argv) {
    small();
    return argc > 1 ? replay(argv[1]) : 0;
}
