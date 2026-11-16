/* Adaptive expert cache: which routed experts live in accelerator memory, and which to copy there next.
 *
 * Slots 0 .. n_card-1 are on the p150a, n_card .. n_slots-1 on an optional second device (an NVIDIA GPU); an expert
 * is in at most one slot. Each token:
 *   tcache_lookup()    for the token's routed experts (all layers): slot or -1 (miss: the CPU computes it), and
 *                      marks them used;
 *   tcache_end_token() picks up to `budget` of the token's misses (lowest expert ids first) and the least recently
 *                      used slots to evict, and returns the copies to make. The copies overlap the next token.
 * ponytail: a copy counts as done at the next token; plan.py checks that the link finishes `budget` copies within a
 * token. Add an in-flight state if the real DMA turns out slower.
 * Same policy as trace.py's simulate_cache (the replay test in test_cache.c compares the two on the real trace).
 * Expert ids are global: layer * 512 + expert. */
#ifndef TUFF_CACHE_H
#define TUFF_CACHE_H
#include <stdint.h>

typedef struct {
    int n_experts, n_slots, n_card, budget;
    int32_t *slot_of;          /* [n_experts] slot, or -1 */
    int32_t *expert_in;        /* [n_slots] expert, or -1 */
    int32_t *prev, *next;      /* recency list over slots: head = most recent, tail = least */
    int32_t head, tail;
    int32_t *miss;             /* this token's misses */
    int n_miss, miss_cap;
    uint8_t *seen;             /* [n_experts] dedup of misses within a token */
    uint64_t hits, lookups, copies, tokens;
} tcache;

typedef struct { int32_t expert, slot, evicted; } tcache_copy;   /* evicted: expert dropped from the slot, or -1 */

/* init: expert ids, best first (e.g. Basalt's profile); the first n_slots fill the slots, better = more recent */
int tcache_init(tcache *c, int n_experts, int n_card, int n_other, int budget, const int32_t *init, int n_init);
void tcache_free(tcache *c);
void tcache_lookup(tcache *c, const int32_t *ids, int n, int32_t *slot);
int tcache_end_token(tcache *c, tcache_copy *out);              /* returns the number of copies (<= budget) */
#endif
