#include "cache.h"
#include <stdlib.h>
#include <string.h>

static void unlink_slot(tcache *c, int s) {
    if (c->prev[s] >= 0) c->next[c->prev[s]] = c->next[s]; else c->head = c->next[s];
    if (c->next[s] >= 0) c->prev[c->next[s]] = c->prev[s]; else c->tail = c->prev[s];
}

static void push_front(tcache *c, int s) {
    c->prev[s] = -1;
    c->next[s] = c->head;
    if (c->head >= 0) c->prev[c->head] = s; else c->tail = s;
    c->head = s;
}

int tcache_init(tcache *c, int n_experts, int n_card, int n_other, int budget, const int32_t *init, int n_init) {
    memset(c, 0, sizeof *c);
    c->n_experts = n_experts; c->n_card = n_card; c->n_slots = n_card + n_other; c->budget = budget;
    c->miss_cap = 4096;
    c->slot_of = malloc(sizeof(int32_t) * n_experts);
    c->seen = calloc(n_experts, 1);
    c->expert_in = malloc(sizeof(int32_t) * (c->n_slots + 1));
    c->prev = malloc(sizeof(int32_t) * (c->n_slots + 1));
    c->next = malloc(sizeof(int32_t) * (c->n_slots + 1));
    c->miss = malloc(sizeof(int32_t) * c->miss_cap);
    if (!c->slot_of || !c->seen || !c->expert_in || !c->prev || !c->next || !c->miss) return -1;
    for (int e = 0; e < n_experts; e++) c->slot_of[e] = -1;
    c->head = c->tail = -1;
    /* worst-ranked first, so the best ends up most recent */
    int n = n_init < c->n_slots ? n_init : c->n_slots;
    for (int s = 0; s < c->n_slots; s++) c->expert_in[s] = -1;
    for (int s = c->n_slots - 1; s >= 0; s--) {
        if (s < n) {
            if (init[s] < 0 || init[s] >= n_experts || c->slot_of[init[s]] >= 0) return -1;
            c->expert_in[s] = init[s];
            c->slot_of[init[s]] = s;
        }
        push_front(c, s);              /* empty slots too: they are evicted first, as the least recent */
    }
    return 0;
}

void tcache_free(tcache *c) {
    free(c->slot_of); free(c->seen); free(c->expert_in); free(c->prev); free(c->next); free(c->miss);
    memset(c, 0, sizeof *c);
}

void tcache_lookup(tcache *c, const int32_t *ids, int n, int32_t *slot) {
    for (int i = 0; i < n; i++) {
        int e = ids[i], s = c->slot_of[e];
        slot[i] = s;
        if (s >= 0) {
            c->hits++;
            if (c->head != s) { unlink_slot(c, s); push_front(c, s); }
        } else if (!c->seen[e]) {
            c->seen[e] = 1;
            if (c->n_miss == c->miss_cap) {
                int32_t *m = realloc(c->miss, sizeof(int32_t) * 2 * c->miss_cap);
                if (!m) continue;          /* ponytail: drop the miss from the copy queue; it still runs on the CPU */
                c->miss = m; c->miss_cap *= 2;
            }
            c->miss[c->n_miss++] = e;
        }
    }
    c->lookups += n;
}

static int cmp_i32(const void *a, const void *b) { return (*(const int32_t *)a > *(const int32_t *)b) - (*(const int32_t *)a < *(const int32_t *)b); }

int tcache_end_token(tcache *c, tcache_copy *out) {
    qsort(c->miss, c->n_miss, sizeof(int32_t), cmp_i32);
    int n = c->n_miss < c->budget ? c->n_miss : c->budget;
    if (n > c->n_slots) n = c->n_slots;
    for (int i = 0; i < n; i++) {
        int e = c->miss[i], s = c->tail, old = c->expert_in[s];
        if (old >= 0) c->slot_of[old] = -1;
        c->expert_in[s] = e;
        c->slot_of[e] = s;
        unlink_slot(c, s); push_front(c, s);                 /* just used */
        out[i] = (tcache_copy){e, s, old};
    }
    for (int i = 0; i < c->n_miss; i++) c->seen[c->miss[i]] = 0;
    c->n_miss = 0;
    c->copies += n;
    c->tokens++;
    return n;
}
