#define _GNU_SOURCE
#include "store.h"
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static const char *fmt_name(enum bfp_fmt f) { return f == BFP8 ? "bfp8" : "bfp4"; }

int tstore_open(tstore *s, const char *dir, enum bfp_fmt fmt, int layers, int lock) {
    const int D = TSTORE_D, I = TSTORE_I;
    memset(s, 0, sizeof *s);
    if (layers < 1 || layers > TSTORE_MAX_LAYERS) return -1;
    s->fmt = fmt;
    s->piece[0] = (size_t)(2 * I / 16) * D;
    s->piece[1] = bfp_mant_bytes(fmt, 2 * I, D);
    s->piece[2] = (size_t)(D / 16) * I;
    s->piece[3] = bfp_mant_bytes(fmt, D, I);
    s->expert_bytes = s->piece[0] + s->piece[1] + s->piece[2] + s->piece[3];
    for (int L = 0; L < layers; L++) {
        char path[4096];
        snprintf(path, sizeof path, "%s/experts.%02d.%s", dir, L, fmt_name(fmt));
        int fd = open(path, O_RDONLY);
        struct stat st;
        if (fd < 0 || fstat(fd, &st) < 0) { perror(path); if (fd >= 0) close(fd); tstore_close(s); return -1; }
        int n = (int)(st.st_size / s->expert_bytes);
        if (n < 1 || (size_t)st.st_size != n * s->expert_bytes || (L && n != s->experts)) {
            fprintf(stderr, "%s: %lld bytes is not a whole number of %zu-byte experts (or differs from layer 0)\n",
                    path, (long long)st.st_size, s->expert_bytes);
            close(fd); tstore_close(s); return -1;
        }
        void *p = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (p == MAP_FAILED) { perror(path); tstore_close(s); return -1; }
        madvise(p, st.st_size, MADV_RANDOM);   /* experts are read whole but in routing order, not file order */
        if (lock && mlock(p, st.st_size) < 0) perror("mlock (continuing unlocked)");
        s->base[L] = p; s->len[L] = st.st_size; s->experts = n; s->layers = L + 1;
    }
    return 0;
}

void tstore_close(tstore *s) {
    for (int L = 0; L < TSTORE_MAX_LAYERS; L++)
        if (s->base[L]) munmap(s->base[L], s->len[L]);
    memset(s, 0, sizeof *s);
}

void tstore_expert(const tstore *s, int L, int e, bfp_mat *gate, bfp_mat *up, bfp_mat *down) {
    const int D = TSTORE_D, I = TSTORE_I;
    uint8_t *x = s->base[L] + (size_t)e * s->expert_bytes, *gu_exp = x, *gu_mant = x + s->piece[0];
    uint8_t *dn = gu_mant + s->piece[1];
    size_t panels = I / 16;                                  /* gate = panels 0..39, up = 40..79 */
    *gate = (bfp_mat){s->fmt, I, D, gu_exp, gu_mant};
    *up = (bfp_mat){s->fmt, I, D, gu_exp + panels * D, gu_mant + bfp_mant_bytes(s->fmt, I, D)};
    *down = (bfp_mat){s->fmt, D, I, dn, dn + s->piece[2]};
}

void tstore_expert_tt(const tstore *s, int L, int e, uint8_t *dst) {
    const int D = TSTORE_D, I = TSTORE_I;
    uint8_t *x = s->base[L] + (size_t)e * s->expert_bytes;
    bfp_mat gu = {s->fmt, 2 * I, D, x, x + s->piece[0]};
    bfp_mat dn = {s->fmt, D, I, x + s->piece[0] + s->piece[1], x + s->piece[0] + s->piece[1] + s->piece[2]};
    bfp_to_tt_tiles(&gu, dst);
    bfp_to_tt_tiles(&dn, dst + bfp_tt_bytes(s->fmt, 2 * I, D));
}
