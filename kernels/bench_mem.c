/* Host DRAM read bandwidth, the number plan.py's CPU_EFF stands in for: all threads sum a buffer far larger than the
 * caches with 256-bit loads (a read-only STREAM). ./bench_mem [MB=4096] [passes=10] */
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

int main(int argc, char **argv) {
    size_t mb = argc > 1 ? atoi(argv[1]) : 4096, n = mb * 1000000 / sizeof(__m256);
    int passes = argc > 2 ? atoi(argv[2]) : 10;
    __m256 *buf = aligned_alloc(64, n * sizeof(__m256));
    if (!buf) { perror("alloc"); return 1; }
#pragma omp parallel for schedule(static)          /* first touch from the threads that read it (NUMA) */
    for (size_t i = 0; i < n; i++) buf[i] = _mm256_set1_ps(1e-9f * (float)(i & 1023));
    double best = 1e30;
    float sink = 0;
    for (int p = 0; p < passes; p++) {
        double t = now();
        float s = 0;
#pragma omp parallel reduction(+ : s)
        {
            __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
#pragma omp for schedule(static)
            for (size_t i = 0; i < n; i += 4) {
                a0 = _mm256_add_ps(a0, buf[i]); a1 = _mm256_add_ps(a1, buf[i + 1]);
                a2 = _mm256_add_ps(a2, buf[i + 2]); a3 = _mm256_add_ps(a3, buf[i + 3]);
            }
            __m256 v = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
            float f[8];
            _mm256_storeu_ps(f, v);
            for (int k = 0; k < 8; k++) s += f[k];
        }
        sink += s;
        t = now() - t;
        if (t < best) best = t;
    }
    printf("read %zu MB: %.1f GB/s (best of %d)%s\n", mb, n * sizeof(__m256) / best / 1e9, passes, sink == 42 ? "!" : "");
    free(buf);
    return 0;
}
