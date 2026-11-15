# -march=haswell = AVX2+FMA, the E5-2690 v4 (Broadwell) baseline; Cascade Lake runs it too.
CC ?= gcc
CFLAGS ?= -O3 -march=haswell -fopenmp -Wall -Wextra
LDLIBS = -lm

all: test_bfp bench_bfp

test_bfp: test_bfp.c bfp.c bfp.h
	$(CC) $(CFLAGS) -O1 -g -fsanitize=address,undefined test_bfp.c bfp.c -o $@ $(LDLIBS)

bench_bfp: bench_bfp.c bfp.c bfp.h
	$(CC) $(CFLAGS) bench_bfp.c bfp.c -o $@ $(LDLIBS)

test: test_bfp
	OMP_NUM_THREADS=4 ./test_bfp

clean:
	rm -f test_bfp bench_bfp

.PHONY: all test clean
