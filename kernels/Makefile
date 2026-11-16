# -march=haswell = AVX2+FMA, the E5-2690 v4 (Broadwell) baseline; Cascade Lake runs it too.
CC ?= gcc
CFLAGS ?= -O3 -march=haswell -fopenmp -Wall -Wextra
LDLIBS = -lm

all: test_bfp test_cache test_store bench_bfp

test_bfp: test_bfp.c bfp.c tt_tile.c bfp.h
	$(CC) $(CFLAGS) -O1 -g -fsanitize=address,undefined test_bfp.c bfp.c tt_tile.c -o $@ $(LDLIBS)

bench_bfp: bench_bfp.c bfp.c bfp.h
	$(CC) $(CFLAGS) bench_bfp.c bfp.c -o $@ $(LDLIBS)

test_cache: test_cache.c cache.c cache.h
	$(CC) $(CFLAGS) -O1 -g -fsanitize=address,undefined test_cache.c cache.c -o $@ $(LDLIBS)

test_store: test_store.c store.c store.h bfp.c bfp.h
	$(CC) $(CFLAGS) -O1 -g -fsanitize=address,undefined test_store.c store.c bfp.c -o $@ $(LDLIBS)

# NVIDIA GPU (RTX 4060 Ti = sm_89). Not part of `make all`: CI only compiles it; run ./test_cuda on the real box.
NVCC ?= nvcc
CUDA_ARCH ?= sm_89
bfp_host.o: bfp.c bfp.h
	$(CC) $(CFLAGS) -c bfp.c -o $@

test_cuda: test_cuda.cu bfp_cuda.cu bfp_cuda.h bfp_host.o
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -Xcompiler -fopenmp test_cuda.cu bfp_cuda.cu bfp_host.o -o $@ -lgomp -lm

test: test_bfp test_cache test_store
	OMP_NUM_THREADS=4 ./test_bfp
	./test_cache

clean:
	rm -f test_bfp test_cache test_store bench_bfp test_cuda bfp_host.o

.PHONY: all test clean
