# The C++ engine. `make` builds it on the CPU reference backend (what CI runs); on the box:
#   make TUFF_TTNN=1 TT_METAL_HOME=/path/to/tt-metal      the card through tt-metal
#   make TUFF_CUDA=1                                       the 4060 Ti (nvcc, sm_89), for `run --gpu`
CXX ?= g++
NVCC ?= nvcc
CUDA_ARCH ?= sm_89
CXXFLAGS ?= -O2 -std=c++17 -march=haswell -fopenmp -Wall -Wextra
K = ../kernels
SRC = main.cpp qfn.cpp weights.cpp cpu_ops.cpp
COBJ = cache.o store.o bfp.o tt_tile.o
HDR = ops.h qfn.h json.h

ifdef TUFF_TTNN
SRC += ttnn_ops.cpp
CXXFLAGS += -DTUFF_TTNN -std=c++20 -I$(TT_METAL_HOME) -I$(TT_METAL_HOME)/ttnn/cpp -I$(TT_METAL_HOME)/tt_metal/api
LDLIBS += -L$(TT_METAL_HOME)/build/lib -lttnn -ltt_metal
endif
ifdef TUFF_CUDA
COBJ += cuda_ops.o bfp_cuda.o
CXXFLAGS += -DTUFF_CUDA
CUDA_HOME ?= /usr/local/cuda
LDLIBS += -L$(CUDA_HOME)/lib64 -lcudart
endif

tuff_engine: $(SRC) $(COBJ) $(HDR)
	$(CXX) $(CXXFLAGS) $(SRC) $(COBJ) -o $@ $(LDLIBS) -lm

%.o: $(K)/%.c
	$(CC) -O3 -march=haswell -fopenmp -c $< -o $@

cuda_ops.o: cuda_ops.cu ops.h
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -std=c++17 -DTUFF_CUDA -c cuda_ops.cu -o $@

bfp_cuda.o: $(K)/bfp_cuda.cu $(K)/bfp_cuda.h
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -c $(K)/bfp_cuda.cu -o $@

clean:
	rm -f tuff_engine *.o

.PHONY: clean
