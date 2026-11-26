# The C++ engine. `make` builds it on the CPU reference backend (what CI checks); on the box with tt-metal:
#   make TUFF_TTNN=1 TT_METAL_HOME=/path/to/tt-metal
CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -march=haswell -fopenmp -Wall -Wextra
K = ../kernels
SRC = main.cpp qfn.cpp cpu_ops.cpp
CSRC = $(K)/cache.c $(K)/store.c $(K)/bfp.c $(K)/tt_tile.c

ifdef TUFF_TTNN
SRC += ttnn_ops.cpp
CXXFLAGS += -DTUFF_TTNN -std=c++20 -I$(TT_METAL_HOME) -I$(TT_METAL_HOME)/ttnn/cpp -I$(TT_METAL_HOME)/tt_metal/api
LDLIBS += -L$(TT_METAL_HOME)/build/lib -lttnn -ltt_metal
endif

tuff_engine: $(SRC) $(CSRC) ops.h qfn.h
	$(CC) -O3 -march=haswell -fopenmp -c $(CSRC)
	$(CXX) $(CXXFLAGS) $(SRC) cache.o store.o bfp.o tt_tile.o -o $@ $(LDLIBS) -lm

clean:
	rm -f tuff_engine *.o

.PHONY: clean
