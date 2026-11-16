#!/bin/sh
# Measure the real host once it is built (E5-2690 v4 now, 8259CL later) and write results/host_$(hostname).md.
# Replaces plan.py's guesses: CPU_EFF (DRAM read efficiency) and the cold-expert speed per core / per socket.
# Read-only: changes no system setting. Run from the repo root:  sh bench/host.sh
set -eu
make -s -C kernels bench_bfp
cc -O3 -march=native -fopenmp kernels/bench_mem.c -o kernels/bench_mem
out=results/host_$(hostname).md
cores=$(lscpu -p=core,socket | grep -v '^#' | sort -u | wc -l)
{
  echo "## Host benchmark: $(hostname), $(date -u +%Y-%m-%d)"
  echo
  echo '```'
  lscpu | grep -E "Model name|^Socket|^Core|^Thread|^NUMA node|L3"
  free -g | head -2
  echo "transparent hugepages: $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo n/a)"
  command -v numactl > /dev/null && numactl -H | head -3
  dmidecode -t memory 2>/dev/null | grep -E "^\s+(Speed|Configured Memory Speed|Size):" | sort | uniq -c || true
  echo '```'
  echo
  echo "### DRAM read bandwidth (plan.py: CPU_EFF = this / peak; E5-2690 v4 peak = 68.3 GB/s)"
  echo
  echo '```'
  for t in 1 $((cores / 2)) $cores $((cores * 2)); do
    printf "%3d threads: " $t; OMP_NUM_THREADS=$t OMP_PROC_BIND=spread OMP_PLACES=cores ./kernels/bench_mem 4096 5
  done
  echo '```'
  echo
  echo "### Cold experts from an 8 GB pool (k = cold experts per layer; bench_bfp prints GB/s and the 48-layer time)"
  echo
  echo '```'
  for t in 1 $((cores / 2)) $cores; do
    for f in bfp4 bfp8; do
      echo "t=$t $f k=6:  $(OMP_NUM_THREADS=$t OMP_PROC_BIND=spread OMP_PLACES=cores ./kernels/bench_bfp 6 $f 8192 200)"
    done
  done
  for n in 1 2 3 4; do      # MTP verification: n tokens per weight read
    echo "t=$cores bfp4 k=12 tokens=$n: $(OMP_NUM_THREADS=$cores OMP_PROC_BIND=spread OMP_PLACES=cores ./kernels/bench_bfp 12 bfp4 8192 100 $n)"
  done
  echo '```'
  if command -v nvidia-smi > /dev/null && command -v nvcc > /dev/null; then
    echo
    echo "### NVIDIA GPU (plan.py: GPU_EFF = measured GB/s / 288 for the 4060 Ti)"
    echo
    echo '```'
    nvidia-smi --query-gpu=name,pcie.link.gen.current,pcie.link.width.current,memory.total --format=csv
    make -s -C kernels test_cuda && ./kernels/test_cuda
    echo '```'
  fi
} | tee "$out"
echo "wrote $out"
