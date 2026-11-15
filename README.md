# tuff

**MoE expert-offloading inference engine for one Tenstorrent Blackhole p150a and a DDR4 Xeon host.**

![status](https://img.shields.io/badge/status-early%20development-orange)
![target](https://img.shields.io/badge/target-Blackhole%20p150a%20%2B%20Xeon%20E5--2690%20v4-6b46c1)
![ci](https://github.com/bmk-keyeast/tuff/actions/workflows/ci.yml/badge.svg)
![license](https://img.shields.io/badge/license-MIT-green)

Mixture-of-Experts models such as gpt-oss-120b, GLM-4.5-Air and Qwen3-235B-A22B are too large for the p150a's 32 GB, but each
token only touches a few experts. tuff keeps the dense part of the model and as many experts as fit on the card, and computes
every other expert **on the host CPU, in place, from DRAM**. Only the hidden-state vector crosses PCIe, never weights.

Sibling project: [blackhole-llm](https://github.com/bmk-keyeast/blackhole-llm) (dense-model analysis and BFP accuracy study).
Part of the KEYEAST club's Tenstorrent work (Changwon Science High School).

> **Status:** the placement planner and the CPU expert kernels work and are tested. The device path (TT-NN), model loading,
> and the end-to-end engine do not exist yet. All throughput numbers below are **model predictions**, not measurements on the target host.

## Target hardware

| | Now | Planned upgrade |
|---|---|---|
| Accelerator | Blackhole p150a: 32 GB GDDR6 @ 512 GB/s | same |
| CPU | Xeon E5-2690 v4 (Broadwell-EP), 14C/28T, AVX2+FMA | Xeon Platinum 8259CL (Cascade Lake, LGA3647), 24C, AVX-512 VNNI |
| Memory | DDR4-2133 ECC RDIMM 4 × 16 GB (4 channels, 68 GB/s peak), expandable to 128 GB | 6 channels DDR4-2933 (141 GB/s peak) |
| Host ↔ card | PCIe 3.0 x16 (15.75 GB/s) | PCIe 3.0 x16 |

PCIe 3.0 is the weak link on both platforms: it is 4-9× slower than the host's own memory. That fact drives the design.

## Design

```
                 per MoE layer, batch-1 decode
  p150a  | attention + router + shared/dense MLP + hot experts |----+----| combine |
         |                                                    | x  |  y |         |
  PCIe   |                                    hidden state -> v    ^ <- partial output
  host   |                                         cold experts (AVX2, from DRAM)  |
```

1. **One weight format on both sides.** Experts are stored in Tenstorrent's native block-float formats (`bfp4_b`, `bfp8_b`:
   16 values share one 8-bit exponent, the same grouping as a TT tile face row). The CPU kernels compute directly on these
   blocks, so moving an expert between host RAM and the card is a block permutation, never a re-quantization.
   The quantizer reproduces blackhole-llm's emulator exactly (43.1 dB / 19.0 dB SQNR on Gaussian weights for BFP8 / BFP4).
2. **Compute cold experts where they live.** Streaming a cold expert to the card costs PCIe bandwidth (~13 GB/s). Computing it on
   the CPU costs DRAM bandwidth (~48 GB/s achievable on 4-channel DDR4-2133), and only `2 × hidden × 2 B` crosses the bus.
3. **Overlap.** Card (hot experts) and CPU (cold experts) work on the same layer at the same time:
   `t_token = t_dense + max(t_hot, t_cold + t_pcie)`.

## Predicted decode speed (batch 1, 4k context)

From [`plan.py`](plan.py), with experts in BFP4, uniform routing (no popularity skew yet), and efficiencies of 43% for the card
(calibrated in blackhole-llm), 70% for DRAM and 80% for PCIe, plus a guessed 30 µs PCIe round trip per layer.
Full tables for all host configurations: [`results/plan.md`](results/plan.md).

| Model (total params) | Experts on card | CPU-only | Stream experts to card | **tuff hybrid** | E5-2690 v4 64 GB | 8259CL |
|---|---:|---:|---:|---:|:-:|---:|
| gpt-oss-120b (117B) | 47% | 12.5 t/s | 10.3 t/s | **31.6 t/s** | fits | 49.7 t/s |
| GLM-4.5-Air (107B) | 44% | 4.4 | 5.1 | **13.2** | fits | 18.2 |
| Qwen3-Next-80B-A3B (79B) | 70% | 18.6 | 32.6 | **68.5** | fits | 84.3 |
| Qwen3-235B-A22B (235B) | 19% | 2.9 | 1.8 | **5.7** | needs 128 GB | 9.4 |
| Qwen3-30B-A3B, gpt-oss-20b, Mixtral-8x7B, DeepSeek-V2-Lite | 100% | | | entirely on the card | | |

On the E5 host the hybrid is predicted to be **2.5×** faster than CPU-only inference and **3×** faster than streaming experts
over PCIe for gpt-oss-120b. Every number depends on the efficiency knobs, so the first hardware milestone is to replace them with
measurements.

## What exists

| Path | What |
|---|---|
| [`plan.py`](plan.py) | Model zoo (8 MoE models, from HF `config.json`), host/card memory budgets, expert placement, decode-time model. Self-checked against published parameter counts |
| [`kernels/bfp.c`](kernels/bfp.c) | BFP8/BFP4 pack/unpack, scalar reference GEMV, AVX2+FMA GEMV, SwiGLU expert, and `moe_forward`: one OpenMP region over all (expert, 16-row panel) pairs of a layer |
| [`kernels/test_bfp.c`](kernels/test_bfp.c) | Exact round-trip, SIMD = scalar = fp64 math, SQNR bounds, fused MoE layer = sum of experts (built with ASan/UBSan) |
| [`kernels/bench_bfp.c`](kernels/bench_bfp.c) | Cold-expert throughput from a DRAM pool larger than the caches, for each model's expert shape |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | Build, test and benchmark on every push (results in the job summary) |

```bash
python plan.py                        # placement + predicted t/s -> results/plan.md
make -C kernels test                  # correctness
make -C kernels bench_bfp && OMP_NUM_THREADS=14 ./kernels/bench_bfp gpt-oss-120b bfp4 8192
```

## Roadmap

- [x] Planner and decode-time model for the E5-2690 v4 / 8259CL hosts
- [x] CPU kernels for TT block-float experts (AVX2), fused MoE layer, tests, CI benchmark
- [ ] Measure on the real host: STREAM bandwidth, `bench_bfp` with 14 threads, NUMA/hugepage effects
- [ ] Model loader: safetensors → BFP panels. gpt-oss ships MXFP4, which shares exponents along the input dim (blocks of 32)
      while TT blocks share along the output dim (blocks of 16), so conversion is lossy and its accuracy cost must be measured
- [ ] CPU-only reference engine (attention + router + experts) matching Hugging Face logits on a small MoE
- [ ] Device path on the p150a with TT-NN: attention, hot experts, hidden-state exchange; measure the real PCIe round trip
- [ ] Popularity-aware placement from routing traces (uniform routing is the worst case)
- [ ] AVX-512 VNNI kernels and 6-channel tuning for the 8259CL upgrade
- [ ] OpenAI-compatible HTTP server

## Host notes for the p150a

The card wants PCIe 5.0 x16; both host platforms are PCIe 3.0. The card still works, and this design keeps the bus traffic to
activations, so the slower link costs latency rather than bandwidth. Also required: an ATX 3.1 PSU with a 12V-2x6 connector
(≥ 800 W suggested), a free slot next to the dual-slot card, and PCIe AER reporting set to "OS First" in the BIOS.

## 한국어 요약

Blackhole p150a 한 장과 DDR4 제온 서버(E5-2690 v4, 이후 8259CL)를 위한 MoE 전문가 오프로딩 추론 엔진입니다.
카드에는 어텐션 같은 밀집 부분과 들어가는 만큼의 전문가를 두고, 나머지 전문가는 호스트 CPU가 DRAM에서 직접 계산합니다.
PCIe 3.0이 호스트 메모리보다 4~9배 느리기 때문에 가중치는 버스로 옮기지 않고 은닉 상태 벡터만 주고받습니다.
전문가 가중치는 Tenstorrent의 BFP4/BFP8 블록 포맷 그대로 저장해서, CPU와 카드가 같은 데이터를 재양자화 없이 함께 씁니다.
현재는 배치 계획기와 CPU 커널(테스트, CI 벤치마크 포함)까지 구현했고, 카드 쪽 경로와 전체 엔진은 앞으로 만들 예정입니다.

## License

MIT. Independent student project, not affiliated with Tenstorrent.
