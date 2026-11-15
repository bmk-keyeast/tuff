# tuff

**A Qwen3.8-Flash-Next inference engine for one Tenstorrent Blackhole p150a and a DDR4 Xeon host.**

![status](https://img.shields.io/badge/status-early%20development-orange)
![model](https://img.shields.io/badge/model-Qwen3.8--Flash--Next-615ced)
![target](https://img.shields.io/badge/target-Blackhole%20p150a%20%2B%20Xeon%20E5--2690%20v4-6b46c1)
![ci](https://github.com/bmk-keyeast/tuff/actions/workflows/ci.yml/badge.svg)
![license](https://img.shields.io/badge/license-MIT-green)

[Strata](https://github.com/Niko1221/Strata) and its fork [Basalt](https://github.com/jesdga95/basalt) run
Qwen3.8-Flash-Next (QFN) on NVIDIA GPUs, keeping hot experts in VRAM and the rest in system RAM. tuff targets the same
model on different hardware: a Tenstorrent Blackhole p150a and an older Xeon server. Like those engines it is built for
this one model, and it is shaped by two facts about the hardware:

- the host's PCIe 3.0 link is 4 to 9 times slower than its own DRAM, so **cold experts are computed on the CPU where they
  live**, and only hidden-state vectors cross the bus;
- the card computes natively in block-float formats, so **experts are stored in Tenstorrent's `bfp4_b`/`bfp8_b` on both
  sides**. Moving an expert between host RAM and the card is a block permutation, never a re-quantization.

Sibling project: [blackhole-llm](https://github.com/bmk-keyeast/blackhole-llm) (BFP accuracy study across 6 models).
Part of the KEYEAST club's Tenstorrent work at Changwon Science High School.

> **Status:** the QFN memory map, the decode-time model and the CPU expert kernels exist and are tested. The device path
> (TT-NN), the `.gguf`/safetensors loader and the end-to-end engine do not exist yet. Every throughput number below is a
> **model prediction**, not a measurement on the target host.

## The model

From `Qwen/Qwen3.8-Flash-Next` `config.json` (`qwen4_exp_text`):

| | |
|---|---|
| Layers | 48: 36 Gated DeltaNet (linear attention) + 12 full attention (24 heads, 2 KV heads, head dim 256, sparse indexer), 3:1 |
| MoE | every layer: 512 routed experts, top-10, SwiGLU 2560 → 640 → 2560, plus one shared expert |
| Residual | hyper-connections, 4 streams |
| Embeddings | 248,320 vocab, plus a 20M-row N-gram embedding table (51.2B params) |
| Drafting | 1 MTP layer |
| Size (our count) | 125.7B main model (24,576 routed experts × 4.92M params), ~7.3B active per token including embedding and LM head |

## Target hardware

| | Now | Planned |
|---|---|---|
| Card | Blackhole p150a, 32 GB GDDR6 @ 512 GB/s | same |
| CPU | Xeon E5-2690 v4 (Broadwell-EP), 14C/28T, AVX2+FMA | Xeon Platinum 8259CL (LGA3647), 24C, AVX-512 VNNI |
| Memory | DDR4-2133 ECC RDIMM 4 × 16 GB, 4 channels (68 GB/s peak), up to 128 GB | 6 channels DDR4-2933 (141 GB/s peak) |
| Link | PCIe 3.0 x16 (15.75 GB/s) | PCIe 3.0 x16 |

## Placement

| Where | What |
|---|---|
| p150a (32 GB) | attention (all 48 layers), router, shared expert, hyper-connection mixers, LM head, KV cache and DeltaNet state, and the hot routed experts: **~9,800 of 24,576 (40%)** at BFP4 with dense weights in BFP8 |
| host DRAM | the other ~14,800 routed experts (~41 GB in BFP4), computed in place by the CPU; token embeddings; the MTP layer's experts |
| NVMe or DRAM | the N-gram table (~28.8 GB quantized, as in Strata): memory-mapped from NVMe with 64 GB of RAM, resident in RAM with 128 GB. Each token reads only a few rows, and the row indices are known as soon as the previous token is sampled, so they can be prefetched |

Per layer, the card runs attention, sends the hidden state to the host, both sides run their experts in parallel, and the
host sends its partial sum back: `t_token = t_dense + max(t_hot, t_cold + t_pcie)`.

## Predicted decode speed (batch 1, 4k context, experts BFP4)

From [`plan.py`](plan.py). Efficiency knobs: card 43% of peak (calibrated in blackhole-llm), DRAM 70%, PCIe 80%, a guessed
30 µs host↔card round trip per layer. "Hit rate" is the share of the 10 activated experts found on the card: uniform routing
gives the 40% that fit, and a measured popularity profile (Basalt ships one) raises it. MTP drafting is not included.
Full output: [`results/plan.md`](results/plan.md).

| Host | Dense on card | Uniform routing | 60% hits | 80% hits |
|---|---|---:|---:|---:|
| E5-2690 v4, 64 GB | BFP8 | 25.4 t/s | 29.5 | 35.2 |
| E5-2690 v4, 64 GB | BFP4 | 34.7 t/s | 41.2 | 53.5 |
| 8259CL, 6 × 32 GB | BFP8 | 32.4 t/s | 35.4 | 38.2 |
| 8259CL, 6 × 32 GB | BFP4 | 48.2 t/s | 54.0 | 60.7 |

What the model says to work on first:
1. **The card's dense part dominates.** At BFP8, attention, the shared expert and the LM head take 21 ms of a 39 ms token on
   the E5, more than the CPU's cold experts (18 ms). BFP4 attention halves it, and blackhole-llm measured which projections
   tolerate BFP4 (WO almost free, LM head the most sensitive). The card's real efficiency on these shapes is the biggest
   unknown.
2. **Expert popularity matters more than CPU speed.** Going from a uniform 40% to 80% card hits is worth more than the
   8259CL upgrade.
3. **64 GB of RAM is enough** (cold experts ~43 GB). 128 GB moves the N-gram table off NVMe.

## What exists

| Path | What |
|---|---|
| [`plan.py`](plan.py) | QFN parameter accounting (checked against the published 125B / ~6B / 51B), card/host/NVMe memory map, decode-time model |
| [`kernels/bfp.c`](kernels/bfp.c) | BFP8/BFP4 pack/unpack, AVX2+FMA GEMV, SwiGLU expert, `moe_forward`: one OpenMP region over all (expert, 16-row panel) pairs of a layer |
| [`kernels/test_bfp.c`](kernels/test_bfp.c) | Exact round-trip, SIMD = scalar = fp64 math, SQNR bounds at QFN expert shapes, fused layer = sum of experts (ASan/UBSan) |
| [`kernels/bench_bfp.c`](kernels/bench_bfp.c) | Cold-expert throughput at QFN shapes from a DRAM pool larger than the caches |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | Build, test and benchmark on every push (job summary) |

**First CI numbers** (GitHub runner, 4 vCPU, 2 GB expert pool; not the target host):

| Format | Cold experts / layer | Layer time | Achieved | CPU expert time per token |
|---|---:|---:|---:|---:|
| BFP4 | 10 | 1562 µs | 17.7 GB/s | 75.0 ms |
| BFP4 | 6 | 925 µs | 17.9 GB/s | 44.4 ms |
| BFP8 | 10 | 1381 µs | 37.8 GB/s | 66.3 ms |
| BFP8 | 6 | 811 µs | 38.6 GB/s | 38.9 ms |

BFP8 moves twice the bytes and still finishes first, so **the BFP4 kernel is compute-bound** (nibble unpack + int→float per
block), not memory-bound. Per vCPU it decodes ~4.4 GB/s, which on 14 Broadwell cores lands right at the ~48 GB/s DRAM
limit, with no margin. Making BFP4 memory-bound on the E5 is the next kernel task (two blocks per 256-bit unpack, int8 activations
with `vpmaddubsw`, and VNNI on the 8259CL).

```bash
python plan.py                                  # memory map + predicted t/s -> results/plan.md
make -C kernels test                            # correctness
make -C kernels bench_bfp
OMP_NUM_THREADS=14 ./kernels/bench_bfp 6 bfp4 8192   # 6 cold experts per layer, 8 GB pool
```

## Roadmap

- [x] QFN memory map and decode-time model for the E5-2690 v4 and 8259CL hosts
- [x] CPU kernels for TT block-float experts at QFN shapes, fused MoE layer, tests, CI benchmark
- [ ] Make the BFP4 kernel memory-bound (it is compute-bound on CI: 17.7 GB/s vs 37.8 GB/s for BFP8)
- [ ] Measure on the real host: STREAM, `bench_bfp` with 14 threads, NUMA and hugepages
- [ ] Loader: QFN from `Qwen/Qwen3.8-Flash-Next` (BF16) to BFP panels; measure BFP4 expert accuracy (KLD, top-1 agreement) the
      way Basalt reports its packs
- [ ] CPU reference forward for one QFN layer (DeltaNet, full attention with indexer, hyper-connections, MoE) matching
      Hugging Face outputs
- [ ] Device path on the p150a with TT-NN: attention, shared and hot experts, hidden-state exchange; measure the real round trip
- [ ] Expert popularity profile from routing traces → card placement
- [ ] MTP speculative decoding
- [ ] N-gram table on NVMe with prefetch
- [ ] AVX-512 VNNI kernels for the 8259CL
- [ ] OpenAI- and Anthropic-compatible server

## Host notes for the p150a

The card wants PCIe 5.0 x16; both host platforms are PCIe 3.0. The card still works, and this design keeps bus traffic to
activations, so the slower link costs latency rather than bandwidth. Also required: an ATX 3.1 PSU with a 12V-2x6
connector (≥ 800 W suggested), a free slot next to the dual-slot card, and PCIe AER reporting set to "OS First" in the BIOS.

## 한국어 요약

**Qwen3.8-Flash-Next 전용** 추론 엔진입니다. NVIDIA GPU용인 Strata와 Basalt처럼 이 모델 하나만을 위해 만들지만,
대상 하드웨어는 Tenstorrent Blackhole p150a 한 장과 DDR4 제온 서버(E5-2690 v4, 이후 8259CL)입니다.
카드에는 어텐션, 공유 전문가, LM head와 들어가는 만큼의 전문가(약 40%)를 두고, 나머지 전문가는 호스트 CPU가 DRAM에서 직접 계산합니다.
PCIe 3.0이 호스트 메모리보다 훨씬 느리기 때문에 가중치는 버스로 옮기지 않고 은닉 상태 벡터만 주고받습니다.
전문가 가중치는 Tenstorrent의 BFP4/BFP8 블록 포맷 그대로 저장해서, CPU와 카드가 재양자화 없이 같은 데이터를 씁니다.
현재는 메모리 배치 계산, 속도 예측 모델, CPU 전문가 커널(테스트와 CI 벤치마크 포함)까지 구현했습니다.

## License

MIT. Independent student project, not affiliated with Tenstorrent, Qwen, Strata or Basalt.
