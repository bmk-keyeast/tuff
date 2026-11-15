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
  sides**. Moving an expert between host RAM and the card is a byte permutation plus a table recode of each mantissa
  (sign-magnitude on the card, offset binary on the host), never a re-quantization.

Sibling project: [blackhole-llm](https://github.com/bmk-keyeast/blackhole-llm) (BFP accuracy study across 6 models).
Part of the KEYEAST club's Tenstorrent work at Changwon Science High School.

> **Status:** the QFN memory map, the decode-time model, the CPU expert kernels and the routed-expert loader exist and are
> tested. The device path (TT-NN), the loader for the dense weights and the end-to-end engine do not exist yet. Every throughput number below is a
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
| [`kernels/bfp.c`](kernels/bfp.c) | BFP8/BFP4 pack/unpack, shuffle-free AVX2+FMA GEMV, SwiGLU expert, `moe_forward`: one OpenMP region over all (expert, 16-row panel) pairs of a layer |
| [`loader.py`](loader.py) | Reads QFN's fused expert tensors from a local snapshot or over HTTP range requests, packs them bit-identically to the C packer, writes one memory-mappable file per layer (1.42 GB at BFP4, 68 GB for all 48), and measures real-weight accuracy |
| [`kernels/test_bfp.c`](kernels/test_bfp.c) | Exact round-trip, SIMD = scalar = fp64 math, SQNR bounds at QFN expert shapes, fused layer = sum of experts (ASan/UBSan) |
| [`kernels/bench_bfp.c`](kernels/bench_bfp.c) | Cold-expert throughput at QFN shapes from a DRAM pool larger than the caches |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | Build, test and benchmark on every push (job summary) |

**CPU kernel benchmark.** CI builds the phase-0 kernel (`a5a7b99`) next to the current one and runs both on the same runner,
because runner CPUs differ between runs. Numbers below: GitHub runner, 4 vCPU, 2 GB expert pool, 10 cold experts per layer; not the
target host.

| Format | Threads | Phase-0 kernel | Current kernel | Speed-up |
|---|---:|---:|---:|---:|
| BFP4 | 1 | 4.8 GB/s | 10.0 GB/s | 2.1× |
| BFP4 | 4 | 9.6 GB/s | 20.6 GB/s | 2.1× |
| BFP8 | 1 | 10.0 GB/s | 15.4 GB/s | 1.5× |
| BFP8 | 4 | 20.2 GB/s | 35.2 GB/s | 1.7× |

The phase-0 kernel decoded each block with `pshufb` and `pmovsxbd`, and both use only port 5 on Broadwell/Haswell. Now the
host keeps the same blocks in a different byte order, as offset-binary fields in row-major dwords. One 256-bit load then covers
8 rows × 8 columns (BFP4), and each column decodes with shift + and + int→float, with no shuffles. Block scales are built from
the exponent bits once per panel, and the offset is removed once per row. At 1 thread, BFP4 now does ~10 GB/s per core. 14
Broadwell cores would be ~2–3× above the E5's ~48 GB/s of usable DRAM bandwidth, so cold experts should be memory-bound there.
This has not been measured on the E5 yet. Next steps are AVX-512 VNNI for the 8259CL and a measurement on the real host.

**Real QFN experts in TT block-float.** [`accuracy.yml`](.github/workflows/accuracy.yml) reads 48 random routed experts
(8 from each of layers 0, 9, 19, 28, 38 and 47) straight from the 360 GB BF16 checkpoint, ~10 MB each, and compares them with
their BFP versions. Full table: [`results/expert_accuracy.md`](results/expert_accuracy.md).

| | BFP4 | BFP4 gate/up + BFP8 down | BFP8 |
|---|---:|---:|---:|
| Weight SQNR (Gaussian matrix: 19.0 / 43.1 dB) | 18.7 dB | – | 42.4 dB |
| Expert output SQNR, mean (worst of 48) | 14.2 dB (13.1) | 16.1 dB (15.3) | 36.6 dB (35.8) |
| Expert bytes | 2.76 MB | 3.58 MB (+30%) | 5.22 MB |

- QFN's expert weights quantize like Gaussian noise. There are no outlier columns, and no layer stands out; the last layers
  are only 0.2–0.4 dB worse.
- Where the exponent is shared makes no difference: sharing it along the input dimension instead of TT's output dimension
  gives the same 14.2 dB. TT's native layout costs nothing for these weights.
- Keeping `down_proj` in BFP8 buys 1.9 dB of expert-output SQNR for 30% more bytes. Whether that is worth it depends on
  model-level numbers (KLD, top-1 agreement), and those need the full forward pass. Expert inputs here are Gaussian, not
  real hidden states.

```bash
python loader.py accuracy                       # real-weight BFP accuracy -> results/expert_accuracy.md
python loader.py convert /path/to/snapshot experts/ --fmt bfp4   # or SRC=hf to stream from Hugging Face
python plan.py                                  # memory map + predicted t/s -> results/plan.md
make -C kernels test                            # correctness
make -C kernels bench_bfp
OMP_NUM_THREADS=14 ./kernels/bench_bfp 6 bfp4 8192   # 6 cold experts per layer, 8 GB pool
```

## Roadmap

- [x] QFN memory map and decode-time model for the E5-2690 v4 and 8259CL hosts
- [x] CPU kernels for TT block-float experts at QFN shapes, fused MoE layer, tests, CI benchmark
- [x] Shuffle-free BFP decode: BFP4 2.1× faster per core (enough for the E5's DRAM bandwidth on paper)
- [ ] Measure on the real host: STREAM, `bench_bfp` with 14 threads, NUMA and hugepages
- [x] Loader: QFN routed experts from `Qwen/Qwen3.8-Flash-Next` (BF16) to BFP panels, bit-identical to the C packer;
      real-weight SQNR of sampled experts
- [ ] Model-level accuracy (KLD, top-1 agreement) the way Basalt reports its packs; needs the CPU reference forward below
- [ ] Loader for the dense part (attention, shared expert, LM head) and the MTP layer's experts
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
CPU 커널은 셔플 명령 없이 압축을 풀도록 다시 짜서, 같은 CI 머신에서 BFP4가 처음 버전보다 2.1배 빨라졌습니다.
로더(`loader.py`)는 360GB 체크포인트 전체를 받지 않고 필요한 전문가만 HTTP 범위 요청으로 읽어 BFP로 변환하며, C 패커와 바이트 단위로 같은 결과를 냅니다.
실제 QFN 전문가 48개를 측정한 결과, BFP4 가중치 SQNR은 18.7dB로 가우시안(19.0dB)과 거의 같았고, 지수 공유 방향(TT 방식)에 따른 손해도 없었습니다.

## License

MIT. Independent student project, not affiliated with Tenstorrent, Qwen, Strata or Basalt.
