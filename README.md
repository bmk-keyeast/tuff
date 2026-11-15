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

> **Status (before the card arrives):** done and tested: the memory map, the decode-time model, the CPU expert kernels
> (1-token and multi-token for MTP), the routed-expert loader, and a **real 48-layer QFN forward pass** that measures
> model-level accuracy and records expert routing. Not done: the device path (TT-NN), the adaptive cache in engine code,
> and the end-to-end engine. Every throughput number below is a **model prediction** fed by measured routing, not a
> measurement on the target host.

## Headline

Predicted decode, batch 1, 4k context, E5-2690 v4 with 64 GB, routed experts BFP4, dense weights on the card BFP8
(the format whose accuracy cost is measured below as negligible). Hit rate measured on a real routing trace that switches
from prose to code; MTP acceptance taken from Basalt's community benchmarks.

| | no MTP | MTP, prose | MTP, chat | MTP, structured output |
|---|---:|---:|---:|---:|
| **tuff (adaptive cache + MTP)** | **37 t/s** | **60 t/s** | **68 t/s** | **126 t/s** |
| static placement, Basalt's profile, no MTP | 27 t/s | | | |

How it gets there, each step measured on the real model on CI:

1. **BFP4 experts + BFP8 dense cost about as much as Basalt's 4-bit packs**: KLD 0.05–0.09, 94–95% same top-1 token.
2. **A static hot-expert placement breaks when the topic changes**: Basalt's profile keeps 67% of routed experts on the
   card for prose but only 33% for code, worse than a uniform pick (40%).
3. **An adaptive card cache fixes it**: copying up to 25 missed experts per token to the card (LRU eviction) holds ~90%
   hits across the prose→code switch, and plan.py charges those copies to DRAM and PCIe bandwidth.
4. **MTP verification amortizes the dense part**: 4 tokens touch 26 experts per layer, not 40, and the multi-token CPU kernel
   runs 4 tokens in 1.94× the time of one.

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
| p150a (32 GB) | attention (all 48 layers), router, shared expert, hyper-connection mixers, LM head, KV cache and DeltaNet state, and a cache of routed experts: **~9,800 of 24,576 (40%)** at BFP4 with dense weights in BFP8 |
| host DRAM | all routed experts (~68 GB in BFP4 with 128 GB of RAM, or the ~41 GB not on the card with 64 GB), computed in place by the CPU on a card miss; token embeddings; the MTP layer's experts |
| NVMe or DRAM | the N-gram table (~28.8 GB quantized, as in Strata): memory-mapped from NVMe with 64 GB of RAM, resident in RAM with 128 GB. Each token reads only a few rows, and the row indices are known as soon as the previous token is sampled, so they can be prefetched |

Per layer, the card runs attention, sends the hidden state to the host, both sides run their experts in parallel, and the
host sends its partial sum back: `t_token = t_dense + max(t_hot, t_cold + t_pcie)`. Experts that missed are copied to the
card in the background for the next token.

## Real forward pass: accuracy and routing

[`trace.py`](trace.py) runs the real model on a 16 GB GitHub runner without downloading the 360 GB checkpoint: it builds
QFN from the `transformers` source on the meta device, then streams one layer at a time from Hugging Face with HTTP range
requests, fetching only the experts that the tokens actually route to (~470 of 512 per layer, at ~560 MB/s). 1024 tokens of
wikitext and 1024 tokens of Python run through all 48 layers in ~30 min, once per weight format in the same batch.
The bf16 perplexity (wikitext 1.61, code 2.06) shows the streamed forward pass is correct. Full report:
[`results/trace.md`](results/trace.md); the routing trace is in [`results/routes.npz`](results/routes.npz).

**Accuracy** (KLD and top-1 agreement against bf16 on the same tokens):

| Format | Dense on card | Routed experts | KLD wikitext / code | Same top-1 wikitext / code |
|---|---|---|---:|---:|
| bfp4 | bf16 | BFP4 | 0.094 / 0.047 | 93.7% / 95.1% |
| **tuff** | **BFP8** (LM head BFP8) | **BFP4** | **0.087 / 0.051** | **93.9% / 95.3%** |
| tuff-fast | BFP4 (LM head BFP8) | BFP4 | 0.457 / 0.188 | 82.9% / 91.6% |

- BFP8 for the dense part is free: the error is the experts' alone.
- BFP4 for *every* dense projection is not: it would make decode ~55% faster on paper (58 → 166 t/s in the headline table)
  but costs 11 points of top-1 on prose. The next step is per-projection precision, as blackhole-llm found for other models
  (some projections are nearly free in BFP4, others are not).

**Routing** (share of the 10 routed experts per token and layer that a placement keeps on the card, 9,815 experts on card):

| Placement | wikitext | code |
|---|---:|---:|
| uniform | 39.9% | 39.9% |
| Basalt's shipped popularity profile | 67.1% | 32.9% |
| profile from the *other* text | 35.7% | 27.6% |
| profile from the same text (upper bound) | 97.0% | 93.9% |
| **adaptive LRU cache, ≤ 25 copies per token** | **93.8%** | **86.3%** |

41% of a token's experts were also used by the previous token in the same layer, which is why a small LRU cache recovers
almost all of the in-sample upper bound. On the prose→code sequence, the cache holds 89.7% hits (static profile: 50.0%).

**MTP.** For m tokens verified together, a layer reads the union of their experts once: 10.0, 16.1, 21.4, 26.1 unique
experts for m = 1, 2, 3, 4 (measured on the trace). [`moe_forward_n`](kernels/bfp.c) streams each cold expert once per
group of 4 tokens; on CI, 4 tokens take 1.94× the time of 1.

## What exists

| Path | What |
|---|---|
| [`plan.py`](plan.py) | QFN parameter accounting (checked against the published 125B / ~6B / 51B), card/host/NVMe memory map, decode-time model with cache promotions and MTP |
| [`kernels/bfp.c`](kernels/bfp.c) | BFP8/BFP4 pack/unpack, shuffle-free AVX2+FMA GEMV, SwiGLU expert, `moe_forward` (one OpenMP region over all expert panels of a layer), `moe_forward_n` (n tokens per weight read, for MTP) |
| [`loader.py`](loader.py) | Reads QFN's fused expert tensors from a local snapshot or over HTTP range requests, packs them bit-identically to the C packer, writes one memory-mappable file per layer (1.42 GB at BFP4, 68 GB for all 48), and measures real-weight accuracy |
| [`trace.py`](trace.py) | Streamed real forward pass: model-level accuracy per format, routing trace, cache and MTP simulation → [`results/trace.md`](results/trace.md) |
| [`kernels/test_bfp.c`](kernels/test_bfp.c) | Exact round-trip, SIMD = scalar = fp64 math, SQNR bounds at QFN expert shapes, fused layer = sum of experts, multi-token = per-token, Python packer = C packer (ASan/UBSan) |
| [`kernels/bench_bfp.c`](kernels/bench_bfp.c) | Cold-expert throughput at QFN shapes from a DRAM pool larger than the caches, 1–8 tokens |
| [`.github/workflows/`](.github/workflows) | `ci` (build, test, benchmark every push), `accuracy` (sampled experts), `trace` (full forward pass) |

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
This has not been measured on the E5 yet.

**Real QFN experts in TT block-float** ([`results/expert_accuracy.md`](results/expert_accuracy.md), 48 sampled experts):
BFP4 weight SQNR 18.7 dB, the same as a Gaussian matrix (19.0 dB), with no outlier columns and no layer standing out.
Sharing the exponent along TT's output dimension costs nothing against the input dimension.

```bash
python plan.py                                  # memory map + predicted t/s -> results/plan.md
make -C kernels test                            # correctness
make -C kernels bench_bfp
OMP_NUM_THREADS=14 ./kernels/bench_bfp 6 bfp4 8192 400 4   # 6 cold experts, 8 GB pool, 4 tokens per read
python loader.py convert /path/to/snapshot experts/ --fmt bfp4   # or SRC=hf to stream from Hugging Face
python trace.py --tokens 1024 --layers 48 && python trace.py analyze basalt.prof   # ~16 GB RAM; CI: trace workflow
```

## Roadmap

- [x] QFN memory map and decode-time model for the E5-2690 v4 and 8259CL hosts
- [x] CPU kernels for TT block-float experts at QFN shapes, fused MoE layer, tests, CI benchmark
- [x] Shuffle-free BFP decode: BFP4 2.1× faster per core (enough for the E5's DRAM bandwidth on paper)
- [x] Loader: QFN routed experts to BFP panels, bit-identical to the C packer; real-weight SQNR
- [x] Real 48-layer forward pass on CI: model-level KLD / top-1 per format, routing trace
- [x] Expert placement from routing traces: static profiles measured, adaptive LRU cache simulated
- [x] Multi-token expert kernel for MTP verification
- [ ] Per-projection precision for the dense part (which projections tolerate BFP4)
- [ ] Adaptive cache in engine code: miss queue, background PCIe copies, LRU on the card
- [ ] Measure on the real host: STREAM, `bench_bfp` with 14 threads, NUMA and hugepages
- [ ] Device path on the p150a with TT-NN: attention, shared and cached experts, hidden-state exchange; measure the real round trip
- [ ] MTP speculative decoding end to end; measure QFN's own acceptance rate
- [ ] Prefill: stream experts to the card in layer order instead of computing them on the CPU
- [ ] N-gram table on NVMe with prefetch
- [ ] AVX-512 VNNI kernels for the 8259CL
- [ ] OpenAI- and Anthropic-compatible server

## Host notes for the p150a

The card wants PCIe 5.0 x16; both host platforms are PCIe 3.0. The card still works, and this design keeps bus traffic to
activations and a capped number of expert copies, so the slower link costs latency rather than bandwidth. Also required:
an ATX 3.1 PSU with a 12V-2x6 connector (≥ 800 W suggested), a free slot next to the dual-slot card, and PCIe AER
reporting set to "OS First" in the BIOS.

## 한국어 요약

**Qwen3.8-Flash-Next 전용** 추론 엔진입니다. NVIDIA GPU용인 Strata·Basalt처럼 이 모델 하나만을 위해 만들지만,
대상 하드웨어는 Tenstorrent Blackhole p150a 한 장과 DDR4 제온 서버(E5-2690 v4, 이후 8259CL)입니다.
카드에는 어텐션, 공유 전문가, LM head와 전문가 캐시(약 40%)를 두고, 카드에 없는 전문가는 호스트 CPU가 DRAM에서 직접 계산합니다.
전문가는 Tenstorrent의 BFP4/BFP8 블록 포맷 그대로 저장해서 CPU와 카드가 재양자화 없이 같은 데이터를 씁니다.

**지금까지 한 것 (카드 도착 전 단계):**
- CPU 전문가 커널: 셔플 없는 BFP 디코딩으로 처음보다 2.1배 빠름, MTP용 다중 토큰 커널(4토큰을 1토큰의 1.94배 시간에)
- 로더: 360GB 체크포인트에서 필요한 전문가만 받아 BFP로 변환, C 패커와 바이트 단위로 동일
- **실제 QFN 48층 전체 forward를 GitHub Actions에서 실행** (레이어 단위 스트리밍, 30분). BF16 퍼플렉서티가 정상이라 실행이 맞음을 확인
  - 전문가 BFP4 + 카드 쪽 가중치 BFP8: KLD 0.05~0.09, top-1 일치 94~95% (Basalt 4비트 팩 수준). BFP8 dense는 손해가 거의 없음
  - dense까지 전부 BFP4로 하면 top-1이 83%로 떨어져서 그대로는 못 씀 → 프로젝션별로 정밀도를 다르게 줄 예정
- 실제 라우팅 기록 분석: Basalt의 고정 배치는 글에서 67%, 코드에서 33%만 카드에 적중. 적응형 캐시(놓친 전문가를 토큰당 25개까지 카드로 복사)는 주제가 바뀌어도 약 90% 적중

**예측 속도 (E5-2690 v4, 64GB):** MTP 없이 37 t/s, MTP 사용 시 일반 글 60, 대화 68, 구조화 출력 126 t/s.
실제 하드웨어 측정이 아니라 측정한 라우팅으로 계산한 예측입니다.

**다음 할 일:** dense 프로젝션별 정밀도 결정, 적응형 캐시를 엔진 코드로 구현, 카드가 오면 TT-NN 카드 경로와 실측.

## License

MIT. Independent student project, not affiliated with Tenstorrent, Qwen, Strata or Basalt.
