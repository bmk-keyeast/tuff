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
  sides**. The host packer rounds exactly like tt-metal (ties to even, its exponent rule), so moving an expert from host
  RAM to the card is a byte permutation plus a table recode of each mantissa (offset binary on the host, sign-magnitude
  on the card), never a re-quantization: [`kernels/tt_tile.c`](kernels/tt_tile.c) produces tt-metal's tiles byte for
  byte.

Sibling project: [blackhole-llm](https://github.com/bmk-keyeast/blackhole-llm) (BFP accuracy study across 6 models).
Part of the KEYEAST club's Tenstorrent work at Changwon Science High School.

> **Status (before the card arrives):** done and tested: the memory map, the decode-time model, the CPU expert kernels
> (1-token and multi-token for MTP), the routed-expert loader, a **real 48-layer QFN forward pass** that measures
> model-level accuracy and records expert routing, the adaptive expert cache in C (replayed on the real routing), the
> memory-mapped expert store, and the per-group dense precision. Written but not run yet (no GPU in CI): CUDA kernels
> for an optional RTX 4060 Ti. The card code is written and checked on the CPU backend; it has not run on a card yet. Every throughput number below
> is a **model prediction** fed by measured routing, not a measurement on the target host
> ([`bench/host.sh`](bench/host.sh) measures it once the host is built).

## Headline

Predicted decode, batch 1, 4k context, E5-2690 v4 with 64 GB, routed experts BFP4. Hit rate measured on a real routing
trace that switches from prose to code. **MTP, measured**: QFN's own drafter run on the real weights
([`ref_real.py`](ref_real.py)), chained greedy drafts on wikitext: draft 1 accepted 71%, draft 2 52%, draft 3 36%
(2 drafts per step is fastest, 2.08 tokens per step). The chat and structured columns borrow Basalt's community
acceptance for those kinds of text. Dense formats: see [which dense groups tolerate BFP4](#which-dense-weights-tolerate-bfp4).
The model charges the host's control of the card too: per token 61 reads the card waits on (30 µs round trip + 3 calls
each) and 604 calls the host must issue, counted on the card code by [`device/opcount.py`](device/opcount.py) at an
assumed 8 µs per call (to be measured on the card); that costs ~12% against the bandwidth-only numbers. With MTP, each step also runs
the MTP layer once over the accepted tokens to fill its cache (one pass, no draft head).

| Dense weights on the card | KLD (wikitext / code) | no MTP | **MTP, measured (prose)** | MTP, chat (Basalt's rate) | MTP, structured (Basalt's rate) |
|---|---:|---:|---:|---:|---:|
| all BFP8 | 0.086 / 0.051 | 33 t/s | 47 t/s | 59 t/s | 109 t/s |
| **tuff default: LM head + DeltaNet z in BFP4** | **0.113 / 0.060** | **36 t/s** | **52 t/s** | **65 t/s** | **121 t/s** |
| tuff fast: + DeltaNet qkv in BFP4 | 0.131 / 0.073 | 39 t/s | 55 t/s | 68 t/s | 125 t/s |
| static placement (Basalt's profile), all BFP8, no MTP | | 26 t/s | | | |
| **default + an RTX 4060 Ti 16 GB** (dense split, shared expert cache) | 0.113 / 0.060 | **47 t/s** | **65 t/s** | **81 t/s** | **147 t/s** |

How it gets there, each step measured on the real model on CI:

1. **BFP4 experts cost about as much as Basalt's 4-bit packs** (KLD 0.05–0.09, 94–95% same top-1 token), and BFP4 is
   affordable for some dense weights but not all: full attention in BFP4 alone costs 3.8 points of top-1.
2. **A static hot-expert placement breaks when the topic changes**: Basalt's profile keeps 67% of routed experts on the
   card for prose but only 33% for code, worse than a uniform pick (40%).
3. **An adaptive card cache fixes it**: copying up to 25 missed experts per token to the card (LRU eviction) holds ~90%
   hits across the prose→code switch, and plan.py charges those copies to DRAM and PCIe bandwidth.
4. **MTP verification amortizes the dense part**: 4 tokens touch 26 experts per layer, not 40, the multi-token CPU kernel
   runs 4 tokens in 1.94× the time of one, and QFN's own drafter gets 2.08 tokens per step on prose.
5. **The card's dense part is the bottleneck, and a second GPU can share it**: see
   [an RTX 4060 Ti next to the p150a](#an-rtx-4060-ti-next-to-the-p150a).

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
| p150a (32 GB) | attention (all 48 layers), router, shared expert, hyper-connection mixers, LM head, KV cache and DeltaNet state, and a cache of routed experts: **~9,800 of 24,576 (40%)** at BFP4 with dense weights in BFP8 (BFP4 for the LM head and DeltaNet z: ~10,000) |
| host DRAM | all routed experts (68 GB in BFP4), memory-mapped from `loader.py convert`'s files and computed in place by the CPU on a card miss; token embeddings; the MTP layer's experts. The adaptive cache can promote any expert, so **128 GB keeps them all in RAM**; with 64 GB the least-used few GB fault in from NVMe |
| NVMe or DRAM | the N-gram table (~28.8 GB quantized, as in Strata): memory-mapped from NVMe with 64 GB of RAM, resident in RAM with 128 GB. Each token reads only a few rows, and the row indices are known as soon as the previous token is sampled, so they can be prefetched |

Per layer, the card runs attention, sends the hidden state to the host, both sides run their experts in parallel, and the
host sends its partial sum back: `t_token = t_dense + max(t_hot, t_cold + t_pcie)`. Experts that missed are copied to the
card in the background for the next token.

## An RTX 4060 Ti next to the p150a

The model says the card's dense weights (attention, DeltaNet, shared expert, LM head: ~4 GB read per token) take ~75% of
a token, so more bandwidth for them is worth more than anything on the expert side. An RTX 4060 Ti 16 GB (288 GB/s) in
the same host can supply it. The two cards cannot talk to each other directly, so everything they exchange goes through
host memory over PCIe 3.0. Two ways to use it, from [`results/trace.md`](results/trace.md) (E5-2690 v4, 64 GB, tuff
default formats, adaptive cache measured on the prose→code trace):

| Setup | GPU share of dense | Experts on card + GPU | Hit rate | no MTP | MTP, measured (prose) | MTP chat (Basalt's rate) | MTP structured (Basalt's rate) |
|---|---:|---:|---:|---:|---:|---:|---:|
| p150a only | – | 10,019 | 90.1% | 36.4 | 51.9 | 64.6 | 120.7 |
| + 4060 Ti as an expert cache | 0% | 10,019 + 5,631 | 97.1% | 38.5 | 57.0 | 71.2 | 135.4 |
| **+ 4060 Ti, dense split + expert cache** | 41% | 10,581 + 4,791 | 97.0% | **47.1** | **64.9** | **80.6** | **146.7** |

- **Expert cache only** helps little: at 90% card hits the CPU is no longer the slow side.
- **Dense split (tensor parallel)** is what pays: every attention and DeltaNet layer is split by heads, the shared expert
  and LM head by rows, and the two partial sums meet in host memory once per block. The split is exact:
  [`tp_check.py`](tp_check.py) runs transformers' own Qwen4Exp layer code whole and as two halves and checks that the
  halves add up (DeltaNet by key-head groups, attention by KV head with the QSA indexer copied to both, shared expert by
  rows). It costs one extra host round trip per layer, which the model charges.
- **The GPU uses the host's expert bytes as they are**: [`kernels/bfp_cuda.cu`](kernels/bfp_cuda.cu) decodes the same
  offset-binary panels as the CPU kernel (a 128-thread block per 16-row panel, 512-byte coalesced reads), so one copy of
  every expert in host RAM serves the CPU, the GPU and, after the byte permutation, the card. CI compiles it for sm_89;
  its thread-to-dword mapping was replayed in numpy against the reference; [`kernels/test_cuda.cu`](kernels/test_cuda.cu)
  checks it against the CPU and measures its bandwidth on the real GPU.
- Guesses to replace with measurements: the GPU reaching 60% of 288 GB/s at batch 1, and 25 µs per layer for a CUDA hop.
  Hardware: the board needs a second x8 (or wider) slot that fits next to the dual-slot p150a, and the PSU must carry
  both cards (~300 W + ~165 W).

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
| all-BFP8 dense | BFP8 | BFP4 | 0.087 / 0.051 | 93.9% / 95.3% |
| **tuff default** | **BFP8, LM head + DeltaNet z BFP4** | **BFP4** | **0.113 / 0.060** | **92.4% / 94.5%** |
| tuff fast | default + DeltaNet qkv BFP4 | BFP4 | 0.131 / 0.073 | 92.0% / 93.9% |
| all-BFP4 dense | BFP4 (LM head BFP8) | BFP4 | 0.457 / 0.188 | 82.9% / 91.6% |

- BFP8 for the dense part is free: the error is the experts' alone.
- BFP4 for the whole dense part is not, so [`results/dense_groups.md`](results/dense_groups.md) moves one group at a time
  to BFP4 (everything else as in all-BFP8 dense):

#### Which dense weights tolerate BFP4

| Group | Params | Card memory saved | ΔKLD (mean of texts) | Δ top-1 |
|---|---:|---:|---:|---:|
| LM head | 636M | 0.32 GB | +0.007 | −1.0 pt |
| DeltaNet z projection | 566M | 0.28 GB | +0.010 | −0.2 pt |
| DeltaNet qkv projection | 944M | 0.47 GB | +0.026 | −1.3 pt |
| hyper-connection mixers | 629M | 0.31 GB | +0.029 | −1.3 pt |
| shared expert | 236M | 0.12 GB | +0.017 | −0.6 pt |
| DeltaNet output projection | 566M | 0.28 GB | +0.048 | −2.0 pt |
| full attention (12 layers, incl. indexer) | 617M | 0.31 GB | +0.117 | −3.8 pt |

Sorted by KLD per GB saved. The two cheapest groups make the default; adding DeltaNet qkv makes "fast". A run of both
mixes confirmed that the errors roughly add (default: +0.018 measured vs +0.017 predicted; fast: +0.034 vs +0.043).
Full attention stays BFP8 in every mix: it is the reason all-BFP4 dense failed.

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

## The decode recurrence, checked on the real model

[`ref.py`](ref.py) writes QFN's decode step out in plain numpy, as the engine will run it: per token and per layer,
DeltaNet's conv window and recurrent state, the KV cache with the QSA indexer, the PLE n-gram hashing and dilated conv,
the hyper-connections and the MoE. It is the specification for the device path, and it is checked twice:

- **against transformers on a tiny random model** (CI, every push): every token's hidden state matches within 4e-6. The
  only freedom is torch.topk's unspecified order when indexer blocks tie at the cutoff, which ref.py resolves as lowest
  block first.
- **on the real 360 GB checkpoint** ([`ref_real.py`](ref_real.py), streamed layer by layer, 9 min on a GitHub runner):
  for the first 48 tokens of the wikitext text, ref.py routes every token to **the same 10 experts as transformers did,
  in all 48 layers (100.0%)**. Routing in a layer depends on everything before it, so this checks the whole recurrence on
  real weights. Full table: [`results/ref_real.md`](results/ref_real.md).

**QFN's own MTP drafter.** transformers does not implement the MTP layer, so its layout follows Basalt's (MIT): the final
4-stream residual, normed and projected per stream by `fc_hidden`, plus the next token's embedding through `fc_embedding`,
then one full-attention MoE layer, its own mixer and the main LM head; a chained draft feeds the layer's own residual and
guess back in. On 160 wikitext tokens (routing still 100% in all 48 layers), with greedy acceptance counted where the text
follows the main model's greedy picks:

| `pre_fc_norm_hidden` over | draft 1 | draft 2 (after 1 accepted) | draft 3 (after 2) | tokens per step, 2 drafts |
|---|---:|---:|---:|---:|
| **each stream** | **71.4%** (n=105) | **51.7%** (n=60) | **36.4%** (n=22) | **2.08** |
| all 4 streams jointly | 64.8% (n=105) | 52.7% (n=55) | 34.8% (n=23) | 1.99 |

The per-stream reading accepts more, so tuff uses it. Basalt's community median for prose is 66.75% per draft at 1.8
drafts per step, close to this. The headline's "measured" column uses the per-stream rates.

## The card code

[`device/qfn.py`](device/qfn.py) is ref.py's recurrence as the p150a will run it, with tuff's split of the work:

| Card | Host |
|---|---|
| hyper-connections, Gated DeltaNet (conv window and recurrent state stay on the card), full attention with the QSA indexer (KV cache, raw indexer keys and pooled block keys, the latter computed once per completed block), router, shared expert, the routed experts in the card's slots, final mixer, LM head | token embedding and N-gram rows (hashing + table on host RAM / NVMe), the routed experts not in a slot, the adaptive cache: after each token up to 25 missed experts are copied into slots (LRU) |

Per layer only vectors cross the bus: the MoE input goes to the host when an expert misses, and the host's partial sum
comes back.

**MTP speculative decoding on the card.** `Card.verify(tokens)` runs the last token plus the drafts through every layer
in one pass: each matmul takes all m rows (one read of the weights, which is what MTP buys), each slot expert runs once
for every token routed to it, and only the recurrent parts (DeltaNet's conv and delta rule, the indexer's causal
selection, attention) walk the tokens in order. `Card.commit(j)` keeps the first j: DeltaNet's per-token states (ttnn's
delta-rule op returns them), the conv windows and the cache positions roll back to exactly that point.
`MTPDrafter` is QFN's MTP layer on the card; its cache is caught up from the main model's residuals as tokens are
accepted, and chained drafts feed its own residual back in. `Card.generate()` does greedy speculative decoding and is
checked to emit **exactly the tokens plain greedy decoding emits**.

It is written in [`device/ops.py`](device/ops.py)'s operations, which have two backends:

- **`Torch`**: float32 on the CPU, with weights rounded through the card's formats (TT-exact BFP, bf16). CI runs the
  card code on it against ref.py ([`device/test_qfn.py`](device/test_qfn.py), tiny random QFN, 40 tokens): in float32 it
  matches ref.py to 1.6e-6 and routes to the same experts 100% of the time, with experts split between card slots and
  the host and promoted between tokens; in tuff's formats (dense BFP8 with the LM head and DeltaNet z in BFP4, card
  experts BFP4) the median error is 7% and 96% of routing decisions agree. Verifying blocks of 1–6 tokens equals single
  steps (2.3e-6), and speculative greedy decoding equals plain greedy decoding token for token, both with an oracle
  drafter that is sometimes wrong (24 tokens in 8 steps instead of 24, partial acceptance and rollback exercised) and
  with the MTP drafter layer.
- **`TTNN`**: the same operations through ttnn, written from tt-metal's API and sources without a card. Gated DeltaNet
  uses ttnn's own `transformer.fused_recurrent_gated_delta_rule`, which also handles QFN's 16 key heads / 48 value heads
  and returns per-token states for MTP verification; top-k uses `ttnn.topk(stable=True)`, the same tie rule as ref.py.
  **First thing to run on the card**: [`device/test_ops.py`](device/test_ops.py) compares every TTNN operation with
  `Torch` and names the ones to fix.

**The host side in C.** [`device/host_c.py`](device/host_c.py) runs the card code over `kernels/libtuff.so` (ctypes):
cold experts come straight from the converted BFP4 files (tstore, memory-mapped) through one `moe_forward_n` call per
layer and step, so an MTP verification reads each cold expert once per 4 tokens, and the card's slots are managed by
tcache. [`device/test_host_c.py`](device/test_host_c.py) (CI) checks that it gives the same outputs as the numpy host on
the same BFP4 numbers (1.4e-6), the same slot decisions (349 of 1,280 lookups hit, in both), and that speculative greedy
decoding over it equals plain greedy decoding over it.

**The card and a 4060 Ti together.** [`device/tp.py`](device/tp.py) runs the same decode on two devices: DeltaNet split
by key-head groups, attention by KV head (the indexer on both), the shared expert and the LM head by rows, at any GPU
share; the replicated parts (hyper-connections, PLE, router) run on the card and the GPU gets the block input as a
vector; the two partial sums meet in host memory once per block. One expert cache covers both devices' slots (slots
0 .. n_card-1 on the card, the rest on the GPU). [`device/test_tp.py`](device/test_tp.py) (CI) checks that an even and
an uneven split (3:1 DeltaNet key heads) give the same outputs as one device (1.6e-6), the same routing and the same
slot decisions, and that speculative greedy decoding on the two devices equals plain greedy decoding on one.

**Fewer dispatches.** Every ttnn call costs host time to dispatch whatever its size, and a QFN pass is thousands of
small ops, so [`device/opcount.py`](device/opcount.py) counts them ([`results/opcount.md`](results/opcount.md), every
routed expert in a slot, scaled to QFN's 48 layers and top-10):

| one pass, m = 1 token | card ops dispatched | host transfers |
|---|---:|---:|
| before | 10,297 | 602 |
| weights that read the same input stacked into one linear per format, one-op RMSNorm, RoPE as one matmul | 5,698 | 242 |
| + experts as gate (SiLU fused) / up / down, gate weights applied once per layer, routing read once as [m, E] | 4,690 | 194 |
| + traced segments: each DeltaNet layer's dense part, both dense ends of each attention layer (60 replays) | 4,054 | 194 |
| + slot experts as three gathered matmuls over weight banks (`ttnn.sparse_matmul`, indexed) | 1,174 | 170 |
| + each MoE output written by the next layer's segment, attention heads batched, PLE traced | 1,144 | 170 |
| + attention's per-token core and the slot experts traced (block scores over all blocks, rows padded to a fixed length) | **604** | **182** |

(m = 3, an MTP verify pass: 17,952 → 772.) At an assumed 8 µs per call, the first column alone would have cost 82 ms of host time per
token (a ceiling near 12 t/s); now it is 5 ms, which the card's own work mostly hides (`plan.host_busy`). A pass is now a chain of traced segments broken only where the host
decides something: the routing read after each layer (48), the indexer's block choice per token in the attention
layers (12), the LM head (1) — 61 reads, which the 182 transfers include; the rest are small writes (expert ids, gate
weights, row ids) that queue behind the ops. A traced segment ([`ops.trace`](device/ops.py)) is captured once per
shape and replayed from the card's trace buffer; on the CPU backend the emulation forbids host transfers inside it
and writes every replay into the same output tensors, so `test_qfn.py` / `test_tp.py` also check that nothing traced
reads the host or keeps a stale alias. The stacked weights round exactly as before (same BFP blocks: the tuff-format
check gives the same 7.2e-2 / 96.3%).

The TTNN side of this is written from the ttnn docs and not run: `sparse_matmul`'s indexed mode, `fill_cache` into
bfp4 banks, `embedding` for row gathers, `update_cache` for row writes, `begin/end_trace_capture` + `execute_trace`.
`device/test_ops.py` checks each against the CPU backend first thing on the card.

**Prefill.** The prompt goes in passes of up to 512 tokens. A long pass is eager (no trace) but its op count no
longer grows with its length: the causal convs are one shifted slice per tap, and attention scores every token's
blocks in one read, gets the host's choice back as an additive mask [m, t], and attends with all heads in one batched
matmul ([`attention_all`](device/qfn.py)). DeltaNet keeps only the last state of a long pass (per-token states, needed
to roll back a speculative verify, would take m × 3 MiB per layer). `test_qfn.py` checks a prompt in two long passes
(the second starting mid-block) and prompts chunked under a drafter against ref.py / plain greedy.

Not there yet: a CUDA backend for ops.py (the GPU side runs on the
Torch backend; with torch on CUDA it is the same code).

## What exists

| Path | What |
|---|---|
| [`plan.py`](plan.py) | QFN parameter accounting (checked against the published 125B / ~6B / 51B), card/host/NVMe memory map, decode-time model with cache promotions and MTP |
| [`kernels/bfp.c`](kernels/bfp.c) | BFP8/BFP4 pack/unpack, shuffle-free AVX2+FMA GEMV, SwiGLU expert, `moe_forward` (one OpenMP region over all expert panels of a layer), `moe_forward_n` (n tokens per weight read, for MTP) |
| [`loader.py`](loader.py) | Reads QFN's fused expert tensors from a local snapshot or over HTTP range requests, packs them bit-identically to the C packer, writes one memory-mappable file per layer (1.42 GB at BFP4, 68 GB for all 48), and measures real-weight accuracy |
| [`trace.py`](trace.py) | Streamed real forward pass: model-level accuracy per format, routing trace, cache and MTP simulation → [`results/trace.md`](results/trace.md) |
| [`kernels/cache.c`](kernels/cache.c) | `tcache`: the adaptive expert cache (card slots + optional GPU slots, LRU, copies per token); CI replays the real routing trace through it and matches `simulate_cache` within 0.1% |
| [`kernels/store.c`](kernels/store.c) | `tstore`: memory-maps the converted layer files; gate, up and down of any expert are zero-copy `bfp_mat` views |
| [`kernels/bfp_cuda.cu`](kernels/bfp_cuda.cu) | GEMV (1–4 tokens) and MoE layer on an NVIDIA GPU, same byte layout as the host; compiled in CI, not run yet |
| [`tp_check.py`](tp_check.py) | The card/GPU tensor-parallel split is exact, on transformers' Qwen4Exp layer code |
| [`ref.py`](ref.py) | QFN decode as a per-token recurrence in plain numpy, keyed by checkpoint names: DeltaNet conv window and state, KV cache with the QSA indexer, PLE n-gram hashing and dilated conv, hyper-connections, MoE. The spec for the device path; CI checks it against transformers' prefill on a tiny random model |
| [`device/`](device) | The card code: `qfn.py` (decode and MTP verification, card/host split, expert slots, MTP drafter), `ops.py` (Torch and ttnn backends), `host_c.py` (the C host side via ctypes), `test_qfn.py` (== ref.py, speculative == greedy), `test_host_c.py` (C host == numpy host), `test_ops.py` (ttnn vs Torch, per operation, on the card) |
| [`device/tp.py`](device/tp.py) | The p150a and a 4060 Ti together: the dense part split (tensor parallel), one expert cache over both devices' slots, partial sums through the host; `test_tp.py` checks it against one device |
| [`kernels/capi.c`](kernels/capi.c) | `libtuff.so`: tstore, tcache and `moe_forward_n` behind a small C API for the card code |
| [`ref_real.py`](ref_real.py) | ref.py on the real checkpoint, streamed: routing vs transformers in all 48 layers, next-token loss, QFN's MTP draft acceptance → [`results/ref_real.md`](results/ref_real.md) |
| [`kernels/tt_tile.c`](kernels/tt_tile.c) | Host panels → Tenstorrent `bfp4_b`/`bfp8_b` tiles of W^T (8×8 nibble transposes + byte table, 1.1 GB/s per core); CI compares it with a port of tt-metal's `pack_as_bfp_tiles` |
| [`bench/host.sh`](bench/host.sh) | Measures the real host: DRAM read bandwidth, cold experts per thread count, MTP, the GPU → `results/host_*.md` |
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
- [x] Per-group precision for the dense part: LM head and DeltaNet z in BFP4, full attention must stay BFP8
- [x] Adaptive cache policy in engine code (`tcache`), replay-tested; memory-mapped expert store (`tstore`)
- [x] RTX 4060 Ti: decode model, exact tensor-parallel split (checked), CUDA kernels on the host layout (compiled)
- [x] Host packing identical to tt-metal's; host → card tile conversion checked against tt-metal's packer
- [x] Decode recurrence specified in numpy (`ref.py`), checked against transformers and on the real weights (100% routing)
- [x] The card code runs over the C host side (libtuff.so: tstore + moe_forward_n, tcache), == the numpy host
- [x] Card + 4060 Ti split in the card code (tensor parallel, shared expert cache), == one device
- [ ] Copy engine: background PCIe copies of promoted experts to the card and the GPU (pinned host memory)
- [ ] Run `test_cuda` and `bench/host.sh` on the real box; replace GPU_EFF, CPU_EFF and the round-trip guesses
- [ ] Measure on the real host: STREAM, `bench_bfp` with 14 threads, NUMA and hugepages
- [x] Card code for the decode step (TT-NN backend + CPU backend), == ref.py on the CPU backend
- [ ] On the card: `device/test_ops.py`, then the card code; measure the real round trip and the card's bandwidth
- [x] QFN's MTP drafter in the reference, on the real weights: chained drafts accepted 71% / 52% / 36%, 2.08 tokens per step
- [x] MTP speculative decoding in the card code: verify + rollback, QFN's drafter, == plain greedy (CPU backend)
- [x] MTP drafter: its cache filled in one pass per step, optional draft vocabulary for the head
- [x] Card dispatches per pass: 10,297 → 604 ops (fused linears, expert banks + sparse matmul, traced segments), counted by `device/opcount.py`
- [x] Prefill on the card: passes of 512 tokens, attention for all of them at once (masked), last DeltaNet state only
- [ ] Prefill experts: stream them to the card in layer order instead of computing the cold ones on the CPU
- [ ] N-gram table on NVMe with prefetch
- [ ] AVX-512 kernels for the 8259CL, only if `bench/host.sh` shows cold experts compute-bound (on paper AVX2 already
      outruns DRAM on both hosts: ~10 GB/s per core)
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
  - dense를 그룹별로 하나씩 BFP4로 바꿔 측정: LM head와 DeltaNet z는 거의 손해가 없고, 풀 어텐션은 손해가 커서 BFP8 유지
  - 기본값: LM head + DeltaNet z만 BFP4 (KLD 0.113 / 0.060). 빠른 모드: DeltaNet qkv까지 BFP4
- 실제 라우팅 기록 분석: Basalt의 고정 배치는 글에서 67%, 코드에서 33%만 카드에 적중. 적응형 캐시(놓친 전문가를 토큰당 25개까지 카드로 복사)는 주제가 바뀌어도 약 90% 적중

**예측 속도 (E5-2690 v4, 64GB, 기본값):** MTP 없이 36 t/s, **QFN 자체 MTP 초안 모델을 실제 가중치로 잰 채택률(71% / 52% / 36%) 기준 52 t/s**, 4060 Ti와 같이 쓰면 47 t/s / MTP 65 t/s. 대화·구조화 출력은 Basalt의 채택률을 빌리면 65 / 121 t/s. (호스트가 카드를 제어하는 시간까지 넣은 값: 토큰당 읽기 61번, 연산 호출 604번, 호출당 8µs 가정)
실제 하드웨어 측정이 아니라 측정한 라우팅으로 계산한 예측입니다.

- 적응형 캐시를 C 엔진 코드(`tcache`)로 구현: 실제 라우팅 기록을 재생해 Python 시뮬레이션과 0.1% 이내로 일치
- 변환된 전문가 파일을 복사 없이 바로 쓰는 저장소(`tstore`). 전문가 전체가 68GB라 적응형 캐시에는 128GB 램이 맞음
- **RTX 4060 Ti 병렬 처리 검토 결과: 가능하고 효과가 큼.** 카드의 dense 연산을 헤드 단위로 나눠 4060 Ti와 같이 돌리면
  예측 47 t/s (카드만 36), MTP 사용 시 65~147 t/s. 나누는 방식이 수학적으로 정확한지는 HF 코드로 검증(`tp_check.py`).
  4060 Ti용 CUDA 커널도 작성(CI 컴파일 통과, 실제 GPU에서는 아직 미실행)
- 실제 호스트가 준비되면 돌릴 측정 스크립트(`bench/host.sh`)
- 호스트 BFP 반올림을 tt-metal과 똑같이 맞춤(동률은 짝수 쪽). 이전에는 동률에서 달라 카드로 옮길 때 값이 미세하게 바뀔 수 있었음.
  호스트 형식 → 카드 타일 변환기(`tt_tile.c`)가 tt-metal 패커와 바이트 단위로 동일함을 확인
- 디코딩 한 스텝을 numpy로 그대로 적은 기준 구현(`ref.py`): 카드 경로를 구현할 때의 명세. HF와 결과가 같은지 CI에서 확인
- **실제 360GB 모델 가중치로도 검증**: 160토큰, 48개 층 전부에서 HF와 똑같은 전문가를 고름(100%). QFN 자체 MTP 초안 모델의 연속 초안 채택률 71% / 52% / 36% (스텝당 2.08토큰)

- **카드 코드 작성**(`device/`): 디코딩 한 스텝을 카드/호스트로 나눠 실행. CPU 백엔드에서 `ref.py`와 1.6e-6 이내로 같고 전문가 선택 100% 일치.
  ttnn 백엔드는 카드 없이 tt-metal 소스를 보고 작성. DeltaNet은 ttnn에 이미 있는 전용 연산(`fused_recurrent_gated_delta_rule`) 사용
- **MTP 추측 디코딩을 카드 코드에 구현**: 여러 토큰을 한 번에 검증하고(가중치 한 번 읽기), 거절된 토큰만큼 상태를 정확히 되돌림.
  QFN MTP 초안 모델도 카드 코드로 작성. 추측 디코딩 결과가 일반 greedy 생성과 토큰 단위로 완전히 같음을 확인
- **호스트 쪽을 C로 연결**(`libtuff.so`): 놓친 전문가는 변환된 BFP4 파일을 mmap한 저장소에서 `moe_forward_n`으로 한 번에 계산,
  카드 슬롯은 C 캐시(`tcache`)가 관리. numpy 호스트와 결과·슬롯 판단이 같음을 CI에서 확인
- **카드 + 4060 Ti 분할 실행**(`device/tp.py`): dense 연산을 헤드·행 단위로 두 장치에 나누고 부분합은 호스트에서 합침.
  전문가 슬롯은 두 장치를 하나의 캐시로 관리. 한 장치로 돌린 결과와 같고(1.6e-6), 추측 디코딩도 일반 greedy와 같음을 확인
- **카드 연산 수 줄이기**(`device/opcount.py`): 같은 입력을 읽는 가중치를 하나의 linear로 합치고, RMSNorm·RoPE를 한 연산으로,
  슬롯 전문가는 층마다 묶음 matmul 3번(sparse_matmul), 패스 전체를 trace 구간의 연결로 바꿈. 토큰 1개 패스의 카드 연산이
  10,297 → 604개, 호스트가 기다리는 읽기는 층마다 라우팅 1번 + 어텐션 블록 선택뿐. 결과·양자화 오차는 그대로
- **프롬프트 처리(prefill)**: 512토큰씩 한 번에. 어텐션은 마스크로 모든 토큰을 한 번에 계산하고, DeltaNet은 마지막 상태만
  남겨 메모리를 아낌(토큰별 상태는 512토큰이면 층마다 1.5GB). 결과가 ref.py와 같음을 확인

**다음 할 일:** 카드가 오면 `device/test_ops.py`부터 실행해 ttnn 연산을 맞추고 실측, 4060 Ti에서 `test_cuda` 실행, 전문가 복사 엔진.

## License

MIT. Independent student project, not affiliated with Tenstorrent, Qwen, Strata or Basalt.
