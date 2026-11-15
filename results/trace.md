## Qwen3.8-Flash-Next forward pass, streamed from the checkpoint

1024 tokens of each text (wikitext, code), 48 layers, batch = every (expert format, text) pair. Wall time 30 min on a GitHub runner.

### Accuracy of BFP experts (dense weights unquantized)

KLD and top-1 agreement are against the bf16 run on the same tokens.

| Text | Experts | PPL | KLD mean | Same top-1 |
|---|---|---:|---:|---:|
| wikitext | bf16 | 1.614 | – | – |
| wikitext | bfp4 | 1.656 | 0.0942 | 93.65% |
| wikitext | bfp4+8 | 1.673 | 0.0732 | 94.72% |
| code | bf16 | 2.057 | – | – |
| code | bfp4 | 2.078 | 0.0469 | 95.11% |
| code | bfp4+8 | 2.076 | 0.0374 | 95.60% |

### Routing

Hit rate = share of the 10 routed experts per token and layer that a placement keeps on the card. Card capacity from plan.py: 9,815 experts with dense weights in BFP8, 10,586 with dense in BFP4 (of 24,576).

| Placement | Text | hit, 9,815 on card | hit, 10,586 on card |
|---|---|---:|---:|
| uniform | any | 39.9% | 43.1% |
| in-sample (same text, upper bound) | wikitext | 97.0% | 97.8% |
| in-sample (same text, upper bound) | code | 93.9% | 95.1% |
| the other text's counts | wikitext | 35.7% | 40.0% |
| the other text's counts | code | 27.6% | 30.8% |
| Basalt's shipped profile | wikitext | 67.1% | 70.3% |
| Basalt's shipped profile | code | 32.9% | 37.1% |

Share of a token's experts that the previous token also used, same layer: wikitext 40.9%, code 36.8%.
With bfp4 experts, 92.9% of the routed experts are the same as with bf16.
With bfp4+8 experts, 94.0% of the routed experts are the same as with bf16.

### Adaptive card cache (simulated on the trace)

9,815 experts on the card (dense BFP8), starting from Basalt's profile. A miss runs on the CPU and the expert is copied to the card for the next token, evicting the least recently used; 'copies' caps the copies per token (0 = static placement). t/s from plan.py, which charges the copies to host DRAM bandwidth and to the PCIe 3.0 link (E5-2690 v4, dense BFP8, no MTP).

| Text | 0 copies | 10 copies | 25 copies | 50 copies | 100 copies |
|---|---:|---:|---:|---:|---:|
| wikitext | 67.1% (31.3 t/s) | 89.3% (37.4 t/s) | 93.8% (37.0 t/s) | 94.9% (36.9 t/s) | 95.1% (36.9 t/s) |
| code | 32.9% (24.1 t/s) | 64.5% (30.1 t/s) | 86.3% (35.8 t/s) | 90.4% (36.5 t/s) | 91.3% (36.3 t/s) |
| wikitext then code | 50.0% (27.2 t/s) | 75.9% (33.2 t/s) | 89.7% (37.3 t/s) | 92.4% (37.1 t/s) | 93.1% (37.1 t/s) |

### Experts per layer when m consecutive tokens are verified together (MTP)

Mean over layers, texts and windows; placement = Basalt's profile with 9,815 experts on the card.

| m tokens | unique experts | on card | cold (CPU) |
|---:|---:|---:|---:|
| 1 | 10.0 | 5.0 | 5.0 |
| 2 | 16.1 | 8.0 | 8.1 |
| 3 | 21.4 | 10.6 | 10.7 |
| 4 | 26.1 | 13.0 | 13.1 |
| 6 | 34.7 | 17.3 | 17.4 |
| 8 | 42.3 | 21.1 | 21.2 |

Predicted decode with MTP (E5-2690 v4, 64 GB), assuming Basalt's measured draft lengths and acceptance; the experts per step are interpolated from the table above.

| Text kind | drafts / step | tokens / step | dense BFP8 | dense BFP4 |
|---|---:|---:|---:|---:|
| prose | 1.8 | 2.20 | 39.4 t/s | 49.2 t/s |
| chat | 2.35 | 2.71 | 44.4 t/s | 54.8 t/s |
| structured | 6.9 | 7.79 | 78.5 t/s | 92.8 t/s |

### Predicted decode: adaptive cache + MTP

Adaptive cache with up to 25 copies per token, hit rate measured on 'wikitext then code'; MTP as above, with each step's union of experts split by that hit rate. Batch 1, 4k context.

| Host | Dense on card | Hit rate | no MTP | prose | chat | structured |
|---|---|---:|---:|---:|---:|---:|
| E5-2690v4, 4x16GB | bfp8 | 89.7% | 37.3 | 59.6 | 68.1 | 126.1 |
| E5-2690v4, 4x16GB | bfp4 | 91.1% | 58.3 | 84.7 | 95.4 | 166.3 |
| 8259CL, 6x32GB | bfp8 | 89.7% | 37.4 | 60.4 | 69.3 | 130.5 |
| 8259CL, 6x32GB | bfp4 | 91.1% | 58.3 | 86.5 | 97.8 | 173.9 |

### Predicted decode with the measured hit rate

plan.py's model with the hit rate measured for Basalt's profile on wikitext (E5-2690 v4, 64 GB, batch 1, no MTP).

| Dense on card | Hit rate | t/s |
|---|---:|---:|
| bfp8 | 67.1% | 31.3 |
| bfp4 | 70.3% | 46.8 |
