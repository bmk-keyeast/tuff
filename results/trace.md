## Qwen3.8-Flash-Next forward pass, streamed from the checkpoint

1024 tokens of each text (wikitext, code), 48 layers, batch = every (expert format, text) pair. Wall time 28 min on a GitHub runner.

### Accuracy per format

bf16 = checkpoint weights. tuff = every dense projection and the LM head in BFP8, routed experts in BFP4. A group name (plan.DENSE) = tuff with that dense group in BFP4. KLD and top-1 agreement are against bf16 on the same tokens.

| Text | Format | Dense on card | PPL | KLD mean | Same top-1 |
|---|---|---:|---:|---:|---:|
| wikitext | bf16 | bf16 | 1.613 | – | – |
| wikitext | tuff | 4.57 GB | 1.674 | 0.0863 | 93.94% |
| wikitext | tuff-mix | 3.97 GB | 1.716 | 0.1131 | 92.38% |
| wikitext | tuff-mix2 | 3.50 GB | 1.747 | 0.1307 | 91.98% |
| code | bf16 | bf16 | 2.057 | – | – |
| code | tuff | 4.57 GB | 2.086 | 0.0507 | 95.41% |
| code | tuff-mix | 3.97 GB | 2.090 | 0.0597 | 94.53% |
| code | tuff-mix2 | 3.50 GB | 2.082 | 0.0733 | 93.94% |

### Routing

Hit rate = share of the 10 routed experts per token and layer that a placement keeps on the card. Card capacity from plan.py: 9,802 experts with dense weights in BFP8, 10,580 with dense in BFP4 (of 24,576).

| Placement | Text | hit, 9,802 on card | hit, 10,580 on card |
|---|---|---:|---:|
| uniform | any | 39.9% | 43.1% |
| in-sample (same text, upper bound) | wikitext | 97.0% | 97.8% |
| in-sample (same text, upper bound) | code | 93.9% | 95.1% |
| the other text's counts | wikitext | 35.6% | 40.0% |
| the other text's counts | code | 27.5% | 30.7% |
| Basalt's shipped profile | wikitext | 67.1% | 70.3% |
| Basalt's shipped profile | code | 32.7% | 37.0% |

Share of a token's experts that the previous token also used, same layer: wikitext 40.9%, code 36.8%.
With tuff experts, 92.8% of the routed experts are the same as with bf16.
With tuff-mix experts, 92.3% of the routed experts are the same as with bf16.
With tuff-mix2 experts, 90.8% of the routed experts are the same as with bf16.

### Adaptive card cache (simulated on the trace)

9,802 experts on the card (dense BFP8), starting from Basalt's profile. A miss runs on the CPU and the expert is copied to the card for the next token, evicting the least recently used; 'copies' caps the copies per token (0 = static placement). t/s from plan.py, which charges the copies to host DRAM bandwidth and to the PCIe 3.0 link (E5-2690 v4, dense BFP8, no MTP).

| Text | 0 copies | 10 copies | 25 copies | 50 copies | 100 copies |
|---|---:|---:|---:|---:|---:|
| wikitext | 67.1% (29.4 t/s) | 89.2% (33.1 t/s) | 93.8% (32.8 t/s) | 94.9% (32.8 t/s) | 95.1% (32.7 t/s) |
| code | 32.7% (23.0 t/s) | 64.5% (28.3 t/s) | 86.2% (33.3 t/s) | 90.4% (33.1 t/s) | 91.3% (33.0 t/s) |
| wikitext then code | 49.9% (25.8 t/s) | 75.8% (31.1 t/s) | 89.7% (33.1 t/s) | 92.3% (32.9 t/s) | 93.1% (32.9 t/s) |

### Experts per layer when m consecutive tokens are verified together (MTP)

Mean over layers, texts and windows; placement = Basalt's profile with 9,802 experts on the card.

| m tokens | unique experts | on card | cold (CPU) |
|---:|---:|---:|---:|
| 1 | 10.0 | 5.0 | 5.0 |
| 2 | 16.1 | 8.0 | 8.1 |
| 3 | 21.4 | 10.6 | 10.7 |
| 4 | 26.1 | 13.0 | 13.1 |
| 6 | 34.7 | 17.3 | 17.4 |
| 8 | 42.3 | 21.1 | 21.2 |

### Predicted decode: adaptive cache + MTP

Adaptive cache with up to 25 copies per token, hit rate measured on 'wikitext then code'. MTP assumes Basalt's measured draft lengths and acceptance (community medians), with each step's experts interpolated from the table above and split by that hit rate. Batch 1, 4k context.

| Host | Dense on card | Hit rate | no MTP | prose | chat | structured | QFN MTP (measured) |
|---|---|---:|---:|---:|---:|---:|---:|
| E5-2690v4, 4x16GB | bfp8 | 89.7% | 33.1 | 52.7 | 60.1 | 110.3 | 48.4 |
| E5-2690v4, 4x16GB | tuff-mix | 90.1% | 36.4 | 58.0 | 66.3 | 122.7 | 53.3 |
| E5-2690v4, 4x16GB | tuff-mix2 | 90.4% | 39.4 | 61.4 | 69.8 | 126.8 | 56.3 |
| 8259CL, 6x32GB | bfp8 | 89.7% | 33.1 | 53.3 | 61.1 | 113.7 | 49.1 |
| 8259CL, 6x32GB | tuff-mix | 90.1% | 36.4 | 58.8 | 67.4 | 126.8 | 54.1 |
| 8259CL, 6x32GB | tuff-mix2 | 90.4% | 39.4 | 62.3 | 71.1 | 131.2 | 57.3 |

### With an RTX 4060 Ti 16 GB next to the p150a

Dense on the card as tuff's default (LM head and DeltaNet z in BFP4, the rest BFP8). One adaptive cache over the card's and the GPU's expert slots (up to 25 copies per token, hit rate measured on 'wikitext then code'); hits split between the devices by their slot counts. 'Split dense' also divides every dense matmul between the two (tensor parallel, partial sums meet in host memory, one extra round trip per layer). E5-2690 v4, 64 GB, batch 1, 4k context; MTP as above.

| Setup | GPU share of dense | Experts on card + GPU | Hit rate | no MTP | prose | chat | structured | QFN MTP (measured) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p150a only | 0% | 10,019 + 0 | 90.1% | 36.4 | 58.0 | 66.3 | 122.7 | 53.3 |
| + 4060 Ti, expert cache | 0% | 10,019 + 5,631 | 97.1% | 38.5 | 63.7 | 73.2 | 138.0 | 58.7 |
| + 4060 Ti, expert cache + split dense | 41% | 10,581 + 4,791 | 97.0% | 47.1 | 73.2 | 83.2 | 149.7 | 67.2 |

### Predicted decode with the measured hit rate

plan.py's model with the hit rate measured for Basalt's profile on wikitext (E5-2690 v4, 64 GB, batch 1, no MTP).

| Dense on card | Hit rate | t/s |
|---|---:|---:|
| bfp8 | 67.1% | 29.4 |
| bfp4 | 70.3% | 42.9 |

### Which dense groups tolerate BFP4

Each row is tuff with one dense group in BFP4 instead of BFP8; ΔKLD and Δtop-1 are against tuff, mean of the texts. Rows are sorted by ΔKLD per GB of card memory saved. 'Cumulative' moves the groups to BFP4 from the top down and assumes their errors add (a run of the chosen mix checks that); t/s from plan.py with the adaptive cache (E5-2690 v4, 25 copies per token, no MTP).

| Group | Params | Card GB saved | ΔKLD | Δ top-1 | Cumulative ΔKLD | Cumulative t/s |
|---|---:|---:|---:|---:|---:|---:|
| (tuff: all BFP8) | | | | | 0 | 33.1 |
