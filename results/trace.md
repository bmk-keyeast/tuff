## Qwen3.8-Flash-Next forward pass, streamed from the checkpoint

1024 tokens of each text (wikitext, code), 48 layers, batch = every (expert format, text) pair. Wall time 65 min on a GitHub runner.

### Accuracy per format

bf16 = checkpoint weights. tuff = every dense projection and the LM head in BFP8, routed experts in BFP4. A group name (plan.DENSE) = tuff with that dense group in BFP4. KLD and top-1 agreement are against bf16 on the same tokens.

| Text | Format | Dense on card | PPL | KLD mean | Same top-1 |
|---|---|---:|---:|---:|---:|
| wikitext | bf16 | bf16 | 1.613 | – | – |
| wikitext | tuff | 4.57 GB | 1.675 | 0.0866 | 93.94% |
| wikitext | lin_qkv | 4.10 GB | 1.736 | 0.1283 | 92.77% |
| wikitext | lin_z | 4.29 GB | 1.699 | 0.1043 | 93.55% |
| wikitext | lin_out | 4.29 GB | 1.709 | 0.1510 | 91.79% |
| wikitext | full | 4.26 GB | 1.843 | 0.2480 | 88.86% |
| wikitext | shared | 4.45 GB | 1.683 | 0.1147 | 93.06% |
| wikitext | hc | 4.25 GB | 1.724 | 0.1303 | 92.28% |
| wikitext | lm_head | 4.25 GB | 1.687 | 0.0947 | 92.77% |
| code | bf16 | bf16 | 2.057 | – | – |
| code | tuff | 4.57 GB | 2.086 | 0.0510 | 95.31% |
| code | lin_qkv | 4.10 GB | 2.077 | 0.0614 | 93.94% |
| code | lin_z | 4.29 GB | 2.078 | 0.0528 | 95.41% |
| code | lin_out | 4.29 GB | 2.101 | 0.0825 | 93.55% |
| code | full | 4.26 GB | 2.077 | 0.1239 | 92.77% |
| code | shared | 4.45 GB | 2.083 | 0.0561 | 95.01% |
| code | hc | 4.25 GB | 2.114 | 0.0659 | 94.33% |
| code | lm_head | 4.25 GB | 2.096 | 0.0575 | 94.53% |

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
With tuff experts, 92.7% of the routed experts are the same as with bf16.
With lin_qkv experts, 91.1% of the routed experts are the same as with bf16.
With lin_z experts, 92.3% of the routed experts are the same as with bf16.
With lin_out experts, 89.6% of the routed experts are the same as with bf16.
With full experts, 88.0% of the routed experts are the same as with bf16.
With shared experts, 92.1% of the routed experts are the same as with bf16.
With hc experts, 90.4% of the routed experts are the same as with bf16.
With lm_head experts, 92.7% of the routed experts are the same as with bf16.

### Adaptive card cache (simulated on the trace)

9,802 experts on the card (dense BFP8), starting from Basalt's profile. A miss runs on the CPU and the expert is copied to the card for the next token, evicting the least recently used; 'copies' caps the copies per token (0 = static placement). t/s from plan.py, which charges the copies to host DRAM bandwidth and to the PCIe 3.0 link (E5-2690 v4, dense BFP8, no MTP).

| Text | 0 copies | 10 copies | 25 copies | 50 copies | 100 copies |
|---|---:|---:|---:|---:|---:|
| wikitext | 67.1% (31.1 t/s) | 89.2% (37.2 t/s) | 93.8% (36.8 t/s) | 94.9% (36.7 t/s) | 95.1% (36.7 t/s) |
| code | 32.7% (24.0 t/s) | 64.5% (29.9 t/s) | 86.2% (35.6 t/s) | 90.4% (36.3 t/s) | 91.3% (36.1 t/s) |
| wikitext then code | 49.9% (27.1 t/s) | 75.8% (33.0 t/s) | 89.7% (37.0 t/s) | 92.3% (36.9 t/s) | 93.1% (36.9 t/s) |

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

| Host | Dense on card | Hit rate | no MTP | prose | chat | structured |
|---|---|---:|---:|---:|---:|---:|
| E5-2690v4, 4x16GB | bfp8 | 89.7% | 37.0 | 59.3 | 67.8 | 125.7 |
| 8259CL, 6x32GB | bfp8 | 89.7% | 37.2 | 60.2 | 69.0 | 130.1 |

### Predicted decode with the measured hit rate

plan.py's model with the hit rate measured for Basalt's profile on wikitext (E5-2690 v4, 64 GB, batch 1, no MTP).

| Dense on card | Hit rate | t/s |
|---|---:|---:|
| bfp8 | 67.1% | 31.1 |
| bfp4 | 70.3% | 46.6 |

### Which dense groups tolerate BFP4

Each row is tuff with one dense group in BFP4 instead of BFP8; ΔKLD and Δtop-1 are against tuff, mean of the texts. Rows are sorted by ΔKLD per GB of card memory saved. 'Cumulative' moves the groups to BFP4 from the top down and assumes their errors add (a run of the chosen mix checks that); t/s from plan.py with the adaptive cache (E5-2690 v4, 25 copies per token, no MTP).

| Group | Params | Card GB saved | ΔKLD | Δ top-1 | Cumulative ΔKLD | Cumulative t/s |
|---|---:|---:|---:|---:|---:|---:|
| (tuff: all BFP8) | | | | | 0 | 37.0 |
| lm_head | 636M | 0.32 | +0.0073 | -0.98 pt | 0.0073 | 39.2 |
| lin_z | 566M | 0.28 | +0.0098 | -0.15 pt | 0.0171 | 41.3 |
| lin_qkv | 944M | 0.47 | +0.0261 | -1.27 pt | 0.0431 | 45.3 |
| hc | 629M | 0.31 | +0.0293 | -1.32 pt | 0.0725 | 48.4 |
| shared | 236M | 0.12 | +0.0166 | -0.59 pt | 0.0890 | 49.6 |
| lin_out | 566M | 0.28 | +0.0480 | -1.96 pt | 0.1370 | 53.0 |
| full | 617M | 0.31 | +0.1171 | -3.81 pt | 0.2541 | 57.2 |
