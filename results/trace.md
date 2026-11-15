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

### Predicted decode with the measured hit rate

plan.py's model with the hit rate measured for Basalt's profile on wikitext (E5-2690 v4, 64 GB, batch 1, no MTP).

| Dense on card | Hit rate | t/s |
|---|---:|---:|
| bfp8 | 67.1% | 31.3 |
| bfp4 | 70.3% | 46.8 |
