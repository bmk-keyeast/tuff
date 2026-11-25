## ref.py on the real Qwen3.8-Flash-Next weights

First 160 tokens of the wikitext text in [`results/routes.npz`](routes.npz), every layer, float64. Routing agreement = share of each token's 10 routed experts that ref.py picks the same as transformers did in trace.py's bf16 run. Next-token loss over the 159 predictions: 1.499 nats (perplexity 4.48; short context). Overall routing agreement 100.0%. Wall time 18 min.

MTP drafter, chained greedy drafts (draft k counted where drafts 1..k-1 were accepted and the text agrees with the main model's greedy picks), with pre_fc_norm_hidden over all streams: draft 1 64.8% (n=105), draft 2 52.7% (n=55), draft 3 34.8% (n=23); per stream: draft 1 71.4% (n=105), draft 2 51.7% (n=60), draft 3 36.4% (n=22).

| Layer | Type | Routing agreement |
|---:|---|---:|
| 0 | linear_attention | 100.0% |
| 1 | linear_attention | 100.0% |
| 2 | linear_attention | 100.0% |
| 3 | indexed_attention | 100.0% |
| 4 | linear_attention | 100.0% |
| 5 | linear_attention | 100.0% |
| 6 | linear_attention | 100.0% |
| 7 | indexed_attention | 100.0% |
| 8 | linear_attention | 100.0% |
| 9 | linear_attention | 100.0% |
| 10 | linear_attention | 100.0% |
| 11 | indexed_attention | 100.0% |
| 12 | linear_attention | 100.0% |
| 13 | linear_attention | 100.0% |
| 14 | linear_attention | 100.0% |
| 15 | indexed_attention | 100.0% |
| 16 | linear_attention | 100.0% |
| 17 | linear_attention | 100.0% |
| 18 | linear_attention | 100.0% |
| 19 | indexed_attention | 100.0% |
| 20 | linear_attention | 100.0% |
| 21 | linear_attention | 100.0% |
| 22 | linear_attention | 100.0% |
| 23 | indexed_attention | 100.0% |
| 24 | linear_attention | 100.0% |
| 25 | linear_attention | 100.0% |
| 26 | linear_attention | 100.0% |
| 27 | indexed_attention | 100.0% |
| 28 | linear_attention | 100.0% |
| 29 | linear_attention | 100.0% |
| 30 | linear_attention | 100.0% |
| 31 | indexed_attention | 100.0% |
| 32 | linear_attention | 100.0% |
| 33 | linear_attention | 100.0% |
| 34 | linear_attention | 100.0% |
| 35 | indexed_attention | 100.0% |
| 36 | linear_attention | 100.0% |
| 37 | linear_attention | 100.0% |
| 38 | linear_attention | 100.0% |
| 39 | indexed_attention | 100.0% |
| 40 | linear_attention | 100.0% |
| 41 | linear_attention | 100.0% |
| 42 | linear_attention | 100.0% |
| 43 | indexed_attention | 100.0% |
| 44 | linear_attention | 100.0% |
| 45 | linear_attention | 100.0% |
| 46 | linear_attention | 100.0% |
| 47 | indexed_attention | 100.0% |
