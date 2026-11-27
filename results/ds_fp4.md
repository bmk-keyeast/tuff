## DeepSeek-V4.1-Flash-1017 FP4 experts in K200 / TT formats (real weights)

8 random routed experts (seed 0) from each of layers 0, 10, 20, 30, 39, read from the checkpoint with HTTP range requests. The reference is the checkpoint's own FP4 expert. SQNR in dB, higher is better: weights, and the SwiGLU expert's output (with the model's swiglu_limit clamps) on 64 Gaussian inputs. For scale: QFN's BF16 experts in BFP4 gave 14.2 dB output SQNR and cost KLD 0.05-0.09 / 94-95% same top-1 (results/expert_accuracy.md, results/trace.md).

Share of FP4 weights at ±0.5 or ±1.5 (the levels a fixed INT4 grid loses): 24.4%.

| Format | Layer 0 | Layer 10 | Layer 20 | Layer 30 | Layer 39 | **All** |
|---|---:|---:|---:|---:|---:|---:|
| INT4, ties to even, weights | 16.6 (min 16.4) | 16.7 (min 16.7) | 16.7 (min 16.6) | 16.6 (min 16.4) | 16.4 (min 16.2) | 16.6 (min 16.2) |
| INT4, ties to even, output | 12.2 (min 11.9) | 12.0 (min 11.7) | 11.8 (min 11.5) | 11.6 (min 11.1) | 11.8 (min 11.2) | 11.9 (min 11.1) |
| INT4, ties alternating, weights | 16.6 (min 16.4) | 16.7 (min 16.7) | 16.7 (min 16.6) | 16.6 (min 16.4) | 16.4 (min 16.2) | 16.6 (min 16.2) |
| INT4, ties alternating, output | 12.5 (min 12.2) | 12.3 (min 12.0) | 12.1 (min 11.7) | 11.9 (min 11.4) | 12.1 (min 11.5) | 12.2 (min 11.4) |
| BFP4, exponent along input (= INT4, per-16 step), weights | 18.9 (min 18.7) | 19.0 (min 18.9) | 19.0 (min 18.9) | 18.9 (min 18.7) | 18.5 (min 18.4) | 18.8 (min 18.4) |
| BFP4, exponent along input (= INT4, per-16 step), output | 14.6 (min 14.2) | 14.4 (min 14.1) | 14.2 (min 13.9) | 14.0 (min 13.6) | 14.1 (min 13.4) | 14.2 (min 13.4) |
| BFP4, exponent along output (TT default tiles), weights | 19.2 (min 19.0) | 19.3 (min 19.2) | 19.2 (min 19.2) | 19.2 (min 18.9) | 18.6 (min 18.5) | 19.1 (min 18.5) |
| BFP4, exponent along output (TT default tiles), output | 15.0 (min 14.7) | 14.7 (min 14.4) | 14.5 (min 14.2) | 14.3 (min 13.8) | 14.2 (min 13.6) | 14.5 (min 13.6) |
| INT8 / BFP8 along input, weights | exact | exact | exact | exact | exact | exact |
| INT8 / BFP8 along input, output | exact | exact | exact | exact | exact | exact |
