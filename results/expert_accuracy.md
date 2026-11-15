## Qwen3.8-Flash-Next routed experts in TT block-float (real weights)

8 random experts (seed 0) from each of layers 0, 9, 19, 28, 38, 47, read straight from the BF16 checkpoint with HTTP range requests. SQNR in dB, higher is better. Expert output = the SwiGLU expert's output on 64 Gaussian input vectors, quantized vs BF16 weights. Gaussian matrices give 19.0 dB (BFP4) and 43.1 dB (BFP8) weight SQNR (`kernels/test_bfp`).

| Layer | Weights BFP4 | Weights BFP8 | Output, all BFP4 | Output, BFP4 gate/up + BFP8 down | Output, all BFP8 | Output, BFP4 with exponent along input dim |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 18.6 (min 18.5) | 42.4 (min 42.3) | 14.8 (min 14.0) | 17.0 (min 15.7) | 37.0 (min 36.6) | 14.6 (min 14.0) |
| 9 | 18.8 (min 18.7) | 42.5 (min 42.5) | 14.4 (min 14.2) | 16.3 (min 16.1) | 36.7 (min 36.6) | 14.4 (min 14.2) |
| 19 | 18.9 (min 18.8) | 42.6 (min 42.5) | 14.2 (min 14.1) | 15.9 (min 15.8) | 36.6 (min 36.5) | 14.2 (min 14.1) |
| 28 | 18.8 (min 18.8) | 42.5 (min 42.5) | 14.1 (min 14.0) | 15.8 (min 15.6) | 36.5 (min 36.4) | 14.1 (min 13.9) |
| 38 | 18.7 (min 18.6) | 42.4 (min 42.4) | 14.0 (min 13.9) | 15.8 (min 15.6) | 36.5 (min 36.4) | 14.0 (min 13.8) |
| 47 | 18.5 (min 18.1) | 42.2 (min 41.8) | 13.8 (min 13.1) | 15.7 (min 15.3) | 36.3 (min 35.8) | 13.9 (min 13.3) |
| **all** | 18.7 (min 18.1) | 42.4 (min 41.8) | 14.2 (min 13.1) | 16.1 (min 15.3) | 36.6 (min 35.8) | 14.2 (min 13.3) |
