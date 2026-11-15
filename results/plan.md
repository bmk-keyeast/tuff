Batch-1 decode, 4096-token context. Experts bfp4, everything else bfp8 on the card. Uniform routing (no expert-popularity skew). Efficiencies: card 43%, DRAM 70%, PCIe 80%, 30 us PCIe round trip per MoE layer.


### E5-2690v4 64GB (DRAM 68 GB/s peak, PCIe 3.0 x16)

| Model | Params (B) | Experts on card | Host RAM for experts (GB) | Fits | CPU-only t/s | Stream-to-card t/s | **Hybrid t/s** | Hybrid time split: dense / hot / cold+PCIe (ms) |
|---|---:|---:|---:|:-:|---:|---:|---:|---|
| Qwen3-30B-A3B | 30.5 | 6144/6144 (100%) | 0.6 | yes | 18.8 | 86.7 | **86.7** | 6.9 / 4.6 / 1.5 |
| Qwen3-Next-80B-A3B | 79.0 | 17231/24576 (70%) | 13.6 | yes | 18.6 | 32.6 | **68.5** | 7.8 / 2.7 / 6.8 |
| gpt-oss-20b | 20.9 | 768/768 (100%) | 1.2 | yes | 17.4 | 80.2 | **80.2** | 6.4 / 6.1 / 0.7 |
| gpt-oss-120b | 116.8 | 2172/4608 (47%) | 35.3 | yes | 12.5 | 10.3 | **31.6** | 8.2 / 4.3 / 23.4 |
| GLM-4.5-Air | 106.9 | 2550/5760 (44%) | 32.5 | yes | 4.4 | 5.1 | **13.2** | 33.6 / 7.0 / 42.3 |
| DeepSeek-V2-Lite | 15.8 | 1664/1664 (100%) | 0.4 | yes | 19.1 | 88.2 | **88.2** | 7.9 / 3.4 / 0.8 |
| Mixtral-8x7B | 46.7 | 256/256 (100%) | 0.3 | yes | 5.8 | 26.9 | **26.9** | 8.4 / 28.8 / 1.0 |
| Qwen3-235B-A22B | 235.1 | 2256/12032 (19%) | 105.0 | no | 2.9 | 1.8 | **5.7** | 37.5 / 6.8 / 138.7 |

### E5-2690v4 128GB (DRAM 68 GB/s peak, PCIe 3.0 x16)

| Model | Params (B) | Experts on card | Host RAM for experts (GB) | Fits | CPU-only t/s | Stream-to-card t/s | **Hybrid t/s** | Hybrid time split: dense / hot / cold+PCIe (ms) |
|---|---:|---:|---:|:-:|---:|---:|---:|---|
| Qwen3-30B-A3B | 30.5 | 6144/6144 (100%) | 0.6 | yes | 18.8 | 86.7 | **86.7** | 6.9 / 4.6 / 1.5 |
| Qwen3-Next-80B-A3B | 79.0 | 17231/24576 (70%) | 13.6 | yes | 18.6 | 32.6 | **68.5** | 7.8 / 2.7 / 6.8 |
| gpt-oss-20b | 20.9 | 768/768 (100%) | 1.2 | yes | 17.4 | 80.2 | **80.2** | 6.4 / 6.1 / 0.7 |
| gpt-oss-120b | 116.8 | 2172/4608 (47%) | 35.3 | yes | 12.5 | 10.3 | **31.6** | 8.2 / 4.3 / 23.4 |
| GLM-4.5-Air | 106.9 | 2550/5760 (44%) | 32.5 | yes | 4.4 | 5.1 | **13.2** | 33.6 / 7.0 / 42.3 |
| DeepSeek-V2-Lite | 15.8 | 1664/1664 (100%) | 0.4 | yes | 19.1 | 88.2 | **88.2** | 7.9 / 3.4 / 0.8 |
| Mixtral-8x7B | 46.7 | 256/256 (100%) | 0.3 | yes | 5.8 | 26.9 | **26.9** | 8.4 / 28.8 / 1.0 |
| Qwen3-235B-A22B | 235.1 | 2256/12032 (19%) | 105.0 | yes | 2.9 | 1.8 | **5.7** | 37.5 / 6.8 / 138.7 |

### 8259CL 192GB (DRAM 141 GB/s peak, PCIe 3.0 x16)

| Model | Params (B) | Experts on card | Host RAM for experts (GB) | Fits | CPU-only t/s | Stream-to-card t/s | **Hybrid t/s** | Hybrid time split: dense / hot / cold+PCIe (ms) |
|---|---:|---:|---:|:-:|---:|---:|---:|---|
| Qwen3-30B-A3B | 30.5 | 6144/6144 (100%) | 0.6 | yes | 38.8 | 86.7 | **86.7** | 6.9 / 4.6 / 1.5 |
| Qwen3-Next-80B-A3B | 79.0 | 17231/24576 (70%) | 13.6 | yes | 38.3 | 32.6 | **84.3** | 7.8 / 2.7 / 4.0 |
| gpt-oss-20b | 20.9 | 768/768 (100%) | 1.2 | yes | 35.9 | 80.2 | **80.2** | 6.4 / 6.1 / 0.7 |
| gpt-oss-120b | 116.8 | 2172/4608 (47%) | 35.3 | yes | 25.8 | 10.3 | **49.7** | 8.2 / 4.3 / 11.9 |
| GLM-4.5-Air | 106.9 | 2550/5760 (44%) | 32.5 | yes | 9.0 | 5.1 | **18.2** | 33.6 / 7.0 / 21.2 |
| DeepSeek-V2-Lite | 15.8 | 1664/1664 (100%) | 0.4 | yes | 39.5 | 88.2 | **88.2** | 7.9 / 3.4 / 0.8 |
| Mixtral-8x7B | 46.7 | 256/256 (100%) | 0.3 | yes | 12.0 | 26.9 | **26.9** | 8.4 / 28.8 / 1.0 |
| Qwen3-235B-A22B | 235.1 | 2256/12032 (19%) | 105.0 | yes | 6.1 | 1.8 | **9.4** | 37.5 / 6.8 / 68.8 |
