## Qwen3.8-Flash-Next on p150a + Xeon

Parameters (this count): main model 125.7B (published: ~125B), active per token 7.3B (published: ~6B; ours includes embedding and LM head), N-gram embeddings 51.2B (published: ~51B), routed experts 24,576 x 4.92M params.

### Memory map (32k context)

| Host | Dense on card | Routed experts on card | Cold experts + embeddings + MTP experts in host RAM | Fits | N-gram table (28.8 GB) |
|---|---|---:|---:|:-:|---|
| E5-2690v4, 4x16GB | bfp8, 4.5 GB | 9,815 / 24,576 (40%) | 43.5 GB | yes | NVMe (mmap) |
| E5-2690v4, 4x16GB | bfp4, 2.4 GB | 10,586 / 24,576 (43%) | 41.4 GB | yes | NVMe (mmap) |
| E5-2690v4, 128GB | bfp8, 4.5 GB | 9,815 / 24,576 (40%) | 43.5 GB | yes | host RAM |
| E5-2690v4, 128GB | bfp4, 2.4 GB | 10,586 / 24,576 (43%) | 41.4 GB | yes | host RAM |
| 8259CL, 6x32GB | bfp8, 4.5 GB | 9,815 / 24,576 (40%) | 43.5 GB | yes | host RAM |
| 8259CL, 6x32GB | bfp4, 2.4 GB | 10,586 / 24,576 (43%) | 41.4 GB | yes | host RAM |

### Decode speed, batch 1, 4k context, experts BFP4

Efficiencies: card 43% of 512 GB/s, DRAM 70% of peak, PCIe 80%, 30 us round trip per layer. 'Card hit rate' = share of the 10 activated experts per layer found on the card. Uniform routing gives the share of experts that fit; a measured popularity profile (Basalt ships one) raises it. MTP speculative decoding is not included.

| Host | Dense format | uniform routing | 60% hits | 80% hits | time split at uniform: dense / hot / cold+PCIe (ms) |
|---|---|---:|---:|---:|---|
| E5-2690v4, 4x16GB | bfp8 | 25.4 t/s | 29.5 | 35.2 | 21.4 / 2.4 / 18.0 |
| E5-2690v4, 4x16GB | bfp4 | 34.7 t/s | 41.2 | 53.5 | 11.7 / 2.6 / 17.1 |
| E5-2690v4, 128GB | bfp8 | 25.4 t/s | 29.5 | 35.2 | 21.4 / 2.4 / 18.0 |
| E5-2690v4, 128GB | bfp4 | 34.7 t/s | 41.2 | 53.5 | 11.7 / 2.6 / 17.1 |
| 8259CL, 6x32GB | bfp8 | 32.4 t/s | 35.4 | 38.2 | 21.4 / 2.4 / 9.5 |
| 8259CL, 6x32GB | bfp4 | 48.2 t/s | 54.0 | 60.7 | 11.7 / 2.6 / 9.1 |
