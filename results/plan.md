## Qwen3.8-Flash-Next on p150a + Xeon

Parameters (this count): main model 125.7B (published: ~125B), active per token 7.3B (published: ~6B; ours includes embedding and LM head), N-gram embeddings 51.2B (published: ~51B), routed experts 24,576 x 4.92M params.

### Memory map (32k context)

| Host | Dense on card | Routed experts on card | Cold experts + embeddings + MTP experts in host RAM | Fits | All experts in RAM (adaptive cache) | N-gram table (28.8 GB) |
|---|---|---:|---:|:-:|:-:|---|
| E5-2690v4, 4x16GB | bfp8, 4.6 GB | 9,802 / 24,576 (40%) | 43.5 GB | yes | no: rarely used experts on NVMe | NVMe (mmap) |
| E5-2690v4, 4x16GB | bfp4, 2.4 GB | 10,580 / 24,576 (43%) | 41.4 GB | yes | no: rarely used experts on NVMe | NVMe (mmap) |
| E5-2690v4, 128GB | bfp8, 4.6 GB | 9,802 / 24,576 (40%) | 43.5 GB | yes | yes | host RAM |
| E5-2690v4, 128GB | bfp4, 2.4 GB | 10,580 / 24,576 (43%) | 41.4 GB | yes | yes | host RAM |
| 8259CL, 6x32GB | bfp8, 4.6 GB | 9,802 / 24,576 (40%) | 43.5 GB | yes | yes | host RAM |
| 8259CL, 6x32GB | bfp4, 2.4 GB | 10,580 / 24,576 (43%) | 41.4 GB | yes | yes | host RAM |

### Decode speed, batch 1, 4k context, experts BFP4

Efficiencies: card 43% of 512 GB/s, DRAM 70% of peak, PCIe 80%, 30 us per host read (61 per token) + 3 card calls after it at 8 us (the host issues 604 calls per token, device/opcount.py). 'Card hit rate' = share of the 10 activated experts per layer found on the card. Uniform routing gives the share of experts that fit; a measured popularity profile (Basalt ships one) raises it. MTP speculative decoding is not included.

| Host | Dense format | uniform routing | 60% hits | 80% hits | time split at uniform: dense / hot / cold+PCIe (ms) |
|---|---|---:|---:|---:|---|
| E5-2690v4, 4x16GB | bfp8 | 24.2 t/s | 27.8 | 32.9 | 21.5 / 2.4 / 19.9 |
| E5-2690v4, 4x16GB | bfp4 | 32.5 t/s | 38.2 | 48.5 | 11.7 / 2.6 / 19.0 |
| E5-2690v4, 128GB | bfp8 | 24.2 t/s | 27.8 | 32.9 | 21.5 / 2.4 / 19.9 |
| E5-2690v4, 128GB | bfp4 | 32.5 t/s | 38.2 | 48.5 | 11.7 / 2.6 / 19.0 |
| 8259CL, 6x32GB | bfp8 | 30.4 t/s | 33.1 | 33.8 | 21.5 / 2.4 / 11.4 |
| 8259CL, 6x32GB | bfp4 | 44.1 t/s | 48.9 | 50.4 | 11.7 / 2.6 / 10.9 |
