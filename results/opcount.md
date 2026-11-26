# Card ops per decode pass (device/opcount.py)

Every routed expert in a slot; a cold expert adds a host read and write per layer instead.

m = 1 token(s) per pass

| part | card ops | host reads | host writes | trace replays | in QFN | QFN card ops |
|---|---:|---:|---:|---:|---:|---:|
| DeltaNet layer (its MoE but the slot experts) | 3 | 1 | 0 | 1 | 36 | 108 |
| attention layer (its MoE but the slot experts) | 29 | 2 | 2 | 4 | 12 | 348 |
| routed experts in slots, per layer (any number) | 3 | 0 | 2 | 1 | 48 | 144 |
| PLE (layer 1) | 2 | 0 | 1 | 1 | 1 | 2 |
| LM head + argmax | 2 | 1 | 0 | 0 | 1 | 2 |
| **one pass** | | | | | | **604** ops, 61 host reads, 121 host writes, 133 trace replays |

m = 3 token(s) per pass

| part | card ops | host reads | host writes | trace replays | in QFN | QFN card ops |
|---|---:|---:|---:|---:|---:|---:|
| DeltaNet layer (its MoE but the slot experts) | 3 | 1 | 0 | 1 | 36 | 108 |
| attention layer (its MoE but the slot experts) | 43 | 4 | 6 | 8 | 12 | 516 |
| routed experts in slots, per layer (any number) | 3 | 0 | 2 | 1 | 48 | 144 |
| PLE (layer 1) | 2 | 0 | 1 | 1 | 1 | 2 |
| LM head + argmax | 2 | 1 | 0 | 0 | 1 | 2 |
| **one pass** | | | | | | **772** ops, 85 host reads, 169 host writes, 181 trace replays |

m = 24 token(s) per pass

| part | card ops | host reads | host writes | trace replays | in QFN | QFN card ops |
|---|---:|---:|---:|---:|---:|---:|
| DeltaNet layer (its MoE but the slot experts) | 106 | 2 | 0 | 0 | 36 | 3,810 |
| attention layer (its MoE but the slot experts) | 152 | 2 | 2 | 0 | 12 | 1,830 |
| routed experts in slots, per layer (any number) | 10 | 0 | 2 | 0 | 48 | 480 |
| PLE (layer 1) | 40 | 0 | 1 | 0 | 1 | 40 |
| LM head + argmax | 2 | 1 | 0 | 0 | 1 | 2 |
| **one pass** | | | | | | **6,162** ops, 85 host reads, 145 host writes, 0 trace replays |
