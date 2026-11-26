# Card ops per decode pass (device/opcount.py)

Every routed expert in a slot (the most ops a pass can take); a cold expert costs the host instead.

m = 1 token(s) per pass

| part | card ops | host transfers | through the host (TTNN, for now) | trace replays | QFN count | QFN card ops |
|---|---:|---:|---:|---:|---:|---:|
| DeltaNet layer | 8 | 1 | 0 | 1 | 36 | 288 |
| attention layer | 91 | 3 | 6 | 2 | 12 | 1,092 |
| PLE (layer 1) | 32 | 1 | 0 | 0 | 1 | 32 |
| routed experts in slots, per expert | 6 | 0 | 0 | 0 | 480 | 2,640 |
| LM head + argmax | 2 | 1 | 0 | 0 | 1 | 2 |
| **one pass** | | | | | | **4,054** ops, 194 host transfers, 72 through the host, 60 trace replays |

m = 3 token(s) per pass

| part | card ops | host transfers | through the host (TTNN, for now) | trace replays | QFN count | QFN card ops |
|---|---:|---:|---:|---:|---:|---:|
| DeltaNet layer | 8 | 1 | 0 | 1 | 36 | 288 |
| attention layer | 223 | 7 | 12 | 2 | 12 | 2,676 |
| PLE (layer 1) | 46 | 1 | 0 | 0 | 1 | 46 |
| routed experts in slots, per expert | 5 | 0 | 0 | 0 | 1065 | 4,980 |
| LM head + argmax | 2 | 1 | 0 | 0 | 1 | 2 |
| **one pass** | | | | | | **7,992** ops, 242 host transfers, 144 through the host, 60 trace replays |
