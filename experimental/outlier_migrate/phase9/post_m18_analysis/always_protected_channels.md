# Post-M18 Analysis 5: Always-Protected Channels
## Method
For each model/layer, I intersected the prompt-averaged top-1% channel sets across all decode positions available in the packet. The reported fraction is intersection size divided by the layer top-1% count at position 100.
| Model | Source packet | Positions | Mean always-protected fraction | Median | Layers with zero stable core | Mean always-protected count |
|---|---|---|---:|---:|---:|---:|
| Granite-Small Phase 1 | `experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z` | `[100, 500, 1000, 5000, 10000, 20000]` | 0.516463414634 | 0.512195121951 | 0/40 | 21.175000000000 |
| Nemotron-3-Nano Phase 2 | `experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z` | `[100, 500, 1000, 5000, 10000, 20000]` | 0.616809116809 | 0.611111111111 | 0/52 | 16.653846153846 |
| DeepSeek-R1-Distill-Qwen-1.5B Phase 5' | `experimental/outlier_migrate/phase5_prime/results/om_phase5p_20260512T053800Z` | `[100, 500, 1000, 5000, 10000, 20000]` | 0.508928571429 | 0.500000000000 | 0/28 | 8.142857142857 |
| Falcon-H1 Phase 7 | `experimental/outlier_migrate/phase7/results/om_phase7_falcon_h1_20260512T223600Z` | `[100, 500, 1000, 5000, 10000, 20000]` | 0.570707070707 | 0.545454545455 | 0/36 | 6.277777777778 |

## Interpretation
Every measured model retains some stable high-magnitude core, but that core is far smaller than the full top-1% protected set. This is consistent with the main decomposition: static protection can preserve a small persistent core while still missing many channels that become important later. A static core-only method would likely have too little coverage; a larger-budget method is exactly what M11b is designed to test next.
