# Toy Query-Pool Benchmark

| Scenario | Method | Budget | Task acc | Rec MSE | Route entropy | Collision | Dead slots | Top margin |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| aligned | topk | 4 | 0.2396 | 0.9595 | 1.3043 | 0.0781 | 0.0000 | 0.3866 |
| aligned | query_pool | 4 | 0.3125 | 1.0469 | 1.0386 | 0.0990 | 0.0938 | 0.5257 |
| rotated | topk | 4 | 0.2760 | 0.9135 | 1.2909 | 0.0677 | 0.0312 | 0.3968 |
| rotated | query_pool | 4 | 0.3125 | 0.9707 | 1.0754 | 0.1198 | 0.0312 | 0.4867 |
| outlier | topk | 4 | 0.2448 | 0.9422 | 1.0643 | 0.1250 | 0.0000 | 0.5368 |
| outlier | query_pool | 4 | 0.2917 | 0.9818 | 1.2435 | 0.1042 | 0.0000 | 0.4467 |
| slot_permuted | topk | 4 | 0.2604 | 0.9424 | 1.2936 | 0.0833 | 0.0000 | 0.3915 |
| slot_permuted | query_pool | 4 | 0.3594 | 1.0183 | 1.0697 | 0.1146 | 0.0625 | 0.4791 |
