# Golden Summary (M0–M3)

All results are full runs (OpenBookQA: 500 samples, ARC-C: 1150 samples).

| Milestone | Scheme | Cache proportion | Dataset | Accuracy |
|---|---|---|---|---|
| M0 baseline | fp16 | 1.0 | openbookqa | 0.528 |
| M0 baseline | fp16 | 1.0 | arc_c | 0.551 |
| M2 int8 | int8 | 1.0 | openbookqa | 0.528 |
| M2 int8 | int8 | 1.0 | arc_c | 0.550 |
| M2 int4 | int4 | 1.0 | openbookqa | 0.526 |
| M2 int4 | int4 | 1.0 | arc_c | 0.554 |
| M3 back | int8 | 0.10 | openbookqa | 0.492 |
| M3 back | int8 | 0.10 | arc_c | 0.537 |
| M3 front | int8 | 0.10 | openbookqa | 0.386 |
| M3 front | int8 | 0.10 | arc_c | 0.407 |
| M3 back | int8 | 0.25 | openbookqa | 0.508 |
| M3 back | int8 | 0.25 | arc_c | 0.562 |
| M3 front | int8 | 0.25 | openbookqa | 0.388 |
| M3 front | int8 | 0.25 | arc_c | 0.383 |
| M3 back | int8 | 0.50 | openbookqa | 0.520 |
| M3 back | int8 | 0.50 | arc_c | 0.572 |
| M3 front | int8 | 0.50 | openbookqa | 0.430 |
| M3 front | int8 | 0.50 | arc_c | 0.463 |
| M3 back | int8 | 0.75 | openbookqa | 0.522 |
| M3 back | int8 | 0.75 | arc_c | 0.557 |
| M3 front | int8 | 0.75 | openbookqa | 0.446 |
| M3 front | int8 | 0.75 | arc_c | 0.402 |
| M3 back | int8 | 1.00 | openbookqa | 0.528 |
| M3 back | int8 | 1.00 | arc_c | 0.550 |
| M3 front | int8 | 1.00 | openbookqa | 0.528 |
| M3 front | int8 | 1.00 | arc_c | 0.550 |

## Extension Summary (M5–M8)

These are full runs. M7 includes alignment ablation on the same model pair and heterogeneity with alignment on (alignment-off failed for the hetero pair).

| Milestone | Setting | Dataset | Accuracy | Notes |
|---|---|---|---|---|
| M5 QAT | projector QAT int8 | openbookqa | 0.396 | Full run |
| M5 QAT | projector QAT int8 | arc_c | 0.402 | Full run |
| M6 mixed precision | INT8 + last4 layers FP16 | openbookqa | 0.528 | Full run |
| M6 mixed precision | INT8 + last4 layers FP16 | arc_c | 0.553 | Full run |
| M6 mixed precision | INT8 + last2 layers FP16 | openbookqa | 0.530 | Full run |
| M6 mixed precision | INT8 + last2 layers FP16 | arc_c | 0.550 | Full run |
| M6 mixed precision | INT8 + last8 layers FP16 | openbookqa | 0.524 | Full run |
| M6 mixed precision | INT8 + last8 layers FP16 | arc_c | 0.552 | Full run |
| M7 alignment ablation | alignment off | openbookqa | 0.528 | Same-model pair |
| M7 alignment ablation | alignment off | arc_c | 0.550 | Same-model pair |
| M7 alignment ablation | alignment on | openbookqa | 0.468 | Same-model pair |
| M7 alignment ablation | alignment on | arc_c | 0.496 | Same-model pair |
| M7 heterogeneity | alignment on | openbookqa | 0.442 | Qwen3->Llama3.2 |
| M7 heterogeneity | alignment on | arc_c | 0.478 | Qwen3->Llama3.2 |
| M8 selective transfer | int8 front p=0.5 | openbookqa | 0.448 | Full run |
| M8 selective transfer | int8 front p=0.5 | arc_c | 0.476 | Full run |
| M8 selective transfer | int8 vnorm_topk p=0.5 | openbookqa | 0.524 | Full run |
| M8 selective transfer | int8 vnorm_topk p=0.5 | arc_c | 0.562 | Full run |
| M8 selective transfer | int8 knorm_topk p=0.5 | openbookqa | 0.518 | Full run |
| M8 selective transfer | int8 knorm_topk p=0.5 | arc_c | 0.563 | Full run |
| M8 selective transfer | int8 proj_vnorm_topk p=0.5 | openbookqa | 0.502 | Full run |
| M8 selective transfer | int8 proj_vnorm_topk p=0.5 | arc_c | 0.541 | Full run |
| M8 selective transfer | int8 random p=0.5 | openbookqa | 0.500 | Full run |
| M8 selective transfer | int8 random p=0.5 | arc_c | 0.530 | Full run |
| M8 selective transfer | int4 front p=0.5 | openbookqa | 0.446 | Full run |
| M8 selective transfer | int4 front p=0.5 | arc_c | 0.478 | Full run |
| M8 selective transfer | int4 vnorm_topk p=0.5 | openbookqa | 0.520 | Full run |
| M8 selective transfer | int4 vnorm_topk p=0.5 | arc_c | 0.563 | Full run |
