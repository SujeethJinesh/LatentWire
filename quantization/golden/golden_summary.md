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
| M3 back | int8 | 1.0 | openbookqa | 0.528 |
| M3 back | int8 | 1.0 | arc_c | 0.550 |
| M3 back | int8 | 0.75 | openbookqa | 0.522 |
| M3 back | int8 | 0.75 | arc_c | 0.557 |
| M3 back | int8 | 0.50 | openbookqa | 0.520 |
| M3 back | int8 | 0.50 | arc_c | 0.572 |
| M3 back | int8 | 0.25 | openbookqa | 0.508 |
| M3 back | int8 | 0.25 | arc_c | 0.562 |
| M3 back | int8 | 0.10 | openbookqa | 0.492 |
| M3 back | int8 | 0.10 | arc_c | 0.537 |
| M3 front | int8 | 1.0 | openbookqa | 0.528 |
| M3 front | int8 | 1.0 | arc_c | 0.550 |
| M3 front | int8 | 0.75 | openbookqa | 0.446 |
| M3 front | int8 | 0.75 | arc_c | 0.402 |
| M3 front | int8 | 0.50 | openbookqa | 0.430 |
| M3 front | int8 | 0.50 | arc_c | 0.463 |
| M3 front | int8 | 0.25 | openbookqa | 0.388 |
| M3 front | int8 | 0.25 | arc_c | 0.383 |
| M3 front | int8 | 0.10 | openbookqa | 0.386 |
| M3 front | int8 | 0.10 | arc_c | 0.407 |
