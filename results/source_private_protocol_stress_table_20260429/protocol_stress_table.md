# Source-Private Protocol Stress Table

- rows: `22`
- pass rows: `15`
- fail / near-miss rows: `7`
- minimum passing delta vs target: `0.213`
- maximum passing payload bytes: `16.0`
- stress families: `canonical candidate-order remap, diagnostic codebook remap, slot-feature remap`

## Rows

| Family | Stressor | Surface | Status | N | Bytes | Accuracy | Target | Best control | Delta vs target | Note |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| deterministic diagnostic packet | diagnostic codebook remap | seed 29 budget 2 | `pass` | 500 | 2.0 | 1.000 | 0.250 | 0.254 | 0.750 | codebook=a1521325; preview=G0,H1,J2,K3 |
| deterministic diagnostic packet | diagnostic codebook remap | seed 29 budget 4 | `pass` | 500 | 4.0 | 1.000 | 0.250 | 0.256 | 0.750 | codebook=a1521325; preview=G0,H1,J2,K3 |
| deterministic diagnostic packet | diagnostic codebook remap | seed 29 budget 8 | `pass` | 500 | 8.0 | 1.000 | 0.250 | 0.252 | 0.750 | codebook=a1521325; preview=G0,H1,J2,K3 |
| deterministic diagnostic packet | diagnostic codebook remap | seed 29 budget 16 | `pass` | 500 | 16.0 | 1.000 | 0.250 | 0.252 | 0.750 | codebook=a1521325; preview=G0,H1,J2,K3 |
| deterministic diagnostic packet | diagnostic codebook remap | seed 31 budget 2 | `pass` | 500 | 2.0 | 1.000 | 0.250 | 0.252 | 0.750 | codebook=93fa3909; preview=J0,K1,L2,M3 |
| deterministic diagnostic packet | diagnostic codebook remap | seed 31 budget 4 | `pass` | 500 | 4.0 | 1.000 | 0.250 | 0.252 | 0.750 | codebook=93fa3909; preview=J0,K1,L2,M3 |
| deterministic diagnostic packet | diagnostic codebook remap | seed 31 budget 8 | `pass` | 500 | 8.0 | 1.000 | 0.250 | 0.254 | 0.750 | codebook=93fa3909; preview=J0,K1,L2,M3 |
| deterministic diagnostic packet | diagnostic codebook remap | seed 31 budget 16 | `pass` | 500 | 16.0 | 1.000 | 0.250 | 0.254 | 0.750 | codebook=93fa3909; preview=J0,K1,L2,M3 |
| deterministic diagnostic packet | diagnostic codebook remap | seed 37 budget 2 | `pass` | 500 | 2.0 | 1.000 | 0.250 | 0.250 | 0.750 | codebook=a5fe24f3; preview=Q0,R1,S2,T3 |
| deterministic diagnostic packet | diagnostic codebook remap | seed 37 budget 4 | `pass` | 500 | 4.0 | 1.000 | 0.250 | 0.252 | 0.750 | codebook=a5fe24f3; preview=Q0,R1,S2,T3 |
| deterministic diagnostic packet | diagnostic codebook remap | seed 37 budget 8 | `pass` | 500 | 8.0 | 1.000 | 0.250 | 0.252 | 0.750 | codebook=a5fe24f3; preview=Q0,R1,S2,T3 |
| deterministic diagnostic packet | diagnostic codebook remap | seed 37 budget 16 | `pass` | 500 | 16.0 | 1.000 | 0.250 | 0.252 | 0.750 | codebook=a5fe24f3; preview=Q0,R1,S2,T3 |
| learned scalar packet | slot-feature remap | remap 101 | `pass` | 512 | 6.0 | 0.463 | 0.250 | 0.264 | 0.213 | raw_sign=0.332; ci_low_vs_target=0.156 |
| learned scalar packet | slot-feature remap | remap 103 | `pass` | 512 | 6.0 | 0.508 | 0.250 | 0.266 | 0.258 | raw_sign=0.316; ci_low_vs_target=0.199 |
| learned scalar packet | slot-feature remap | remap 107 | `pass` | 512 | 6.0 | 0.492 | 0.250 | 0.250 | 0.242 | raw_sign=0.330; ci_low_vs_target=0.186 |
| canonical RASP relative-score packet | canonical candidate-order remap | remap 101 | `near-miss` | 512 | 4.0 | 0.494 | 0.250 | 0.295 | 0.244 | scalar=0.426; relative_minus_scalar=0.068; ci_low_vs_target=0.184 |
| canonical RASP relative-score packet | canonical candidate-order remap | remap 103 | `near-miss` | 512 | 4.0 | 0.520 | 0.250 | 0.256 | 0.270 | scalar=0.496; relative_minus_scalar=0.023; ci_low_vs_target=0.213 |
| canonical RASP relative-score packet | canonical candidate-order remap | remap 107 | `near-miss` | 512 | 4.0 | 0.506 | 0.250 | 0.355 | 0.256 | scalar=0.502; relative_minus_scalar=0.004; ci_low_vs_target=0.199 |
| canonical RASP relative-score packet | canonical candidate-order remap | remap 109 | `near-miss` | 512 | 4.0 | 0.477 | 0.250 | 0.279 | 0.227 | scalar=0.451; relative_minus_scalar=0.025; ci_low_vs_target=0.170 |
| canonical RASP relative-score packet | canonical candidate-order remap | remap 113 | `near-miss` | 512 | 4.0 | 0.473 | 0.250 | 0.279 | 0.223 | scalar=0.436; relative_minus_scalar=0.037; ci_low_vs_target=0.164 |
| canonical RASP relative-score packet | canonical candidate-order remap | remap 127 | `near-miss` | 512 | 4.0 | 0.453 | 0.250 | 0.275 | 0.203 | scalar=0.428; relative_minus_scalar=0.025; ci_low_vs_target=0.146 |
| canonical RASP relative-score packet | canonical candidate-order remap | remap 131 | `near-miss` | 512 | 4.0 | 0.506 | 0.250 | 0.281 | 0.256 | scalar=0.434; relative_minus_scalar=0.072; ci_low_vs_target=0.197 |

## Open Gap

This table covers deterministic codebook remap, learned slot-feature remap, and canonical candidate-order remap. It does not yet cover learned target-decoder prompt paraphrases; that remains the next protocol-generalization stress.
