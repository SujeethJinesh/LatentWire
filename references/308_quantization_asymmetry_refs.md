# 308 Quantization / Asymmetric KV Inspiration Memo

Primary-source memo for quantization ideas that can become LatentWire
ablations. The point is not to claim quantization as the method; it is to steal
the mathematical controls that make compressed state usable: calibration-aware
scales, outlier isolation, rotations, mixed precision, and asymmetric K/V
allocation.

## Key references

| Method | Primary source | Mechanism worth stealing | LatentWire ablation |
|---|---|---|---|
| **AWQ** | [paper](https://arxiv.org/abs/2306.00978), [code](https://github.com/mit-han-lab/llm-awq) | Protect salient channels using activation-aware calibration instead of uniform quantization | Calibrate head/slot salience from target behavior; compare protected-head routing against uniform and shuffled salience controls. |
| **SmoothQuant** | [paper](https://arxiv.org/abs/2211.10438), [code](https://github.com/mit-han-lab/smoothquant) | Move activation outliers into weights by a diagonal rescaling | Use bounded preconditioning/gauge transforms before transport; log scale range, norm ratio, cosine drift, and entropy collapse. |
| **KIVI** | [paper](https://arxiv.org/abs/2402.02750), [code](https://github.com/jy-yuan/KIVI) | Treat keys and values differently because their quantization/error structure differs | Split route budget and value budget; report separate K-style routing fidelity and V-style reconstruction fidelity. |
| **KVQuant** | [paper](https://arxiv.org/abs/2401.18079), [code](https://github.com/SqueezeAILab/KVQuant) | Keep long-context KV cache usable by outlier-aware non-uniform quantization | Add outlier-channel telemetry and test whether route failures are caused by rare high-magnitude channels. |
| **QServe** | [paper](https://arxiv.org/abs/2405.04532), [code](https://github.com/mit-han-lab/qserve) | Joint system/quantization design with low-bit weights, activations, and KV cache | Separate algorithmic quality from bytes/latency; every result needs accuracy, transferred bytes, and cache-layout cost. |
| **QuaRot** | [paper](https://arxiv.org/abs/2404.00456), [code](https://github.com/spcl/QuaRot) | Use rotations to remove outliers and make low-bit inference stable | Test orthogonal or Hadamard-style gauge fixes before alignment; compare against free diagonal scaling. |
| **SpinQuant** | [paper](https://arxiv.org/abs/2405.16406), [code](https://github.com/facebookresearch/SpinQuant) | Learn rotations that improve quantization robustness | Try learned near-orthogonal rotations as a constrained alignment/preconditioning layer, not an unconstrained adapter. |
| **EXL2 / ExLlamaV2** | [code](https://github.com/turboderp-org/exllamav2) | Practical mixed-bit allocation with variable bitrate targets | Use variable head/slot budgets and report a frontier instead of one operating point. |

## LatentWire takeaways

- **K and V should not share one budget by default.** K-like signals choose where
  to route; V-like signals carry content. A single top-k budget can be wrong
  even when total bytes are fixed.
- **Free diagonal scaling is too dangerous.** It can improve aligned toy data,
  but it changes norms and route entropy enough to break rotated/outlier
  stresses. Bounded or orthogonal preconditioning is the safer control.
- **Outlier channels need explicit telemetry.** Log per-head max/median norm
  ratios, outlier-channel mass, retained outlier fraction, and whether
  failures concentrate in high-magnitude coordinates.
- **Bytes need to be first-class.** Quantization papers win by showing quality
  at matched memory/latency. LatentWire should report accuracy, transferred KV
  bytes, retained fraction, latency, and calibration compute for every branch.

## Concrete ablations

1. **Asymmetric K/V budget sweep**: run `(route,value)` splits like `(1,3)`,
   `(2,2)`, `(3,1)`, and `(2,4)` under the same total or explicit byte budget.
2. **Bounded diagonal vs orthogonal preconditioning**: compare free diagonal,
   bounded diagonal, Hadamard/random orthogonal, and learned near-orthogonal
   transforms before routing.
3. **Outlier-protected routing**: reserve a small budget for high-norm
   heads/channels and compare against pure query-aware routing.
4. **Mixed-bit / mixed-byte frontier**: emulate EXL2-style variable allocation
   by giving high-salience heads more slots or precision and low-salience heads
   fewer slots.
5. **No-op cost controls**: match bytes with random protected heads and
   shuffled salience maps so improvements cannot be explained by spending more
   cache.

Interpretability fields to keep:

- `kv_route_budget`
- `kv_value_budget`
- `kv_route_entropy`
- `kv_value_entropy`
- `kv_route_value_overlap`
- `kv_route_value_jaccard`
- `kv_route_value_kl`
- `kv_route_value_cosine`
- `kv_gate_mean`
- `outlier_mass_retained`
- `scale_norm_ratio`
- `precondition_cosine_drift`
