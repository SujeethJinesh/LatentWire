# Toy Query-Pool vs Top-k Selection for Synthetic KV Transport

## Purpose

This memo defines a small synthetic experiment to isolate one question:

> When the bridge budget is fixed, is LatentWire better served by a sparse `top-k` selector over raw KV slots, or by a small learned `query_pool_transport` interface that first pools query-conditioned evidence and then emits a transport correction?

The experiment is intentionally toy-sized so we can measure route geometry, collapse, and reconstruction behavior before spending another large evaluation budget.

## Synthetic setup

Let each example contain a latent semantic state `z_i \in \mathbb{R}^d`, a source KV bank, and a target query:

```math
z_i \sim \mathcal{N}(0, I_d)
```

```math
k_{i,j} = A_s z_i + \epsilon_{i,j}, \qquad
v_{i,j} = B_s z_i + \eta_{i,j}
```

for source slots `j = 1..M`, with small isotropic noise `\epsilon_{i,j}, \eta_{i,j}`. The target-side query is:

```math
q_i = C_t z_i + \xi_i
```

with `C_t` optionally chosen to be:

1. aligned with `A_s, B_s`,
2. permuted by a hidden head/slot shuffle,
3. rotated by an orthogonal gauge matrix,
4. or mismatched with a mild nonlinearity.

The label is either:

```math
y_i = \arg\max_r (W z_i)_r
```

or a reconstruction target:

```math
\hat z_i \approx z_i
```

depending on whether the run is classification-like or communication-like.

## Two bridge families

### 1) Top-k selector

Score each source slot:

```math
s_{i,j} = \frac{q_i^\top k_{i,j}}{\sqrt{d}}
```

Select the top `k` slots:

```math
\mathcal{I}_i = \mathrm{TopK}(s_i, k)
```

and emit the transported summary:

```math
h_i^{(\mathrm{topk})} = \sum_{j \in \mathcal{I}_i} \alpha_{i,j} v_{i,j},
 \qquad
\alpha_{i,j} = \mathrm{softmax}_{j \in \mathcal{I}_i}(s_{i,j} / \tau)
```

This is the sparse, index-based baseline. It is cheap, but it is vulnerable to selector collapse if the same few slots always win.

### 2) Query-pool transport

Introduce a small query pool `P = {p_m}_{m=1}^P`, with learnable pool vectors:

```math
u_i = \phi(q_i), \qquad
\beta_i = \mathrm{softmax}\left(\frac{u_i P^\top}{\tau}\right)
```

Then use pooled transport weights:

```math
h_i^{(\mathrm{pool})} = \sum_{m=1}^P \beta_{i,m} \, T_m \Big(\sum_{j=1}^M \omega_{i,m,j} v_{i,j}\Big)
```

where `T_m` is a small per-pool transform and `\omega_{i,m}` is either:

- a deterministic pool-specific attention over slots, or
- a low-rank router from `u_i` to `M` slots.

The key distinction is that `top-k` selects slots directly, while `query_pool_transport` first forms a compact query-conditioned interface and only then emits transport.

## Losses

Use the same core loss for both families:

```math
\mathcal{L} = \mathcal{L}_{task} + \lambda_{rec}\mathcal{L}_{rec} + \lambda_{ent}\mathcal{L}_{ent} + \lambda_{cov}\mathcal{L}_{cov}
```

with:

```math
\mathcal{L}_{task} = \mathrm{CE}(g(h_i), y_i)
```

```math
\mathcal{L}_{rec} = \| \hat z_i - z_i \|_2^2
```

```math
\mathcal{L}_{ent} = - \frac{1}{N} \sum_i H(\beta_i)
```

```math
\mathcal{L}_{cov} = \sum_{m=1}^P \max(0, \rho - \mathbb{E}_i[\beta_{i,m}])
```

where `\mathcal{L}_{ent}` can be used either as an entropy floor or entropy penalty depending on whether we want to discourage collapse or encourage sharp routing.

## Expected measurements

Run the experiment at matched budget for all methods and report:

1. `task_acc`: classification accuracy or exact recovery rate.
2. `rec_mse`: latent reconstruction error.
3. `route_entropy`: mean entropy of the selector or pool weights.
4. `topk_margin`: gap between the k-th and (k+1)-th scores for top-k selection.
5. `slot_collision_rate`: fraction of examples whose chosen slots overlap with the dataset mode.
6. `dead_slot_rate`: fraction of unused slots or unused pool atoms.
7. `perm_robustness`: performance under source-slot permutation or orthogonal gauge transforms.
8. `budget_curve`: accuracy versus selected-slot budget `k` or pool size `P`.

## Toy hypotheses

The toy experiment should answer these questions cleanly:

1. If the source slots are permuted or gauge-rotated, does `top-k` break more quickly than `query_pool_transport`?
2. Does a small learned query pool reduce selector collapse by spreading mass over semantically distinct route atoms?
3. Is there an intermediate budget where pooling dominates sparse selection even when raw accuracy is close?
4. Does the pooled interface preserve latent geometry better, as measured by reconstruction and permutation robustness?

## Mapping to LatentWire

This maps directly onto the current bridge stack:

- `top-k selector` corresponds to the existing position/slot selection logic in the route and selector traces.
- `query_pool_transport` corresponds to a `runtime_query_features`-conditioned bridge bank, where a small query interface selects transport atoms before emission.
- `route_entropy`, `dead_slot_rate`, and `collision_rate` correspond to the current selector-collapse telemetry.
- `perm_robustness` corresponds to the gauge/permutation controls already motivating the symmetry and quotient-space memos.
- `rec_mse` corresponds to the `latent_kv_bottleneck` and `reconstructive_bridge_adapter` line of attack.

## Why this toy experiment is useful

The current LatentWire negatives suggest that the bridge is not failing because it is too small in the abstract. It is failing because the routing interface is not stable under collapse, permutation, and tokenizer mismatch. This toy setup separates those issues:

- `top-k` tests sparse slot selection.
- `query_pool_transport` tests whether a small learned interface is more stable than direct slot selection.
- the synthetic gauge and permutation controls tell us whether the bridge is actually learning communication or merely memorizing an index order.

If `query_pool_transport` wins on the toy problem under permutation/gauge stress, then it is worth the larger `LatentWire` ablation. If it does not, then the next move should be `headwise_route_atom` or `byte_probe_bridge`, not more dense transport.

## Minimal implementation sketch

If we decide to code this later, the simplest version is:

1. Sample `z_i`, build source K/V slots and target queries.
2. Fit `top-k` and `query_pool_transport` with the same hidden width and same slot budget.
3. Evaluate under aligned, permuted, and orthogonally rotated source spaces.
4. Record the metrics above and compare budget curves.

That gives a clean go/no-go for whether the next real LatentWire branch should be query pooling or headwise route atoms.
