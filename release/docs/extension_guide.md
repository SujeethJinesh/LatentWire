# Extension Guide

## Add a Method

Create `src/outlier_migrate/methods/new_method.py` with a function accepting
`scores: np.ndarray` and `budget: int`, returning a boolean mask. Register it:

```python
from outlier_migrate.methods import register_method

register_method("new_method", new_method)
```

Use `methods/m11b.py` as the template for budget-tuned methods.

## Add a Model

Add a config in `configs/` with `model.name`, `model.architecture`,
`model.hidden_size`, and `model.layers`. If the architecture needs custom
hooking, extend `models.py` with a new adapter function while keeping the
script interface unchanged.

## Add a Metric

Add a pure function to `metrics.py` and a focused CPU test. Reproduction
scripts should write the metric into a JSON output with an explicit tolerance
in `docs/reproducing_results.md`.

## Add a Quantization Format

Add format-specific helpers beside `symmetric_int4_quantize` in
`quantization.py`. Keep protected-channel masks format-independent so FP8,
MXFP4, or NVFP4 experiments can reuse method code.

## Rotation Plus Budget Composition

`methods/composition.py` reserves the ICLR follow-up interface for combining
ParoQuant-style rotations with M11b-style budget selection. The intended flow
is:

```python
rotated_weights, protected_mask = compose_rotation_with_budget(
    weights, channel_scores, budget
)
```

The stub is intentionally unimplemented in this workshop release.
