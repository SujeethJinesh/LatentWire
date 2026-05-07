from types import SimpleNamespace

import torch

from experimental.shared.ssq_lr_s2_state_replay_scout import (
    _byte_plan,
    _candidate_recipe_specs,
    _quantize_mixed_int3_mxfp4_low_error,
)


def _cache_with_states(*states: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(
        layers=[SimpleNamespace(recurrent_states=state.clone()) for state in states]
    )


def test_s2_mixed_int3_mxfp4_plan_counts_mask_and_scale_bytes() -> None:
    cache = _cache_with_states(torch.linspace(-2, 2, 256), torch.linspace(-1, 1, 256))

    plan = _byte_plan(
        cache,
        (0, 1),
        precision="mixed_int3_mxfp4_low_error_10pct",
        block_size=128,
    )

    total_bytes = plan["quantized_state_bytes"] + plan["scale_bytes"] + plan["metadata_bytes"]
    assert plan["metadata_bytes"] == 2.0
    assert plan["scale_bytes"] == 8.0
    assert plan["bf16_state_bytes"] / total_bytes > 4.0
    assert plan["effective_bits"] < 4.0


def test_s2_mixed_int3_mxfp4_quantizer_preserves_shape_and_dtype() -> None:
    state = torch.randn(3, 257, dtype=torch.float16)

    quantized = _quantize_mixed_int3_mxfp4_low_error(
        state,
        block_size=64,
        int3_fraction=0.10,
    )

    assert quantized.shape == state.shape
    assert quantized.dtype == state.dtype
    assert torch.isfinite(quantized.float()).all()


def test_s2_candidate_specs_include_mixed_low_error_recipes() -> None:
    recipe_ids = {spec["recipe_id"] for spec in _candidate_recipe_specs((0, 12, 30))}

    assert "mixed_int3_mxfp4_low_error_10pct" in recipe_ids
    assert "mixed_int3_mxfp4_low_error_25pct" in recipe_ids
