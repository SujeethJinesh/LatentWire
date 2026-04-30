from __future__ import annotations

from scripts.build_source_private_product_codebook_geometry_gate import (
    _dimension_utilities,
    _groups_for_variant,
    run_geometry_gate,
)
from scripts.run_source_private_hidden_repair_packet_smoke import make_benchmark
from scripts.run_source_private_tool_trace_compression_baselines import _fit_ridge_encoder_for_view


def test_geometry_groups_cover_each_dimension_once() -> None:
    train_rows = make_benchmark(examples=32, candidates=4, seed=5, family_set="all")
    encoder = _fit_ridge_encoder_for_view(
        train_rows,
        feature_dim=64,
        ridge=1e-2,
        candidate_view="slot",
        fit_intercept=False,
    )
    utilities = _dimension_utilities(train_rows, encoder=encoder, feature_dim=64, candidate_view="slot")

    groups = _groups_for_variant(
        variant="utility_balanced",
        feature_dim=64,
        budget_bytes=4,
        utilities=utilities,
        seed=7,
    )

    flattened = sorted(int(dim) for group in groups for dim in group)
    assert flattened == list(range(64))
    assert len(groups) == 4
    assert all(len(group) > 0 for group in groups)


def test_opq_groups_are_canonical_partitions_for_rotated_space() -> None:
    utilities = [0.0] * 64

    groups = _groups_for_variant(
        variant="opq_procrustes",
        feature_dim=64,
        budget_bytes=4,
        utilities=utilities,
        seed=7,
    )

    assert [len(group) for group in groups] == [16, 16, 16, 16]
    assert sorted(int(dim) for group in groups for dim in group) == list(range(64))


def test_protected_hadamard_groups_are_canonical_partitions_for_rotated_space() -> None:
    utilities = [0.0] * 64

    groups = _groups_for_variant(
        variant="protected_hadamard",
        feature_dim=64,
        budget_bytes=4,
        utilities=utilities,
        seed=7,
    )

    assert [len(group) for group in groups] == [16, 16, 16, 16]
    assert sorted(int(dim) for group in groups for dim in group) == list(range(64))


def test_product_codebook_geometry_gate_writes_artifacts(tmp_path) -> None:
    payload = run_geometry_gate(
        output_dir=tmp_path / "geometry",
        train_examples=64,
        eval_examples=32,
        train_seed=5,
        eval_seed=6,
        remap_seeds=[101],
        budgets=[2],
        variants=["canonical", "utility_round_robin"],
        train_family_set="all",
        eval_family_set="all",
        candidates=4,
        feature_dim=64,
        ridge=1e-2,
        candidate_view="slot",
        fit_intercept=False,
        bootstrap_samples=64,
        opq_iterations=1,
    )

    assert payload["gate"] == "source_private_product_codebook_geometry_gate"
    assert len(payload["rows"]) == 2
    assert payload["rows"][0]["variant"] == "canonical"
    assert "source_minus_canonical" in payload["rows"][1]
    assert (tmp_path / "geometry" / "product_codebook_geometry_gate.md").exists()
    assert (tmp_path / "geometry" / "remap_101" / "budget_2" / "canonical" / "predictions.jsonl").exists()
