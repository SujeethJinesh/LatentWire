import json

import numpy as np

from scripts import build_source_private_arc_challenge_hidden_query_common_basis_gate as gate
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate


def _rows() -> list[arc_gate.ArcRow]:
    return [
        arc_gate.ArcRow(
            row_id="r0",
            content_id="c0",
            question="q0",
            choices=("a", "b", "c", "d"),
            choice_labels=("A", "B", "C", "D"),
            answer_index=1,
            answer_label="B",
        ),
        arc_gate.ArcRow(
            row_id="r1",
            content_id="c1",
            question="q1",
            choices=("e", "f", "g", "h"),
            choice_labels=("A", "B", "C", "D"),
            answer_index=2,
            answer_label="C",
        ),
    ]


def test_flat_source_view_and_candidate_roll_shapes() -> None:
    rows = _rows()
    hidden = np.arange(2 * 4 * 3, dtype=np.float64).reshape(2, 4, 3) + 1.0
    query = np.arange(2 * 4 * 2, dtype=np.float64).reshape(2, 4, 2) + 2.0
    state = gate.HiddenQueryState(hidden=hidden, query=query, metadata={})

    hidden_view = gate._flat_source_view(rows=rows, state=state, view="hidden_residual")
    query_view = gate._flat_source_view(rows=rows, state=state, view="query_residual")
    combined_view = gate._flat_source_view(rows=rows, state=state, view="hidden_query_residual")

    assert hidden_view.shape == (8, 3)
    assert query_view.shape == (8, 2)
    assert combined_view.shape == (8, 5)
    assert np.allclose(np.linalg.norm(hidden_view, axis=1), 1.0)

    rolled = gate._roll_candidate_features(rows, hidden_view)
    assert rolled.shape == hidden_view.shape
    assert np.allclose(rolled[0], hidden_view[1])
    assert np.allclose(rolled[3], hidden_view[0])


def test_pca_ridge_map_learns_synthetic_linear_target() -> None:
    rng = np.random.default_rng(7)
    source = rng.normal(size=(12, 5))
    true_w = rng.normal(size=(5, 3))
    target = source @ true_w
    mapper = gate._fit_pca_ridge_map(
        source_features=source,
        target_features=target,
        fit_flat_indices=np.arange(12),
        pca_dim=5,
        ridge=1e-6,
    )
    predicted = gate._apply_pca_ridge_map(source, mapper)

    assert mapper["pca_dim"] == 5
    assert predicted.shape == target.shape
    assert np.mean(gate._rowwise_cosine(predicted, target)) > 0.99


def test_paired_delta_ci_reports_mean_delta() -> None:
    exp = np.asarray([True, True, False, True])
    base = np.asarray([True, False, False, False])
    ci = gate._paired_delta_ci(exp, base, seed=1, samples=0)

    assert ci["mean_delta"] == 0.5
    assert ci["ci95_low"] == 0.5
    assert ci["ci95_high"] == 0.5


def test_source_cache_contract_reads_audit_paths(tmp_path) -> None:
    val_cache = tmp_path / "qwen15_validation.jsonl"
    test_cache = tmp_path / "qwen15_test.jsonl"
    val_cache.write_text(
        json.dumps({"source_family": "qwen2.5_1.5b", "source_selected_index": 0}) + "\n",
        encoding="utf-8",
    )
    test_cache.write_text(
        json.dumps({"source_family": "qwen2.5_1.5b", "source_selected_index": 1}) + "\n",
        encoding="utf-8",
    )
    audit = {
        "alternate_source_family": "qwen2.5_1.5b",
        "alternate_source_model": "/models/qwen15",
        "alt_validation_cache": str(val_cache),
        "alt_test_cache": str(test_cache),
    }
    (tmp_path / "source_cache_audit.json").write_text(json.dumps(audit), encoding="utf-8")

    contract = gate._source_cache_contract(tmp_path, "auto")

    assert contract["source_family"] == "qwen2.5_1.5b"
    assert contract["source_cache_prefix"] == "qwen25_15b"
    assert contract["source_model"] == "/models/qwen15"
    assert contract["validation_source_cache"] == val_cache
    assert contract["test_source_cache"] == test_cache
