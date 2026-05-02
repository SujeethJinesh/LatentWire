from __future__ import annotations

import json

import numpy as np

from scripts import build_source_private_arc_challenge_fourier_anchor_syndrome_gate as gate


def _write_jsonl(path, rows) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _row(row_id: str, answer_index: int) -> dict:
    choices = [f"{row_id} ice", f"{row_id} fire", f"{row_id} rain"]
    return {
        "id": row_id,
        "content_id": f"content-{row_id}",
        "question": f"What is true for {row_id}?",
        "choices": choices,
        "choice_labels": ["A", "B", "C"],
        "answer_index": answer_index,
        "answer_label": ["A", "B", "C"][answer_index],
    }


def _cache_row(row: dict, source_index: int) -> dict:
    return {
        "row_id": row["id"],
        "content_id": row["content_id"],
        "source_family": "toy_source",
        "source_model": "toy-model",
        "source_score_mode": "fixture_choice",
        "source_selected_index": source_index,
        "source_selected_choice_sha256": "",
        "source_visible_fields": ["question", "choices"],
        "forbidden_source_fields": ["answer", "answerKey", "answer_index", "answer_label", "gold"],
    }


def test_dct_low_frequency_features_are_deterministic_and_normalized() -> None:
    features = np.asarray([[1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 0.0, -1.0]], dtype=np.float64)

    first = gate._dct_low_frequency_features(features, output_dim=3)
    second = gate._dct_low_frequency_features(features, output_dim=3)

    assert np.allclose(first, second)
    assert first.shape == (2, 3)
    assert np.allclose(np.linalg.norm(first, axis=1), np.ones(2))


def test_fourier_anchor_syndrome_gate_writes_artifacts(tmp_path) -> None:
    train_rows = [_row(f"train-{index}", index % 3) for index in range(12)]
    validation_rows = [_row(f"validation-{index}", (index + 1) % 3) for index in range(6)]
    test_rows = [_row(f"test-{index}", (index + 2) % 3) for index in range(6)]
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    test_path = tmp_path / "test.jsonl"
    validation_cache = tmp_path / "validation_cache.jsonl"
    test_cache = tmp_path / "test_cache.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(validation_path, validation_rows)
    _write_jsonl(test_path, test_rows)
    _write_jsonl(validation_cache, [_cache_row(row, (index + 1) % 3) for index, row in enumerate(validation_rows)])
    _write_jsonl(test_cache, [_cache_row(row, (index + 2) % 3) for index, row in enumerate(test_rows)])

    payload = gate.build_fourier_anchor_syndrome_gate(
        output_dir=tmp_path / "out",
        train_path=train_path,
        validation_path=validation_path,
        test_path=test_path,
        validation_source_cache=validation_cache,
        test_source_cache=test_cache,
        seeds=[3],
        budget_bytes=4,
        anchor_count=12,
        spectral_dim=6,
        code_dim=6,
        bootstrap_samples=20,
        min_lift_over_target=-1.0,
        min_gap_over_control=-1.0,
        min_gap_over_text=-1.0,
    )

    assert payload["gate"] == "source_private_arc_challenge_fourier_anchor_syndrome_gate"
    assert payload["basis_contract"]["spectral_dim"] == 6
    assert "spectral_bin_permutation" in payload["splits"]["test"]["variants"]
    assert "anchor_value_shuffle" in payload["splits"]["validation"]["variants"]
    assert payload["method_contract"]["forbidden_eval_source_inputs"]
    assert payload["source_cache_audit"]["source_families"] == ["toy_source"]
    assert "source_cache_audit" in payload["method_contract"]["source_packet_origin"]
    assert (tmp_path / "out" / "arc_challenge_fourier_anchor_syndrome_gate.json").exists()
    assert (tmp_path / "out" / "per_variant_seed_metrics.csv").exists()
    assert (tmp_path / "out" / "matched_predictions.jsonl").exists()
