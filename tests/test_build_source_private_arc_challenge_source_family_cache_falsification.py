from __future__ import annotations

import json

from scripts import build_source_private_arc_challenge_source_family_cache_falsification as gate


def _write_jsonl(path, rows) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _row(row_id: str, answer_index: int) -> dict:
    choices = [f"{row_id} alpha", f"{row_id} beta", f"{row_id} gamma"]
    return {
        "id": row_id,
        "content_id": f"content-{row_id}",
        "question": f"Which option fits {row_id}?",
        "choices": choices,
        "choice_labels": ["A", "B", "C"],
        "answer_index": answer_index,
        "answer_label": ["A", "B", "C"][answer_index],
    }


def _cache_row(row: dict, source_index: int, *, family: str) -> dict:
    return {
        "row_id": row["id"],
        "content_id": row["content_id"],
        "source_family": family,
        "source_selected_index": source_index,
        "source_visible_fields": ["question", "choices"],
        "forbidden_source_fields": ["answer", "answerKey", "answer_index", "answer_label", "gold"],
    }


def test_source_family_cache_falsification_writes_disagreement_artifacts(tmp_path) -> None:
    train_rows = [_row(f"train-{index}", index % 3) for index in range(12)]
    validation_rows = [_row(f"validation-{index}", 1) for index in range(6)]
    test_rows = [_row(f"test-{index}", 1) for index in range(6)]
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    test_path = tmp_path / "test.jsonl"
    qwen_validation_cache = tmp_path / "qwen_validation_cache.jsonl"
    qwen_test_cache = tmp_path / "qwen_test_cache.jsonl"
    alt_validation_cache = tmp_path / "alt_validation_cache.jsonl"
    alt_test_cache = tmp_path / "alt_test_cache.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(validation_path, validation_rows)
    _write_jsonl(test_path, test_rows)
    _write_jsonl(qwen_validation_cache, [_cache_row(row, 0, family="qwen") for row in validation_rows])
    _write_jsonl(qwen_test_cache, [_cache_row(row, 0, family="qwen") for row in test_rows])
    _write_jsonl(alt_validation_cache, [_cache_row(row, 1, family="alt") for row in validation_rows])
    _write_jsonl(alt_test_cache, [_cache_row(row, 1, family="alt") for row in test_rows])

    payload = gate.build_source_family_cache_falsification(
        output_dir=tmp_path / "out",
        train_path=train_path,
        validation_path=validation_path,
        test_path=test_path,
        qwen_validation_cache=qwen_validation_cache,
        qwen_test_cache=qwen_test_cache,
        alt_validation_cache=alt_validation_cache,
        alt_test_cache=alt_test_cache,
        alternate_source_family="toy_alt",
        alternate_source_model=tmp_path / "toy-model",
        materialize_alt_caches=False,
        force_rematerialize=False,
        source_lm_device="cpu",
        source_lm_dtype="float32",
        source_lm_max_length=32,
        source_lm_normalization="mean",
        source_lm_prompt_mode="qa",
        local_files_only=True,
        seeds=[3],
        budget_bytes=4,
        anchor_count=12,
        spectral_dim=6,
        code_dim=6,
        bootstrap_samples=20,
        min_disagreement_count=1,
        min_lift_over_target=-1.0,
        min_gap_over_control=-1.0,
        min_gap_over_text=-1.0,
        min_gap_over_qwen=-1.0,
    )

    assert payload["gate"] == "source_private_arc_challenge_source_family_cache_falsification"
    assert payload["source_cache_audit"]["materialized_this_run"] is False
    assert payload["splits"]["test"]["source_cache_agreement"]["disagreement_count"] == len(test_rows)
    assert payload["splits"]["test"]["qwen_disagreement_slice"]["aggregate"]["n"] == len(test_rows)
    assert payload["basis"]["spectral_dim"] == 6
    assert payload["basis"]["seeds"] == [3]
    assert payload["basis"]["bootstrap_samples"] == 20
    assert (tmp_path / "out" / "source_family_cache_falsification.json").exists()
    assert (tmp_path / "out" / "per_source_split_seed_metrics.csv").exists()
    assert (tmp_path / "out" / "qwen_disagreement_predictions.jsonl").exists()
    disagreement_rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "qwen_disagreement_predictions.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    assert {row["condition"] for row in disagreement_rows} == {
        gate.arc_gate.MATCHED_CONDITION,
        gate.QWEN_SUBSTITUTED_CONDITION,
    }
    matched_rows = [row for row in disagreement_rows if row["condition"] == gate.arc_gate.MATCHED_CONDITION]
    qwen_rows = [row for row in disagreement_rows if row["condition"] == gate.QWEN_SUBSTITUTED_CONDITION]
    assert {row["metadata"]["source_selected_index"] for row in matched_rows} == {1}
    assert {row["metadata"]["source_selected_index"] for row in qwen_rows} == {0}


def test_source_family_cache_falsification_rejects_forbidden_cache_payload(tmp_path) -> None:
    train_rows = [_row(f"train-{index}", index % 3) for index in range(12)]
    validation_rows = [_row(f"validation-{index}", 1) for index in range(6)]
    test_rows = [_row(f"test-{index}", 1) for index in range(6)]
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    test_path = tmp_path / "test.jsonl"
    qwen_validation_cache = tmp_path / "qwen_validation_cache.jsonl"
    qwen_test_cache = tmp_path / "qwen_test_cache.jsonl"
    alt_validation_cache = tmp_path / "alt_validation_cache.jsonl"
    alt_test_cache = tmp_path / "alt_test_cache.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(validation_path, validation_rows)
    _write_jsonl(test_path, test_rows)
    _write_jsonl(qwen_validation_cache, [_cache_row(row, 0, family="qwen") for row in validation_rows])
    _write_jsonl(qwen_test_cache, [_cache_row(row, 0, family="qwen") for row in test_rows])
    leaking_rows = [_cache_row(row, 1, family="alt") for row in validation_rows]
    leaking_rows[0]["answer_index"] = validation_rows[0]["answer_index"]
    _write_jsonl(alt_validation_cache, leaking_rows)
    _write_jsonl(alt_test_cache, [_cache_row(row, 1, family="alt") for row in test_rows])

    try:
        gate.build_source_family_cache_falsification(
            output_dir=tmp_path / "out",
            train_path=train_path,
            validation_path=validation_path,
            test_path=test_path,
            qwen_validation_cache=qwen_validation_cache,
            qwen_test_cache=qwen_test_cache,
            alt_validation_cache=alt_validation_cache,
            alt_test_cache=alt_test_cache,
            alternate_source_family="toy_alt",
            alternate_source_model=tmp_path / "toy-model",
            materialize_alt_caches=False,
            force_rematerialize=False,
            source_lm_device="cpu",
            source_lm_dtype="float32",
            source_lm_max_length=32,
            source_lm_normalization="mean",
            source_lm_prompt_mode="qa",
            local_files_only=True,
            seeds=[3],
            budget_bytes=4,
            anchor_count=12,
            spectral_dim=6,
            code_dim=6,
            bootstrap_samples=20,
            min_disagreement_count=1,
            min_lift_over_target=-1.0,
            min_gap_over_control=-1.0,
            min_gap_over_text=-1.0,
            min_gap_over_qwen=-1.0,
        )
    except ValueError as exc:
        assert "source cache audit failed" in str(exc)
        assert "leaked_payload_keys" in str(exc)
    else:
        raise AssertionError("forbidden answer field in source cache should fail the audit")
