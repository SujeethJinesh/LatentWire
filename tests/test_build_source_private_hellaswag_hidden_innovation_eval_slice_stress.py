from __future__ import annotations

import json

from scripts import build_source_private_hellaswag_hidden_innovation_eval_slice_stress as eval_slice


def _canonical_row(index: int) -> dict[str, object]:
    return {
        "id": f"row-{index}",
        "content_id": f"content-{index}",
        "question": f"question {index}",
        "choices": [f"a{index}", f"b{index}", f"c{index}", f"d{index}"],
        "choice_labels": ["A", "B", "C", "D"],
        "answer_index": index % 4,
        "answer_label": ["A", "B", "C", "D"][index % 4],
        "source_name": "unit",
    }


def test_slice_jsonl_materializes_requested_rows(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(json.dumps(_canonical_row(index), sort_keys=True) for index in range(5)) + "\n",
        encoding="utf-8",
    )

    metadata = eval_slice._slice_jsonl(
        source_path=source,
        output_path=tmp_path / "slice.jsonl",
        start=1,
        count=3,
    )

    rows = [json.loads(line) for line in (tmp_path / "slice.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == ["row-1", "row-2", "row-3"]
    assert metadata["slice_rows"] == 3
    assert metadata["slice_start"] == 1
    assert metadata["slice_end_exclusive"] == 4


def test_eval_slice_gate_writes_artifacts_with_mocked_model_calls(tmp_path, monkeypatch) -> None:
    source = tmp_path / "validation.jsonl"
    source.write_text(
        "\n".join(json.dumps(_canonical_row(index), sort_keys=True) for index in range(2048)) + "\n",
        encoding="utf-8",
    )

    def fake_cache_eval_source_state(**kwargs):
        kwargs["score_cache"].write_text("{}", encoding="utf-8")
        kwargs["hidden_npz"].write_bytes(b"fake hidden")
        kwargs["hidden_meta"].write_text("{}", encoding="utf-8")
        return {
            "score_cache": str(kwargs["score_cache"]),
            "score_cache_sha256": "fake",
            "score_cache_hit": False,
            "hidden_cache": str(kwargs["hidden_npz"]),
            "hidden_cache_meta": str(kwargs["hidden_meta"]),
            "hidden_cache_sha256": "fake",
            "hidden_cache_hit": False,
            "score_model": {},
            "hidden_model": {},
        }

    def fake_bagged_build_gate(*, output_dir, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "hellaswag_hidden_innovation_bagged_gate.json").write_text(
            "{}",
            encoding="utf-8",
        )
        (output_dir / "jackknife_rows.csv").write_text("name,pass_gate\nall,true\n", encoding="utf-8")
        (output_dir / "manifest.json").write_text("{}", encoding="utf-8")
        return {
            "pass_gate": True,
            "headline": {
                "selected_eval_accuracy": 0.55,
                "source_label_copy_eval_accuracy": 0.45,
                "trained_choice_bias_label_copy_eval_accuracy": 0.46,
                "best_label_copy_eval_accuracy": 0.46,
                "selected_minus_best_label_copy": 0.09,
                "paired_ci95_low_vs_best_label_copy": 0.04,
                "paired_ci95_high_vs_best_label_copy": 0.13,
                "source_rank_only_bagged_control_accuracy": 0.45,
                "selected_minus_source_rank_only_bagged_control": 0.10,
                "paired_ci95_low_vs_source_rank_only_bagged": 0.05,
                "paired_ci95_high_vs_source_rank_only_bagged": 0.14,
                "score_only_bagged_control_accuracy": 0.45,
                "selected_minus_score_only_bagged_control": 0.10,
                "paired_ci95_low_vs_score_only_bagged": 0.05,
                "zero_hidden_control_accuracy": 0.45,
                "selected_minus_zero_hidden_control": 0.10,
                "wrong_example_hidden_control_accuracy": 0.43,
                "candidate_roll_hidden_control_accuracy": 0.42,
                "score_channel_roll_hidden_control_accuracy": 0.41,
                "train_sample_seed_count": 3,
                "component_model_count": 9,
            },
            "jackknife_summary": {
                "all_pass": True,
                "row_count": 3,
                "pass_count": 3,
                "selected_minus_best_label_copy_min": 0.04,
                "paired_ci95_low_vs_best_label_copy_min": 0.01,
            },
            "packet_contract": {
                "raw_payload_bytes": 2,
                "framed_record_bytes": 5,
                "source_text_exposed": False,
                "source_kv_exposed": False,
                "raw_hidden_vector_transmitted": False,
                "raw_scores_transmitted": False,
            },
            "timing": {"total_seconds": 1.0},
        }

    monkeypatch.setattr(eval_slice, "_cache_eval_source_state", fake_cache_eval_source_state)
    monkeypatch.setattr(eval_slice.bagged, "build_gate", fake_bagged_build_gate)

    payload = eval_slice.build_gate(
        output_dir=tmp_path / "out",
        eval_full_path=source,
        eval_slice_start=1024,
        eval_slice_rows=1024,
        bootstrap_samples=10,
    )

    assert payload["pass_gate"] is True
    assert payload["headline"]["eval_slice_start"] == 1024
    assert payload["headline"]["eval_rows"] == 1024
    assert payload["headline"]["raw_payload_bytes"] == 2
    assert (tmp_path / "out" / "hellaswag_hidden_innovation_eval_slice_stress.json").exists()
    assert (tmp_path / "out" / "hellaswag_hidden_innovation_eval_slice_stress.md").exists()
    assert (tmp_path / "out" / "manifest.json").exists()


def test_eval_slice_gate_reuses_explicit_eval_caches(tmp_path, monkeypatch) -> None:
    source = tmp_path / "validation.jsonl"
    source.write_text(
        "\n".join(json.dumps(_canonical_row(index), sort_keys=True) for index in range(2048)) + "\n",
        encoding="utf-8",
    )
    external_score_cache = tmp_path / "external" / "score_cache.json"
    external_hidden_cache = tmp_path / "external" / "hidden_cache.npz"
    observed: dict[str, object] = {}

    def fake_cache_eval_source_state(**kwargs):
        observed["cache_score_cache"] = kwargs["score_cache"]
        observed["cache_hidden_npz"] = kwargs["hidden_npz"]
        observed["cache_hidden_meta"] = kwargs["hidden_meta"]
        kwargs["score_cache"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["hidden_npz"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["score_cache"].write_text("{}", encoding="utf-8")
        kwargs["hidden_npz"].write_bytes(b"fake hidden")
        kwargs["hidden_meta"].write_text("{}", encoding="utf-8")
        return {
            "score_cache": str(kwargs["score_cache"]),
            "score_cache_sha256": "fake",
            "score_cache_hit": True,
            "hidden_cache": str(kwargs["hidden_npz"]),
            "hidden_cache_meta": str(kwargs["hidden_meta"]),
            "hidden_cache_sha256": "fake",
            "hidden_cache_hit": True,
            "score_model": {},
            "hidden_model": {},
        }

    def fake_bagged_build_gate(*, output_dir, **kwargs):
        observed["bagged_eval_score_cache"] = kwargs["eval_score_cache"]
        observed["bagged_eval_hidden_cache"] = kwargs["eval_hidden_cache"]
        observed["bagged_aggregation_policy"] = kwargs["aggregation_policy"]
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "hellaswag_hidden_innovation_bagged_gate.json").write_text(
            "{}",
            encoding="utf-8",
        )
        (output_dir / "jackknife_rows.csv").write_text("name,pass_gate\nall,true\n", encoding="utf-8")
        (output_dir / "manifest.json").write_text("{}", encoding="utf-8")
        return {
            "pass_gate": True,
            "headline": {
                "selected_eval_accuracy": 0.55,
                "source_label_copy_eval_accuracy": 0.45,
                "trained_choice_bias_label_copy_eval_accuracy": 0.46,
                "best_label_copy_eval_accuracy": 0.46,
                "selected_minus_best_label_copy": 0.09,
                "paired_ci95_low_vs_best_label_copy": 0.04,
                "paired_ci95_high_vs_best_label_copy": 0.13,
                "source_rank_only_bagged_control_accuracy": 0.45,
                "selected_minus_source_rank_only_bagged_control": 0.10,
                "paired_ci95_low_vs_source_rank_only_bagged": 0.05,
                "paired_ci95_high_vs_source_rank_only_bagged": 0.14,
                "score_only_bagged_control_accuracy": 0.45,
                "selected_minus_score_only_bagged_control": 0.10,
                "paired_ci95_low_vs_score_only_bagged": 0.05,
                "zero_hidden_control_accuracy": 0.45,
                "selected_minus_zero_hidden_control": 0.10,
                "wrong_example_hidden_control_accuracy": 0.43,
                "candidate_roll_hidden_control_accuracy": 0.42,
                "score_channel_roll_hidden_control_accuracy": 0.41,
                "train_sample_seed_count": 3,
                "component_model_count": 9,
            },
            "jackknife_summary": {
                "all_pass": True,
                "row_count": 3,
                "pass_count": 3,
                "selected_minus_best_label_copy_min": 0.04,
                "paired_ci95_low_vs_best_label_copy_min": 0.01,
            },
            "packet_contract": {
                "raw_payload_bytes": 2,
                "framed_record_bytes": 5,
                "source_text_exposed": False,
                "source_kv_exposed": False,
                "raw_hidden_vector_transmitted": False,
                "raw_scores_transmitted": False,
            },
            "timing": {"total_seconds": 1.0},
        }

    monkeypatch.setattr(eval_slice, "_cache_eval_source_state", fake_cache_eval_source_state)
    monkeypatch.setattr(eval_slice.bagged, "build_gate", fake_bagged_build_gate)

    payload = eval_slice.build_gate(
        output_dir=tmp_path / "out",
        eval_full_path=source,
        eval_slice_start=1024,
        eval_slice_rows=1024,
        eval_score_cache=external_score_cache,
        eval_hidden_cache=external_hidden_cache,
        aggregation_policy="mean_zscore_vote_on_score_agreement",
        bootstrap_samples=10,
    )

    assert payload["eval_cache_metadata"]["score_cache_hit"] is True
    assert observed["cache_score_cache"] == external_score_cache
    assert observed["cache_hidden_npz"] == external_hidden_cache
    assert observed["cache_hidden_meta"] == external_hidden_cache.with_suffix(".json")
    assert observed["bagged_eval_score_cache"] == external_score_cache
    assert observed["bagged_eval_hidden_cache"] == external_hidden_cache
    assert observed["bagged_aggregation_policy"] == "mean_zscore_vote_on_score_agreement"


def test_eval_slice_terminal_tail_rule_is_explicit(tmp_path, monkeypatch) -> None:
    terminal_source = tmp_path / "validation_10042.jsonl"
    terminal_source.write_text(
        "\n".join(json.dumps(_canonical_row(index), sort_keys=True) for index in range(10042)) + "\n",
        encoding="utf-8",
    )
    nonterminal_source = tmp_path / "validation_11000.jsonl"
    nonterminal_source.write_text(
        "\n".join(json.dumps(_canonical_row(index), sort_keys=True) for index in range(11000)) + "\n",
        encoding="utf-8",
    )

    def fake_cache_eval_source_state(**kwargs):
        kwargs["score_cache"].write_text("{}", encoding="utf-8")
        kwargs["hidden_npz"].write_bytes(b"fake hidden")
        kwargs["hidden_meta"].write_text("{}", encoding="utf-8")
        return {
            "score_cache": str(kwargs["score_cache"]),
            "score_cache_sha256": "fake",
            "score_cache_hit": False,
            "hidden_cache": str(kwargs["hidden_npz"]),
            "hidden_cache_meta": str(kwargs["hidden_meta"]),
            "hidden_cache_sha256": "fake",
            "hidden_cache_hit": False,
            "score_model": {},
            "hidden_model": {},
        }

    def fake_bagged_build_gate(*, output_dir, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "hellaswag_hidden_innovation_bagged_gate.json").write_text(
            "{}",
            encoding="utf-8",
        )
        (output_dir / "manifest.json").write_text("{}", encoding="utf-8")
        return {
            "pass_gate": True,
            "headline": {
                "selected_eval_accuracy": 0.55,
                "source_label_copy_eval_accuracy": 0.45,
                "trained_choice_bias_label_copy_eval_accuracy": 0.46,
                "best_label_copy_eval_accuracy": 0.46,
                "selected_minus_best_label_copy": 0.09,
                "paired_ci95_low_vs_best_label_copy": 0.04,
                "paired_ci95_high_vs_best_label_copy": 0.13,
                "source_rank_only_bagged_control_accuracy": 0.45,
                "selected_minus_source_rank_only_bagged_control": 0.10,
                "paired_ci95_low_vs_source_rank_only_bagged": 0.05,
                "paired_ci95_high_vs_source_rank_only_bagged": 0.14,
                "score_only_bagged_control_accuracy": 0.45,
                "selected_minus_score_only_bagged_control": 0.10,
                "paired_ci95_low_vs_score_only_bagged": 0.05,
                "zero_hidden_control_accuracy": 0.45,
                "selected_minus_zero_hidden_control": 0.10,
                "wrong_example_hidden_control_accuracy": 0.43,
                "candidate_roll_hidden_control_accuracy": 0.42,
                "score_channel_roll_hidden_control_accuracy": 0.41,
                "train_sample_seed_count": 3,
                "component_model_count": 9,
            },
            "jackknife_summary": {
                "all_pass": True,
                "row_count": 3,
                "pass_count": 3,
                "selected_minus_best_label_copy_min": 0.04,
                "paired_ci95_low_vs_best_label_copy_min": 0.01,
            },
            "packet_contract": {
                "raw_payload_bytes": 2,
                "framed_record_bytes": 5,
                "source_text_exposed": False,
                "source_kv_exposed": False,
                "raw_hidden_vector_transmitted": False,
                "raw_scores_transmitted": False,
            },
            "timing": {"total_seconds": 1.0},
        }

    monkeypatch.setattr(eval_slice, "_cache_eval_source_state", fake_cache_eval_source_state)
    monkeypatch.setattr(eval_slice.bagged, "build_gate", fake_bagged_build_gate)

    terminal_payload = eval_slice.build_gate(
        output_dir=tmp_path / "terminal",
        eval_full_path=terminal_source,
        eval_slice_start=9216,
        eval_slice_rows=826,
        bootstrap_samples=10,
    )
    assert terminal_payload["pass_gate"] is True
    assert terminal_payload["headline"]["terminal_tail_slice"] is True
    assert terminal_payload["headline"]["standard_sized_slice"] is False

    nonterminal_payload = eval_slice.build_gate(
        output_dir=tmp_path / "nonterminal",
        eval_full_path=nonterminal_source,
        eval_slice_start=9216,
        eval_slice_rows=826,
        bootstrap_samples=10,
    )
    assert nonterminal_payload["pass_gate"] is False
    assert nonterminal_payload["headline"]["terminal_tail_slice"] is False
    assert nonterminal_payload["headline"]["standard_sized_slice"] is False
