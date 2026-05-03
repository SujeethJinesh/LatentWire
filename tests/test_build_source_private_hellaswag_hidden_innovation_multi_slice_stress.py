from __future__ import annotations

import json

from scripts import build_source_private_hellaswag_hidden_innovation_multi_slice_stress as multi_slice


def _packet_contract() -> dict[str, object]:
    return {
        "raw_payload_bytes": 2,
        "framed_record_bytes": 5,
        "source_text_exposed": False,
        "source_kv_exposed": False,
        "raw_hidden_vector_transmitted": False,
        "raw_scores_transmitted": False,
    }


def _headline(start: int, *, delta: float = 0.04) -> dict[str, object]:
    return {
        "eval_slice_start": start,
        "eval_slice_end_exclusive": start + 1024,
        "eval_rows": 1024,
        "selected_eval_accuracy": 0.50,
        "source_label_copy_eval_accuracy": 0.45,
        "trained_choice_bias_label_copy_eval_accuracy": 0.46,
        "best_label_copy_eval_accuracy": 0.46,
        "selected_minus_best_label_copy": delta,
        "paired_ci95_low_vs_best_label_copy": 0.01,
        "paired_ci95_high_vs_best_label_copy": 0.06,
        "source_rank_only_bagged_control_accuracy": 0.45,
        "selected_minus_source_rank_only_bagged_control": 0.05,
        "paired_ci95_low_vs_source_rank_only_bagged": 0.02,
        "paired_ci95_high_vs_source_rank_only_bagged": 0.06,
        "score_only_bagged_control_accuracy": 0.45,
        "selected_minus_score_only_bagged_control": 0.05,
        "paired_ci95_low_vs_score_only_bagged": 0.02,
        "zero_hidden_control_accuracy": 0.45,
        "selected_minus_zero_hidden_control": 0.05,
        "wrong_example_hidden_control_accuracy": 0.43,
        "candidate_roll_hidden_control_accuracy": 0.42,
        "score_channel_roll_hidden_control_accuracy": 0.41,
    }


def _bagged_payload(*, delta: float = 0.04) -> dict[str, object]:
    headline = _headline(0, delta=delta)
    headline.pop("eval_slice_start")
    headline.pop("eval_slice_end_exclusive")
    headline.pop("eval_rows")
    return {
        "gate": "source_private_hellaswag_hidden_innovation_bagged_gate",
        "pass_gate": True,
        "headline": headline,
        "control_readouts": {
            "source_label_copy": {"accuracy": 0.45, "correct": 461, "rows": 1024},
            "score_only_bagged_control": {"accuracy": 0.45, "correct": 461, "rows": 1024},
            "zero_hidden_control": {"accuracy": 0.45, "correct": 461, "rows": 1024},
        },
        "jackknife_summary": {
            "pass_count": 3,
            "row_count": 3,
            "selected_minus_best_label_copy_min": 0.03,
            "paired_ci95_low_vs_best_label_copy_min": 0.01,
        },
        "packet_contract": _packet_contract(),
    }


def _eval_slice_payload(start: int, *, delta: float = 0.04) -> dict[str, object]:
    headline = _headline(start, delta=delta)
    headline |= {
        "jackknife_pass_count": 3,
        "jackknife_row_count": 3,
        "jackknife_min_delta_vs_best_label_copy": 0.03,
        "jackknife_min_ci95_low_vs_best_label_copy": 0.01,
    }
    return {
        "gate": "source_private_hellaswag_hidden_innovation_eval_slice_stress",
        "pass_gate": True,
        "headline": headline,
        "packet_contract": _packet_contract(),
    }


def _write_json(path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def test_multi_slice_gate_passes_for_three_contiguous_private_slices(tmp_path) -> None:
    paths = [tmp_path / "slice0.json", tmp_path / "slice1.json", tmp_path / "slice2.json"]
    _write_json(paths[0], _bagged_payload())
    _write_json(paths[1], _eval_slice_payload(1024))
    _write_json(paths[2], _eval_slice_payload(2048))

    payload = multi_slice.build_gate(output_dir=tmp_path / "out", slice_artifacts=tuple(paths))

    assert payload["pass_gate"] is True
    assert payload["headline"]["slice_count"] == 3
    assert payload["headline"]["total_eval_rows"] == 3072
    assert payload["headline"]["pass_slice_count"] == 3
    assert payload["headline"]["contiguous_validation_prefix"] is True
    assert payload["headline"]["min_delta_vs_best_label_copy"] >= 0.02
    assert payload["headline"]["all_rank_score_channel_controls_available"] is True
    assert payload["headline"]["source_private_packet"] is True
    assert (tmp_path / "out" / "hellaswag_hidden_innovation_multi_slice_stress.json").exists()
    assert (tmp_path / "out" / "hellaswag_hidden_innovation_multi_slice_stress.md").exists()
    assert (tmp_path / "out" / "slice_rows.csv").exists()


def test_multi_slice_gate_fails_if_a_slice_is_below_delta(tmp_path) -> None:
    paths = [tmp_path / "slice0.json", tmp_path / "slice1.json", tmp_path / "slice2.json"]
    _write_json(paths[0], _bagged_payload())
    _write_json(paths[1], _eval_slice_payload(1024, delta=0.01))
    _write_json(paths[2], _eval_slice_payload(2048))

    payload = multi_slice.build_gate(output_dir=tmp_path / "out", slice_artifacts=tuple(paths))

    assert payload["pass_gate"] is False
    assert payload["headline"]["pass_slice_count"] == 2
    assert payload["headline"]["min_delta_vs_best_label_copy"] == 0.01


def test_multi_slice_gate_fails_if_slices_are_not_contiguous(tmp_path) -> None:
    paths = [tmp_path / "slice0.json", tmp_path / "slice1.json", tmp_path / "slice2.json"]
    _write_json(paths[0], _bagged_payload())
    _write_json(paths[1], _eval_slice_payload(2048))
    _write_json(paths[2], _eval_slice_payload(3072))

    payload = multi_slice.build_gate(output_dir=tmp_path / "out", slice_artifacts=tuple(paths))

    assert payload["pass_gate"] is False
    assert payload["headline"]["contiguous_validation_prefix"] is False
