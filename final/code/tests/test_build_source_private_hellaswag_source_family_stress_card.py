from __future__ import annotations

import json

from scripts import build_source_private_hellaswag_source_family_stress_card as card


def _write(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _global_payload() -> dict[str, object]:
    return {
        "pass_gate": True,
        "headline": {
            "eval_rows": 10042,
            "eval_slice_count": 10,
            "mean_zscore_slice_pass_count": 10,
            "mean_zscore_subbag_pass_count": 3,
            "source_private_packet_framed_bytes": 5,
            "source_private_packet_raw_bytes": 2,
            "train_sample_seed_count": 3,
        },
        "policy_rows": [
            {
                "policy": "mean_zscore",
                "selected_eval_accuracy": 0.53,
                "best_label_copy_eval_accuracy": 0.48,
                "source_label_copy_eval_accuracy": 0.48,
                "score_only_bagged_control_accuracy": 0.48,
                "zero_hidden_control_accuracy": 0.48,
                "wrong_example_hidden_control_accuracy": 0.45,
                "candidate_roll_hidden_control_accuracy": 0.41,
                "selected_minus_best_label_copy": 0.05,
                "selected_minus_score_only_bagged_control": 0.05,
                "paired_ci95_low_vs_best_label_copy": 0.04,
                "paired_ci95_low_vs_score_only_bagged": 0.04,
            }
        ],
        "source_models": ["qwen"],
        "timing": {"total_seconds": 10.0},
    }


def _slice_payload(pass_gate: bool = True) -> dict[str, object]:
    return {
        "pass_gate": pass_gate,
        "headline": {
            "eval_rows": 1024,
            "eval_slice_start": 1024,
            "eval_slice_end_exclusive": 2048,
            "selected_eval_accuracy": 0.50,
            "best_label_copy_eval_accuracy": 0.45,
            "source_label_copy_eval_accuracy": 0.44,
            "score_only_bagged_control_accuracy": 0.44,
            "zero_hidden_control_accuracy": 0.44,
            "wrong_example_hidden_control_accuracy": 0.40,
            "candidate_roll_hidden_control_accuracy": 0.36,
            "selected_minus_best_label_copy": 0.05,
            "selected_minus_score_only_bagged_control": 0.06,
            "paired_ci95_low_vs_best_label_copy": 0.02,
            "paired_ci95_low_vs_score_only_bagged": 0.03,
            "jackknife_pass_count": 3,
            "jackknife_row_count": 3,
            "raw_payload_bytes": 2,
            "framed_record_bytes": 5,
        },
        "eval_cache_metadata": {"score_model": {"model_path": "tinyllama"}},
        "timing": {"total_seconds": 20.0},
    }


def _full_eval_payload(pass_gate: bool = True) -> dict[str, object]:
    return {
        "pass_gate": False,
        "headline": {
            "eval_rows": 10042,
            "eval_slice_start": 0,
            "eval_slice_end_exclusive": 10042,
        },
        "bagged_gate": {
            "pass_gate": pass_gate,
            "headline": {
                "selected_eval_accuracy": 0.51,
                "best_label_copy_eval_accuracy": 0.46,
                "source_label_copy_eval_accuracy": 0.45,
                "score_only_bagged_control_accuracy": 0.45,
                "zero_hidden_control_accuracy": 0.45,
                "wrong_example_hidden_control_accuracy": 0.42,
                "candidate_roll_hidden_control_accuracy": 0.39,
                "selected_minus_best_label_copy": 0.05,
                "selected_minus_score_only_bagged_control": 0.06,
                "paired_ci95_low_vs_best_label_copy": 0.03,
                "paired_ci95_low_vs_score_only_bagged": 0.04,
            },
            "jackknife_summary": {
                "pass_count": 3,
                "row_count": 3,
            },
            "packet_contract": {
                "raw_payload_bytes": 2,
                "framed_record_bytes": 5,
            },
        },
        "eval_cache_metadata": {"score_model": {"model_path": "tinyllama"}},
        "timing": {"total_seconds": 200.0},
    }


def test_source_family_card_promotes_non_qwen_slice(tmp_path) -> None:
    qwen = tmp_path / "qwen.json"
    tiny = tmp_path / "tiny.json"
    _write(qwen, _global_payload())
    _write(tiny, _slice_payload())

    payload = card.build_card(output_dir=tmp_path / "out", qwen_global=qwen, tinyllama_slice=tiny)

    assert payload["pass_gate"] is True
    assert payload["headline"]["source_family_count"] == 2
    assert payload["headline"]["tinyllama_heldout_slice_pass"] is True
    assert payload["headline"]["tinyllama_full_validation_present"] is False
    assert payload["headline"]["iclr_ready"] is False
    assert (tmp_path / "out" / "hellaswag_source_family_stress_card.json").exists()
    assert (tmp_path / "out" / "hellaswag_source_family_stress_card.csv").exists()
    assert (tmp_path / "out" / "manifest.json").exists()


def test_source_family_card_fails_when_non_qwen_slice_fails(tmp_path) -> None:
    qwen = tmp_path / "qwen.json"
    tiny = tmp_path / "tiny.json"
    _write(qwen, _global_payload())
    _write(tiny, _slice_payload(pass_gate=False))

    payload = card.build_card(output_dir=tmp_path / "out", qwen_global=qwen, tinyllama_slice=tiny)

    assert payload["pass_gate"] is False
    assert payload["headline"]["qwen_full_validation_pass"] is True
    assert payload["headline"]["tinyllama_heldout_slice_pass"] is False


def test_source_family_card_promotes_full_non_qwen_when_present(tmp_path) -> None:
    qwen = tmp_path / "qwen.json"
    tiny = tmp_path / "tiny.json"
    full = tmp_path / "tiny_full.json"
    _write(qwen, _global_payload())
    _write(tiny, _slice_payload())
    _write(full, _full_eval_payload())

    payload = card.build_card(
        output_dir=tmp_path / "out",
        qwen_global=qwen,
        tinyllama_slice=tiny,
        tinyllama_full=full,
    )

    assert payload["pass_gate"] is True
    assert payload["headline"]["tinyllama_full_validation_present"] is True
    assert payload["headline"]["tinyllama_full_validation_pass"] is True
    assert payload["headline"]["tinyllama_full_delta_vs_best_label_copy"] == 0.05
    assert len(payload["rows"]) == 3
