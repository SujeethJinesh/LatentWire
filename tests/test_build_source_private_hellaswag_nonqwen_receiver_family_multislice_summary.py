from __future__ import annotations

import json
import pytest

from scripts import build_source_private_hellaswag_nonqwen_receiver_family_multislice_summary as summary


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact(*, start: int, target: float, packet: float, receiver: float, oracle: float):
    return {
        "gate": "source_private_hellaswag_nonqwen_receiver_family_packet_gate",
        "pass_gate": False,
        "headline": {
            "eval_rows": 10,
            "native_systems_complete": False,
            "packet_minus_target_only": packet - target,
            "packet_only_eval_accuracy": packet,
            "receiver_ci95_low_vs_packet_only": receiver - packet - 0.01,
            "receiver_ci95_low_vs_target_only": receiver - target - 0.01,
            "receiver_eval_accuracy": receiver,
            "receiver_improvement_gate": receiver > packet,
            "receiver_minus_packet_only": receiver - packet,
            "receiver_minus_target_only": receiver - target,
            "row_count": 12,
            "selected_receiver_kind": "candidate_ridge_receiver",
            "slice_end_exclusive": start + 12,
            "slice_start": start,
            "source_family": "TinyLlama",
            "source_utility_gate": packet - target >= 0.05,
            "strict_receiver_packet_lift_required": 0.005,
            "strict_target_lift_required": 0.05,
            "target_family": "Phi-3-mini",
            "target_family_transfer_gate": receiver > target,
            "target_lm_device": "cpu",
            "target_lm_dtype": "float32",
            "target_only_eval_accuracy": target,
            "target_or_packet_oracle_eval_accuracy": oracle,
            "target_score_cache_hit": True,
            "target_score_latency_s": 1.5,
            "train_rows": 2,
        },
        "source_packet": {
            "raw_payload_bytes": 2,
            "framed_record_bytes": 5,
            "exposes_source_text": False,
            "exposes_source_kv": False,
            "exposes_raw_hidden": False,
            "exposes_raw_scores": False,
        },
        "receiver_gate": {
            "source_packet": {
                "control_fields": [
                    "wrong_example_hidden_prediction",
                    "candidate_roll_hidden_prediction",
                    "zero_hidden_prediction",
                    "source_label_prediction",
                ]
            }
        },
    }


def test_nonqwen_receiver_multislice_summary_marks_packet_utility_without_receiver_pass(
    tmp_path,
):
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    _write_json(left, _artifact(start=1024, target=0.2, packet=0.5, receiver=0.45, oracle=0.6))
    _write_json(right, _artifact(start=1036, target=0.3, packet=0.6, receiver=0.55, oracle=0.7))

    payload = summary.build_summary(
        output_dir=tmp_path / "out",
        artifacts=(left, right),
        run_date="2026-05-03",
    )

    h = payload["headline"]
    assert payload["pass_gate"] is False
    assert h["slice_count"] == 2
    assert h["contiguous"] is True
    assert h["source_utility_slice_count"] == 2
    assert h["target_family_transfer_slice_count"] == 2
    assert h["receiver_improvement_slice_count"] == 0
    assert h["weighted_packet_minus_target_only"] == pytest.approx(0.3)
    assert h["weighted_receiver_minus_packet_only"] == pytest.approx(-0.05)
    assert h["source_private_packet"] is True
    assert (tmp_path / "out" / "slice_rows.csv").exists()
