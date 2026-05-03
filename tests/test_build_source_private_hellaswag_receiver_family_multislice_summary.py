from __future__ import annotations

import json

import pytest

from scripts import build_source_private_hellaswag_receiver_family_multislice_summary as summary


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact(
    *,
    root,
    name: str,
    start: int,
    target: float,
    packet: float,
    receiver: float,
    oracle: float,
):
    packet_artifact = root / f"{name}_packet_artifact.json"
    _write_json(
        packet_artifact,
        {
            "packet_contract": {
                "raw_payload_bytes": 2,
                "framed_record_bytes": 5,
                "source_text_exposed": False,
                "source_kv_exposed": False,
                "raw_hidden_vector_transmitted": False,
                "raw_scores_transmitted": False,
            }
        },
    )
    return {
        "gate": "source_private_hellaswag_receiver_family_packet_gate",
        "pass_gate": False,
        "target_family_transfer_gate": receiver > target,
        "receiver_improvement_gate": receiver > packet,
        "headline": {
            "eval_rows": 10,
            "packet_field": "selected_prediction",
            "packet_margin_field": "selected_margin",
            "packet_only_eval_accuracy": packet,
            "receiver_ci95_high_vs_packet_only": receiver - packet + 0.01,
            "receiver_ci95_high_vs_target_only": receiver - target + 0.01,
            "receiver_ci95_low_vs_packet_only": receiver - packet - 0.01,
            "receiver_ci95_low_vs_target_only": receiver - target - 0.01,
            "receiver_eval_accuracy": receiver,
            "receiver_minus_packet_only": receiver - packet,
            "receiver_minus_target_only": receiver - target,
            "row_count": 12,
            "selected_receiver_kind": "candidate_ridge_receiver",
            "selected_receiver_train_accuracy": 0.5,
            "source_family": "Qwen2.5",
            "strict_packet_delta_required": 0.005,
            "strict_target_delta_required": 0.02,
            "target_family": "Phi-3-mini",
            "target_only_eval_accuracy": target,
            "target_or_packet_oracle_eval_accuracy": oracle,
            "train_rows": 2,
        },
        "source_packet": {
            "artifact_path": str(packet_artifact),
            "artifact_sha256": "fixture",
            "predictions_jsonl": "fixture.jsonl",
            "predictions_sha256": "fixture",
        },
        "target_scores": {
            "artifact_path": "target_global_artifact.json",
            "artifact_sha256": "fixture",
            "row_count": 12,
            "score_slice_count": 1,
            "slices": [{"start": start, "end": start + 12, "score_cache": "fixture.json"}],
        },
        "control_rows": [{"name": "zero_hidden_prediction"}],
    }


def test_receiver_family_multislice_summary_records_packet_transfer_without_receiver_pass(
    tmp_path,
):
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    _write_json(
        left,
        _artifact(
            root=tmp_path,
            name="left",
            start=1024,
            target=0.25,
            packet=0.45,
            receiver=0.44,
            oracle=0.60,
        ),
    )
    _write_json(
        right,
        _artifact(
            root=tmp_path,
            name="right",
            start=1036,
            target=0.30,
            packet=0.50,
            receiver=0.40,
            oracle=0.65,
        ),
    )

    payload = summary.build_summary(
        output_dir=tmp_path / "out",
        artifacts=(left, right),
        run_date="2026-05-03",
    )

    h = payload["headline"]
    assert payload["pass_gate"] is False
    assert h["slice_count"] == 2
    assert h["contiguous"] is True
    assert h["target_family_transfer_slice_count"] == 2
    assert h["receiver_improvement_slice_count"] == 0
    assert h["weighted_packet_minus_target_only"] == pytest.approx(0.2)
    assert h["weighted_receiver_minus_packet_only"] == pytest.approx(-0.055)
    assert h["source_private_packet"] is True
    assert h["raw_payload_bytes"] == 2
    assert h["framed_record_bytes"] == 5
    assert (tmp_path / "out" / "slice_rows.csv").exists()
