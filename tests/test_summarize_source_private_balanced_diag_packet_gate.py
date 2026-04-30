from __future__ import annotations

import json
import pathlib

from scripts import summarize_source_private_balanced_diag_packet_gate as summary_script


def _write_direct(path: pathlib.Path, *, packet_correct: list[bool], target_correct: list[bool], family_set: str = "all") -> None:
    path.mkdir(parents=True)
    rows = []
    ids = [f"ex_{idx:04d}" for idx in range(len(packet_correct))]
    for example_id, packet, target in zip(ids, packet_correct, target_correct, strict=True):
        rows.append(
            {
                "example_id": example_id,
                "family_name": f"family_{int(example_id[-4:]) % 2}",
                "answer_label": f"answer_{example_id}",
                "conditions": {
                    "matched_repair_packet": {"correct": packet},
                    "target_only": {"correct": target},
                },
            }
        )
    (path / "predictions_budget2.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    packet_accuracy = sum(packet_correct) / len(packet_correct)
    target_accuracy = sum(target_correct) / len(target_correct)
    sweep = {
        "budget_summaries": [
            {
                "budget_bytes": 2,
                "pass_gate": True,
                "matched_selector_accuracy": packet_accuracy,
                "best_no_source_accuracy": target_accuracy,
                "best_source_destroying_control_accuracy": target_accuracy,
                "best_reviewer_negative_control_accuracy": target_accuracy,
                "metrics": {
                    "matched_repair_packet": {"mean_payload_bytes": 2.0, "p50_latency_ms": 0.1},
                },
            }
        ]
    }
    (path / "sweep_summary.json").write_text(json.dumps(sweep), encoding="utf-8")
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "args": {
                    "diagnostic_table_mode": "plausible_decoys",
                    "examples": len(packet_correct),
                    "family_set": family_set,
                    "seed": 29,
                }
            }
        ),
        encoding="utf-8",
    )


def _write_public(path: pathlib.Path, *, public_correct: list[bool], target_correct: list[bool], eval_family_set: str = "all") -> None:
    path.mkdir(parents=True)
    rows = []
    ids = [f"ex_{idx:04d}" for idx in range(len(public_correct))]
    for example_id, public, target in zip(ids, public_correct, target_correct, strict=True):
        rows.append(
            {
                "example_id": example_id,
                "family_name": f"family_{int(example_id[-4:]) % 2}",
                "answer_label": f"answer_{example_id}",
                "public_correct": public,
                "target_correct": target,
            }
        )
    (path / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    public_accuracy = sum(public_correct) / len(public_correct)
    target_accuracy = sum(target_correct) / len(target_correct)
    payload = {
        "candidate_view": "diag_only",
        "diagnostic_table_mode": "plausible_decoys",
        "eval_examples": len(public_correct),
        "eval_family_set": eval_family_set,
        "eval_seed": 29,
        "pass_gate": True,
        "summary": {
            "public_only_accuracy": public_accuracy,
            "target_only_accuracy": target_accuracy,
            "train_eval_id_intersection_count": 0,
            "p50_latency_ms": 0.2,
        },
        "train_family_set": "core",
    }
    (path / "run_summary.json").write_text(json.dumps(payload), encoding="utf-8")


def test_balanced_diag_summary_passes_when_packet_beats_public(tmp_path) -> None:
    direct = tmp_path / "direct"
    public = tmp_path / "public"
    target = [True, False, False, False] * 8
    _write_direct(direct, packet_correct=[True] * 32, target_correct=target)
    _write_public(public, public_correct=target, target_correct=target)

    payload = summary_script.summarize_pairs(
        pairs=[(direct, public)],
        output_dir=tmp_path / "out",
        budget_bytes=2,
        min_packet_minus_public_ci_low=0.10,
        max_public_lift=0.05,
    )

    assert payload["headline"]["pass_gate"] is True
    assert (tmp_path / "out" / "summary.md").exists()


def test_balanced_diag_summary_fails_when_public_matches_packet(tmp_path) -> None:
    direct = tmp_path / "direct"
    public = tmp_path / "public"
    target = [True, False, False, False] * 8
    _write_direct(direct, packet_correct=[True] * 32, target_correct=target)
    _write_public(public, public_correct=[True] * 32, target_correct=target)

    payload = summary_script.summarize_pairs(
        pairs=[(direct, public)],
        output_dir=tmp_path / "out",
        budget_bytes=2,
        min_packet_minus_public_ci_low=0.10,
        max_public_lift=0.05,
    )

    assert payload["headline"]["pass_gate"] is False


def test_balanced_diag_summary_requires_content_parity(tmp_path) -> None:
    direct = tmp_path / "direct"
    public = tmp_path / "public"
    target = [True, False, False, False] * 8
    _write_direct(direct, packet_correct=[True] * 32, target_correct=target)
    _write_public(public, public_correct=target, target_correct=target)
    rows = [json.loads(line) for line in (public / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[0]["family_name"] = "different_family"
    (public / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    payload = summary_script.summarize_pairs(
        pairs=[(direct, public)],
        output_dir=tmp_path / "out",
        budget_bytes=2,
        min_packet_minus_public_ci_low=0.10,
        max_public_lift=0.05,
    )

    assert payload["headline"]["pass_gate"] is False
    assert payload["headline"]["all_content_parity"] is False
