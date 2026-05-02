from __future__ import annotations

import json

from scripts import build_source_private_benchmark_selection_gate as gate


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _seed_payload(*, budget: int, matched: float, target: float, text: float, seeds: int = 5):
    return {
        "pass_gate": True,
        "budget_bytes": budget,
        "eval_rows": 10,
        "aggregate": {
            "all_seeds_pass": True,
            "best_destructive_accuracy": target,
            "matched_accuracy_mean": matched,
            "matched_accuracy_min": matched,
            "matched_accuracy_max": matched,
            "target_accuracy": target,
            "same_byte_structured_text_accuracy": text,
            "matched_minus_target_min": matched - target,
            "matched_minus_same_byte_text_min": matched - text,
            "matched_minus_best_destructive_min": matched - target,
            "paired_ci95_low_vs_target_min": 0.03,
            "seed_count": seeds,
            "pass_count": seeds,
        },
    }


def _prediction_rows(*, row_count: int, target_correct: set[int], packet_correct: set[int]):
    rows = []
    for index in range(row_count):
        answer = index % 4
        for condition, correct_set in (
            ("target_only", target_correct),
            ("matched_source_private_packet", packet_correct),
        ):
            correct = index in correct_set
            rows.append(
                {
                    "row_id": str(index),
                    "condition": condition,
                    "answer_index": answer,
                    "correct": correct,
                    "prediction_index": answer if correct else (answer + 1) % 4,
                }
            )
    return rows


def test_benchmark_selection_gate_selects_openbookqa(tmp_path):
    openbook_seed = tmp_path / "openbook_seed.json"
    arc_seed = tmp_path / "arc_seed.json"
    csqa_seed = tmp_path / "csqa_seed.json"
    openbook_predictions = tmp_path / "openbook_predictions.jsonl"
    arc_predictions = tmp_path / "arc_predictions.jsonl"
    csqa_predictions = tmp_path / "csqa_predictions.jsonl"
    _write_json(openbook_seed, _seed_payload(budget=3, matched=0.38, target=0.27, text=0.35))
    _write_json(arc_seed, _seed_payload(budget=12, matched=0.34, target=0.26, text=0.31))
    _write_json(csqa_seed, _seed_payload(budget=2, matched=0.44, target=0.21, text=0.43))
    _write_jsonl(openbook_predictions, _prediction_rows(row_count=10, target_correct={0, 1, 2}, packet_correct={2, 3, 4, 5}))
    _write_jsonl(arc_predictions, _prediction_rows(row_count=10, target_correct={0, 1, 2}, packet_correct={2, 3, 4, 5}))
    _write_jsonl(csqa_predictions, _prediction_rows(row_count=10, target_correct={0, 1, 2}, packet_correct={2, 3, 4, 5}))

    benchmarks = (
        {
            "row_id": "openbook",
            "dataset": "OpenBookQA",
            "split": "test",
            "priority": 0,
            "seed_artifact": openbook_seed,
            "oracle_predictions_jsonl": openbook_predictions,
        },
        {
            "row_id": "arc",
            "dataset": "ARC-Challenge",
            "split": "test",
            "priority": 1,
            "seed_artifact": arc_seed,
            "oracle_predictions_jsonl": arc_predictions,
        },
        {
            "row_id": "csqa",
            "dataset": "CommonsenseQA",
            "split": "validation",
            "priority": 2,
            "seed_artifact": csqa_seed,
            "oracle_predictions_jsonl": csqa_predictions,
        },
    )

    payload = gate.build_gate(output_dir=tmp_path / "out", benchmarks=benchmarks)

    assert payload["pass_gate"] is True
    assert payload["headline"]["selected_benchmark"] == "OpenBookQA test"
    assert payload["headline"]["receiver_candidate_count"] == 2
    assert payload["headline"]["diagnostic_text_saturated_count"] == 1
    rows = {row["row_id"]: row for row in payload["rows"]}
    assert rows["openbook"]["selection_role"] == "receiver_candidate"
    assert rows["csqa"]["selection_role"] == "diagnostic_text_saturated"
    assert rows["openbook"]["oracle_headroom_vs_packet"] == 0.2
    assert (tmp_path / "out" / "benchmark_selection_gate.json").exists()
    assert (tmp_path / "out" / "benchmark_selection_gate.csv").exists()
    assert (tmp_path / "out" / "benchmark_selection_gate.md").exists()


def test_benchmark_selection_gate_fails_when_text_margin_saturates(tmp_path):
    seed = tmp_path / "seed.json"
    predictions = tmp_path / "predictions.jsonl"
    _write_json(seed, _seed_payload(budget=2, matched=0.44, target=0.21, text=0.43))
    _write_jsonl(predictions, _prediction_rows(row_count=10, target_correct={0, 1, 2}, packet_correct={2, 3, 4, 5}))

    payload = gate.build_gate(
        output_dir=tmp_path / "out",
        benchmarks=(
            {
                "row_id": "csqa",
                "dataset": "CommonsenseQA",
                "split": "validation",
                "priority": 0,
                "seed_artifact": seed,
                "oracle_predictions_jsonl": predictions,
            },
        ),
    )

    assert payload["pass_gate"] is False
    assert payload["headline"]["selected_benchmark"] == "none"
    assert payload["rows"][0]["selection_role"] == "diagnostic_text_saturated"
