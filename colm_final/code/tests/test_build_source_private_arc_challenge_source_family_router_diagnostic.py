from __future__ import annotations

import csv
import json

from scripts import build_source_private_arc_challenge_source_family_router_diagnostic as gate


def _write_jsonl(path, rows) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _prediction(
    *,
    split: str,
    seed: int,
    content_id: str,
    condition: str,
    correct: bool,
    scores: list[float],
    source_selected_index: int,
) -> dict:
    best = max(range(len(scores)), key=lambda index: (scores[index], -index))
    return {
        "split": split,
        "seed": seed,
        "content_id": content_id,
        "row_id": content_id,
        "condition": condition,
        "answer_index": 0,
        "prediction_index": best,
        "correct": correct,
        "payload_bytes": 12,
        "latency_ms": 0.1,
        "metadata": {
            "scores": scores,
            "best_score": max(scores),
            "packet_code_l2": sum(abs(value) for value in scores),
            "source_selected_index": source_selected_index,
        },
    }


def _pair_rows(split: str, seed: int, content_id: str, *, alt_correct: bool, qwen_correct: bool) -> list[dict]:
    alt_scores = [3.0, 0.0, -1.0] if alt_correct else [0.3, 0.2, 0.1]
    qwen_scores = [3.0, 0.0, -1.0] if qwen_correct else [0.3, 0.2, 0.1]
    return [
        _prediction(
            split=split,
            seed=seed,
            content_id=content_id,
            condition=gate.ALT_CONDITION,
            correct=alt_correct,
            scores=alt_scores,
            source_selected_index=0,
        ),
        _prediction(
            split=split,
            seed=seed,
            content_id=content_id,
            condition=gate.QWEN_CONDITION,
            correct=qwen_correct,
            scores=qwen_scores,
            source_selected_index=0,
        ),
    ]


def test_router_diagnostic_writes_rule_and_oracle_artifacts(tmp_path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    rows = []
    agreement_rows = []
    patterns = [
        ("validation", "validation-a", True, False),
        ("validation", "validation-b", False, True),
        ("validation", "validation-c", True, False),
        ("validation", "validation-d", False, True),
        ("test", "test-a", True, False),
        ("test", "test-b", False, True),
        ("test", "test-c", True, False),
        ("test", "test-d", False, True),
    ]
    for split, content_id, alt_correct, qwen_correct in patterns:
        rows.extend(
            _pair_rows(
                split=split,
                seed=47,
                content_id=content_id,
                alt_correct=alt_correct,
                qwen_correct=qwen_correct,
            )
        )
        agreement_rows.append(
            {
                "split": split,
                "row_id": content_id,
                "content_id": content_id,
                "answer_index": "0",
                "alt_source_selected_index": "0",
                "qwen_source_selected_index": "1",
                "agree": "False",
                "alt_source_correct": str(alt_correct),
                "qwen_source_correct": str(qwen_correct),
            }
        )
    _write_jsonl(input_dir / "qwen_disagreement_predictions.jsonl", rows)
    with (input_dir / "source_cache_agreement.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(agreement_rows[0]))
        writer.writeheader()
        writer.writerows(agreement_rows)
    (input_dir / "source_family_cache_falsification.json").write_text(
        json.dumps({"pass_gate": False}, sort_keys=True),
        encoding="utf-8",
    )

    payload = gate.build_router_diagnostic(
        input_dir=input_dir,
        output_dir=output_dir,
        bootstrap_samples=20,
        min_gap_over_qwen=-1.0,
    )

    assert payload["gate"] == "source_private_arc_challenge_source_family_router_diagnostic"
    assert payload["parent_pass_gate"] is False
    assert payload["source_choice_complementarity"]["test"]["source_choice_oracle_accuracy"] == 1.0
    assert payload["selected_rule_test_summary"]["aggregate"]["oracle_accuracy_mean"] == 1.0
    assert payload["selected_rule"]["metric"] in gate.METRICS
    assert (output_dir / "source_family_router_diagnostic.json").exists()
    assert (output_dir / "source_family_router_diagnostic.md").exists()
    assert (output_dir / "router_rule_metrics.csv").exists()
    assert (output_dir / "selected_router_predictions.jsonl").exists()
    selected_rows = [
        json.loads(line)
        for line in (output_dir / "selected_router_predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {row["split"] for row in selected_rows} == {"validation", "test"}
    assert {row["metric"] for row in selected_rows} == {payload["selected_rule"]["metric"]}
