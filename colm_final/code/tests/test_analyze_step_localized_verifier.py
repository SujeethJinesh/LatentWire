from __future__ import annotations

import json
from pathlib import Path

from scripts import analyze_step_localized_verifier as verifier


def _row(index: int, method: str, correct: bool, **extra: object) -> dict[str, object]:
    row: dict[str, object] = {
        "index": index,
        "method": method,
        "correct": correct,
        "example_id": f"ex-{index}",
    }
    row.update(extra)
    return row


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def test_extract_step_verifier_features_localizes_invalid_equation() -> None:
    row = {
        "prediction": "First compute 2 + 2 = 5.\nThen answer: 5",
        "normalized_prediction": "5",
        "candidate_format_score": 1.0,
        "candidate_completion_score": 1.0,
        "candidate_vote_margin": 0,
        "candidate_answer_agreement": 1,
    }

    features = verifier.extract_step_verifier_features(row)

    assert features["step_count"] == 2
    assert features["equation_step_count"] == 1
    assert features["valid_equation_step_count"] == 0
    assert features["first_error_step_1indexed"] == 1
    assert features["has_invalid_equation"] is True
    assert features["localized_error_count"] >= 1


def test_summarize_source_compares_scalar_and_step_policies(tmp_path: Path) -> None:
    source = tmp_path / "repair.jsonl"
    rows: list[dict[str, object]] = []
    selected_correct = [True, False, False, True]
    repair_correct = [True, True, False, True]
    predictions = [
        "2 + 2 = 4.\nFinal answer: 4",
        "3 + 4 = 8.\nFinal answer: 8",
        "The result is unclear",
        "5 - 1 = 4.\nAnswer: 4",
    ]
    for idx in range(4):
        rows.append(_row(idx, "target_alone", False))
        rows.append(
            _row(
                idx,
                "selected_route_no_repair",
                selected_correct[idx],
                prediction=predictions[idx],
                normalized_prediction=["4", "8", "0", "4"][idx],
                candidate_format_score=[4.0, 4.0, -1.0, 3.0][idx],
                candidate_completion_score=1.0,
                candidate_vote_margin=1,
                candidate_answer_agreement=1,
                generated_tokens=16,
            )
        )
        rows.append(_row(idx, "target_self_repair", repair_correct[idx]))
        rows.append(
            _row(
                idx,
                "process_repair_selected_route",
                repair_correct[idx],
                repair_prompt_chars=100,
                generated_tokens=20,
            )
        )
    _write_jsonl(source, rows)

    summary = verifier.summarize_source(source, bootstrap_samples=50, bootstrap_seed=7)

    assert summary.source == source.name
    policies = {row["policy"] for row in summary.rows}
    assert "scalar_meta_gate" in policies
    assert "step_localized_gate" in policies
    assert "critique_plus_repair_gate" in policies
    repair_all = next(row for row in summary.rows if row["policy"] == "repair_all_selected")
    assert repair_all["accuracy"] == 0.75
    assert repair_all["accuracy_ci_low"] is not None


def test_main_writes_json_and_markdown(tmp_path: Path) -> None:
    source = tmp_path / "repair.jsonl"
    output_json = tmp_path / "verifier.json"
    output_md = tmp_path / "verifier.md"
    _write_jsonl(
        source,
        [
            _row(0, "target_alone", False),
            _row(
                0,
                "selected_route_no_repair",
                True,
                prediction="1 + 1 = 2. Final answer: 2",
                normalized_prediction="2",
            ),
            _row(0, "process_repair_selected_route", True),
        ],
    )

    payload = verifier.main(
        [
            "--inputs",
            str(source),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--bootstrap-samples",
            "20",
        ]
    )

    assert json.loads(output_json.read_text()) == payload
    markdown = output_md.read_text()
    assert "# Step-Localized Verifier Replay" in markdown
    assert "step_localized_gate" in markdown
