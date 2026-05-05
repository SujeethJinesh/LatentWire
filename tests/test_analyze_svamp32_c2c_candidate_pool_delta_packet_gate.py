from __future__ import annotations

import json
import pathlib

from scripts import analyze_svamp32_c2c_candidate_pool_delta_packet_gate as gate


def test_candidate_pool_canonicalizes_and_sorts_numeric_predictions() -> None:
    candidates = gate._candidate_pool(
        answers=["5.0", "#### 5"],
        rows={
            "target_alone": {"normalized_prediction": "12"},
            "source_alone": {"prediction": "Final answer: 3.0"},
            "text_to_text": {"normalized_prediction": "5"},
            "c2c_teacher": {"prediction": "#### 7"},
        },
    )

    assert [candidate.value for candidate in candidates] == ["3", "5", "7", "12"]
    by_value = {candidate.value: candidate.origins for candidate in candidates}
    assert by_value["5"] == ("gold", "text_to_text")


def test_evaluate_conditions_promotes_matched_when_controls_fail() -> None:
    rows = [
        gate.RowScores(
            index=0,
            example_id="row-a",
            answers=("5",),
            candidates=(
                gate.Candidate("3", 3.0, ("target_alone",)),
                gate.Candidate("5", 5.0, ("gold", "c2c_teacher")),
            ),
            target_scores=(2.0, 1.0),
            teacher_scores=(1.0, 2.0),
            packet_values=(-2.0, 2.0),
            packet_quantized=(-7, 7),
            packet_scale=1.0 / 7.0,
        ),
        gate.RowScores(
            index=1,
            example_id="row-b",
            answers=("8",),
            candidates=(
                gate.Candidate("8", 8.0, ("gold", "c2c_teacher")),
                gate.Candidate("9", 9.0, ("target_alone",)),
            ),
            target_scores=(1.0, 2.0),
            teacher_scores=(2.0, 1.0),
            packet_values=(2.0, -2.0),
            packet_quantized=(7, -7),
            packet_scale=1.0 / 7.0,
        ),
    ]

    payload = gate.evaluate_conditions(
        rows,
        target_ids={"teacher_only": {"row-a", "row-b"}, "clean_residual_targets": {"row-a", "row-b"}},
        conditions=("matched", "target_only", "zero_delta", "candidate_roll", "row_shuffle"),
        rng_seed=1,
    )

    assert payload["condition_summaries"]["matched"]["correct_count"] == 2
    assert payload["condition_summaries"]["target_only"]["correct_count"] == 0
    assert payload["condition_summaries"]["candidate_roll"]["correct_count"] == 0
    assert payload["source_necessary_clean_ids"] == ["row-a", "row-b"]


def test_analyze_writes_manifest_and_byte_contract(tmp_path: pathlib.Path) -> None:
    rows = [
        gate.RowScores(
            index=0,
            example_id="row-a",
            answers=("5",),
            candidates=(
                gate.Candidate("3", 3.0, ("target_alone",)),
                gate.Candidate("5", 5.0, ("gold", "c2c_teacher")),
            ),
            target_scores=(2.0, 1.0),
            teacher_scores=(1.0, 2.0),
            packet_values=(-2.0, 2.0),
            packet_quantized=(-7, 7),
            packet_scale=1.0 / 7.0,
        ),
        gate.RowScores(
            index=1,
            example_id="row-b",
            answers=("8",),
            candidates=(
                gate.Candidate("8", 8.0, ("gold", "c2c_teacher")),
                gate.Candidate("9", 9.0, ("target_alone",)),
            ),
            target_scores=(1.0, 2.0),
            teacher_scores=(2.0, 1.0),
            packet_values=(2.0, -2.0),
            packet_quantized=(7, -7),
            packet_scale=1.0 / 7.0,
        ),
    ]
    target_set = tmp_path / "target_set.json"
    target_set.write_text(
        json.dumps({"ids": {"teacher_only": ["row-a", "row-b"], "clean_residual_targets": ["row-a", "row-b"]}}),
        encoding="utf-8",
    )

    payload = gate.analyze(
        rows=rows,
        target_set_path=target_set,
        run_config={"coeff_bits": 4},
        run_date="2026-05-05",
        output_json=tmp_path / "gate.json",
        output_md=tmp_path / "gate.md",
    )

    assert payload["status"] == "c2c_candidate_pool_delta_packet_capacity_clears_controls_not_deployable"
    assert payload["packet_contract"]["avg_packet_bytes_per_row"] == 2.0
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "manifest.md").exists()
