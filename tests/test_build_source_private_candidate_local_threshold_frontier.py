from __future__ import annotations

import json

from scripts.build_source_private_candidate_local_threshold_frontier import build_threshold_frontier


def _row(
    *,
    condition: str,
    example_id: str,
    answer_index: int,
    prior_index: int,
    scores: list[float] | None,
) -> dict[str, object]:
    if scores is None:
        prediction_index = prior_index
    else:
        best = max(scores)
        tied = [idx for idx, score in enumerate(scores) if abs(score - best) <= 1e-8]
        prediction_index = prior_index if prior_index in tied else tied[0]
    return {
        "condition": condition,
        "example_id": example_id,
        "answer_index": answer_index,
        "prior_index": prior_index,
        "correct": prediction_index == answer_index,
        "metadata": {"scores": scores} if scores is not None else {"decoder": "prior"},
    }


def _write_predictions(root, direction: str, learned_scores: list[list[float]], control_scores: list[list[float]]) -> None:
    path = root / direction
    path.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, (learned, control) in enumerate(zip(learned_scores, control_scores, strict=True)):
        example_id = f"{direction}_{idx}"
        rows.append(_row(condition="target_only", example_id=example_id, answer_index=0, prior_index=1, scores=None))
        rows.append(
            _row(
                condition="learned_synonym_dictionary_packet",
                example_id=example_id,
                answer_index=0,
                prior_index=1,
                scores=learned,
            )
        )
        for condition in (
            "zero_source",
            "shuffled_source",
            "answer_masked_source",
            "public_only_sidecar",
            "target_derived_sidecar",
            "random_same_byte",
            "atom_id_derangement",
            "private_random_source_atoms",
            "permuted_teacher_receiver",
        ):
            rows.append(
                _row(
                    condition=condition,
                    example_id=example_id,
                    answer_index=0,
                    prior_index=1,
                    scores=control,
                )
            )
    (path / "predictions_budget8.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_threshold_frontier_finds_live_clean_band_and_leaky_control(tmp_path) -> None:
    live = tmp_path / "live"
    leaky = tmp_path / "leaky"
    for direction in ("core_to_holdout", "holdout_to_core", "same_family_all"):
        _write_predictions(
            live,
            direction,
            learned_scores=[[0.7, 0.1], [0.65, 0.2], [0.8, 0.0], [0.75, 0.1]],
            control_scores=[[0.2, 0.6], [0.1, 0.7], [0.3, 0.5], [0.2, 0.55]],
        )
        _write_predictions(
            leaky,
            direction,
            learned_scores=[[0.7, 0.1], [0.65, 0.2], [0.8, 0.0], [0.75, 0.1]],
            control_scores=[[0.6, 0.2], [0.7, 0.1], [0.5, 0.3], [0.55, 0.2]],
        )

    payload = build_threshold_frontier(
        method_roots={
            "live_candidate_local_residual_norm": [live],
            "rr_anchor_coordinate_dot": [leaky],
            "public_random_rotation_sign": [leaky],
        },
        output_dir=tmp_path / "out",
        thresholds=(0.0, 0.48, 0.85),
    )

    assert payload["headline"]["pass_gate"] is True
    assert payload["headline"]["live_clean_threshold_range"]["exists"] is True
    assert payload["headline"]["rr_clean_threshold_range"]["exists"] is False
    assert payload["headline"]["random_rotation_sign_clean_threshold_range"]["exists"] is False
    by_method_threshold = {
        (row["method_id"], row["threshold"]): row for row in payload["summary_rows"]
    }
    assert by_method_threshold[("live_candidate_local_residual_norm", 0.48)]["clean_rows"] == 3
    assert by_method_threshold[("rr_anchor_coordinate_dot", 0.48)]["best_control_accuracy_max"] == 1.0
    assert (tmp_path / "out" / "candidate_local_threshold_frontier.json").exists()
    assert (tmp_path / "out" / "candidate_local_threshold_frontier_direction_rows.csv").exists()
