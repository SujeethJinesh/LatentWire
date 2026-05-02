from __future__ import annotations

import json

from scripts import build_source_private_candidate_local_margin_atlas as atlas


def _row(
    *,
    condition: str,
    example_id: str,
    answer_index: int,
    prior_index: int,
    scores: list[float] | None,
) -> dict[str, object]:
    prediction_index = prior_index if scores is None else max(range(len(scores)), key=lambda idx: scores[idx])
    return {
        "condition": condition,
        "example_id": example_id,
        "family_name": "fam_a" if example_id.endswith("0") else "fam_b",
        "answer_index": answer_index,
        "prior_index": prior_index,
        "correct": prediction_index == answer_index,
        "payload_bytes": 8 if condition != "target_only" else 0,
        "latency_ms": 0.1,
        "metadata": {"scores": scores} if scores is not None else {"decoder": "prior"},
    }


def _write_predictions(
    root,
    direction: str,
    *,
    learned_scores: list[float],
    control_scores: list[float],
    oracle_scores: list[float] | None = None,
) -> None:
    path = root / direction
    path.mkdir(parents=True, exist_ok=True)
    rows = []
    oracle = oracle_scores or learned_scores
    for idx in range(4):
        example_id = f"{direction}_{idx}"
        rows.append(_row(condition="target_only", example_id=example_id, answer_index=0, prior_index=1, scores=None))
        rows.append(
            _row(
                condition="learned_synonym_dictionary_packet",
                example_id=example_id,
                answer_index=0,
                prior_index=1,
                scores=learned_scores,
            )
        )
        rows.append(
            _row(
                condition="oracle_learned_candidate_atoms",
                example_id=example_id,
                answer_index=0,
                prior_index=1,
                scores=oracle,
            )
        )
        for condition in atlas.STRICT_SOURCE_DESTROYING_CONTROLS:
            rows.append(
                _row(
                    condition=condition,
                    example_id=example_id,
                    answer_index=0,
                    prior_index=1,
                    scores=control_scores,
                )
            )
    (path / "predictions_budget8.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_threshold_prediction_ties_and_fallback_match_receiver_rule() -> None:
    tied = _row(condition="learned_synonym_dictionary_packet", example_id="ex0", answer_index=0, prior_index=1, scores=[0.6, 0.6])
    low = _row(condition="learned_synonym_dictionary_packet", example_id="ex1", answer_index=0, prior_index=1, scores=[0.4, 0.1])

    assert atlas._predicted_index_at_threshold(tied) == 1
    assert atlas._predicted_index_at_threshold(low) == 1
    example = atlas._example_margin_row(
        method_id="m",
        root=atlas.ROOT,
        direction="core_to_holdout",
        row=tied,
    )
    assert example["answer_margin"] == 0.0
    assert example["winner_margin"] == 0.0
    assert example["answer_rank"] == 1
    assert example["accepted_at_048"] is True
    assert example["correct_at_048"] is False


def test_margin_atlas_separates_live_from_control_leaky_common_basis(tmp_path) -> None:
    roots = {
        "live_candidate_local_residual_norm": tmp_path / "live",
        "rr_anchor_coordinate_dot": tmp_path / "rr",
        "procrustes_common_basis": tmp_path / "procrustes",
        "public_random_rotation_sign": tmp_path / "sign",
    }
    for direction in ("core_to_holdout", "holdout_to_core", "same_family_all"):
        _write_predictions(
            roots["live_candidate_local_residual_norm"],
            direction,
            learned_scores=[0.7, 0.1],
            control_scores=[0.2, 0.4],
            oracle_scores=[0.9, 0.0],
        )
        _write_predictions(
            roots["rr_anchor_coordinate_dot"],
            direction,
            learned_scores=[0.7, 0.1] if direction != "holdout_to_core" else [0.2, 0.3],
            control_scores=[0.2, 0.4],
            oracle_scores=[0.9, 0.0],
        )
        _write_predictions(
            roots["procrustes_common_basis"],
            direction,
            learned_scores=[0.7, 0.1],
            control_scores=[0.7, 0.1],
            oracle_scores=[0.9, 0.0],
        )
        _write_predictions(
            roots["public_random_rotation_sign"],
            direction,
            learned_scores=[0.7, 0.1],
            control_scores=[0.55, 0.1],
            oracle_scores=[0.9, 0.0],
        )

    payload = atlas.build_margin_atlas(
        method_roots={method: [root] for method, root in roots.items()},
        output_dir=tmp_path / "out",
    )

    assert payload["headline"]["pass_gate"] is True
    assert payload["headline"]["live_matched_positive_margin_rate"] == 1.0
    assert payload["headline"]["live_best_control_stored_accuracy"] == 0.0
    assert payload["headline"]["procrustes_best_control_positive_margin_rate"] == 1.0
    assert payload["example_rows"]
    assert (tmp_path / "out" / "margin_atlas.json").exists()
    assert (tmp_path / "out" / "margin_atlas_example_rows.csv").exists()
    assert (tmp_path / "out" / "margin_atlas.svg").exists()
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert "margin_atlas_example_rows.csv" in manifest["artifacts"]
    assert "margin_atlas.svg" in manifest["artifacts"]
