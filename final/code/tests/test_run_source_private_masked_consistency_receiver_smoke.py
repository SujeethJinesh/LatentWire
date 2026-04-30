from __future__ import annotations

import json

from scripts import run_source_private_masked_consistency_receiver_smoke as gate


def _state() -> gate.ConsistencyReceiverState:
    return gate._fit_state(
        train_examples=64,
        eval_examples=16,
        train_seed=5,
        eval_seed=6,
        train_family_set="all",
        eval_family_set="all",
        candidates=4,
        feature_dim=64,
        budget_bytes=2,
        ridge=1e-2,
        candidate_view="full",
        fit_intercept=False,
    )


def test_receiver_features_have_one_row_per_candidate() -> None:
    state = _state()
    example = state.eval_rows[0]
    payload, mask, _ = gate._payload_for_condition(
        condition="matched_consistency_packet",
        example=example,
        rows=state.eval_rows,
        index=0,
        state=state,
        rng=gate.random.Random(7),
    )

    features = gate._receiver_features(example, payload, mask, state)

    assert features.shape[0] == len(example.candidates)
    assert features.shape[1] >= 10
    assert features[:, 1].tolist() == [1.0] * len(example.candidates)


def test_fit_receiver_predicts_valid_candidate() -> None:
    state = _state()
    weights = gate._fit_receiver(
        state,
        seed=7,
        receiver_ridge=1e-2,
        mask_rounds=1,
        random_rounds=1,
        matched_weight=4.0,
        mask_weight=2.0,
        control_weight=2.0,
        target_only_weight=2.0,
    )
    example = state.eval_rows[0]
    payload, mask, _ = gate._payload_for_condition(
        condition="matched_consistency_packet",
        example=example,
        rows=state.eval_rows,
        index=0,
        state=state,
        rng=gate.random.Random(7),
    )

    prediction, scores = gate._learned_prediction(example, payload, mask, state, weights)

    assert prediction in {candidate.label for candidate in example.candidates}
    assert len(scores) == len(example.candidates)


def test_run_gate_writes_smoke_artifacts(tmp_path) -> None:
    payload = gate.run_gate(
        output_dir=tmp_path,
        train_examples=64,
        eval_examples=16,
        train_seed=7,
        eval_seed=8,
        train_family_set="all",
        eval_family_set="all",
        candidates=4,
        feature_dim=64,
        budget_bytes=2,
        ridge=1e-2,
        receiver_ridge=1e-2,
        candidate_view="full",
        fit_intercept=False,
        seed=9,
        mask_rounds=1,
        random_rounds=1,
        matched_weight=4.0,
        mask_weight=2.0,
        control_weight=2.0,
        target_only_weight=2.0,
        tolerance_vs_hamming=0.25,
        conditions=["target_only", "matched_consistency_packet", *gate.CONTROL_CONDITIONS],
    )

    assert payload["summary"]["exact_id_parity"] is True
    assert payload["summary"]["n"] == 16
    assert "matched_consistency_packet" in payload["summary"]["learned_metrics"]
    assert (tmp_path / "predictions.jsonl").exists()
    assert (tmp_path / "summary.md").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert "run_summary.json" in manifest["artifacts"]
