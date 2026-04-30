from __future__ import annotations

import numpy as np

from scripts import run_source_private_masked_pq_consistency_receiver as gate
from scripts import run_source_private_product_codebook_target_decoder_smoke as pq_gate


def _state() -> pq_gate.ProductCodebookReceiverState:
    return pq_gate.build_receiver_state(
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
        candidate_view="slot",
        fit_intercept=False,
        remap_slot_seed=101,
    )


def test_candidate_features_have_one_row_per_candidate() -> None:
    state = _state()
    example = state.eval_rows[0]
    payload = gate._payload_for_rows(
        condition="matched_product_codebook",
        example=example,
        state=state,
        rows=state.eval_rows,
        index=0,
        rng=gate.random.Random(7),
    )

    features = gate._candidate_features(example, payload, state)

    assert features.shape[0] == len(example.candidates)
    assert features.shape[1] >= 8
    assert np.all(features[:, 1] == 1.0)


def test_fit_score_receiver_predicts_valid_candidate() -> None:
    state = _state()
    weights = gate._fit_score_receiver(
        state,
        ridge=1e-2,
        seed=7,
        mask_rounds=1,
        random_rounds=1,
        matched_weight=4.0,
        mask_weight=2.0,
        control_weight=1.0,
        target_only_weight=1.0,
    )
    example = state.eval_rows[0]
    payload = gate._payload_for_rows(
        condition="matched_product_codebook",
        example=example,
        state=state,
        rows=state.eval_rows,
        index=0,
        rng=gate.random.Random(7),
    )

    prediction = gate._receiver_prediction(example, payload, state, weights)

    assert prediction in {candidate.label for candidate in example.candidates}


def test_summarize_reports_packet_and_l2_gates() -> None:
    state = _state()
    weights = gate._fit_score_receiver(
        state,
        ridge=1e-2,
        seed=7,
        mask_rounds=1,
        random_rounds=1,
        matched_weight=4.0,
        mask_weight=2.0,
        control_weight=1.0,
        target_only_weight=1.0,
    )
    rows = gate._predict_rows(
        state,
        weights,
        seed=8,
        conditions=["target_only", "matched_product_codebook", *gate.CONTROL_CONDITIONS],
    )

    summary = gate._summarize(rows, conditions=["target_only", "matched_product_codebook", *gate.CONTROL_CONDITIONS])

    assert summary["n"] == len(state.eval_rows)
    assert summary["exact_id_parity"] is True
    assert "learned_minus_l2" in summary
    assert "matched_product_codebook" in summary["learned_metrics"]
