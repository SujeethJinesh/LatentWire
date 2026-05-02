from __future__ import annotations

import numpy as np

from scripts import run_source_private_arc_candidate_alignment_receiver_preflight as gate
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate


def _rows() -> list[arc_gate.ArcRow]:
    return [
        arc_gate.ArcRow(
            row_id="r0",
            content_id="c0",
            question="Which option is first?",
            choices=("alpha", "beta", "gamma"),
            choice_labels=("A", "B", "C"),
            answer_index=0,
            answer_label="A",
        ),
        arc_gate.ArcRow(
            row_id="r1",
            content_id="c1",
            question="Which option is second?",
            choices=("alpha", "beta", "gamma"),
            choice_labels=("A", "B", "C"),
            answer_index=1,
            answer_label="B",
        ),
        arc_gate.ArcRow(
            row_id="r2",
            content_id="c2",
            question="Which option is third?",
            choices=("alpha", "beta", "gamma"),
            choice_labels=("A", "B", "C"),
            answer_index=2,
            answer_label="C",
        ),
        arc_gate.ArcRow(
            row_id="r3",
            content_id="c3",
            question="Which option repeats first?",
            choices=("alpha", "beta", "gamma"),
            choice_labels=("A", "B", "C"),
            answer_index=0,
            answer_label="A",
        ),
    ]


def _source_rows() -> list[np.ndarray]:
    rows = []
    for answer in (0, 1, 2, 0):
        row = np.full((3, 2), -1.0, dtype=np.float64)
        row[answer] = np.asarray([4.0, 2.0], dtype=np.float64)
        rows.append(row)
    return rows


def test_sign_sketch_and_packet_bytes_are_fixed_size() -> None:
    source_rows = _source_rows()
    projection = np.eye(2, dtype=np.float64)

    sketches = gate._sketch_rows(source_rows, projection, quantization="sign")

    assert set(np.unique(sketches[0])) <= {-1.0, 1.0}
    assert gate._packet_bytes(max_candidate_count=5, sketch_dim=16, quantization="sign") == 10
    assert gate._packet_bytes(max_candidate_count=5, sketch_dim=16, quantization="int8") == 82
    assert gate._packet_bytes(max_candidate_count=5, sketch_dim=16, quantization="none") == 320


def test_candidate_alignment_receiver_uses_matched_slots() -> None:
    rows = _rows()
    source_rows = _source_rows()
    public_rows = [np.zeros((3, 2), dtype=np.float64) for _ in rows]
    fit_indices = [0, 1, 2]
    eval_indices = [3]
    receiver = gate._fit_receiver(
        rows,
        source_rows,
        public_rows,
        fit_indices,
        max_candidate_count=3,
        target_only=False,
        label_shuffle=False,
        l2_grid=(0.01, 0.1, 1.0),
    )

    matched = gate._scores_by_row(
        rows,
        receiver,
        source_rows,
        public_rows,
        eval_indices,
        max_candidate_count=3,
    )[3]
    rolled_source = gate._source_control_rows(
        source_rows,
        fit_indices=fit_indices,
        eval_indices=eval_indices,
        control="candidate_roll_source",
        seed=7,
    )
    rolled = gate._scores_by_row(
        rows,
        receiver,
        rolled_source,
        public_rows,
        eval_indices,
        max_candidate_count=3,
    )[3]

    assert gate._prediction(matched) == 0
    assert gate._prediction(rolled) != 0


def test_source_residual_vector_has_no_public_only_signal() -> None:
    public = np.asarray([2.0, -3.0, 5.0], dtype=np.float64)
    zero_source = np.zeros(3, dtype=np.float64)

    residual = gate._source_residual_design_vector(
        source_sketch=zero_source,
        public_sketch=public,
    )

    assert np.allclose(residual, 0.0)


def test_consistency_repair_features_have_no_zero_source_signal() -> None:
    public = np.asarray([[2.0, -3.0], [0.5, 4.0], [-1.0, 1.0]], dtype=np.float64)
    zero_source = np.zeros_like(public)

    features = gate._consistency_repair_feature_rows(
        source_row=zero_source,
        public_row=public,
        target_scores=[0.2, 0.5, -0.1],
    )

    assert np.allclose(features, 0.0)


def test_set_repair_features_are_permutation_equivariant_and_zero_source_clean() -> None:
    source = np.asarray([[1.0, -2.0], [3.0, 0.5], [-1.0, 4.0]], dtype=np.float64)
    public = np.asarray([[0.5, 1.5], [-2.0, 1.0], [1.0, -0.25]], dtype=np.float64)
    target = np.asarray([0.2, -0.4, 0.9], dtype=np.float64)
    perm = np.asarray([2, 0, 1])

    features = gate._set_repair_feature_rows(
        source_row=source,
        public_row=public,
        target_scores=target,
    )
    permuted_features = gate._set_repair_feature_rows(
        source_row=source[perm],
        public_row=public[perm],
        target_scores=target[perm],
    )
    zero_features = gate._set_repair_feature_rows(
        source_row=np.zeros_like(source),
        public_row=public,
        target_scores=target,
    )

    assert np.allclose(permuted_features, features[perm])
    assert np.allclose(zero_features, 0.0)


def test_set_repair_accept_abstain_threshold_blocks_harmful_delta() -> None:
    target = [0.0, 1.0, 0.0]
    weak_delta = [0.5, 0.0, 0.0]
    strong_delta = [2.0, 0.0, 0.0]

    assert gate._apply_repair_acceptance(target, weak_delta, threshold=0.6) == target
    assert gate._prediction(gate._apply_repair_acceptance(target, strong_delta, threshold=0.6)) == 0


def test_set_repair_receiver_freezes_public_base_and_fixes_matched_error() -> None:
    rows = _rows()
    source_rows = _source_rows()
    public_rows = [np.zeros((3, 2), dtype=np.float64) for _ in rows]
    target_scores = {index: [0.0, 1.0, 0.0] for index in range(len(rows))}
    fit_indices = [0, 1, 2]
    eval_indices = [3]
    receiver = gate._fit_set_repair(
        rows,
        source_rows,
        public_rows,
        target_scores,
        fit_indices,
        label_shuffle=False,
        desired_margin=1.0,
        mask_rounds=0,
        mask_keep_prob=1.0,
        control_conditions=("candidate_roll_source",),
        matched_weight=3.0,
        mask_weight=1.0,
        control_weight=0.25,
        l2_grid=(0.01, 0.1, 1.0),
        accept_threshold_grid=(0.0, 0.1),
        seed=7,
    )

    matched = gate._set_repair_scores_by_row(
        rows,
        receiver,
        source_rows,
        public_rows,
        target_scores,
        eval_indices,
    )[3]
    zero_source = gate._source_control_rows(
        source_rows,
        fit_indices=fit_indices,
        eval_indices=eval_indices,
        control="zero_source",
        seed=7,
    )
    zero = gate._set_repair_scores_by_row(
        rows,
        receiver,
        zero_source,
        public_rows,
        target_scores,
        eval_indices,
    )[3]

    assert receiver["weights"][0] == 0.0
    assert gate._prediction(target_scores[3]) == 1
    assert gate._prediction(matched) == 0
    assert np.allclose(zero, target_scores[3])


def test_consistency_repair_freezes_target_public_for_zero_source() -> None:
    rows = _rows()
    source_rows = _source_rows()
    public_rows = [np.zeros((3, 2), dtype=np.float64) for _ in rows]
    target_scores = {index: [0.0, 1.0, 0.0] for index in range(len(rows))}
    fit_indices = [0, 1, 2]
    eval_indices = [3]
    receiver = gate._fit_consistency_repair(
        rows,
        source_rows,
        public_rows,
        target_scores,
        fit_indices,
        label_shuffle=False,
        desired_margin=1.0,
        mask_rounds=1,
        mask_keep_prob=1.0,
        control_conditions=("candidate_roll_source",),
        matched_weight=3.0,
        mask_weight=1.0,
        control_weight=0.5,
        l2_grid=(0.01, 0.1, 1.0),
        seed=7,
    )

    matched_delta = gate._consistency_repair_scores_by_row(
        rows,
        receiver,
        source_rows,
        public_rows,
        target_scores,
        eval_indices,
    )
    matched = gate._add_score_rows(target_scores, matched_delta, eval_indices)[3]
    zero_source = gate._source_control_rows(
        source_rows,
        fit_indices=fit_indices,
        eval_indices=eval_indices,
        control="zero_source",
        seed=7,
    )
    zero_delta = gate._consistency_repair_scores_by_row(
        rows,
        receiver,
        zero_source,
        public_rows,
        target_scores,
        eval_indices,
    )
    zero = gate._add_score_rows(target_scores, zero_delta, eval_indices)[3]

    assert receiver["metadata"]["control_pair_count"] > 0
    assert gate._prediction(target_scores[3]) == 1
    assert gate._prediction(matched) == 0
    assert np.allclose(zero, target_scores[3])


def test_residual_receiver_freezes_public_base_and_fixes_matched_error() -> None:
    rows = _rows()
    source_rows = _source_rows()
    public_rows = [np.zeros((3, 2), dtype=np.float64) for _ in rows]
    target_scores = {index: [0.0, 1.0, 0.0] for index in range(len(rows))}
    fit_indices = [0, 1, 2]
    eval_indices = [3]
    receiver = gate._fit_residual_correction(
        rows,
        source_rows,
        public_rows,
        target_scores,
        fit_indices,
        label_shuffle=False,
        residual_fit_policy="target_errors",
        desired_margin=1.0,
        l2_grid=(0.01, 0.1, 1.0),
    )

    matched_delta = gate._residual_scores_by_row(
        rows,
        receiver,
        source_rows,
        public_rows,
        eval_indices,
    )
    matched = gate._add_score_rows(target_scores, matched_delta, eval_indices)[3]
    zero_source = gate._source_control_rows(
        source_rows,
        fit_indices=fit_indices,
        eval_indices=eval_indices,
        control="zero_source",
        seed=7,
    )
    zero_delta = gate._residual_scores_by_row(
        rows,
        receiver,
        zero_source,
        public_rows,
        eval_indices,
    )
    zero = gate._add_score_rows(target_scores, zero_delta, eval_indices)[3]

    assert receiver["metadata"]["used_fit_rows"] == 2
    assert gate._prediction(target_scores[3]) == 1
    assert gate._prediction(matched) == 0
    assert np.allclose(zero, target_scores[3])


def test_source_control_rows_are_destructive_and_deterministic() -> None:
    source_rows = _source_rows()

    zero = gate._source_control_rows(
        source_rows,
        fit_indices=[0, 1, 2],
        eval_indices=[3],
        control="zero_source",
        seed=11,
    )
    noise_a = gate._source_control_rows(
        source_rows,
        fit_indices=[0, 1, 2],
        eval_indices=[3],
        control="same_norm_noise",
        seed=11,
    )
    noise_b = gate._source_control_rows(
        source_rows,
        fit_indices=[0, 1, 2],
        eval_indices=[3],
        control="same_norm_noise",
        seed=11,
    )

    assert np.allclose(zero[0], 0.0)
    assert np.allclose(noise_a[3], noise_b[3])
    assert np.allclose(np.linalg.norm(noise_a[3]), np.linalg.norm(source_rows[3]), atol=1e-6)
    assert not np.allclose(noise_a[3], source_rows[3])


def test_target_derived_source_control_substitutes_public_sketches() -> None:
    source_rows = _source_rows()
    public_rows = [np.full_like(row, fill_value=float(index + 1)) for index, row in enumerate(source_rows)]

    target_derived = gate._source_variant_for_control(
        source_rows,
        public_rows,
        fit_indices=[0, 1, 2],
        eval_indices=[3],
        control="target_derived_source",
        seed=11,
    )

    assert np.allclose(target_derived[3], public_rows[3])
    assert not np.allclose(target_derived[3], source_rows[3])


def test_condition_metrics_tracks_paired_control_deltas() -> None:
    rows = []
    for condition in gate.REPORT_CONDITIONS:
        rows.append(
            {
                "content_id": "c0",
                "condition": condition,
                "correct": condition == gate.MATCHED_CONDITION,
                "margin": 1.0 if condition == gate.MATCHED_CONDITION else -1.0,
            }
        )

    metrics = gate._condition_metrics(rows, seed=3, bootstrap_samples=20)

    assert metrics[gate.MATCHED_CONDITION]["accuracy"] == 1.0
    assert metrics["target_public_only"]["accuracy"] == 0.0
    assert metrics[gate.MATCHED_CONDITION]["paired_accuracy_vs_target_public_only"]["mean"] == 1.0
    assert metrics[gate.MATCHED_CONDITION]["mean_margin_delta_vs_target_public_only"] == 2.0
