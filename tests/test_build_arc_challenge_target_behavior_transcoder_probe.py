from __future__ import annotations

from scripts import build_arc_challenge_target_behavior_transcoder_probe as probe


def test_parse_args_accepts_target_behavior_transcoder_probe_flags() -> None:
    args = probe.parse_args(
        [
            "--train-disagreement-limit",
            "4",
            "--test-disagreement-limit",
            "4",
            "--packet-rank",
            "12",
            "--packet-top-k",
            "2",
            "--packet-bits",
            "3",
            "--batchtopk-epochs",
            "17",
            "--target-hidden-layer",
            "-2",
        ]
    )

    assert args.train_disagreement_limit == 4
    assert args.test_disagreement_limit == 4
    assert args.packet_rank == 12
    assert args.packet_top_k == 2
    assert args.packet_bits == 3
    assert args.batchtopk_epochs == 17
    assert args.target_hidden_layer == -2


def test_condition_metrics_pairs_target_probe_conditions() -> None:
    rows = [
        {
            "content_id": "a",
            "condition": probe.MATCHED_CONDITION,
            "correct": True,
            "margin": 0.5,
            "packet_fired": True,
            "packet_helped": True,
            "packet_harmed": False,
        },
        {
            "content_id": "a",
            "condition": "target_only",
            "correct": False,
            "margin": -0.1,
            "packet_fired": False,
            "packet_helped": False,
            "packet_harmed": False,
        },
        {
            "content_id": "a",
            "condition": "zero_packet",
            "correct": False,
            "margin": -0.2,
            "packet_fired": False,
            "packet_helped": False,
            "packet_harmed": False,
        },
    ]
    for condition in probe.CONTROL_CONDITIONS:
        if condition in {"target_only", "zero_packet"}:
            continue
        rows.append(
            {
                "content_id": "a",
                "condition": condition,
                "correct": False,
                "margin": -0.3,
                "packet_fired": False,
                "packet_helped": False,
                "packet_harmed": False,
            }
        )

    metrics = probe._condition_metrics(rows, seed=1, bootstrap_samples=10)

    assert metrics[probe.MATCHED_CONDITION]["accuracy"] == 1.0
    assert metrics["target_only"]["accuracy"] == 0.0
    assert metrics[probe.MATCHED_CONDITION]["paired_accuracy_vs_target_only"]["mean"] == 1.0


def test_fixed_weight_sweep_reports_best_heldout_weight() -> None:
    rows = [
        {
            "content_id": "a",
            "condition": "target_only",
            "correct": False,
            "answer_index": 1,
            "scores": [1.0, 0.0],
            "packet_residual": [0.0, 0.0],
        },
        {
            "content_id": "a",
            "condition": probe.MATCHED_CONDITION,
            "correct": False,
            "answer_index": 1,
            "scores": [1.0, 0.0],
            "packet_residual": [-1.0, 1.0],
        },
    ]
    for condition in probe.CONTROL_CONDITIONS:
        if condition == "target_only":
            continue
        rows.append(
            {
                "content_id": "a",
                "condition": condition,
                "correct": False,
                "answer_index": 1,
                "scores": [1.0, 0.0],
                "packet_residual": [0.0, 0.0],
            }
        )

    sweep = probe._fixed_weight_sweep(rows, weights=(0.0, 1.0))

    assert sweep[probe.MATCHED_CONDITION]["best"]["residual_weight"] == 1.0
    assert sweep[probe.MATCHED_CONDITION]["best"]["accuracy"] == 1.0
    assert sweep[probe.MATCHED_CONDITION]["best"]["helped_vs_target"] == 1


def test_target_oracle_scope_metadata_is_not_source_private_claim() -> None:
    metadata = probe._target_oracle_scope_metadata()

    assert metadata["claim_scope"] == "target_hidden_oracle_feasibility_only"
    assert metadata["source_model_used"] is False
    assert metadata["source_private"] is False
    assert metadata["target_hidden_runtime_used"] is True
