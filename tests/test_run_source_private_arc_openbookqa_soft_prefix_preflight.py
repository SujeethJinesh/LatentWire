from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate
from scripts import run_source_private_arc_openbookqa_soft_prefix_preflight as preflight


def _rows() -> list[arc_gate.ArcRow]:
    return [
        arc_gate.ArcRow(
            row_id="r0",
            content_id="c0",
            question="Which object is magnetic?",
            choices=("wood", "iron", "paper"),
            choice_labels=("A", "B", "C"),
            answer_index=1,
            answer_label="B",
        ),
        arc_gate.ArcRow(
            row_id="r1",
            content_id="c1",
            question="What do plants need?",
            choices=("sunlight", "sand"),
            choice_labels=("A", "B"),
            answer_index=0,
            answer_label="A",
        ),
    ]


def test_prompt_and_source_choice_texts_are_answer_key_forbidden() -> None:
    row = _rows()[0]

    prompt = preflight._mcq_prompt(row)
    source_texts = preflight._source_choice_texts([row])

    assert "Choices:" in prompt
    assert "B. iron" in prompt
    assert "Best answer:" in prompt
    assert len(source_texts) == 3
    assert "Candidate under consideration: B. iron" in source_texts[1]
    assert row.answer_label not in preflight._source_prompt(row).split("Useful evidence:")[-1]


def test_select_rows_with_cache_requires_valid_source_predictions() -> None:
    rows = _rows()
    selected, predictions = preflight._select_rows_with_cache(
        rows,
        {"c0": 1, "c1": 0, "missing": 0},
        row_limit=2,
    )

    assert [row.content_id for row in selected] == ["c0", "c1"]
    assert predictions == [1, 0]


def test_soft_prefix_connector_shapes() -> None:
    source = torch.randn(5)
    target = torch.randn(3)
    matched = preflight.SourceSoftPrefixConnector(
        source_dim=5,
        target_dim=3,
        target_embed_dim=7,
        hidden_dim=11,
        prefix_len=4,
        use_source=True,
        use_target=True,
    )
    slots = preflight.SourceSoftPrefixConnector(
        source_dim=5,
        target_dim=3,
        target_embed_dim=7,
        hidden_dim=11,
        prefix_len=2,
        use_source=False,
        use_target=False,
    )

    assert matched(source, target).shape == (4, 7)
    assert slots(source, target).shape == (2, 7)


def test_query_soft_prefix_connector_shapes() -> None:
    source_tokens = torch.randn(5, 6)
    target = torch.randn(3)
    connector = preflight.SourceQuerySoftPrefixConnector(
        source_dim=6,
        target_dim=3,
        target_embed_dim=7,
        hidden_dim=11,
        prefix_len=4,
        use_target=True,
    )

    assert connector(source_tokens, target).shape == (4, 7)


def test_contrastive_margin_penalty_rewards_matched_over_control() -> None:
    matched = torch.tensor([0.0, 3.0, 1.0])
    weak_control = torch.tensor([0.0, 1.0, 2.0])
    strong_control = torch.tensor([0.0, 2.9, 1.0])

    no_penalty = preflight._contrastive_margin_penalty(
        matched_scores=matched,
        control_scores=weak_control,
        answer_index=1,
        margin=0.2,
        loss_cap=0.5,
    )
    penalty = preflight._contrastive_margin_penalty(
        matched_scores=matched,
        control_scores=strong_control,
        answer_index=1,
        margin=0.2,
        loss_cap=0.5,
    )
    capped = preflight._contrastive_margin_penalty(
        matched_scores=torch.tensor([5.0, 0.0, 4.0]),
        control_scores=torch.tensor([0.0, 5.0, 1.0]),
        answer_index=1,
        margin=1.0,
        loss_cap=0.5,
    )

    assert torch.allclose(no_penalty, torch.zeros(()), atol=1e-6)
    assert penalty.item() > 0.0
    assert torch.allclose(capped, torch.tensor(0.5), atol=1e-6)


def test_fit_source_control_variants_are_destructive() -> None:
    source = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2)

    zero = preflight._fit_source_control_variant(
        source,
        fit_indices=[0, 1, 2],
        fit_position=0,
        row_index=0,
        control="zero_source",
        seed=7,
        epoch=0,
        device="cpu",
    )
    shuffled = preflight._fit_source_control_variant(
        source,
        fit_indices=[0, 1, 2],
        fit_position=0,
        row_index=0,
        control="shuffled_source",
        seed=7,
        epoch=0,
        device="cpu",
    )
    rolled = preflight._fit_source_control_variant(
        source,
        fit_indices=[0, 1, 2],
        fit_position=0,
        row_index=0,
        control="candidate_roll_source",
        seed=7,
        epoch=0,
        device="cpu",
    )
    noise_a = preflight._fit_source_control_variant(
        source,
        fit_indices=[0, 1, 2],
        fit_position=0,
        row_index=0,
        control="same_norm_noise",
        seed=7,
        epoch=0,
        device="cpu",
    )
    noise_b = preflight._fit_source_control_variant(
        source,
        fit_indices=[0, 1, 2],
        fit_position=0,
        row_index=0,
        control="same_norm_noise",
        seed=7,
        epoch=0,
        device="cpu",
    )

    assert torch.allclose(zero, torch.zeros_like(source[0]))
    assert torch.allclose(shuffled, source[1])
    assert torch.allclose(rolled, torch.roll(source[0], shifts=1, dims=0))
    assert preflight._candidate_roll_source_summary(torch.arange(4, dtype=torch.float32)) is None
    assert torch.allclose(noise_a, noise_b)
    assert torch.allclose(noise_a.norm(), source[0].norm(), atol=1e-5)


def test_fixed_token_pool_residualizes_and_resizes() -> None:
    tokens = np.asarray(
        [
            [1.0, 0.0],
            [3.0, 0.0],
            [5.0, 0.0],
        ],
        dtype=np.float64,
    )

    pooled = preflight._fixed_token_pool(tokens, pool_size=5, residualized=True)

    assert pooled.shape == (5, 2)
    assert np.allclose(pooled[1], [0.0, 0.0], atol=1e-6)
    assert np.allclose(pooled[4], [0.0, 0.0], atol=1e-6)
    assert np.allclose(np.linalg.norm(pooled[[0, 2, 3]], axis=1), 1.0, atol=1e-6)


def test_standardize_rank3_uses_all_train_tokens() -> None:
    matrix = torch.tensor(
        [
            [[1.0, 10.0], [3.0, 30.0]],
            [[5.0, 50.0], [7.0, 70.0]],
            [[9.0, 90.0], [11.0, 110.0]],
        ]
    )

    standardized, metadata = preflight._standardize(matrix, [0, 1])

    assert metadata["tensor_rank"] == 3
    assert torch.allclose(standardized[:2].reshape(-1, 2).mean(dim=0), torch.zeros(2), atol=1e-6)


def test_candidate_score_pool_residual_features_use_cached_selection() -> None:
    pooled, metadata = preflight._choice_candidate_pool_features(
        _rows(),
        [1, 0],
        source_feature_mode="cached_choice_score_pool_residual",
        feature_dim=2,
        source_model="",
        source_device="auto_cpu",
        source_dtype="float32",
        source_max_length=32,
        source_hidden_layer=-1,
        source_token_pool_size=3,
        local_files_only=True,
    )

    assert metadata["kind"] == "cached_source_selection_score_candidate_pool"
    assert metadata["uses_source_predictions"] is True
    assert pooled.shape == (2, 3, 1)
    assert np.allclose(pooled[0, :, 0], [-1.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0])
    assert np.allclose(pooled[1, :, 0], [0.5, -0.5, 0.5])


def test_candidate_hidden_score_pool_concatenates_features(monkeypatch) -> None:
    rows = _rows()

    def fake_hidden(*args, **kwargs):
        del args, kwargs
        return np.asarray(
            [
                [1.0, 0.0],
                [3.0, 0.0],
                [5.0, 0.0],
                [2.0, 2.0],
                [2.0, 6.0],
            ],
            dtype=np.float64,
        ), {"kind": "fake_hidden"}

    monkeypatch.setattr(preflight, "_hf_choice_hidden_features", fake_hidden)

    pooled, metadata = preflight._choice_candidate_pool_features(
        rows,
        [1, 0],
        source_feature_mode="hf_choice_hidden_score_candidate_pool_residual",
        feature_dim=2,
        source_model="",
        source_device="auto_cpu",
        source_dtype="float32",
        source_max_length=32,
        source_hidden_layer=-1,
        source_token_pool_size=3,
        local_files_only=True,
    )

    assert metadata["kind"] == "hf_choice_hidden_score_candidate_pool"
    assert metadata["hidden_metadata"]["kind"] == "fake_hidden"
    assert pooled.shape == (2, 3, 3)
    assert np.allclose(np.linalg.norm(pooled[0, [0, 2]], axis=1), 1.0, atol=1e-6)


def test_public_candidate_innovation_uses_fit_indices_only() -> None:
    rows = _rows()
    source = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
        ],
        dtype=np.float64,
    )
    public = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=np.float64,
    )

    innovation, metadata = preflight._public_candidate_innovation_features(
        source,
        public,
        fit_flat_indices=preflight._flat_candidate_indices_for_rows(rows, [0]),
        ridge=1.0,
    )

    assert metadata["fit_candidate_count"] == 3
    assert np.allclose(innovation[:3], 0.0, atol=1e-6)
    assert innovation[3, 0] > 9.0
    assert innovation[4, 0] > 19.0


def test_candidate_hidden_public_innovation_pool_is_prediction_free(monkeypatch) -> None:
    rows = _rows()

    def fake_hidden(*args, **kwargs):
        del args, kwargs
        return np.asarray(
            [
                [1.0, 0.0],
                [3.0, 0.0],
                [5.0, 0.0],
                [2.0, 2.0],
                [2.0, 6.0],
            ],
            dtype=np.float64,
        ), {"kind": "fake_hidden"}

    def fake_public(*args, **kwargs):
        del args, kwargs
        return np.asarray(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
                [100.0, 0.0],
                [200.0, 0.0],
            ],
            dtype=np.float64,
        ), {"kind": "fake_public"}

    def fail_score_rows(*args, **kwargs):
        del args, kwargs
        raise AssertionError("hidden-only public innovation must not read source predictions")

    monkeypatch.setattr(preflight, "_hf_choice_hidden_features", fake_hidden)
    monkeypatch.setattr(preflight, "_public_candidate_hashed_features", fake_public)
    monkeypatch.setattr(preflight, "_source_selection_score_rows", fail_score_rows)

    pooled, metadata = preflight._choice_candidate_pool_features(
        rows,
        [1, 0],
        source_feature_mode="hf_choice_hidden_public_innovation_candidate_pool",
        feature_dim=2,
        source_model="",
        source_device="auto_cpu",
        source_dtype="float32",
        source_max_length=32,
        source_hidden_layer=-1,
        source_token_pool_size=3,
        local_files_only=True,
        fit_indices=[0],
        innovation_ridge=10.0,
    )

    assert metadata["kind"] == "hf_choice_hidden_public_innovation_candidate_pool"
    assert metadata["uses_source_predictions"] is False
    assert metadata["hidden_metadata"]["kind"] == "fake_hidden"
    assert metadata["innovation_metadata"]["public_metadata"]["kind"] == "fake_public"
    assert metadata["innovation_metadata"]["fit_candidate_count"] == 3
    assert pooled.shape == (2, 3, 2)


def test_candidate_hidden_public_innovation_requires_fit_indices(monkeypatch) -> None:
    def fake_hidden(*args, **kwargs):
        del args, kwargs
        return np.zeros((5, 2), dtype=np.float64), {"kind": "fake_hidden"}

    monkeypatch.setattr(preflight, "_hf_choice_hidden_features", fake_hidden)

    with pytest.raises(ValueError, match="requires fit_indices"):
        preflight._choice_candidate_pool_features(
            _rows(),
            [1, 0],
            source_feature_mode="hf_choice_hidden_public_innovation_candidate_pool",
            feature_dim=2,
            source_model="",
            source_device="auto_cpu",
            source_dtype="float32",
            source_max_length=32,
            source_hidden_layer=-1,
            source_token_pool_size=3,
            local_files_only=True,
        )


def test_selected_choice_residual_features_are_row_centered(monkeypatch) -> None:
    rows = _rows()

    def fake_features(*args, **kwargs):
        del args, kwargs
        return torch.tensor(
            [
                [1.0, 0.0],
                [3.0, 0.0],
                [5.0, 0.0],
                [2.0, 2.0],
                [2.0, 6.0],
            ]
        ).numpy()

    monkeypatch.setattr(preflight.arc_gate, "_features", fake_features)

    selected, metadata = preflight._selected_choice_features(
        rows,
        [1, 0],
        source_feature_mode="hashed_selected_residual",
        feature_dim=2,
        source_model="",
        source_device="auto_cpu",
        source_dtype="float32",
        source_max_length=32,
        source_hidden_layer=-1,
        source_token_pool_size=4,
        local_files_only=True,
    )

    assert metadata["row_centered_selected_residual"] is True
    assert selected.shape == (2, 2)
    assert torch.allclose(selected[0], torch.zeros(2), atol=1e-6)
    assert torch.allclose(selected[1], torch.tensor([0.0, -1.0]), atol=1e-6)

    absolute, absolute_metadata = preflight._selected_choice_features(
        rows,
        [1, 0],
        source_feature_mode="hashed_selected",
        feature_dim=2,
        source_model="",
        source_device="auto_cpu",
        source_dtype="float32",
        source_max_length=32,
        source_hidden_layer=-1,
        source_token_pool_size=4,
        local_files_only=True,
    )

    assert absolute_metadata["row_centered_selected_residual"] is False
    assert torch.allclose(absolute[0], torch.tensor([3.0, 0.0]), atol=1e-6)
    assert torch.allclose(absolute[1], torch.tensor([2.0, 2.0]), atol=1e-6)


def test_residual_feature_modes_are_cli_options() -> None:
    hashed = preflight.parse_args(["--source-feature-mode", "hashed_selected_residual"])
    hidden = preflight.parse_args(["--source-feature-mode", "hf_selected_hidden_residual"])
    token_pool = preflight.parse_args(
        [
            "--source-feature-mode",
            "hf_choice_token_hidden_pool_residual",
            "--source-token-pool-size",
            "6",
        ]
    )
    candidate_pool = preflight.parse_args(
        [
            "--source-feature-mode",
            "hf_choice_hidden_score_candidate_pool_residual",
        ]
    )
    public_innovation = preflight.parse_args(
        [
            "--source-feature-mode",
            "hf_choice_hidden_public_innovation_candidate_pool",
            "--innovation-ridge",
            "25",
            "--contrastive-weight",
            "0.5",
            "--contrastive-loss-cap",
            "0.25",
            "--contrastive-controls",
            "zero_source,shuffled_source,candidate_roll_source",
        ]
    )

    assert hashed.source_feature_mode == "hashed_selected_residual"
    assert hidden.source_feature_mode == "hf_selected_hidden_residual"
    assert token_pool.source_feature_mode == "hf_choice_token_hidden_pool_residual"
    assert token_pool.source_token_pool_size == 6
    assert candidate_pool.source_feature_mode == "hf_choice_hidden_score_candidate_pool_residual"
    assert public_innovation.source_feature_mode == "hf_choice_hidden_public_innovation_candidate_pool"
    assert public_innovation.innovation_ridge == 25.0
    assert public_innovation.contrastive_weight == 0.5
    assert public_innovation.contrastive_loss_cap == 0.25
    assert preflight._parse_contrastive_controls(public_innovation.contrastive_controls) == (
        "zero_source",
        "shuffled_source",
        "candidate_roll_source",
    )


def test_unknown_contrastive_controls_raise() -> None:
    with pytest.raises(ValueError, match="unknown contrastive controls"):
        preflight._parse_contrastive_controls("zero_source,bad_control")


def test_condition_metrics_adds_paired_controls() -> None:
    rows = [
        {
            "content_id": "c0",
            "condition": preflight.MATCHED_CONDITION,
            "correct": True,
            "margin": 0.8,
        },
        {"content_id": "c0", "condition": "target_only", "correct": False, "margin": -0.1},
        {
            "content_id": "c1",
            "condition": preflight.MATCHED_CONDITION,
            "correct": False,
            "margin": -0.2,
        },
        {"content_id": "c1", "condition": "target_only", "correct": False, "margin": -0.4},
    ]
    for condition in preflight.REPORT_CONDITIONS:
        if condition in {preflight.MATCHED_CONDITION, "target_only"}:
            continue
        rows.extend(
            [
                {"content_id": "c0", "condition": condition, "correct": False, "margin": 0.0},
                {"content_id": "c1", "condition": condition, "correct": False, "margin": 0.0},
            ]
        )

    metrics = preflight._condition_metrics(rows, seed=3, bootstrap_samples=20)

    assert metrics[preflight.MATCHED_CONDITION]["accuracy"] == 0.5
    assert metrics["target_only"]["accuracy"] == 0.0
    paired = metrics[preflight.MATCHED_CONDITION]["paired_accuracy_vs_target_only"]
    assert paired["mean"] == 0.5
    assert metrics[preflight.MATCHED_CONDITION]["mean_margin_delta_vs_target_only"] > 0.0
