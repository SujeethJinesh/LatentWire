from __future__ import annotations

import numpy as np

from scripts import build_source_private_hellaswag_discrete_query_innovation_codec_gate as gate


def test_candidate_token_features_shape_and_packet_flags() -> None:
    scores = np.asarray(
        [
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    packet = np.asarray([3, 0], dtype=np.int64)

    tokens = gate._candidate_token_features(scores, packet)

    assert tokens.shape == (2, 4, 14)
    assert np.all(tokens[np.arange(2), packet, 3] == 1.0)
    assert np.sum(tokens[:, :, 3]) == 2.0


def test_query_summaries_are_deterministic_and_temperature_sensitive() -> None:
    tokens = np.arange(2 * 4 * 5, dtype=np.float64).reshape(2, 4, 5) / 10.0
    queries = gate._fixed_query_parameters(5, query_count=3, seed=11)

    summary_a = gate._query_summaries(tokens, queries, temperature=0.75)
    summary_b = gate._query_summaries(tokens, queries, temperature=0.75)
    summary_c = gate._query_summaries(tokens, queries, temperature=1.5)

    assert summary_a.shape[0] == 2
    assert np.allclose(summary_a, summary_b)
    assert not np.allclose(summary_a, summary_c)


def test_query_residual_encoder_preserves_candidate_low_bits() -> None:
    scores = np.asarray(
        [
            [4.0, 1.0, 0.0, 0.0],
            [0.0, 4.0, 1.0, 0.0],
            [0.0, 0.0, 4.0, 1.0],
            [1.0, 0.0, 0.0, 4.0],
            [3.0, 2.0, 0.0, 0.0],
            [0.0, 3.0, 2.0, 0.0],
            [0.0, 0.0, 3.0, 2.0],
            [2.0, 0.0, 0.0, 3.0],
        ],
        dtype=np.float64,
    )
    packet = np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
    answers = packet.copy()
    qwen_scores = np.zeros_like(scores)
    tokens = gate._candidate_token_features(scores, packet)
    targets = gate._innovation_targets(
        answers=answers,
        qwen_scores=qwen_scores,
        qwen_target=np.zeros(len(answers), dtype=np.int64),
        qwen_hybrid=np.zeros(len(answers), dtype=np.int64),
        prior="uniform",
    )
    encoded = gate._encode_query_residual_codes(
        config={
            "name": "toy",
            "kind": "query_residual_vq",
            "target_prior": "uniform",
            "query_count": 2,
            "query_seed": 5,
            "query_temperature": 1.0,
            "encoder_ridge": 1.0,
            "clusters": 4,
            "cluster_seed": 7,
            "kmeans_iterations": 5,
        },
        train_tokens=tokens,
        eval_tokens=tokens,
        train_packet=packet,
        eval_packet=packet,
        train_targets=targets,
        fit_indices=np.arange(len(answers), dtype=np.int64),
    )

    assert encoded["codebook_size"] == 16
    assert np.all(encoded["train_code"] % 4 == packet)
    assert np.all(encoded["eval_code"] % 4 == packet)
    assert encoded["eval_code"].max() < encoded["codebook_size"]


def test_source_state_floor_ratios_use_framed_packet_denominator() -> None:
    ratios = gate._source_state_floor_ratios(4)

    assert ratios["fp16_one_token_kv_floor_bytes_ratio_vs_framed_packet"] == 3072.0
    assert ratios["qjl_1bit_one_token_kv_floor_bytes_ratio_vs_framed_packet"] == 192.0
