from __future__ import annotations

"""Run a source-private candidate-alignment receiver preflight on ARC.

This gate is intentionally smaller than the soft-prefix LM receiver.  It asks
whether answer-key-forbidden source candidate slots can help an external
candidate ranker after public target-side candidate information and destructive
source controls are held fixed.  The default source packet is a sign sketch of
each candidate slot, so the communicated object has an explicit byte budget.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import pathlib
import random
import resource
import statistics
import sys
import time
from typing import Any, Sequence

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402
from scripts import run_source_private_arc_openbookqa_soft_prefix_preflight as soft_prefix  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_arc_candidate_alignment_receiver_preflight_20260502_arc_hidden_public_innovation_n8"
)
DEFAULT_ARC_VALIDATION = soft_prefix.DEFAULT_ARC_VALIDATION
DEFAULT_ARC_SOURCE_CACHE = soft_prefix.DEFAULT_ARC_SOURCE_CACHE
DEFAULT_QWEN_SOURCE = soft_prefix.DEFAULT_QWEN_SOURCE

MATCHED_CONDITION = "matched_candidate_alignment_receiver"
CONTROL_CONDITIONS = (
    "target_public_only",
    "zero_source",
    "shuffled_source",
    "same_norm_noise",
    "train_mean_source",
    "target_derived_source",
    "label_shuffled",
    "candidate_roll_source",
    "candidate_derangement",
    "same_byte_visible_text",
    "source_label_copy_audit_upper_bound",
)
PASS_CONTROL_CONDITIONS = tuple(
    condition for condition in CONTROL_CONDITIONS if condition != "source_label_copy_audit_upper_bound"
)
REPORT_CONDITIONS = (MATCHED_CONDITION, *CONTROL_CONDITIONS)
DEFAULT_L2_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
RECEIVER_MODES = (
    "direct",
    "target_residual",
    "target_consistency_repair",
    "target_set_repair",
    "equivariant_set_delta_accept",
)
RESIDUAL_FIT_POLICIES = ("target_errors", "all")
CONSISTENCY_REPAIR_CONTROL_CONDITIONS = (
    "shuffled_source",
    "same_norm_noise",
    "train_mean_source",
    "target_derived_source",
    "candidate_roll_source",
)
SET_REPAIR_CONTROL_CONDITIONS = CONSISTENCY_REPAIR_CONTROL_CONDITIONS
DEFAULT_ACCEPT_THRESHOLD_GRID = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display(path: pathlib.Path | str) -> str:
    resolved = _resolve(path)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _sha256_file(path: pathlib.Path | str) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _peak_rss_mib() -> float:
    usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return usage / divisor


def _paired_bootstrap(deltas: list[float], *, seed: int, samples: int) -> dict[str, float]:
    if not deltas:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    rng = random.Random(seed)
    n = len(deltas)
    means = [statistics.fmean(deltas[rng.randrange(n)] for _ in range(n)) for _ in range(samples)]
    return {
        "mean": float(statistics.fmean(deltas)),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
    }


def _prediction(scores: Sequence[float]) -> int:
    return int(max(range(len(scores)), key=lambda index: (float(scores[index]), -index)))


def _margin(scores: Sequence[float], answer_index: int) -> float:
    gold = float(scores[int(answer_index)])
    distractors = [float(score) for index, score in enumerate(scores) if index != int(answer_index)]
    return gold - max(distractors) if distractors else gold


def _public_candidate_rows(
    rows: Sequence[arc_gate.ArcRow],
    flat_features: np.ndarray,
) -> list[np.ndarray]:
    public_rows: list[np.ndarray] = []
    offset = 0
    for row in rows:
        end = offset + len(row.choices)
        public_rows.append(np.asarray(flat_features[offset:end], dtype=np.float64))
        offset = end
    if offset != int(flat_features.shape[0]):
        raise ValueError("public candidate feature count does not match row choices")
    return public_rows


def _standardize_candidate_rows(
    candidate_rows: Sequence[np.ndarray],
    fit_indices: Sequence[int],
) -> tuple[list[np.ndarray], dict[str, Any]]:
    fit = np.vstack([candidate_rows[int(index)] for index in fit_indices])
    mean = fit.mean(axis=0, keepdims=True)
    std = fit.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    standardized = [(np.asarray(row, dtype=np.float64) - mean) / std for row in candidate_rows]
    return standardized, {
        "mean_l2": float(np.linalg.norm(mean)),
        "std_min": float(std.min()),
        "std_max": float(std.max()),
        "fit_candidate_count": int(fit.shape[0]),
        "feature_dim": int(fit.shape[1]),
    }


def _signed_projection(input_dim: int, sketch_dim: int, *, seed: int) -> np.ndarray:
    if input_dim < 1 or sketch_dim < 1:
        raise ValueError("projection dimensions must be positive")
    rng = np.random.default_rng(int(seed))
    signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float64), size=(input_dim, sketch_dim))
    return signs / math.sqrt(float(input_dim))


def _sketch_rows(
    candidate_rows: Sequence[np.ndarray],
    projection: np.ndarray,
    *,
    quantization: str,
) -> list[np.ndarray]:
    sketches = [np.asarray(row, dtype=np.float64) @ projection for row in candidate_rows]
    if quantization == "none":
        return sketches
    if quantization == "int8":
        quantized_rows: list[np.ndarray] = []
        for row in sketches:
            scale = max(float(np.max(np.abs(row))) / 127.0, 1e-12)
            codes = np.clip(np.round(row / scale), -127.0, 127.0)
            quantized_rows.append((codes * scale).astype(np.float64))
        return quantized_rows
    if quantization == "sign":
        return [np.where(row >= 0.0, 1.0, -1.0).astype(np.float64) for row in sketches]
    raise ValueError(f"unknown source sketch quantization {quantization!r}")


def _packet_bytes(*, max_candidate_count: int, sketch_dim: int, quantization: str) -> int:
    if quantization == "none":
        return int(max_candidate_count) * int(sketch_dim) * 4
    if quantization == "int8":
        return int(max_candidate_count) * int(sketch_dim) + 2
    if quantization == "sign":
        return int(math.ceil(int(max_candidate_count) * int(sketch_dim) / 8.0))
    raise ValueError(f"unknown source sketch quantization {quantization!r}")


def _candidate_design_vector(
    *,
    candidate_index: int,
    choice_count: int,
    max_candidate_count: int,
    source_sketch: np.ndarray,
    public_sketch: np.ndarray,
    target_only: bool,
) -> np.ndarray:
    denom = max(choice_count - 1, 1)
    candidate_fraction = float(candidate_index) / float(denom)
    choice_fraction = float(choice_count) / float(max_candidate_count)
    public = np.asarray(public_sketch, dtype=np.float64)
    if target_only:
        return np.concatenate(
            [
                np.asarray(
                    [
                        candidate_fraction,
                        choice_fraction,
                        float(np.linalg.norm(public)),
                    ],
                    dtype=np.float64,
                ),
                public,
            ]
        )

    source = np.asarray(source_sketch, dtype=np.float64)
    compatibility = source * public
    dot = float(source @ public / math.sqrt(float(source.shape[0])))
    return np.concatenate(
        [
            np.asarray(
                [
                    candidate_fraction,
                    choice_fraction,
                    dot,
                    float(np.linalg.norm(source)),
                    float(np.linalg.norm(public)),
                ],
                dtype=np.float64,
            ),
            source,
            public,
            compatibility,
        ]
    )


def _source_residual_design_vector(
    *,
    source_sketch: np.ndarray,
    public_sketch: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source_sketch, dtype=np.float64)
    public = np.asarray(public_sketch, dtype=np.float64)
    compatibility = source * public
    dot = float(source @ public / math.sqrt(float(source.shape[0])))
    return np.concatenate(
        [
            np.asarray([dot, float(np.linalg.norm(source))], dtype=np.float64),
            source,
            compatibility,
        ]
    )


def _max_other(values: np.ndarray, index: int) -> float:
    if values.shape[0] <= 1:
        return 0.0
    return float(np.max(np.delete(values, int(index))))


def _consistency_repair_feature_rows(
    *,
    source_row: np.ndarray,
    public_row: np.ndarray,
    target_scores: Sequence[float],
) -> np.ndarray:
    source = np.asarray(source_row, dtype=np.float64)
    public = np.asarray(public_row, dtype=np.float64)
    target = np.asarray(target_scores, dtype=np.float64)
    if public.shape[0] != target.shape[0]:
        raise ValueError("target score count must match candidate count")
    if source.shape[1] != public.shape[1]:
        raise ValueError("source/public sketch dimensions must match for consistency repair")
    if source.shape[0] < target.shape[0]:
        raise ValueError("source candidate count must cover target candidates")
    source = source[: target.shape[0]]
    public = public[: target.shape[0]]

    sqrt_dim = math.sqrt(float(source.shape[1]))
    dots = np.sum(source * public, axis=1) / sqrt_dim
    norms = np.linalg.norm(source, axis=1)
    sign_agreements = np.mean(np.sign(source) * np.sign(public), axis=1)
    target_mean = float(np.mean(target))
    target_top = float(np.max(target))
    dot_mean = float(np.mean(dots))
    norm_mean = float(np.mean(norms))
    agreement_mean = float(np.mean(sign_agreements))
    dot_spread = float(np.std(dots))
    norm_spread = float(np.std(norms))
    agreement_spread = float(np.std(sign_agreements))

    rows: list[np.ndarray] = []
    for candidate_index in range(source.shape[0]):
        src = source[candidate_index]
        pub = public[candidate_index]
        compatibility = src * pub
        target_centered = float(target[candidate_index] - target_mean)
        target_gap = float(target[candidate_index] - target_top)
        dot = float(dots[candidate_index])
        norm = float(norms[candidate_index])
        agreement = float(sign_agreements[candidate_index])
        dot_gap = dot - _max_other(dots, candidate_index)
        norm_gap = norm - _max_other(norms, candidate_index)
        agreement_gap = agreement - _max_other(sign_agreements, candidate_index)
        rows.append(
            np.concatenate(
                [
                    np.asarray(
                        [
                            dot,
                            norm,
                            norm * norm,
                            agreement,
                            dot - dot_mean,
                            norm - norm_mean,
                            agreement - agreement_mean,
                            dot_gap,
                            norm_gap,
                            agreement_gap,
                            float(dot_spread),
                            float(norm_spread),
                            float(agreement_spread),
                            dot * target_centered,
                            dot * target_gap,
                            norm * target_centered,
                            norm * target_gap,
                            agreement * target_centered,
                            agreement * target_gap,
                        ],
                        dtype=np.float64,
                    ),
                    src,
                    compatibility,
                    np.sign(src) * np.sign(pub),
                    src * target_centered,
                    compatibility * target_centered,
                    src * target_gap,
                ]
            )
        )
    return np.vstack(rows)


def _set_repair_feature_rows(
    *,
    source_row: np.ndarray,
    public_row: np.ndarray,
    target_scores: Sequence[float],
) -> np.ndarray:
    source = np.asarray(source_row, dtype=np.float64)
    public = np.asarray(public_row, dtype=np.float64)
    target = np.asarray(target_scores, dtype=np.float64)
    if public.shape[0] != target.shape[0]:
        raise ValueError("target score count must match candidate count")
    if source.shape[1] != public.shape[1]:
        raise ValueError("source/public sketch dimensions must match for set repair")
    if source.shape[0] < target.shape[0]:
        raise ValueError("source candidate count must cover target candidates")
    source = source[: target.shape[0]]
    public = public[: target.shape[0]]

    sqrt_dim = math.sqrt(float(source.shape[1]))
    compatibility = source * public
    signed_compatibility = np.sign(source) * np.sign(public)
    dots = np.sum(compatibility, axis=1) / sqrt_dim
    norms = np.linalg.norm(source, axis=1)
    agreements = np.mean(signed_compatibility, axis=1)
    target_mean = float(np.mean(target))
    target_top = float(np.max(target))
    target_centered = target - target_mean
    target_gap = target - target_top
    source_mean = np.mean(source, axis=0)
    compatibility_mean = np.mean(compatibility, axis=0)
    signed_compatibility_mean = np.mean(signed_compatibility, axis=0)
    dot_mean = float(np.mean(dots))
    norm_mean = float(np.mean(norms))
    agreement_mean = float(np.mean(agreements))
    dot_max = float(np.max(dots))
    norm_max = float(np.max(norms))
    agreement_max = float(np.max(agreements))

    rows: list[np.ndarray] = []
    for candidate_index in range(source.shape[0]):
        src = source[candidate_index]
        comp = compatibility[candidate_index]
        signed = signed_compatibility[candidate_index]
        centered = float(target_centered[candidate_index])
        gap = float(target_gap[candidate_index])
        dot = float(dots[candidate_index])
        norm = float(norms[candidate_index])
        agreement = float(agreements[candidate_index])
        local = np.asarray(
            [
                dot,
                norm,
                norm * norm,
                agreement,
                dot - dot_mean,
                norm - norm_mean,
                agreement - agreement_mean,
                dot - dot_max,
                norm - norm_max,
                agreement - agreement_max,
                dot - _max_other(dots, candidate_index),
                norm - _max_other(norms, candidate_index),
                agreement - _max_other(agreements, candidate_index),
                dot * centered,
                dot * gap,
                norm * centered,
                norm * gap,
                agreement * centered,
                agreement * gap,
            ],
            dtype=np.float64,
        )
        rows.append(
            np.concatenate(
                [
                    local,
                    src,
                    comp,
                    signed,
                    src - source_mean,
                    comp - compatibility_mean,
                    signed - signed_compatibility_mean,
                    src * centered,
                    comp * centered,
                    src * gap,
                    comp * gap,
                ]
            )
        )
    return np.vstack(rows)


def _label_for_row(
    rows: Sequence[arc_gate.ArcRow],
    *,
    row_index: int,
    row_position: int,
    row_indices: Sequence[int],
    label_shuffle: bool,
) -> int:
    if not label_shuffle:
        return int(rows[int(row_index)].answer_index)
    source_position = (int(row_position) + 1) % len(row_indices)
    shifted_answer = int(rows[int(row_indices[source_position])].answer_index)
    return shifted_answer % len(rows[int(row_index)].choices)


def _build_design(
    rows: Sequence[arc_gate.ArcRow],
    source_sketch_rows: Sequence[np.ndarray],
    public_sketch_rows: Sequence[np.ndarray],
    row_indices: Sequence[int],
    *,
    max_candidate_count: int,
    target_only: bool,
    label_shuffle: bool,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    features: list[np.ndarray] = []
    labels: list[float] = []
    refs: list[tuple[int, int]] = []
    for position, row_index in enumerate(row_indices):
        row = rows[int(row_index)]
        answer = _label_for_row(
            rows,
            row_index=int(row_index),
            row_position=position,
            row_indices=row_indices,
            label_shuffle=label_shuffle,
        )
        source_row = source_sketch_rows[int(row_index)]
        public_row = public_sketch_rows[int(row_index)]
        for candidate_index in range(len(row.choices)):
            features.append(
                _candidate_design_vector(
                    candidate_index=candidate_index,
                    choice_count=len(row.choices),
                    max_candidate_count=max_candidate_count,
                    source_sketch=source_row[candidate_index],
                    public_sketch=public_row[candidate_index],
                    target_only=target_only,
                )
            )
            labels.append(1.0 if candidate_index == answer else -1.0)
            refs.append((int(row_index), int(candidate_index)))
    if not features:
        raise ValueError("no candidate rows were built")
    return np.vstack(features), np.asarray(labels, dtype=np.float64), refs


def _build_pairwise_design(
    rows: Sequence[arc_gate.ArcRow],
    source_sketch_rows: Sequence[np.ndarray],
    public_sketch_rows: Sequence[np.ndarray],
    row_indices: Sequence[int],
    *,
    max_candidate_count: int,
    target_only: bool,
    label_shuffle: bool,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[float] = []
    for position, row_index in enumerate(row_indices):
        row = rows[int(row_index)]
        answer = _label_for_row(
            rows,
            row_index=int(row_index),
            row_position=position,
            row_indices=row_indices,
            label_shuffle=label_shuffle,
        )
        candidate_features = [
            _candidate_design_vector(
                candidate_index=candidate_index,
                choice_count=len(row.choices),
                max_candidate_count=max_candidate_count,
                source_sketch=source_sketch_rows[int(row_index)][candidate_index],
                public_sketch=public_sketch_rows[int(row_index)][candidate_index],
                target_only=target_only,
            )
            for candidate_index in range(len(row.choices))
        ]
        for candidate_index, feature in enumerate(candidate_features):
            if candidate_index == answer:
                continue
            diff = candidate_features[answer] - feature
            features.append(diff)
            labels.append(1.0)
            features.append(-diff)
            labels.append(-1.0)
    if not features:
        raise ValueError("no pairwise candidate rows were built")
    return np.vstack(features), np.asarray(labels, dtype=np.float64)


def _build_residual_pairwise_design(
    rows: Sequence[arc_gate.ArcRow],
    source_sketch_rows: Sequence[np.ndarray],
    public_sketch_rows: Sequence[np.ndarray],
    target_score_rows: dict[int, list[float]],
    row_indices: Sequence[int],
    *,
    label_shuffle: bool,
    residual_fit_policy: str,
    desired_margin: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if residual_fit_policy not in RESIDUAL_FIT_POLICIES:
        raise ValueError(f"unknown residual fit policy {residual_fit_policy!r}")
    features: list[np.ndarray] = []
    labels: list[float] = []
    used_rows = 0
    skipped_target_correct = 0
    for position, row_index in enumerate(row_indices):
        row = rows[int(row_index)]
        target_scores = [float(score) for score in target_score_rows[int(row_index)]]
        target_prediction = _prediction(target_scores)
        if residual_fit_policy == "target_errors" and target_prediction == int(row.answer_index):
            skipped_target_correct += 1
            continue
        answer = _label_for_row(
            rows,
            row_index=int(row_index),
            row_position=position,
            row_indices=row_indices,
            label_shuffle=label_shuffle,
        )
        candidate_features = [
            _source_residual_design_vector(
                source_sketch=source_sketch_rows[int(row_index)][candidate_index],
                public_sketch=public_sketch_rows[int(row_index)][candidate_index],
            )
            for candidate_index in range(len(row.choices))
        ]
        used_rows += 1
        for candidate_index, feature in enumerate(candidate_features):
            if candidate_index == answer:
                continue
            target_diff = float(target_scores[answer]) - float(target_scores[candidate_index])
            desired_residual_diff = float(desired_margin) - target_diff
            diff = candidate_features[answer] - feature
            features.append(diff)
            labels.append(desired_residual_diff)
            features.append(-diff)
            labels.append(-desired_residual_diff)
    if not features and residual_fit_policy == "target_errors":
        return _build_residual_pairwise_design(
            rows,
            source_sketch_rows,
            public_sketch_rows,
            target_score_rows,
            row_indices,
            label_shuffle=label_shuffle,
            residual_fit_policy="all",
            desired_margin=desired_margin,
        )
    if not features:
        raise ValueError("no residual pairwise rows were built")
    return (
        np.vstack(features),
        np.asarray(labels, dtype=np.float64),
        {
            "used_fit_rows": int(used_rows),
            "skipped_target_correct_rows": int(skipped_target_correct),
            "residual_fit_policy": residual_fit_policy,
            "effective_residual_fit_policy": residual_fit_policy,
            "desired_margin": float(desired_margin),
            "pair_count": int(len(features)),
        },
    )


def _build_consistency_repair_pairwise_design(
    rows: Sequence[arc_gate.ArcRow],
    source_sketch_rows: Sequence[np.ndarray],
    public_sketch_rows: Sequence[np.ndarray],
    target_score_rows: dict[int, list[float]],
    row_indices: Sequence[int],
    *,
    label_shuffle: bool,
    desired_margin: float,
    mask_rounds: int,
    mask_keep_prob: float,
    control_conditions: Sequence[str],
    matched_weight: float,
    mask_weight: float,
    control_weight: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    features: list[np.ndarray] = []
    labels: list[float] = []
    weights: list[float] = []
    matched_pair_count = 0
    masked_pair_count = 0
    control_pair_count = 0
    rng = np.random.default_rng(int(seed))
    source_controls: dict[str, list[np.ndarray]] = {
        control: _source_variant_for_control(
            source_sketch_rows,
            public_sketch_rows,
            fit_indices=row_indices,
            eval_indices=row_indices,
            control=control,
            seed=seed + 17 * (position + 1),
        )
        for position, control in enumerate(control_conditions)
    }

    def add_view(
        *,
        row_index: int,
        source_row: np.ndarray,
        answer: int,
        residual_target: str,
        sample_weight: float,
    ) -> int:
        row = rows[int(row_index)]
        target_scores = [float(score) for score in target_score_rows[int(row_index)]]
        candidate_features = _consistency_repair_feature_rows(
            source_row=source_row,
            public_row=public_sketch_rows[int(row_index)],
            target_scores=target_scores,
        )
        added = 0
        for candidate_index, feature in enumerate(candidate_features):
            if candidate_index == answer:
                continue
            diff = candidate_features[answer] - feature
            if residual_target == "gold_margin":
                target_diff = float(target_scores[answer]) - float(target_scores[candidate_index])
                label = float(desired_margin) - target_diff
            elif residual_target == "preserve_target":
                label = 0.0
            else:
                raise ValueError(f"unknown residual_target {residual_target!r}")
            features.append(diff)
            labels.append(label)
            weights.append(float(sample_weight))
            features.append(-diff)
            labels.append(-label)
            weights.append(float(sample_weight))
            added += 2
        if added == 0 and len(row.choices) > 1:
            raise ValueError("consistency repair failed to add pairwise rows")
        return added

    for position, row_index in enumerate(row_indices):
        answer = _label_for_row(
            rows,
            row_index=int(row_index),
            row_position=position,
            row_indices=row_indices,
            label_shuffle=label_shuffle,
        )
        matched_pair_count += add_view(
            row_index=int(row_index),
            source_row=source_sketch_rows[int(row_index)],
            answer=answer,
            residual_target="gold_margin",
            sample_weight=matched_weight,
        )
        for mask_index in range(int(mask_rounds)):
            mask = (rng.random(source_sketch_rows[int(row_index)].shape[1]) < float(mask_keep_prob)).astype(
                np.float64
            )
            if float(np.sum(mask)) <= 0.0:
                mask[int(rng.integers(0, mask.shape[0]))] = 1.0
            masked_pair_count += add_view(
                row_index=int(row_index),
                source_row=source_sketch_rows[int(row_index)] * mask[None, :],
                answer=answer,
                residual_target="gold_margin",
                sample_weight=mask_weight,
            )
        for control in control_conditions:
            control_pair_count += add_view(
                row_index=int(row_index),
                source_row=source_controls[control][int(row_index)],
                answer=answer,
                residual_target="preserve_target",
                sample_weight=control_weight,
            )
    if not features:
        raise ValueError("no consistency repair pairwise rows were built")
    return (
        np.vstack(features),
        np.asarray(labels, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
        {
            "used_fit_rows": int(len(row_indices)),
            "matched_pair_count": int(matched_pair_count),
            "masked_pair_count": int(masked_pair_count),
            "control_pair_count": int(control_pair_count),
            "control_conditions": list(control_conditions),
            "desired_margin": float(desired_margin),
            "mask_rounds": int(mask_rounds),
            "mask_keep_prob": float(mask_keep_prob),
            "matched_weight": float(matched_weight),
            "mask_weight": float(mask_weight),
            "control_weight": float(control_weight),
        },
    )


def _build_set_repair_pairwise_design(
    rows: Sequence[arc_gate.ArcRow],
    source_sketch_rows: Sequence[np.ndarray],
    public_sketch_rows: Sequence[np.ndarray],
    target_score_rows: dict[int, list[float]],
    row_indices: Sequence[int],
    *,
    label_shuffle: bool,
    desired_margin: float,
    mask_rounds: int,
    mask_keep_prob: float,
    control_conditions: Sequence[str],
    matched_weight: float,
    mask_weight: float,
    control_weight: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    features: list[np.ndarray] = []
    labels: list[float] = []
    weights: list[float] = []
    matched_pair_count = 0
    masked_pair_count = 0
    control_pair_count = 0
    rng = np.random.default_rng(int(seed))
    source_controls: dict[str, list[np.ndarray]] = {
        control: _source_variant_for_control(
            source_sketch_rows,
            public_sketch_rows,
            fit_indices=row_indices,
            eval_indices=row_indices,
            control=control,
            seed=seed + 23 * (position + 1),
        )
        for position, control in enumerate(control_conditions)
    }

    def add_view(
        *,
        row_index: int,
        source_row: np.ndarray,
        answer: int,
        residual_target: str,
        sample_weight: float,
    ) -> int:
        row = rows[int(row_index)]
        target_scores = [float(score) for score in target_score_rows[int(row_index)]]
        candidate_features = _set_repair_feature_rows(
            source_row=source_row,
            public_row=public_sketch_rows[int(row_index)],
            target_scores=target_scores,
        )
        added = 0
        for candidate_index, feature in enumerate(candidate_features):
            if candidate_index == answer:
                continue
            diff = candidate_features[answer] - feature
            if residual_target == "gold_margin":
                target_diff = float(target_scores[answer]) - float(target_scores[candidate_index])
                label = float(desired_margin) - target_diff
            elif residual_target == "preserve_target":
                label = 0.0
            else:
                raise ValueError(f"unknown residual_target {residual_target!r}")
            features.append(diff)
            labels.append(label)
            weights.append(float(sample_weight))
            features.append(-diff)
            labels.append(-label)
            weights.append(float(sample_weight))
            added += 2
        if added == 0 and len(row.choices) > 1:
            raise ValueError("set repair failed to add pairwise rows")
        return added

    for position, row_index in enumerate(row_indices):
        answer = _label_for_row(
            rows,
            row_index=int(row_index),
            row_position=position,
            row_indices=row_indices,
            label_shuffle=label_shuffle,
        )
        matched_pair_count += add_view(
            row_index=int(row_index),
            source_row=source_sketch_rows[int(row_index)],
            answer=answer,
            residual_target="gold_margin",
            sample_weight=matched_weight,
        )
        for _ in range(int(mask_rounds)):
            mask = (rng.random(source_sketch_rows[int(row_index)].shape[1]) < float(mask_keep_prob)).astype(
                np.float64
            )
            if float(np.sum(mask)) <= 0.0:
                mask[int(rng.integers(0, mask.shape[0]))] = 1.0
            masked_pair_count += add_view(
                row_index=int(row_index),
                source_row=source_sketch_rows[int(row_index)] * mask[None, :],
                answer=answer,
                residual_target="gold_margin",
                sample_weight=mask_weight,
            )
        for control in control_conditions:
            control_pair_count += add_view(
                row_index=int(row_index),
                source_row=source_controls[control][int(row_index)],
                answer=answer,
                residual_target="preserve_target",
                sample_weight=control_weight,
            )
    if not features:
        raise ValueError("no set repair pairwise rows were built")
    return (
        np.vstack(features),
        np.asarray(labels, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
        {
            "used_fit_rows": int(len(row_indices)),
            "matched_pair_count": int(matched_pair_count),
            "masked_pair_count": int(masked_pair_count),
            "control_pair_count": int(control_pair_count),
            "control_conditions": list(control_conditions),
            "desired_margin": float(desired_margin),
            "mask_rounds": int(mask_rounds),
            "mask_keep_prob": float(mask_keep_prob),
            "matched_weight": float(matched_weight),
            "mask_weight": float(mask_weight),
            "control_weight": float(control_weight),
        },
    )


def _fit_ridge(features: np.ndarray, labels: np.ndarray, *, l2: float) -> np.ndarray:
    x = np.concatenate([np.ones((features.shape[0], 1), dtype=np.float64), features], axis=1)
    penalty = float(l2) * np.eye(x.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ labels)


def _fit_ridge_no_intercept(features: np.ndarray, labels: np.ndarray, *, l2: float) -> np.ndarray:
    penalty = float(l2) * np.eye(features.shape[1], dtype=np.float64)
    weights = np.linalg.solve(features.T @ features + penalty, features.T @ labels)
    return np.concatenate([np.zeros(1, dtype=np.float64), weights])


def _fit_weighted_ridge_no_intercept(
    features: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    *,
    l2: float,
) -> np.ndarray:
    weights = np.sqrt(np.maximum(np.asarray(sample_weights, dtype=np.float64), 0.0))
    xw = np.asarray(features, dtype=np.float64) * weights[:, None]
    yw = np.asarray(labels, dtype=np.float64) * weights
    penalty = float(l2) * np.eye(features.shape[1], dtype=np.float64)
    solved = np.linalg.solve(xw.T @ xw + penalty, xw.T @ yw)
    return np.concatenate([np.zeros(1, dtype=np.float64), solved])


def _score_design(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x = np.concatenate([np.ones((features.shape[0], 1), dtype=np.float64), features], axis=1)
    return x @ weights


def _fit_residual_correction(
    rows: Sequence[arc_gate.ArcRow],
    source_sketch_rows: Sequence[np.ndarray],
    public_sketch_rows: Sequence[np.ndarray],
    target_score_rows: dict[int, list[float]],
    fit_indices: Sequence[int],
    *,
    label_shuffle: bool,
    residual_fit_policy: str,
    desired_margin: float,
    l2_grid: Sequence[float],
) -> dict[str, Any]:
    features, labels, metadata = _build_residual_pairwise_design(
        rows,
        source_sketch_rows,
        public_sketch_rows,
        target_score_rows,
        fit_indices,
        label_shuffle=label_shuffle,
        residual_fit_policy=residual_fit_policy,
        desired_margin=desired_margin,
    )
    cv_rows: list[dict[str, float]] = []
    selected_l2 = float(l2_grid[len(l2_grid) // 2])
    fit_indices = list(fit_indices)
    if len(fit_indices) >= 2:
        for l2 in l2_grid:
            fold_accs: list[float] = []
            for heldout in fit_indices:
                train_indices = [index for index in fit_indices if index != heldout]
                train_x, train_y, _ = _build_residual_pairwise_design(
                    rows,
                    source_sketch_rows,
                    public_sketch_rows,
                    target_score_rows,
                    train_indices,
                    label_shuffle=label_shuffle,
                    residual_fit_policy=residual_fit_policy,
                    desired_margin=desired_margin,
                )
                weights = _fit_ridge_no_intercept(train_x, train_y, l2=float(l2))
                residual_scores = _residual_scores_by_row(
                    rows,
                    {
                        "weights": weights,
                        "target_only": False,
                        "label_shuffle": label_shuffle,
                    },
                    source_sketch_rows,
                    public_sketch_rows,
                    [heldout],
                )
                final_scores = {
                    int(heldout): [
                        float(base) + float(delta)
                        for base, delta in zip(
                            target_score_rows[int(heldout)],
                            residual_scores[int(heldout)],
                            strict=True,
                        )
                    ]
                }
                fold_accs.append(
                    1.0
                    if _prediction(final_scores[int(heldout)]) == int(rows[int(heldout)].answer_index)
                    else 0.0
                )
            cv_rows.append({"l2": float(l2), "row_grouped_cv_accuracy": float(statistics.fmean(fold_accs))})
        selected = max(cv_rows, key=lambda row: (row["row_grouped_cv_accuracy"], row["l2"]))
        selected_l2 = float(selected["l2"])
    else:
        cv_rows.append({"l2": selected_l2, "row_grouped_cv_accuracy": 0.0})
    weights = _fit_ridge_no_intercept(features, labels, l2=selected_l2)
    return {
        "weights": weights,
        "selected_l2": selected_l2,
        "cv_rows": cv_rows,
        "feature_dim": int(features.shape[1]),
        "target_only": False,
        "label_shuffle": bool(label_shuffle),
        "training_objective": "residual_pairwise_ridge",
        "metadata": metadata,
    }


def _fit_consistency_repair(
    rows: Sequence[arc_gate.ArcRow],
    source_sketch_rows: Sequence[np.ndarray],
    public_sketch_rows: Sequence[np.ndarray],
    target_score_rows: dict[int, list[float]],
    fit_indices: Sequence[int],
    *,
    label_shuffle: bool,
    desired_margin: float,
    mask_rounds: int,
    mask_keep_prob: float,
    control_conditions: Sequence[str],
    matched_weight: float,
    mask_weight: float,
    control_weight: float,
    l2_grid: Sequence[float],
    seed: int,
) -> dict[str, Any]:
    features, labels, sample_weights, metadata = _build_consistency_repair_pairwise_design(
        rows,
        source_sketch_rows,
        public_sketch_rows,
        target_score_rows,
        fit_indices,
        label_shuffle=label_shuffle,
        desired_margin=desired_margin,
        mask_rounds=mask_rounds,
        mask_keep_prob=mask_keep_prob,
        control_conditions=control_conditions,
        matched_weight=matched_weight,
        mask_weight=mask_weight,
        control_weight=control_weight,
        seed=seed,
    )
    cv_rows: list[dict[str, float]] = []
    selected_l2 = float(l2_grid[len(l2_grid) // 2])
    fit_indices = list(fit_indices)
    if len(fit_indices) >= 2:
        for l2 in l2_grid:
            fold_accs: list[float] = []
            for heldout in fit_indices:
                train_indices = [index for index in fit_indices if index != heldout]
                train_x, train_y, train_w, _ = _build_consistency_repair_pairwise_design(
                    rows,
                    source_sketch_rows,
                    public_sketch_rows,
                    target_score_rows,
                    train_indices,
                    label_shuffle=label_shuffle,
                    desired_margin=desired_margin,
                    mask_rounds=mask_rounds,
                    mask_keep_prob=mask_keep_prob,
                    control_conditions=control_conditions,
                    matched_weight=matched_weight,
                    mask_weight=mask_weight,
                    control_weight=control_weight,
                    seed=seed + int(heldout) * 1009,
                )
                weights = _fit_weighted_ridge_no_intercept(train_x, train_y, train_w, l2=float(l2))
                repair_scores = _consistency_repair_scores_by_row(
                    rows,
                    {"weights": weights},
                    source_sketch_rows,
                    public_sketch_rows,
                    target_score_rows,
                    [heldout],
                )
                final_scores = {
                    int(heldout): [
                        float(base) + float(delta)
                        for base, delta in zip(
                            target_score_rows[int(heldout)],
                            repair_scores[int(heldout)],
                            strict=True,
                        )
                    ]
                }
                fold_accs.append(
                    1.0
                    if _prediction(final_scores[int(heldout)]) == int(rows[int(heldout)].answer_index)
                    else 0.0
                )
            cv_rows.append({"l2": float(l2), "row_grouped_cv_accuracy": float(statistics.fmean(fold_accs))})
        selected = max(cv_rows, key=lambda row: (row["row_grouped_cv_accuracy"], row["l2"]))
        selected_l2 = float(selected["l2"])
    else:
        cv_rows.append({"l2": selected_l2, "row_grouped_cv_accuracy": 0.0})
    weights = _fit_weighted_ridge_no_intercept(features, labels, sample_weights, l2=selected_l2)
    return {
        "weights": weights,
        "selected_l2": selected_l2,
        "cv_rows": cv_rows,
        "feature_dim": int(features.shape[1]),
        "target_only": False,
        "label_shuffle": bool(label_shuffle),
        "training_objective": "consistency_repair_pairwise_ridge",
        "metadata": metadata,
    }


def _repair_accept_score(target_scores: Sequence[float], delta_scores: Sequence[float]) -> float:
    target = np.asarray(target_scores, dtype=np.float64)
    delta = np.asarray(delta_scores, dtype=np.float64)
    if target.shape[0] <= 1:
        return 0.0
    proposal_scores = target + delta
    target_prediction = _prediction(target)
    proposal = _prediction(proposal_scores)
    if proposal == target_prediction:
        return 0.0
    return float(proposal_scores[proposal] - proposal_scores[target_prediction])


def _apply_repair_acceptance(
    target_scores: Sequence[float],
    delta_scores: Sequence[float],
    *,
    threshold: float,
) -> list[float]:
    if _repair_accept_score(target_scores, delta_scores) <= float(threshold):
        return [float(score) for score in target_scores]
    return [float(base) + float(delta) for base, delta in zip(target_scores, delta_scores, strict=True)]


def _fit_set_repair(
    rows: Sequence[arc_gate.ArcRow],
    source_sketch_rows: Sequence[np.ndarray],
    public_sketch_rows: Sequence[np.ndarray],
    target_score_rows: dict[int, list[float]],
    fit_indices: Sequence[int],
    *,
    label_shuffle: bool,
    desired_margin: float,
    mask_rounds: int,
    mask_keep_prob: float,
    control_conditions: Sequence[str],
    matched_weight: float,
    mask_weight: float,
    control_weight: float,
    l2_grid: Sequence[float],
    accept_threshold_grid: Sequence[float],
    seed: int,
) -> dict[str, Any]:
    features, labels, sample_weights, metadata = _build_set_repair_pairwise_design(
        rows,
        source_sketch_rows,
        public_sketch_rows,
        target_score_rows,
        fit_indices,
        label_shuffle=label_shuffle,
        desired_margin=desired_margin,
        mask_rounds=mask_rounds,
        mask_keep_prob=mask_keep_prob,
        control_conditions=control_conditions,
        matched_weight=matched_weight,
        mask_weight=mask_weight,
        control_weight=control_weight,
        seed=seed,
    )
    fit_indices = list(fit_indices)
    cv_rows: list[dict[str, float]] = []
    selected_l2 = float(l2_grid[len(l2_grid) // 2])
    selected_threshold = float(accept_threshold_grid[0])
    if len(fit_indices) >= 2:
        for l2 in l2_grid:
            threshold_accs = {float(threshold): [] for threshold in accept_threshold_grid}
            threshold_accepts = {float(threshold): [] for threshold in accept_threshold_grid}
            threshold_helps = {float(threshold): [] for threshold in accept_threshold_grid}
            threshold_harms = {float(threshold): [] for threshold in accept_threshold_grid}
            for heldout in fit_indices:
                train_indices = [index for index in fit_indices if index != heldout]
                train_x, train_y, train_w, _ = _build_set_repair_pairwise_design(
                    rows,
                    source_sketch_rows,
                    public_sketch_rows,
                    target_score_rows,
                    train_indices,
                    label_shuffle=label_shuffle,
                    desired_margin=desired_margin,
                    mask_rounds=mask_rounds,
                    mask_keep_prob=mask_keep_prob,
                    control_conditions=control_conditions,
                    matched_weight=matched_weight,
                    mask_weight=mask_weight,
                    control_weight=control_weight,
                    seed=seed + int(heldout) * 1009,
                )
                weights = _fit_weighted_ridge_no_intercept(train_x, train_y, train_w, l2=float(l2))
                delta_scores = _set_repair_delta_scores_by_row(
                    rows,
                    {"weights": weights},
                    source_sketch_rows,
                    public_sketch_rows,
                    target_score_rows,
                    [heldout],
                )[int(heldout)]
                target_scores = target_score_rows[int(heldout)]
                target_correct = _prediction(target_scores) == int(rows[int(heldout)].answer_index)
                for threshold in accept_threshold_grid:
                    threshold = float(threshold)
                    final_scores = _apply_repair_acceptance(
                        target_scores,
                        delta_scores,
                        threshold=threshold,
                    )
                    accepted = final_scores != [float(score) for score in target_scores]
                    final_correct = _prediction(final_scores) == int(rows[int(heldout)].answer_index)
                    threshold_accs[threshold].append(1.0 if final_correct else 0.0)
                    threshold_accepts[threshold].append(1.0 if accepted else 0.0)
                    threshold_helps[threshold].append(1.0 if final_correct and not target_correct else 0.0)
                    threshold_harms[threshold].append(1.0 if target_correct and not final_correct else 0.0)
            for threshold in accept_threshold_grid:
                threshold = float(threshold)
                cv_rows.append(
                    {
                        "l2": float(l2),
                        "accept_threshold": threshold,
                        "row_grouped_cv_accuracy": float(statistics.fmean(threshold_accs[threshold])),
                        "row_grouped_accept_rate": float(statistics.fmean(threshold_accepts[threshold])),
                        "row_grouped_help_rate": float(statistics.fmean(threshold_helps[threshold])),
                        "row_grouped_harm_rate": float(statistics.fmean(threshold_harms[threshold])),
                    }
                )
        selected = max(
            cv_rows,
            key=lambda row: (
                row["row_grouped_cv_accuracy"],
                -row["row_grouped_harm_rate"],
                row["row_grouped_help_rate"],
                -row["row_grouped_accept_rate"],
                row["accept_threshold"],
                row["l2"],
            ),
        )
        selected_l2 = float(selected["l2"])
        selected_threshold = float(selected["accept_threshold"])
    else:
        cv_rows.append(
            {
                "l2": selected_l2,
                "accept_threshold": selected_threshold,
                "row_grouped_cv_accuracy": 0.0,
                "row_grouped_accept_rate": 0.0,
                "row_grouped_help_rate": 0.0,
                "row_grouped_harm_rate": 0.0,
            }
        )
    weights = _fit_weighted_ridge_no_intercept(features, labels, sample_weights, l2=selected_l2)
    return {
        "weights": weights,
        "selected_l2": selected_l2,
        "selected_accept_threshold": selected_threshold,
        "cv_rows": cv_rows,
        "feature_dim": int(features.shape[1]),
        "target_only": False,
        "label_shuffle": bool(label_shuffle),
        "training_objective": "set_repair_pairwise_ridge_with_accept",
        "metadata": metadata,
    }


def _residual_scores_by_row(
    rows: Sequence[arc_gate.ArcRow],
    receiver: dict[str, Any],
    source_sketch_rows: Sequence[np.ndarray],
    public_sketch_rows: Sequence[np.ndarray],
    row_indices: Sequence[int],
) -> dict[int, list[float]]:
    weights = np.asarray(receiver["weights"], dtype=np.float64)
    by_row: dict[int, list[float]] = {}
    for row_index in row_indices:
        row = rows[int(row_index)]
        features = np.vstack(
            [
                _source_residual_design_vector(
                    source_sketch=source_sketch_rows[int(row_index)][candidate_index],
                    public_sketch=public_sketch_rows[int(row_index)][candidate_index],
                )
                for candidate_index in range(len(row.choices))
            ]
        )
        by_row[int(row_index)] = [float(score) for score in _score_design(features, weights)]
    return by_row


def _consistency_repair_scores_by_row(
    rows: Sequence[arc_gate.ArcRow],
    receiver: dict[str, Any],
    source_sketch_rows: Sequence[np.ndarray],
    public_sketch_rows: Sequence[np.ndarray],
    target_score_rows: dict[int, list[float]],
    row_indices: Sequence[int],
) -> dict[int, list[float]]:
    weights = np.asarray(receiver["weights"], dtype=np.float64)
    by_row: dict[int, list[float]] = {}
    for row_index in row_indices:
        features = _consistency_repair_feature_rows(
            source_row=source_sketch_rows[int(row_index)],
            public_row=public_sketch_rows[int(row_index)],
            target_scores=target_score_rows[int(row_index)],
        )
        if features.shape[1] + 1 != weights.shape[0]:
            raise ValueError("consistency repair feature dimension does not match receiver weights")
        by_row[int(row_index)] = [float(score) for score in _score_design(features, weights)]
    return by_row


def _set_repair_delta_scores_by_row(
    rows: Sequence[arc_gate.ArcRow],
    receiver: dict[str, Any],
    source_sketch_rows: Sequence[np.ndarray],
    public_sketch_rows: Sequence[np.ndarray],
    target_score_rows: dict[int, list[float]],
    row_indices: Sequence[int],
) -> dict[int, list[float]]:
    weights = np.asarray(receiver["weights"], dtype=np.float64)
    by_row: dict[int, list[float]] = {}
    for row_index in row_indices:
        features = _set_repair_feature_rows(
            source_row=source_sketch_rows[int(row_index)],
            public_row=public_sketch_rows[int(row_index)],
            target_scores=target_score_rows[int(row_index)],
        )
        if features.shape[1] + 1 != weights.shape[0]:
            raise ValueError("set repair feature dimension does not match receiver weights")
        by_row[int(row_index)] = [float(score) for score in _score_design(features, weights)]
    return by_row


def _set_repair_scores_by_row(
    rows: Sequence[arc_gate.ArcRow],
    receiver: dict[str, Any],
    source_sketch_rows: Sequence[np.ndarray],
    public_sketch_rows: Sequence[np.ndarray],
    target_score_rows: dict[int, list[float]],
    row_indices: Sequence[int],
) -> dict[int, list[float]]:
    delta_rows = _set_repair_delta_scores_by_row(
        rows,
        receiver,
        source_sketch_rows,
        public_sketch_rows,
        target_score_rows,
        row_indices,
    )
    threshold = float(receiver.get("selected_accept_threshold", 0.0))
    return {
        int(row_index): _apply_repair_acceptance(
            target_score_rows[int(row_index)],
            delta_rows[int(row_index)],
            threshold=threshold,
        )
        for row_index in row_indices
    }


def _add_score_rows(
    base_rows: dict[int, list[float]],
    delta_rows: dict[int, list[float]],
    row_indices: Sequence[int],
) -> dict[int, list[float]]:
    combined: dict[int, list[float]] = {}
    for row_index in row_indices:
        combined[int(row_index)] = [
            float(base) + float(delta)
            for base, delta in zip(base_rows[int(row_index)], delta_rows[int(row_index)], strict=True)
        ]
    return combined


def _accuracy_for_scores(
    rows: Sequence[arc_gate.ArcRow],
    refs: Sequence[tuple[int, int]],
    scores: Sequence[float],
) -> float:
    by_row: dict[int, list[tuple[float, int]]] = {}
    for score, (row_index, candidate_index) in zip(scores, refs, strict=True):
        by_row.setdefault(int(row_index), []).append((float(score), int(candidate_index)))
    correct = 0
    for row_index, values in by_row.items():
        prediction = max(values, key=lambda item: (item[0], -item[1]))[1]
        correct += int(prediction == int(rows[row_index].answer_index))
    return float(correct / len(by_row)) if by_row else 0.0


def _select_l2(
    rows: Sequence[arc_gate.ArcRow],
    source_sketch_rows: Sequence[np.ndarray],
    public_sketch_rows: Sequence[np.ndarray],
    fit_indices: Sequence[int],
    *,
    max_candidate_count: int,
    target_only: bool,
    training_objective: str,
    l2_grid: Sequence[float],
) -> tuple[float, list[dict[str, float]]]:
    fit_indices = list(fit_indices)
    if len(fit_indices) < 2:
        selected = float(l2_grid[len(l2_grid) // 2])
        return selected, [{"l2": selected, "row_grouped_cv_accuracy": 0.0}]
    cv_rows: list[dict[str, float]] = []
    for l2 in l2_grid:
        fold_accs: list[float] = []
        for heldout in fit_indices:
            train_indices = [index for index in fit_indices if index != heldout]
            if training_objective == "pairwise_ridge":
                train_x, train_y = _build_pairwise_design(
                    rows,
                    source_sketch_rows,
                    public_sketch_rows,
                    train_indices,
                    max_candidate_count=max_candidate_count,
                    target_only=target_only,
                    label_shuffle=False,
                )
            elif training_objective == "pointwise_ridge":
                train_x, train_y, _ = _build_design(
                    rows,
                    source_sketch_rows,
                    public_sketch_rows,
                    train_indices,
                    max_candidate_count=max_candidate_count,
                    target_only=target_only,
                    label_shuffle=False,
                )
            else:
                raise ValueError(f"unknown training objective {training_objective!r}")
            weights = _fit_ridge(train_x, train_y, l2=float(l2))
            holdout_x, _, holdout_refs = _build_design(
                rows,
                source_sketch_rows,
                public_sketch_rows,
                [heldout],
                max_candidate_count=max_candidate_count,
                target_only=target_only,
                label_shuffle=False,
            )
            fold_accs.append(
                _accuracy_for_scores(rows, holdout_refs, _score_design(holdout_x, weights))
            )
        cv_rows.append(
            {
                "l2": float(l2),
                "row_grouped_cv_accuracy": float(statistics.fmean(fold_accs)),
            }
        )
    selected = max(cv_rows, key=lambda row: (row["row_grouped_cv_accuracy"], row["l2"]))
    return float(selected["l2"]), cv_rows


def _fit_receiver(
    rows: Sequence[arc_gate.ArcRow],
    source_sketch_rows: Sequence[np.ndarray],
    public_sketch_rows: Sequence[np.ndarray],
    fit_indices: Sequence[int],
    *,
    max_candidate_count: int,
    target_only: bool,
    label_shuffle: bool,
    l2_grid: Sequence[float],
    training_objective: str = "pointwise_ridge",
) -> dict[str, Any]:
    selected_l2, cv_rows = _select_l2(
        rows,
        source_sketch_rows,
        public_sketch_rows,
        fit_indices,
        max_candidate_count=max_candidate_count,
        target_only=target_only,
        training_objective=training_objective,
        l2_grid=l2_grid,
    )
    if training_objective == "pairwise_ridge":
        features, labels = _build_pairwise_design(
            rows,
            source_sketch_rows,
            public_sketch_rows,
            fit_indices,
            max_candidate_count=max_candidate_count,
            target_only=target_only,
            label_shuffle=label_shuffle,
        )
    elif training_objective == "pointwise_ridge":
        features, labels, _ = _build_design(
            rows,
            source_sketch_rows,
            public_sketch_rows,
            fit_indices,
            max_candidate_count=max_candidate_count,
            target_only=target_only,
            label_shuffle=label_shuffle,
        )
    else:
        raise ValueError(f"unknown training objective {training_objective!r}")
    weights = _fit_ridge(features, labels, l2=selected_l2)
    return {
        "weights": weights,
        "selected_l2": selected_l2,
        "cv_rows": cv_rows,
        "feature_dim": int(features.shape[1]),
        "target_only": bool(target_only),
        "label_shuffle": bool(label_shuffle),
        "training_objective": training_objective,
    }


def _scores_by_row(
    rows: Sequence[arc_gate.ArcRow],
    receiver: dict[str, Any],
    source_sketch_rows: Sequence[np.ndarray],
    public_sketch_rows: Sequence[np.ndarray],
    row_indices: Sequence[int],
    *,
    max_candidate_count: int,
) -> dict[int, list[float]]:
    features, _, refs = _build_design(
        rows,
        source_sketch_rows,
        public_sketch_rows,
        row_indices,
        max_candidate_count=max_candidate_count,
        target_only=bool(receiver["target_only"]),
        label_shuffle=False,
    )
    flat_scores = _score_design(features, np.asarray(receiver["weights"], dtype=np.float64))
    by_row: dict[int, list[float]] = {}
    for score, (row_index, candidate_index) in zip(flat_scores, refs, strict=True):
        row_scores = by_row.setdefault(
            int(row_index),
            [-1.0e9 for _ in rows[int(row_index)].choices],
        )
        row_scores[int(candidate_index)] = float(score)
    return by_row


def _source_control_rows(
    source_sketch_rows: Sequence[np.ndarray],
    *,
    fit_indices: Sequence[int],
    eval_indices: Sequence[int],
    control: str,
    seed: int,
) -> list[np.ndarray]:
    source = [np.asarray(row, dtype=np.float64) for row in source_sketch_rows]
    if control == "zero_source":
        return [np.zeros_like(row) for row in source]
    if control == "train_mean_source":
        train_mean = np.mean([source[int(index)] for index in fit_indices], axis=0)
        return [train_mean.copy() for _ in source]
    if control == "candidate_roll_source":
        return [np.roll(row, shift=1, axis=0) for row in source]
    if control == "shuffled_source":
        variants = [row.copy() for row in source]
        eval_indices = list(eval_indices)
        for position, row_index in enumerate(eval_indices):
            other = eval_indices[(position + 1) % len(eval_indices)]
            variants[int(row_index)] = source[int(other)].copy()
        return variants
    if control == "same_norm_noise":
        variants = [row.copy() for row in source]
        for row_index in eval_indices:
            rng = np.random.default_rng(int(seed) * 1009 + int(row_index) * 31)
            noise = rng.normal(size=source[int(row_index)].shape)
            noise_norm = max(float(np.linalg.norm(noise)), 1e-12)
            source_norm = max(float(np.linalg.norm(source[int(row_index)])), 1e-12)
            variants[int(row_index)] = noise / noise_norm * source_norm
        return variants
    raise ValueError(f"unknown source control {control!r}")


def _source_variant_for_control(
    source_sketch_rows: Sequence[np.ndarray],
    public_sketch_rows: Sequence[np.ndarray],
    *,
    fit_indices: Sequence[int],
    eval_indices: Sequence[int],
    control: str,
    seed: int,
) -> list[np.ndarray]:
    if control == "target_derived_source":
        return [np.asarray(row, dtype=np.float64).copy() for row in public_sketch_rows]
    return _source_control_rows(
        source_sketch_rows,
        fit_indices=fit_indices,
        eval_indices=eval_indices,
        control=control,
        seed=seed,
    )


def _visible_text_scores(
    row: arc_gate.ArcRow,
    *,
    source_prediction: int,
    same_byte_budget: int,
) -> list[float]:
    hint = row.choices[int(source_prediction)].encode("utf-8")[: int(same_byte_budget)].decode(
        "utf-8",
        errors="ignore",
    )
    hint_tokens = set(arc_gate._tokens(hint))
    scores: list[float] = []
    for choice in row.choices:
        choice_tokens = set(arc_gate._tokens(choice))
        overlap = len(hint_tokens & choice_tokens)
        prefix_match = float(choice.lower().startswith(hint.lower())) if hint else 0.0
        scores.append(float(overlap) + prefix_match)
    return scores


def _prediction_rows(
    rows: Sequence[arc_gate.ArcRow],
    source_predictions: Sequence[int],
    score_by_condition: dict[str, dict[int, list[float]]],
    eval_indices: Sequence[int],
) -> list[dict[str, Any]]:
    prediction_rows: list[dict[str, Any]] = []
    for row_index in eval_indices:
        row = rows[int(row_index)]
        for condition in REPORT_CONDITIONS:
            scores = score_by_condition[condition][int(row_index)]
            prediction = _prediction(scores)
            prediction_rows.append(
                {
                    "row_id": row.row_id,
                    "content_id": row.content_id,
                    "condition": condition,
                    "answer_index": int(row.answer_index),
                    "answer_label": row.answer_label,
                    "prediction_index": int(prediction),
                    "prediction_label": row.choice_labels[prediction],
                    "correct": bool(prediction == int(row.answer_index)),
                    "margin": float(_margin(scores, int(row.answer_index))),
                    "scores": [float(score) for score in scores],
                    "source_selected_index": int(source_predictions[int(row_index)]),
                    "source_selected_label": row.choice_labels[int(source_predictions[int(row_index)])],
                }
            )
    return prediction_rows


def _summarize_condition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "correct": 0, "accuracy": 0.0, "mean_margin": 0.0}
    correct = sum(1 for row in rows if row["correct"])
    return {
        "n": len(rows),
        "correct": int(correct),
        "accuracy": float(correct / len(rows)),
        "mean_margin": float(statistics.fmean(float(row["margin"]) for row in rows)),
    }


def _condition_metrics(rows: list[dict[str, Any]], *, seed: int, bootstrap_samples: int) -> dict[str, Any]:
    metrics = {
        condition: _summarize_condition([row for row in rows if row["condition"] == condition])
        for condition in REPORT_CONDITIONS
    }
    by_id: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_id.setdefault(str(row["content_id"]), {})[str(row["condition"])] = row
    for condition in CONTROL_CONDITIONS:
        deltas = [
            float(group[MATCHED_CONDITION]["correct"]) - float(group[condition]["correct"])
            for group in by_id.values()
            if MATCHED_CONDITION in group and condition in group
        ]
        margin_deltas = [
            float(group[MATCHED_CONDITION]["margin"]) - float(group[condition]["margin"])
            for group in by_id.values()
            if MATCHED_CONDITION in group and condition in group
        ]
        metrics[MATCHED_CONDITION][f"paired_accuracy_vs_{condition}"] = _paired_bootstrap(
            deltas,
            seed=seed + len(condition),
            samples=bootstrap_samples,
        )
        metrics[MATCHED_CONDITION][f"mean_margin_delta_vs_{condition}"] = (
            float(statistics.fmean(margin_deltas)) if margin_deltas else 0.0
        )
    return metrics


def _acceptance_metrics(
    rows: Sequence[arc_gate.ArcRow],
    score_by_condition: dict[str, dict[int, list[float]]],
    eval_indices: Sequence[int],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    target_scores_by_row = score_by_condition["target_public_only"]
    for condition in REPORT_CONDITIONS:
        accepted = 0
        helps = 0
        harms = 0
        ties = 0
        for row_index in eval_indices:
            row = rows[int(row_index)]
            target_scores = target_scores_by_row[int(row_index)]
            scores = score_by_condition[condition][int(row_index)]
            was_accepted = not np.allclose(scores, target_scores, rtol=0.0, atol=1e-9)
            target_correct = _prediction(target_scores) == int(row.answer_index)
            final_correct = _prediction(scores) == int(row.answer_index)
            accepted += int(was_accepted)
            helps += int((not target_correct) and final_correct)
            harms += int(target_correct and (not final_correct))
            ties += int(target_correct == final_correct)
        n = max(len(eval_indices), 1)
        metrics[condition] = {
            "accepted": int(accepted),
            "accept_rate": float(accepted / n),
            "helps": int(helps),
            "harms": int(harms),
            "ties_vs_target_correctness": int(ties),
        }
    return metrics


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "row_id",
        "content_id",
        "condition",
        "answer_index",
        "prediction_index",
        "correct",
        "margin",
        "source_selected_index",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    headline = payload["headline"]
    lines = [
        "# ARC Candidate-Alignment Receiver Preflight",
        "",
        f"- date: `{payload['date']}`",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- implementation gate only: `{payload['implementation_gate_only']}`",
        f"- fit/eval rows: `{payload['fit_rows']}` / `{payload['eval_rows']}`",
        f"- packet bytes: `{payload['systems']['source_packet_bytes']}B`",
        f"- matched accuracy: `{headline['matched_accuracy']:.3f}`",
        f"- best control by accuracy: `{headline['best_control_by_accuracy']}`",
        f"- best control accuracy: `{headline['best_control_accuracy']:.3f}`",
        f"- matched margin: `{headline['matched_mean_margin']:.6f}`",
        f"- best control margin: `{headline['best_control_mean_margin']:.6f}`",
        f"- matched minus best-control margin: `{headline['matched_minus_best_control_margin']:.6f}`",
        f"- zero-source exact target-public match: `{headline['zero_source_exact_match']}`",
        "",
        "## Conditions",
        "",
        "| Condition | Accuracy | Correct / N | Mean Margin |",
        "|---|---:|---:|---:|",
    ]
    for condition, metrics in payload["condition_metrics"].items():
        lines.append(
            f"| `{condition}` | {metrics['accuracy']:.3f} | {metrics['correct']} / {metrics['n']} | "
            f"{metrics['mean_margin']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
            "",
            "Lay explanation: this experiment gives each answer choice a tiny source-model hint and "
            "trains a simple referee to rank the choices. The broken controls shuffle, erase, randomize, "
            "or rotate those hints. If the real hints do not beat the broken hints, we have not proven "
            "real model-to-model communication.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_preflight(
    *,
    output_dir: pathlib.Path,
    eval_path: pathlib.Path,
    source_cache_path: pathlib.Path,
    row_limit: int,
    fit_fraction: float,
    source_feature_mode: str,
    source_feature_dim: int,
    source_token_pool_size: int,
    source_model: str,
    source_device: str,
    source_dtype: str,
    source_max_length: int,
    source_hidden_layer: int,
    target_feature_dim: int,
    sketch_dim: int,
    source_sketch_quantization: str,
    receiver_mode: str,
    training_objective: str,
    residual_fit_policy: str,
    residual_desired_margin: float,
    repair_mask_rounds: int,
    repair_mask_keep_prob: float,
    repair_matched_weight: float,
    repair_mask_weight: float,
    repair_control_weight: float,
    accept_threshold_grid: Sequence[float],
    innovation_ridge: float,
    local_files_only: bool,
    seed: int,
    bootstrap_samples: int,
    same_byte_budget: int,
    min_accuracy_gap: float,
    min_margin_gap: float,
    l2_grid: Sequence[float] = DEFAULT_L2_GRID,
) -> dict[str, Any]:
    total_start = time.perf_counter()
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_all = arc_gate._load_rows(_resolve(eval_path))
    source_cache = soft_prefix._read_source_cache(_resolve(source_cache_path))
    rows, source_predictions = soft_prefix._select_rows_with_cache(
        rows_all,
        source_cache,
        row_limit=row_limit,
    )
    fit_indices, eval_indices = soft_prefix._row_indices(
        row_count=len(rows),
        fit_fraction=fit_fraction,
        seed=seed,
    )
    max_candidate_count = max(len(row.choices) for row in rows)

    source_summary, source_meta = soft_prefix._selected_choice_features(
        rows,
        source_predictions,
        source_feature_mode=source_feature_mode,
        feature_dim=source_feature_dim,
        source_model=source_model,
        source_device=source_device,
        source_dtype=source_dtype,
        source_max_length=source_max_length,
        source_hidden_layer=source_hidden_layer,
        source_token_pool_size=source_token_pool_size,
        fit_indices=fit_indices,
        innovation_ridge=innovation_ridge,
        local_files_only=local_files_only,
    )
    if source_summary.dim() != 3:
        raise ValueError("candidate-alignment receiver requires a rank-3 candidate-pool source feature mode")
    source_summary, source_standardizer = soft_prefix._standardize(source_summary, fit_indices)
    source_rows = [np.asarray(row, dtype=np.float64) for row in source_summary.detach().cpu().numpy()]

    public_flat, public_meta = soft_prefix._public_candidate_hashed_features(
        rows,
        feature_dim=target_feature_dim,
    )
    public_rows_raw = _public_candidate_rows(rows, public_flat)
    public_rows, public_standardizer = _standardize_candidate_rows(public_rows_raw, fit_indices)

    source_projection = _signed_projection(source_rows[0].shape[1], sketch_dim, seed=seed + 101)
    public_projection = _signed_projection(public_rows[0].shape[1], sketch_dim, seed=seed + 202)
    source_sketch_rows = _sketch_rows(
        source_rows,
        source_projection,
        quantization=source_sketch_quantization,
    )
    public_sketch_rows = [row @ public_projection for row in public_rows]

    target_receiver = _fit_receiver(
        rows,
        source_sketch_rows,
        public_sketch_rows,
        fit_indices,
        max_candidate_count=max_candidate_count,
        target_only=True,
        label_shuffle=False,
        training_objective=training_objective,
        l2_grid=l2_grid,
    )
    score_by_condition: dict[str, dict[int, list[float]]] = {}
    target_scores_all = _scores_by_row(
        rows,
        target_receiver,
        source_sketch_rows,
        public_sketch_rows,
        list(range(len(rows))),
        max_candidate_count=max_candidate_count,
    )
    score_by_condition["target_public_only"] = _scores_by_row(
        rows,
        target_receiver,
        source_sketch_rows,
        public_sketch_rows,
        eval_indices,
        max_candidate_count=max_candidate_count,
    )
    candidate_derangement_scores: dict[int, list[float]] | None = None

    if receiver_mode == "direct":
        matched_receiver = _fit_receiver(
            rows,
            source_sketch_rows,
            public_sketch_rows,
            fit_indices,
            max_candidate_count=max_candidate_count,
            target_only=False,
            label_shuffle=False,
            training_objective=training_objective,
            l2_grid=l2_grid,
        )
        label_shuffled_receiver = _fit_receiver(
            rows,
            source_sketch_rows,
            public_sketch_rows,
            fit_indices,
            max_candidate_count=max_candidate_count,
            target_only=False,
            label_shuffle=True,
            training_objective=training_objective,
            l2_grid=l2_grid,
        )
        score_by_condition[MATCHED_CONDITION] = _scores_by_row(
            rows,
            matched_receiver,
            source_sketch_rows,
            public_sketch_rows,
            eval_indices,
            max_candidate_count=max_candidate_count,
        )
        score_by_condition["label_shuffled"] = _scores_by_row(
            rows,
            label_shuffled_receiver,
            source_sketch_rows,
            public_sketch_rows,
            eval_indices,
            max_candidate_count=max_candidate_count,
        )
        for control in (
            "zero_source",
            "shuffled_source",
            "same_norm_noise",
            "train_mean_source",
            "target_derived_source",
            "candidate_roll_source",
        ):
            variant_source = _source_variant_for_control(
                source_sketch_rows,
                public_sketch_rows,
                fit_indices=fit_indices,
                eval_indices=eval_indices,
                control=control,
                seed=seed,
            )
            score_by_condition[control] = _scores_by_row(
                rows,
                matched_receiver,
                variant_source,
                public_sketch_rows,
                eval_indices,
                max_candidate_count=max_candidate_count,
            )
        receiver_logs = {
            "matched_receiver": {
                "selected_l2": matched_receiver["selected_l2"],
                "cv_rows": matched_receiver["cv_rows"],
                "feature_dim": matched_receiver["feature_dim"],
            },
            "label_shuffled_receiver": {
                "selected_l2": label_shuffled_receiver["selected_l2"],
                "cv_rows": label_shuffled_receiver["cv_rows"],
                "feature_dim": label_shuffled_receiver["feature_dim"],
            },
        }
    elif receiver_mode == "target_residual":
        residual_receiver = _fit_residual_correction(
            rows,
            source_sketch_rows,
            public_sketch_rows,
            target_scores_all,
            fit_indices,
            label_shuffle=False,
            residual_fit_policy=residual_fit_policy,
            desired_margin=residual_desired_margin,
            l2_grid=l2_grid,
        )
        label_shuffled_receiver = _fit_residual_correction(
            rows,
            source_sketch_rows,
            public_sketch_rows,
            target_scores_all,
            fit_indices,
            label_shuffle=True,
            residual_fit_policy=residual_fit_policy,
            desired_margin=residual_desired_margin,
            l2_grid=l2_grid,
        )
        matched_delta = _residual_scores_by_row(
            rows,
            residual_receiver,
            source_sketch_rows,
            public_sketch_rows,
            eval_indices,
        )
        score_by_condition[MATCHED_CONDITION] = _add_score_rows(
            score_by_condition["target_public_only"],
            matched_delta,
            eval_indices,
        )
        shuffled_delta = _residual_scores_by_row(
            rows,
            label_shuffled_receiver,
            source_sketch_rows,
            public_sketch_rows,
            eval_indices,
        )
        score_by_condition["label_shuffled"] = _add_score_rows(
            score_by_condition["target_public_only"],
            shuffled_delta,
            eval_indices,
        )
        for control in (
            "zero_source",
            "shuffled_source",
            "same_norm_noise",
            "train_mean_source",
            "target_derived_source",
            "candidate_roll_source",
        ):
            variant_source = _source_variant_for_control(
                source_sketch_rows,
                public_sketch_rows,
                fit_indices=fit_indices,
                eval_indices=eval_indices,
                control=control,
                seed=seed,
            )
            control_delta = _residual_scores_by_row(
                rows,
                residual_receiver,
                variant_source,
                public_sketch_rows,
                eval_indices,
            )
            score_by_condition[control] = _add_score_rows(
                score_by_condition["target_public_only"],
                control_delta,
                eval_indices,
            )
        deranged_source = [np.asarray(row, dtype=np.float64).copy() for row in source_sketch_rows]
        for row_index in eval_indices:
            row = deranged_source[int(row_index)]
            candidate_count = len(rows[int(row_index)].choices)
            deranged = row.copy()
            deranged[:candidate_count] = row[:candidate_count][::-1]
            deranged_source[int(row_index)] = deranged
        candidate_derangement_scores = _set_repair_scores_by_row(
            rows,
            set_receiver,
            deranged_source,
            public_sketch_rows,
            target_scores_all,
            eval_indices,
        )
        receiver_logs = {
            "residual_receiver": {
                "selected_l2": residual_receiver["selected_l2"],
                "cv_rows": residual_receiver["cv_rows"],
                "feature_dim": residual_receiver["feature_dim"],
                "metadata": residual_receiver["metadata"],
            },
            "label_shuffled_residual_receiver": {
                "selected_l2": label_shuffled_receiver["selected_l2"],
                "cv_rows": label_shuffled_receiver["cv_rows"],
                "feature_dim": label_shuffled_receiver["feature_dim"],
                "metadata": label_shuffled_receiver["metadata"],
            },
        }
    elif receiver_mode == "target_consistency_repair":
        repair_receiver = _fit_consistency_repair(
            rows,
            source_sketch_rows,
            public_sketch_rows,
            target_scores_all,
            fit_indices,
            label_shuffle=False,
            desired_margin=residual_desired_margin,
            mask_rounds=repair_mask_rounds,
            mask_keep_prob=repair_mask_keep_prob,
            control_conditions=CONSISTENCY_REPAIR_CONTROL_CONDITIONS,
            matched_weight=repair_matched_weight,
            mask_weight=repair_mask_weight,
            control_weight=repair_control_weight,
            l2_grid=l2_grid,
            seed=seed + 303,
        )
        label_shuffled_receiver = _fit_consistency_repair(
            rows,
            source_sketch_rows,
            public_sketch_rows,
            target_scores_all,
            fit_indices,
            label_shuffle=True,
            desired_margin=residual_desired_margin,
            mask_rounds=repair_mask_rounds,
            mask_keep_prob=repair_mask_keep_prob,
            control_conditions=CONSISTENCY_REPAIR_CONTROL_CONDITIONS,
            matched_weight=repair_matched_weight,
            mask_weight=repair_mask_weight,
            control_weight=repair_control_weight,
            l2_grid=l2_grid,
            seed=seed + 607,
        )
        matched_delta = _consistency_repair_scores_by_row(
            rows,
            repair_receiver,
            source_sketch_rows,
            public_sketch_rows,
            target_scores_all,
            eval_indices,
        )
        score_by_condition[MATCHED_CONDITION] = _add_score_rows(
            score_by_condition["target_public_only"],
            matched_delta,
            eval_indices,
        )
        shuffled_delta = _consistency_repair_scores_by_row(
            rows,
            label_shuffled_receiver,
            source_sketch_rows,
            public_sketch_rows,
            target_scores_all,
            eval_indices,
        )
        score_by_condition["label_shuffled"] = _add_score_rows(
            score_by_condition["target_public_only"],
            shuffled_delta,
            eval_indices,
        )
        for control in (
            "zero_source",
            "shuffled_source",
            "same_norm_noise",
            "train_mean_source",
            "target_derived_source",
            "candidate_roll_source",
        ):
            variant_source = _source_variant_for_control(
                source_sketch_rows,
                public_sketch_rows,
                fit_indices=fit_indices,
                eval_indices=eval_indices,
                control=control,
                seed=seed,
            )
            control_delta = _consistency_repair_scores_by_row(
                rows,
                repair_receiver,
                variant_source,
                public_sketch_rows,
                target_scores_all,
                eval_indices,
            )
            score_by_condition[control] = _add_score_rows(
                score_by_condition["target_public_only"],
                control_delta,
                eval_indices,
            )
        receiver_logs = {
            "consistency_repair_receiver": {
                "selected_l2": repair_receiver["selected_l2"],
                "cv_rows": repair_receiver["cv_rows"],
                "feature_dim": repair_receiver["feature_dim"],
                "metadata": repair_receiver["metadata"],
            },
            "label_shuffled_consistency_repair_receiver": {
                "selected_l2": label_shuffled_receiver["selected_l2"],
                "cv_rows": label_shuffled_receiver["cv_rows"],
                "feature_dim": label_shuffled_receiver["feature_dim"],
                "metadata": label_shuffled_receiver["metadata"],
            },
        }
    elif receiver_mode in {"target_set_repair", "equivariant_set_delta_accept"}:
        set_receiver = _fit_set_repair(
            rows,
            source_sketch_rows,
            public_sketch_rows,
            target_scores_all,
            fit_indices,
            label_shuffle=False,
            desired_margin=residual_desired_margin,
            mask_rounds=repair_mask_rounds,
            mask_keep_prob=repair_mask_keep_prob,
            control_conditions=SET_REPAIR_CONTROL_CONDITIONS,
            matched_weight=repair_matched_weight,
            mask_weight=repair_mask_weight,
            control_weight=repair_control_weight,
            l2_grid=l2_grid,
            accept_threshold_grid=accept_threshold_grid,
            seed=seed + 809,
        )
        label_shuffled_receiver = _fit_set_repair(
            rows,
            source_sketch_rows,
            public_sketch_rows,
            target_scores_all,
            fit_indices,
            label_shuffle=True,
            desired_margin=residual_desired_margin,
            mask_rounds=repair_mask_rounds,
            mask_keep_prob=repair_mask_keep_prob,
            control_conditions=SET_REPAIR_CONTROL_CONDITIONS,
            matched_weight=repair_matched_weight,
            mask_weight=repair_mask_weight,
            control_weight=repair_control_weight,
            l2_grid=l2_grid,
            accept_threshold_grid=accept_threshold_grid,
            seed=seed + 1013,
        )
        score_by_condition[MATCHED_CONDITION] = _set_repair_scores_by_row(
            rows,
            set_receiver,
            source_sketch_rows,
            public_sketch_rows,
            target_scores_all,
            eval_indices,
        )
        score_by_condition["label_shuffled"] = _set_repair_scores_by_row(
            rows,
            label_shuffled_receiver,
            source_sketch_rows,
            public_sketch_rows,
            target_scores_all,
            eval_indices,
        )
        for control in (
            "zero_source",
            "shuffled_source",
            "same_norm_noise",
            "train_mean_source",
            "target_derived_source",
            "candidate_roll_source",
        ):
            variant_source = _source_variant_for_control(
                source_sketch_rows,
                public_sketch_rows,
                fit_indices=fit_indices,
                eval_indices=eval_indices,
                control=control,
                seed=seed,
            )
            score_by_condition[control] = _set_repair_scores_by_row(
                rows,
                set_receiver,
                variant_source,
                public_sketch_rows,
                target_scores_all,
                eval_indices,
            )
        receiver_logs = {
            "set_repair_receiver": {
                "selected_l2": set_receiver["selected_l2"],
                "selected_accept_threshold": set_receiver["selected_accept_threshold"],
                "cv_rows": set_receiver["cv_rows"],
                "feature_dim": set_receiver["feature_dim"],
                "metadata": set_receiver["metadata"],
            },
            "label_shuffled_set_repair_receiver": {
                "selected_l2": label_shuffled_receiver["selected_l2"],
                "selected_accept_threshold": label_shuffled_receiver["selected_accept_threshold"],
                "cv_rows": label_shuffled_receiver["cv_rows"],
                "feature_dim": label_shuffled_receiver["feature_dim"],
                "metadata": label_shuffled_receiver["metadata"],
            },
        }
    else:
        raise ValueError(f"unknown receiver mode {receiver_mode!r}")

    if candidate_derangement_scores is not None:
        score_by_condition["candidate_derangement"] = candidate_derangement_scores
    else:
        score_by_condition["candidate_derangement"] = {
            int(row_index): [float(score) for score in np.roll(score_by_condition[MATCHED_CONDITION][int(row_index)], 1)]
            for row_index in eval_indices
        }
    score_by_condition["same_byte_visible_text"] = {
        int(row_index): _visible_text_scores(
            rows[int(row_index)],
            source_prediction=source_predictions[int(row_index)],
            same_byte_budget=same_byte_budget,
        )
        for row_index in eval_indices
    }
    score_by_condition["source_label_copy_audit_upper_bound"] = {}
    for row_index in eval_indices:
        row = rows[int(row_index)]
        scores = [-1.0e9 for _ in row.choices]
        scores[int(source_predictions[int(row_index)])] = 0.0
        score_by_condition["source_label_copy_audit_upper_bound"][int(row_index)] = scores

    prediction_rows = _prediction_rows(rows, source_predictions, score_by_condition, eval_indices)
    metrics = _condition_metrics(prediction_rows, seed=seed + 404, bootstrap_samples=bootstrap_samples)
    matched = metrics[MATCHED_CONDITION]
    best_control_by_accuracy = max(PASS_CONTROL_CONDITIONS, key=lambda name: metrics[name]["accuracy"])
    best_control_by_margin = max(PASS_CONTROL_CONDITIONS, key=lambda name: metrics[name]["mean_margin"])
    zero_source_exact_match = all(
        np.allclose(
            score_by_condition["zero_source"][int(row_index)],
            score_by_condition["target_public_only"][int(row_index)],
            rtol=0.0,
            atol=1e-9,
        )
        for row_index in eval_indices
    )
    zero_source_exact_required = receiver_mode in {
        "target_residual",
        "target_consistency_repair",
        "target_set_repair",
        "equivariant_set_delta_accept",
    }
    headline = {
        "matched_accuracy": matched["accuracy"],
        "matched_mean_margin": matched["mean_margin"],
        "best_control_by_accuracy": best_control_by_accuracy,
        "best_control_accuracy": metrics[best_control_by_accuracy]["accuracy"],
        "best_control_by_margin": best_control_by_margin,
        "best_control_mean_margin": metrics[best_control_by_margin]["mean_margin"],
        "matched_minus_best_control_accuracy": matched["accuracy"] - metrics[best_control_by_accuracy]["accuracy"],
        "matched_minus_best_control_margin": matched["mean_margin"] - metrics[best_control_by_margin]["mean_margin"],
        "zero_source_exact_match": bool(zero_source_exact_match),
        "zero_source_exact_required": bool(zero_source_exact_required),
    }
    pass_gate = bool(
        headline["matched_minus_best_control_accuracy"] >= min_accuracy_gap
        and headline["matched_minus_best_control_margin"] >= min_margin_gap
        and matched[f"paired_accuracy_vs_{best_control_by_accuracy}"]["ci95_low"] > 0.0
        and (zero_source_exact_match or not zero_source_exact_required)
    )
    source_packet_bytes = _packet_bytes(
        max_candidate_count=max_candidate_count,
        sketch_dim=sketch_dim,
        quantization=source_sketch_quantization,
    )
    interpretation = (
        "This preflight promotes the candidate-alignment branch only if a matched source candidate "
        "sketch beats target-public, erased-source, wrong-row, same-norm-noise, train-mean, "
        "label-shuffled, candidate-roll, candidate-derangement, and visible same-byte text controls. "
        "A failure weakens this exact external receiver but still leaves richer equivariant rankers alive."
    )
    payload = {
        "gate": "source_private_arc_candidate_alignment_receiver_preflight",
        "date": "2026-05-02",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "implementation_gate_only": True,
        "pass_gate": pass_gate,
        "fit_rows": len(fit_indices),
        "eval_rows": len(eval_indices),
        "row_limit": len(rows),
        "fit_indices": fit_indices,
        "eval_indices": eval_indices,
        "config": {
            "source_feature_mode": source_feature_mode,
            "source_feature_dim": source_feature_dim,
            "source_token_pool_size": source_token_pool_size,
            "source_model": source_model,
            "source_device": source_device,
            "source_dtype": source_dtype,
            "source_max_length": source_max_length,
            "source_hidden_layer": source_hidden_layer,
            "target_feature_dim": target_feature_dim,
            "sketch_dim": sketch_dim,
            "source_sketch_quantization": source_sketch_quantization,
            "receiver_mode": receiver_mode,
            "training_objective": training_objective,
            "residual_fit_policy": residual_fit_policy,
            "residual_desired_margin": residual_desired_margin,
            "repair_mask_rounds": repair_mask_rounds,
            "repair_mask_keep_prob": repair_mask_keep_prob,
            "repair_matched_weight": repair_matched_weight,
            "repair_mask_weight": repair_mask_weight,
            "repair_control_weight": repair_control_weight,
            "accept_threshold_grid": [float(value) for value in accept_threshold_grid],
            "innovation_ridge": innovation_ridge,
            "same_byte_budget": same_byte_budget,
            "seed": seed,
            "l2_grid": [float(value) for value in l2_grid],
            "min_accuracy_gap": min_accuracy_gap,
            "min_margin_gap": min_margin_gap,
        },
        "systems": {
            "source_packet_bytes": source_packet_bytes,
            "packet_formula": (
                "ceil(max_candidate_count * sketch_dim / 8) for sign sketches; "
                "max_candidate_count * sketch_dim + 2 scale bytes for int8; "
                "float32 sketch bytes for unquantized diagnostic mode"
            ),
            "max_candidate_count": max_candidate_count,
            "sketch_dim": sketch_dim,
            "quantization": source_sketch_quantization,
        },
        "feature_metadata": {
            "source": source_meta,
            "source_standardizer": source_standardizer,
            "public_candidate": public_meta,
            "public_standardizer": public_standardizer,
        },
        "fit_logs": {
            "target_public_receiver": {
                "selected_l2": target_receiver["selected_l2"],
                "cv_rows": target_receiver["cv_rows"],
                "feature_dim": target_receiver["feature_dim"],
            },
            **receiver_logs,
        },
        "headline": headline,
        "acceptance_metrics": _acceptance_metrics(rows, score_by_condition, eval_indices),
        "pass_control_conditions": list(PASS_CONTROL_CONDITIONS),
        "audit_only_conditions": ["source_label_copy_audit_upper_bound"],
        "condition_metrics": metrics,
        "interpretation": interpretation,
        "inputs": {
            "eval_path": _display(eval_path),
            "eval_path_sha256": _sha256_file(eval_path),
            "source_cache_path": _display(source_cache_path),
            "source_cache_path_sha256": _sha256_file(source_cache_path),
        },
        "runtime": {
            "latency_s": float(time.perf_counter() - total_start),
            "peak_rss_mib": _peak_rss_mib(),
        },
    }
    json_path = output_dir / "arc_candidate_alignment_receiver_preflight.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_jsonl(output_dir / "prediction_audit.jsonl", prediction_rows)
    _write_csv(output_dir / "prediction_audit.csv", prediction_rows)
    _write_markdown(output_dir / "arc_candidate_alignment_receiver_preflight.md", payload)
    manifest = {
        "gate": payload["gate"],
        "pass_gate": payload["pass_gate"],
        "created_utc": payload["created_utc"],
        "files": [
            {
                "path": _display(path),
                "sha256": _sha256_file(path),
                "bytes": _resolve(path).stat().st_size,
            }
            for path in (
                json_path,
                output_dir / "prediction_audit.jsonl",
                output_dir / "prediction_audit.csv",
                output_dir / "arc_candidate_alignment_receiver_preflight.md",
            )
        ],
        "inputs": payload["inputs"],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _parse_l2_grid(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("l2 grid must not be empty")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-path", type=pathlib.Path, default=DEFAULT_ARC_VALIDATION)
    parser.add_argument("--source-cache-path", type=pathlib.Path, default=DEFAULT_ARC_SOURCE_CACHE)
    parser.add_argument("--row-limit", type=int, default=8)
    parser.add_argument("--fit-fraction", type=float, default=0.5)
    parser.add_argument(
        "--source-feature-mode",
        choices=(
            "cached_choice_score_pool",
            "cached_choice_score_pool_residual",
            "hf_choice_hidden_candidate_pool",
            "hf_choice_hidden_candidate_pool_residual",
            "hf_choice_hidden_score_candidate_pool",
            "hf_choice_hidden_score_candidate_pool_residual",
            "hf_choice_hidden_public_innovation_candidate_pool",
            "hf_choice_hidden_public_innovation_candidate_pool_residual",
            "hf_choice_hidden_score_public_innovation_candidate_pool",
            "hf_choice_hidden_score_public_innovation_candidate_pool_residual",
        ),
        default="hf_choice_hidden_public_innovation_candidate_pool_residual",
    )
    parser.add_argument("--source-feature-dim", type=int, default=128)
    parser.add_argument("--source-token-pool-size", type=int, default=5)
    parser.add_argument("--source-model", default=DEFAULT_QWEN_SOURCE)
    parser.add_argument("--source-device", default="auto_cpu")
    parser.add_argument("--source-dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--source-max-length", type=int, default=192)
    parser.add_argument("--source-hidden-layer", type=int, default=-1)
    parser.add_argument("--target-feature-dim", type=int, default=64)
    parser.add_argument("--sketch-dim", type=int, default=16)
    parser.add_argument("--source-sketch-quantization", choices=("none", "int8", "sign"), default="sign")
    parser.add_argument("--receiver-mode", choices=RECEIVER_MODES, default="target_residual")
    parser.add_argument(
        "--training-objective",
        choices=("pointwise_ridge", "pairwise_ridge"),
        default="pairwise_ridge",
    )
    parser.add_argument("--residual-fit-policy", choices=RESIDUAL_FIT_POLICIES, default="target_errors")
    parser.add_argument("--residual-desired-margin", type=float, default=1.0)
    parser.add_argument("--repair-mask-rounds", type=int, default=2)
    parser.add_argument("--repair-mask-keep-prob", type=float, default=0.65)
    parser.add_argument("--repair-matched-weight", type=float, default=2.0)
    parser.add_argument("--repair-mask-weight", type=float, default=1.0)
    parser.add_argument("--repair-control-weight", type=float, default=1.5)
    parser.add_argument(
        "--accept-threshold-grid",
        default=",".join(str(value) for value in DEFAULT_ACCEPT_THRESHOLD_GRID),
        help="Comma-separated accept thresholds for set-repair modes.",
    )
    parser.add_argument("--innovation-ridge", type=float, default=10.0)
    parser.add_argument("--local-files-only", choices=("true", "false"), default="true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--same-byte-budget", type=int, default=12)
    parser.add_argument("--min-accuracy-gap", type=float, default=0.0)
    parser.add_argument("--min-margin-gap", type=float, default=0.0)
    parser.add_argument(
        "--l2-grid",
        default=",".join(str(value) for value in DEFAULT_L2_GRID),
        help="Comma-separated ridge values.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_preflight(
        output_dir=args.output_dir,
        eval_path=args.eval_path,
        source_cache_path=args.source_cache_path,
        row_limit=int(args.row_limit),
        fit_fraction=float(args.fit_fraction),
        source_feature_mode=str(args.source_feature_mode),
        source_feature_dim=int(args.source_feature_dim),
        source_token_pool_size=int(args.source_token_pool_size),
        source_model=str(args.source_model),
        source_device=str(args.source_device),
        source_dtype=str(args.source_dtype),
        source_max_length=int(args.source_max_length),
        source_hidden_layer=int(args.source_hidden_layer),
        target_feature_dim=int(args.target_feature_dim),
        sketch_dim=int(args.sketch_dim),
        source_sketch_quantization=str(args.source_sketch_quantization),
        receiver_mode=str(args.receiver_mode),
        training_objective=str(args.training_objective),
        residual_fit_policy=str(args.residual_fit_policy),
        residual_desired_margin=float(args.residual_desired_margin),
        repair_mask_rounds=int(args.repair_mask_rounds),
        repair_mask_keep_prob=float(args.repair_mask_keep_prob),
        repair_matched_weight=float(args.repair_matched_weight),
        repair_mask_weight=float(args.repair_mask_weight),
        repair_control_weight=float(args.repair_control_weight),
        accept_threshold_grid=_parse_l2_grid(str(args.accept_threshold_grid)),
        innovation_ridge=float(args.innovation_ridge),
        local_files_only=str(args.local_files_only).lower() == "true",
        seed=int(args.seed),
        bootstrap_samples=int(args.bootstrap_samples),
        same_byte_budget=int(args.same_byte_budget),
        min_accuracy_gap=float(args.min_accuracy_gap),
        min_margin_gap=float(args.min_margin_gap),
        l2_grid=_parse_l2_grid(str(args.l2_grid)),
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
