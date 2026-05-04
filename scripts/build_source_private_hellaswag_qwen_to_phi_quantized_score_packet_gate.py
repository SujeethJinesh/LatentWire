from __future__ import annotations

"""Equal-byte quantized source-score packet gate for Qwen-to-Phi HellaSwag.

Reviewers flagged that raw source-score-vector quantization was still missing.
This gate compares fixed-byte quantized Qwen score packets against the current
fixed hybrid baseline on the frozen Qwen-to-Phi validation surface. It is a
baseline/falsification gate: if compact quantized scores solve the receiver
problem, LatentWire's packet claim must beat that; if they fail, score-vector
packets are weakened before returning to target-native latent receivers.
"""

import argparse
import datetime as dt
import json
import pathlib
import sys
import time
from typing import Any, Sequence

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_hellaswag_nonqwen_receiver_family_packet_gate as nonqwen  # noqa: E402
from scripts import build_source_private_hellaswag_official_train_receiver_calibration as official  # noqa: E402
from scripts import build_source_private_hellaswag_qwen_to_phi_denoising_syndrome_packet_gate as denoise  # noqa: E402
from scripts import build_source_private_hellaswag_qwen_to_phi_harm_controlled_bucket_gate as bucket_gate  # noqa: E402
from scripts import build_source_private_hellaswag_qwen_to_phi_heldout_uncertainty_router_gate as router_gate  # noqa: E402
from scripts import build_source_private_hellaswag_qwen_to_phi_official_train_receiver_calibrated_gate as receiver_gate  # noqa: E402
from scripts import build_source_private_hellaswag_qwen_to_phi_official_train_source_dictionary_gate as source_gate  # noqa: E402
from scripts import build_source_private_hellaswag_qwen_to_phi_oracle_switch_decomposition_gate as oracle  # noqa: E402
from scripts import build_source_private_hellaswag_score_packet_headroom as headroom  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_qwen_to_phi_quantized_score_packet_gate_20260504_validation1024_2048"
)
DEFAULT_TRAIN_PATH = source_gate.DEFAULT_TRAIN_PATH
DEFAULT_QWEN_TRAIN_CACHE_DIR = source_gate.DEFAULT_QWEN_TRAIN_CACHE_DIR
DEFAULT_SOURCE_SCORE_CACHE = source_gate.DEFAULT_SOURCE_SCORE_CACHE
DEFAULT_SAMPLE_SEEDS = source_gate.DEFAULT_SAMPLE_SEEDS
DEFAULT_SPLIT_SEEDS = source_gate.DEFAULT_SPLIT_SEEDS
DEFAULT_COMPONENT_RIDGES = source_gate.DEFAULT_COMPONENT_RIDGES
DEFAULT_TARGET_MODEL = pathlib.Path(nonqwen.DEFAULT_PHI_MODEL)
DEFAULT_PHI_TRAIN_SCORE_CACHE = receiver_gate.DEFAULT_OUTPUT / "phi_official_train_score_cache.json"
BOOTSTRAP_SAMPLES = denoise.BOOTSTRAP_SAMPLES
CANDIDATE_COUNT = 4
DEFAULT_BUDGET_BYTES = (1, 2, 4, 8)
CODECS = ("uniform_zscore", "rotated_uniform_zscore")
ALPHAS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    path = pathlib.Path(path)
    return path if path.is_absolute() else ROOT / path


def _display_path(path: pathlib.Path | str) -> str:
    path = _resolve(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _answers(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["answer_index"]) for row in rows], dtype=np.int64)


def _field_array(rows: Sequence[dict[str, Any]], field: str) -> np.ndarray:
    return np.asarray([int(row[field]) for row in rows], dtype=np.int64)


def _accuracy(predictions: np.ndarray, answers: np.ndarray) -> float:
    return float(np.mean(np.asarray(predictions, dtype=np.int64) == np.asarray(answers, dtype=np.int64)))


def _z_scores(scores: np.ndarray) -> np.ndarray:
    return router_gate._z_scores(scores)


def _top2(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return router_gate._top2(scores)


def _paired_ci(
    *,
    selected: np.ndarray,
    baseline: np.ndarray,
    answers: np.ndarray,
    seed: int,
    samples: int,
) -> dict[str, float | int]:
    return denoise._paired_ci(
        selected=np.asarray(selected, dtype=np.int64),
        baseline=np.asarray(baseline, dtype=np.int64),
        answers=np.asarray(answers, dtype=np.int64),
        seed=seed,
        samples=samples,
    )


def _orthogonal_matrix(dim: int = CANDIDATE_COUNT, seed: int = 20260504) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return (q * signs).astype(np.float64)


def _uniform_quantize_dequantize(values: np.ndarray, *, bits: int, clip: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    levels = 1 << int(bits)
    if levels <= 1:
        return np.zeros_like(values)
    clipped = np.clip(values, -float(clip), float(clip))
    scaled = (clipped + float(clip)) / (2.0 * float(clip))
    quantized = np.rint(scaled * float(levels - 1))
    return (quantized / float(levels - 1) * 2.0 * float(clip) - float(clip)).astype(np.float64)


def _codec_bits_per_coord(raw_payload_bytes: int) -> int:
    bits = int(raw_payload_bytes) * 8 // CANDIDATE_COUNT
    if bits <= 0:
        raise ValueError("raw_payload_bytes must provide at least one bit per candidate")
    return bits


def _framed_bytes(raw_payload_bytes: int) -> int:
    return int(raw_payload_bytes) + 3


def _reconstruct_scores(
    scores: np.ndarray,
    *,
    codec: str,
    raw_payload_bytes: int,
    clip: float,
    rotation: np.ndarray | None = None,
) -> np.ndarray:
    z = _z_scores(scores)
    bits = _codec_bits_per_coord(raw_payload_bytes)
    if codec == "uniform_zscore":
        return _uniform_quantize_dequantize(z, bits=bits, clip=clip)
    if codec == "rotated_uniform_zscore":
        if rotation is None:
            rotation = _orthogonal_matrix(z.shape[1])
        rotated = z @ rotation
        dequant = _uniform_quantize_dequantize(rotated, bits=bits, clip=clip)
        return (dequant @ rotation.T).astype(np.float64)
    raise ValueError(f"unknown codec {codec!r}")


def _predict_blend(
    *,
    reconstructed_qwen: np.ndarray,
    phi_scores: np.ndarray,
    hybrid: np.ndarray,
    model: dict[str, Any],
) -> np.ndarray:
    pz = _z_scores(phi_scores)
    blended = pz + float(model["alpha"]) * np.asarray(reconstructed_qwen, dtype=np.float64)
    proposal = np.argmax(blended, axis=1).astype(np.int64)
    confidence = np.max(blended, axis=1) - blended[np.arange(len(hybrid)), np.asarray(hybrid, dtype=np.int64)]
    predictions = np.asarray(hybrid, dtype=np.int64).copy()
    mask = confidence > float(model["threshold"])
    predictions[mask] = proposal[mask]
    return predictions.astype(np.int64)


def _evaluate(
    *,
    predictions: np.ndarray,
    baseline: np.ndarray,
    answers: np.ndarray,
    seed: int,
    samples: int,
) -> dict[str, float | int]:
    paired = _paired_ci(
        selected=predictions,
        baseline=baseline,
        answers=answers,
        seed=seed,
        samples=samples,
    )
    return {
        "accuracy": _accuracy(predictions, answers),
        "delta": float(paired["delta"]),
        "ci95_low": float(paired["ci95_low"]),
        "ci95_high": float(paired["ci95_high"]),
        "helps": int(paired["helps"]),
        "harms": int(paired["harms"]),
        "net_help": int(paired["helps"]) - int(paired["harms"]),
        "override_count": int(np.sum(predictions != baseline)),
    }


def _fit_model(
    *,
    reconstructed_qwen: np.ndarray,
    phi_scores: np.ndarray,
    hybrid: np.ndarray,
    answers: np.ndarray,
    dev_indices: np.ndarray,
    bootstrap_samples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dev_indices = np.asarray(dev_indices, dtype=np.int64)
    noop_model = {"alpha": 0.0, "threshold": float("inf"), "selection": "no_op"}
    noop_eval = _evaluate(
        predictions=hybrid[dev_indices],
        baseline=hybrid[dev_indices],
        answers=answers[dev_indices],
        seed=65060504,
        samples=max(200, min(bootstrap_samples, 1000)),
    )
    best_key: tuple[float, float, float, int, int, str] = (
        float(noop_eval["accuracy"]),
        float(noop_eval["delta"]),
        float(noop_eval["ci95_low"]),
        int(noop_eval["net_help"]),
        0,
        "no_op",
    )
    best_model = noop_model
    rows: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        blended = _z_scores(phi_scores[dev_indices]) + float(alpha) * reconstructed_qwen[dev_indices]
        confidence = np.max(blended, axis=1) - blended[np.arange(len(dev_indices)), hybrid[dev_indices]]
        thresholds = sorted(set(float(value) for value in confidence))
        thresholds.append(float(np.max(confidence) + max(1e-9, abs(float(np.max(confidence))) * 1e-6)))
        if len(thresholds) > 96:
            finite = thresholds[:-1]
            thresholds = sorted({finite[int(round(q * (len(finite) - 1)))] for q in np.linspace(0.0, 1.0, 81)})
            thresholds.append(float(np.max(finite) + max(1e-9, abs(float(np.max(finite))) * 1e-6)))
        for threshold in thresholds:
            model = {"alpha": float(alpha), "threshold": float(threshold)}
            predictions = _predict_blend(
                reconstructed_qwen=reconstructed_qwen[dev_indices],
                phi_scores=phi_scores[dev_indices],
                hybrid=hybrid[dev_indices],
                model=model,
            )
            metrics = _evaluate(
                predictions=predictions,
                baseline=hybrid[dev_indices],
                answers=answers[dev_indices],
                seed=65160504 + len(rows),
                samples=max(200, min(bootstrap_samples, 1000)),
            )
            row = {
                "alpha": float(alpha),
                "threshold": float(threshold),
                "official_dev_accuracy": metrics["accuracy"],
                "official_dev_delta_vs_hybrid": metrics["delta"],
                "official_dev_ci95_low_vs_hybrid": metrics["ci95_low"],
                "official_dev_ci95_high_vs_hybrid": metrics["ci95_high"],
                "official_dev_helps_vs_hybrid": metrics["helps"],
                "official_dev_harms_vs_hybrid": metrics["harms"],
                "official_dev_net_help": metrics["net_help"],
                "official_dev_override_count": metrics["override_count"],
            }
            rows.append(row)
            key = (
                float(metrics["accuracy"]),
                float(metrics["delta"]),
                float(metrics["ci95_low"]),
                int(metrics["net_help"]),
                -int(metrics["override_count"]),
                json.dumps({"alpha": alpha, "threshold": threshold}, sort_keys=True),
            )
            if key > best_key:
                best_key = key
                best_model = model
    return best_model, sorted(rows, key=lambda item: item["official_dev_accuracy"], reverse=True)


def _method_row(
    *,
    name: str,
    rows: Sequence[dict[str, Any]],
    predictions: np.ndarray,
    fixed_hybrid: np.ndarray,
    candidate_only: np.ndarray,
    target_only: np.ndarray,
    bootstrap_samples: int,
    raw_payload_bytes: int,
    framed_record_bytes: int,
    source_private: bool,
    raw_score_vector_exposed: bool = False,
    quantized_score_vector_exposed: bool = False,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    answers = _answers(rows)
    vs_hybrid = _paired_ci(
        selected=predictions,
        baseline=fixed_hybrid,
        answers=answers,
        seed=66060504 + sum(ord(ch) for ch in name),
        samples=bootstrap_samples,
    )
    vs_candidate = _paired_ci(
        selected=predictions,
        baseline=candidate_only,
        answers=answers,
        seed=66060604 + sum(ord(ch) for ch in name),
        samples=bootstrap_samples,
    )
    vs_target = _paired_ci(
        selected=predictions,
        baseline=target_only,
        answers=answers,
        seed=66060704 + sum(ord(ch) for ch in name),
        samples=bootstrap_samples,
    )
    return {
        "method": name,
        "eval_rows": len(rows),
        "accuracy": _accuracy(predictions, answers),
        "fixed_hybrid_accuracy": _accuracy(fixed_hybrid, answers),
        "candidate_only_accuracy": _accuracy(candidate_only, answers),
        "target_only_accuracy": _accuracy(target_only, answers),
        "delta_vs_fixed_hybrid": float(vs_hybrid["delta"]),
        "ci95_low_vs_fixed_hybrid": float(vs_hybrid["ci95_low"]),
        "ci95_high_vs_fixed_hybrid": float(vs_hybrid["ci95_high"]),
        "helps_vs_fixed_hybrid": int(vs_hybrid["helps"]),
        "harms_vs_fixed_hybrid": int(vs_hybrid["harms"]),
        "delta_vs_candidate_only": float(vs_candidate["delta"]),
        "ci95_low_vs_candidate_only": float(vs_candidate["ci95_low"]),
        "delta_vs_target_only": float(vs_target["delta"]),
        "ci95_low_vs_target_only": float(vs_target["ci95_low"]),
        "override_count_vs_fixed_hybrid": int(np.sum(predictions != fixed_hybrid)),
        "override_rate_vs_fixed_hybrid": float(np.mean(predictions != fixed_hybrid)),
        "raw_payload_bytes": int(raw_payload_bytes),
        "framed_record_bytes": int(framed_record_bytes),
        "source_private": bool(source_private),
        "source_text_exposed": False,
        "source_kv_exposed": False,
        "source_hidden_vector_exposed": False,
        "raw_source_score_or_logit_vector_exposed": bool(raw_score_vector_exposed),
        "quantized_source_score_vector_exposed": bool(quantized_score_vector_exposed),
        "details": json.dumps(details or {}, sort_keys=True),
    }


def _source_pair_oracle(source_top1: np.ndarray, source_top2: np.ndarray, answers: np.ndarray) -> np.ndarray:
    return np.where(source_top1 == answers, source_top1, np.where(source_top2 == answers, source_top2, source_top1)).astype(np.int64)


def _control_scores(
    *,
    qwen_scores: np.ndarray,
    phi_scores: np.ndarray,
    condition: str,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if condition == "matched":
        return qwen_scores
    if condition == "source_row_shuffle":
        return qwen_scores[rng.permutation(len(qwen_scores))]
    if condition == "candidate_roll_source":
        return np.roll(qwen_scores, shift=1, axis=1)
    if condition == "code_value_permutation":
        return qwen_scores[:, rng.permutation(qwen_scores.shape[1])]
    if condition == "target_derived_source_packet":
        return phi_scores
    if condition == "zero_source_packet":
        return np.zeros_like(qwen_scores)
    if condition == "random_same_byte_source":
        return qwen_scores[rng.integers(0, len(qwen_scores), size=len(qwen_scores))]
    raise ValueError(f"unknown condition {condition!r}")


def _slice_rows(
    *,
    rows: Sequence[dict[str, Any]],
    predictions: np.ndarray,
    fixed_hybrid: np.ndarray,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    return router_gate._slice_rows(
        rows=rows,
        predictions=predictions,
        fixed_hybrid=fixed_hybrid,
        bootstrap_samples=bootstrap_samples,
    )


def _write_markdown(path: pathlib.Path | str, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Qwen-To-Phi Quantized Score Packet Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- calibration rows: `{h['official_train_calibration_rows']}`",
        f"- eval rows: `{h['eval_rows']}`",
        f"- fixed hybrid accuracy: `{h['fixed_hybrid_accuracy']:.6f}`",
        f"- best quantized score packet: `{h['best_quantized_method']}`",
        f"- best quantized accuracy: `{h['best_quantized_accuracy']:.6f}`",
        f"- best quantized delta: `{h['best_quantized_delta_vs_fixed_hybrid']:.6f}`",
        f"- best quantized CI95 low: `{h['best_quantized_ci95_low_vs_fixed_hybrid']:.6f}`",
        f"- raw source-score fusion accuracy: `{h['raw_source_score_logit_fusion_accuracy']:.6f}`",
        f"- source top1/top2 oracle accuracy: `{h['source_top1_or_top2_oracle_accuracy']:.6f}`",
        f"- best destructive: `{h['best_destructive_control_name']}` (`{h['best_destructive_control_accuracy']:.6f}`)",
        "",
        "## Budget Rows",
        "",
        "| Raw bytes | Framed bytes | Best method | Accuracy | Delta | CI95 low | Overrides |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in payload.get("budget_rows", []):
        lines.append(
            "| "
            f"{row['raw_payload_bytes']} | "
            f"{row['framed_record_bytes']} | "
            f"`{row['best_method']}` | "
            f"{row['best_accuracy']:.6f} | "
            f"{row['best_delta_vs_fixed_hybrid']:.6f} | "
            f"{row['best_ci95_low_vs_fixed_hybrid']:.6f} | "
            f"{row['best_override_count_vs_fixed_hybrid']} |"
        )
    lines.extend(
        [
            "",
            "## Slice Rows",
            "",
            "| Slice | Rows | Method acc. | Fixed hybrid acc. | Delta | CI95 low | Helps | Harms |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload.get("slice_rows", []):
        lines.append(
            "| "
            f"{row['slice_start']} | "
            f"{row['eval_rows']} | "
            f"{row['method_accuracy']:.6f} | "
            f"{row['fixed_hybrid_accuracy']:.6f} | "
            f"{row['delta_vs_fixed_hybrid']:.6f} | "
            f"{row['ci95_low_vs_fixed_hybrid']:.6f} | "
            f"{row['helps_vs_fixed_hybrid']} | "
            f"{row['harms_vs_fixed_hybrid']} |"
        )
    lines.extend(
        [
            "",
        "## Interpretation",
        "",
        payload["interpretation"],
        "",
        "## Lay Explanation",
        "",
        payload["lay_explanation"],
        ]
    )
    _resolve(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_gate(
    *,
    output_dir: pathlib.Path | str = DEFAULT_OUTPUT,
    train_path: pathlib.Path | str = DEFAULT_TRAIN_PATH,
    qwen_train_cache_dir: pathlib.Path | str = DEFAULT_QWEN_TRAIN_CACHE_DIR,
    phi_train_score_cache: pathlib.Path | str = DEFAULT_PHI_TRAIN_SCORE_CACHE,
    source_score_cache: pathlib.Path | str = DEFAULT_SOURCE_SCORE_CACHE,
    slices: tuple[dict[str, Any], ...] = denoise.DEFAULT_SLICES,
    sample_seeds: tuple[int, ...] = DEFAULT_SAMPLE_SEEDS,
    split_seeds: tuple[int, ...] = DEFAULT_SPLIT_SEEDS,
    component_ridges: tuple[float, ...] = DEFAULT_COMPONENT_RIDGES,
    fit_rows_per_slice: int = denoise.FIT_ROWS_PER_SLICE,
    select_rows_per_slice: int = denoise.SELECT_ROWS_PER_SLICE,
    train_hidden_rows: int = 512,
    dev_fraction: float = 0.25,
    max_calibration_rows: int | None = None,
    target_model: pathlib.Path | str = DEFAULT_TARGET_MODEL,
    target_device: str = "mps",
    target_dtype: str = "float16",
    target_max_length: int = 256,
    target_normalization: str = "mean",
    target_prompt_mode: str = "continuation",
    local_files_only: bool = True,
    budget_bytes: tuple[int, ...] = DEFAULT_BUDGET_BYTES,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    run_date: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_date = run_date or dt.date.today().isoformat()
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = source_gate._build_qwen_oob_calibration(
        train_path=train_path,
        qwen_train_cache_dir=qwen_train_cache_dir,
        sample_seeds=sample_seeds,
        split_seeds=split_seeds,
        component_ridges=component_ridges,
        train_hidden_rows=train_hidden_rows,
        dev_fraction=dev_fraction,
    )
    calibration = receiver_gate._limit_calibration(calibration, max_calibration_rows)
    fit_indices, dev_indices = official._official_split_indices(
        len(calibration["rows"]),
        dev_fraction=dev_fraction,
        seed=4242,
    )
    calibration_arc_rows, calibration_row_meta = receiver_gate._arc_rows_for_calibration(
        train_path=train_path,
        calibration_rows=calibration["rows"],
    )
    phi_cache_path = _resolve(phi_train_score_cache)
    phi_cache_existed = phi_cache_path.exists()
    phi_scores, _, phi_state, phi_sha = receiver_gate._load_or_build_phi_scores(
        rows=calibration_arc_rows,
        score_cache=phi_cache_path,
        target_model=target_model,
        target_device=target_device,
        target_dtype=target_dtype,
        target_max_length=target_max_length,
        target_normalization=target_normalization,
        target_prompt_mode=target_prompt_mode,
        local_files_only=local_files_only,
    )
    train_clip = float(np.quantile(np.abs(_z_scores(calibration["scores"])[fit_indices]).reshape(-1), 0.995))
    train_clip = max(1.0, min(4.0, train_clip))
    rotation = _orthogonal_matrix(CANDIDATE_COUNT, seed=20260504)
    rows, metadata = denoise._load_rows(
        slices=slices,
        fit_rows_per_slice=fit_rows_per_slice,
        select_rows_per_slice=select_rows_per_slice,
    )
    source_score_metadata = oracle._load_source_scores(rows, source_score_cache)
    eval_rows = [row for row in rows if row["_split"] == "eval"]
    eval_scores = np.asarray([row["qwen_source_scores"] for row in eval_rows], dtype=np.float64)
    eval_phi_scores = np.asarray([row["phi_target_scores"] for row in eval_rows], dtype=np.float64)
    fixed_hybrid = _field_array(eval_rows, "qwen_hybrid_prediction")
    candidate_only = _field_array(eval_rows, "selected_prediction")
    target_only = _field_array(eval_rows, "phi_target_prediction")
    answers = _answers(eval_rows)
    config_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    quantized_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    raw_model = router_gate._fit_logit_fusion(
        qwen_scores=calibration["scores"],
        phi_scores=phi_scores,
        hybrid=calibration["hybrid"],
        answers=calibration["answers"],
        dev_indices=dev_indices,
    )
    raw_predictions = router_gate._predict_logit_fusion(eval_scores, eval_phi_scores, raw_model)
    method_rows.extend(
        [
            _method_row(
                name="fixed_hybrid_vote_on_score_agreement",
                rows=eval_rows,
                predictions=fixed_hybrid,
                fixed_hybrid=fixed_hybrid,
                candidate_only=candidate_only,
                target_only=target_only,
                bootstrap_samples=bootstrap_samples,
                raw_payload_bytes=1,
                framed_record_bytes=4,
                source_private=True,
            ),
            _method_row(
                name="qwen_candidate_only",
                rows=eval_rows,
                predictions=candidate_only,
                fixed_hybrid=fixed_hybrid,
                candidate_only=candidate_only,
                target_only=target_only,
                bootstrap_samples=bootstrap_samples,
                raw_payload_bytes=1,
                framed_record_bytes=4,
                source_private=True,
            ),
            _method_row(
                name="phi_target_only",
                rows=eval_rows,
                predictions=target_only,
                fixed_hybrid=fixed_hybrid,
                candidate_only=candidate_only,
                target_only=target_only,
                bootstrap_samples=bootstrap_samples,
                raw_payload_bytes=0,
                framed_record_bytes=0,
                source_private=True,
            ),
            _method_row(
                name="raw_source_score_logit_fusion_control",
                rows=eval_rows,
                predictions=raw_predictions,
                fixed_hybrid=fixed_hybrid,
                candidate_only=candidate_only,
                target_only=target_only,
                bootstrap_samples=bootstrap_samples,
                raw_payload_bytes=32,
                framed_record_bytes=32,
                source_private=False,
                raw_score_vector_exposed=True,
                details={"alpha": raw_model["alpha"]},
            ),
        ]
    )
    if eval_rows and "source_rank_only_bagged_prediction" in eval_rows[0]:
        method_rows.append(
            _method_row(
                name="source_rank_only_bagged_control",
                rows=eval_rows,
                predictions=_field_array(eval_rows, "source_rank_only_bagged_prediction"),
                fixed_hybrid=fixed_hybrid,
                candidate_only=candidate_only,
                target_only=target_only,
                bootstrap_samples=bootstrap_samples,
                raw_payload_bytes=1,
                framed_record_bytes=4,
                source_private=True,
                details={"existing_audit_control": True},
            )
        )
    source_top1, source_top2 = _top2(eval_scores)
    method_rows.extend(
        [
            _method_row(
                name="source_top1_label_control",
                rows=eval_rows,
                predictions=source_top1,
                fixed_hybrid=fixed_hybrid,
                candidate_only=candidate_only,
                target_only=target_only,
                bootstrap_samples=bootstrap_samples,
                raw_payload_bytes=1,
                framed_record_bytes=4,
                source_private=True,
            ),
            _method_row(
                name="source_top1_top2_oracle_diagnostic",
                rows=eval_rows,
                predictions=_source_pair_oracle(source_top1, source_top2, answers),
                fixed_hybrid=fixed_hybrid,
                candidate_only=candidate_only,
                target_only=target_only,
                bootstrap_samples=bootstrap_samples,
                raw_payload_bytes=0,
                framed_record_bytes=0,
                source_private=True,
                details={"oracle": True, "not_promotable": True},
            ),
        ]
    )
    for codec in CODECS:
        for raw_bytes in budget_bytes:
            train_recon = _reconstruct_scores(
                calibration["scores"],
                codec=codec,
                raw_payload_bytes=int(raw_bytes),
                clip=train_clip,
                rotation=rotation,
            )
            model, config = _fit_model(
                reconstructed_qwen=train_recon,
                phi_scores=phi_scores,
                hybrid=calibration["hybrid"],
                answers=calibration["answers"],
                dev_indices=dev_indices,
                bootstrap_samples=bootstrap_samples,
            )
            for row in config:
                row.update({"codec": codec, "raw_payload_bytes": int(raw_bytes), "bits_per_coord": _codec_bits_per_coord(int(raw_bytes))})
            config_rows.extend(config)
            eval_recon = _reconstruct_scores(
                eval_scores,
                codec=codec,
                raw_payload_bytes=int(raw_bytes),
                clip=train_clip,
                rotation=rotation,
            )
            predictions = _predict_blend(
                reconstructed_qwen=eval_recon,
                phi_scores=eval_phi_scores,
                hybrid=fixed_hybrid,
                model=model,
            )
            row = _method_row(
                name=f"quantized_score_packet_{codec}_{raw_bytes}B",
                rows=eval_rows,
                predictions=predictions,
                fixed_hybrid=fixed_hybrid,
                candidate_only=candidate_only,
                target_only=target_only,
                bootstrap_samples=bootstrap_samples,
                raw_payload_bytes=int(raw_bytes),
                framed_record_bytes=_framed_bytes(int(raw_bytes)),
                source_private=True,
                quantized_score_vector_exposed=True,
                details={
                    "codec": codec,
                    "model": model,
                    "clip": train_clip,
                    "bits_per_coord": _codec_bits_per_coord(int(raw_bytes)),
                    "rotation_seed": 20260504 if codec == "rotated_uniform_zscore" else None,
                },
            )
            method_rows.append(row)
            quantized_rows.append(row)
            for condition in (
                "source_row_shuffle",
                "candidate_roll_source",
                "code_value_permutation",
                "target_derived_source_packet",
                "zero_source_packet",
                "random_same_byte_source",
            ):
                c_scores = _control_scores(
                    qwen_scores=eval_scores,
                    phi_scores=eval_phi_scores,
                    condition=condition,
                    seed=20260504 + sum(ord(ch) for ch in f"{codec}-{raw_bytes}-{condition}"),
                )
                c_recon = _reconstruct_scores(
                    c_scores,
                    codec=codec,
                    raw_payload_bytes=int(raw_bytes),
                    clip=train_clip,
                    rotation=rotation,
                )
                c_predictions = _predict_blend(
                    reconstructed_qwen=c_recon,
                    phi_scores=eval_phi_scores,
                    hybrid=fixed_hybrid,
                    model=model,
                )
                control_row = _method_row(
                    name=f"{condition}_{codec}_{raw_bytes}B_control",
                    rows=eval_rows,
                    predictions=c_predictions,
                    fixed_hybrid=fixed_hybrid,
                    candidate_only=candidate_only,
                    target_only=target_only,
                    bootstrap_samples=bootstrap_samples,
                    raw_payload_bytes=int(raw_bytes),
                    framed_record_bytes=_framed_bytes(int(raw_bytes)),
                    source_private=True,
                    quantized_score_vector_exposed=True,
                    details={"condition": condition, "codec": codec, "model": model},
                )
                method_rows.append(control_row)
                control_rows.append(control_row)
    method_rows = sorted(method_rows, key=lambda row: row["accuracy"], reverse=True)
    quantized_rows = sorted(quantized_rows, key=lambda row: row["accuracy"], reverse=True)
    best_quantized = quantized_rows[0]
    best_destructive = max(control_rows, key=lambda row: row["accuracy"])
    best_details = json.loads(best_quantized["details"])
    best_predictions = _predict_blend(
        reconstructed_qwen=_reconstruct_scores(
            eval_scores,
            codec=best_details["codec"],
            raw_payload_bytes=int(best_quantized["raw_payload_bytes"]),
            clip=train_clip,
            rotation=rotation,
        ),
        phi_scores=eval_phi_scores,
        hybrid=fixed_hybrid,
        model=best_details["model"],
    )
    budget_rows = []
    for raw_bytes in sorted({int(row["raw_payload_bytes"]) for row in quantized_rows}):
        budget_members = [row for row in quantized_rows if int(row["raw_payload_bytes"]) == raw_bytes]
        best_budget = max(budget_members, key=lambda row: row["accuracy"])
        budget_rows.append(
            {
                "raw_payload_bytes": raw_bytes,
                "framed_record_bytes": int(best_budget["framed_record_bytes"]),
                "best_method": best_budget["method"],
                "best_accuracy": best_budget["accuracy"],
                "best_delta_vs_fixed_hybrid": best_budget["delta_vs_fixed_hybrid"],
                "best_ci95_low_vs_fixed_hybrid": best_budget["ci95_low_vs_fixed_hybrid"],
                "best_override_count_vs_fixed_hybrid": best_budget["override_count_vs_fixed_hybrid"],
            }
        )
    slice_rows = _slice_rows(
        rows=eval_rows,
        predictions=best_predictions,
        fixed_hybrid=fixed_hybrid,
        bootstrap_samples=bootstrap_samples,
    )
    train_content_ids = {row.content_id for row in calibration_arc_rows}
    eval_content_ids = {str(row.get("content_id", row["row_id"])) for row in eval_rows}
    pass_gate = bool(
        best_quantized["delta_vs_fixed_hybrid"] >= 0.005
        and best_quantized["ci95_low_vs_fixed_hybrid"] > 0.0
        and best_quantized["ci95_low_vs_candidate_only"] > 0.0
        and best_quantized["accuracy"] > best_destructive["accuracy"]
        and all(row["delta_vs_fixed_hybrid"] >= 0.0 for row in slice_rows)
        and int(best_quantized["helps_vs_fixed_hybrid"]) > int(best_quantized["harms_vs_fixed_hybrid"])
        and len(train_content_ids & eval_content_ids) == 0
    )
    raw_row = next(row for row in method_rows if row["method"] == "raw_source_score_logit_fusion_control")
    oracle_row = next(row for row in method_rows if row["method"] == "source_top1_top2_oracle_diagnostic")
    headline = {
        "official_train_calibration_rows": int(len(calibration["rows"])),
        "official_train_fit_rows": int(len(fit_indices)),
        "official_train_dev_rows": int(len(dev_indices)),
        "official_train_content_overlap_with_eval": int(len(train_content_ids & eval_content_ids)),
        "eval_rows": len(eval_rows),
        "score_clip": train_clip,
        "fixed_hybrid_accuracy": next(row for row in method_rows if row["method"] == "fixed_hybrid_vote_on_score_agreement")["accuracy"],
        "candidate_only_accuracy": next(row for row in method_rows if row["method"] == "qwen_candidate_only")["accuracy"],
        "phi_target_accuracy": next(row for row in method_rows if row["method"] == "phi_target_only")["accuracy"],
        "best_quantized_method": best_quantized["method"],
        "best_quantized_accuracy": best_quantized["accuracy"],
        "best_quantized_delta_vs_fixed_hybrid": best_quantized["delta_vs_fixed_hybrid"],
        "best_quantized_ci95_low_vs_fixed_hybrid": best_quantized["ci95_low_vs_fixed_hybrid"],
        "best_quantized_helps_vs_fixed_hybrid": best_quantized["helps_vs_fixed_hybrid"],
        "best_quantized_harms_vs_fixed_hybrid": best_quantized["harms_vs_fixed_hybrid"],
        "best_quantized_raw_payload_bytes": best_quantized["raw_payload_bytes"],
        "best_quantized_framed_record_bytes": best_quantized["framed_record_bytes"],
        "best_destructive_control_name": best_destructive["method"],
        "best_destructive_control_accuracy": best_destructive["accuracy"],
        "raw_source_score_logit_fusion_accuracy": raw_row["accuracy"],
        "raw_source_score_logit_fusion_delta_vs_fixed_hybrid": raw_row["delta_vs_fixed_hybrid"],
        "source_top1_or_top2_oracle_accuracy": oracle_row["accuracy"],
        "source_top1_or_top2_oracle_delta_vs_fixed_hybrid": oracle_row["delta_vs_fixed_hybrid"],
        "phi_train_score_cache_hit": bool(phi_cache_existed),
        "native_systems_claim_allowed": False,
    }
    payload = {
        "gate": "source_private_hellaswag_qwen_to_phi_quantized_score_packet_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": pass_gate,
        "pass_rule": (
            "Pass only if the official-train selected quantized source-score packet beats fixed hybrid by "
            "at least 0.005 with positive paired CI, beats candidate-only with positive paired CI, beats all "
            "source-destroying controls, is nonnegative on all eval slices, helps more than harms, and has "
            "zero official-train/eval content overlap."
        ),
        "headline": headline,
        "packet_contract": {
            "receiver_visible_payload": (
                "Row-wise quantized 4-candidate Qwen score vector reconstructed from a fixed-byte packet; "
                "Phi-local score vector is decoder side information."
            ),
            "raw_payload_byte_budgets": list(budget_bytes),
            "framed_record_bytes": "raw_payload_bytes + 3",
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_vector_transmitted": False,
            "raw_scores_or_logits_transmitted": False,
            "quantized_source_score_vector_transmitted": True,
            "codecs": list(CODECS),
            "turboquant_inspiration": (
                "rotated_uniform_zscore uses a deterministic random orthogonal rotation followed by scalar "
                "coordinate quantization, mirroring the random-rotation/scalar-quantization principle used by "
                "TurboQuant/QJL-style vector compression, but applied only to 4-way MCQ score packets."
            ),
        },
        "calibration_row_metadata": calibration_row_meta,
        "source_score_metadata": source_score_metadata,
        "slice_metadata": metadata,
        "sample_cache_rows": calibration["sample_cache_rows"],
        "component_rows": calibration["component_rows"],
        "method_rows": method_rows,
        "config_rows": config_rows,
        "budget_rows": budget_rows,
        "slice_rows": slice_rows,
        "systems_packet_sideband": {
            "best_raw_payload_bytes_per_request": int(best_quantized["raw_payload_bytes"]),
            "best_framed_record_bytes_per_request": int(best_quantized["framed_record_bytes"]),
            "fit_and_eval_wall_time_s": float(time.perf_counter() - started),
            "phi_train_score_cache_hit": bool(phi_cache_existed),
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_exposed": False,
            "raw_score_vector_exposed": False,
            "quantized_score_vector_exposed": True,
            "native_gpu_claims_allowed": False,
        },
        "inputs": {
            "train_path": _display_path(train_path),
            "train_sha256": official._sha256_file(train_path),
            "qwen_train_cache_dir": _display_path(qwen_train_cache_dir),
            "phi_train_score_cache": _display_path(phi_cache_path),
            "phi_train_score_cache_sha256": phi_sha,
            "phi_train_score_model": phi_state,
            "source_score_cache": denoise._display_path(source_score_cache),
            "source_score_cache_sha256": denoise._sha256_file(source_score_cache),
        },
        "interpretation": (
            "This gate closes the reviewer-requested calibrated source-score quantization baseline. It tests "
            "whether equal-byte score packets recover the large source top1/top2 oracle headroom before we "
            "spend more effort on heavier target-native latent receivers."
        ),
        "lay_explanation": (
            "Instead of sending Qwen's full private scores, we squashed its four answer scores into tiny "
            "1-8 byte messages and let Phi combine the reconstructed scores with its own scores. If this works, "
            "then a simple score message is enough; if it fails, the problem needs richer latent evidence."
        ),
    }
    bucket_gate._write_json(output_dir / "hellaswag_qwen_to_phi_quantized_score_packet_gate.json", payload)
    bucket_gate._write_csv(output_dir / "method_rows.csv", method_rows)
    bucket_gate._write_csv(output_dir / "config_rows.csv", config_rows)
    bucket_gate._write_csv(output_dir / "budget_rows.csv", budget_rows)
    bucket_gate._write_csv(output_dir / "slice_rows.csv", slice_rows)
    _write_markdown(output_dir / "hellaswag_qwen_to_phi_quantized_score_packet_gate.md", payload)
    bucket_gate._write_json(
        output_dir / "manifest.json",
        {
            "gate": payload["gate"],
            "date": run_date,
            "outputs": [
                "hellaswag_qwen_to_phi_quantized_score_packet_gate.json",
                "hellaswag_qwen_to_phi_quantized_score_packet_gate.md",
                "method_rows.csv",
                "config_rows.csv",
                "budget_rows.csv",
                "slice_rows.csv",
            ],
            "headline": headline,
            "inputs": payload["inputs"],
        },
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-path", type=pathlib.Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--qwen-train-cache-dir", type=pathlib.Path, default=DEFAULT_QWEN_TRAIN_CACHE_DIR)
    parser.add_argument("--phi-train-score-cache", type=pathlib.Path, default=DEFAULT_PHI_TRAIN_SCORE_CACHE)
    parser.add_argument("--source-score-cache", type=pathlib.Path, default=DEFAULT_SOURCE_SCORE_CACHE)
    parser.add_argument("--sample-seeds", type=official._parse_int_tuple, default=DEFAULT_SAMPLE_SEEDS)
    parser.add_argument("--split-seeds", type=official._parse_int_tuple, default=DEFAULT_SPLIT_SEEDS)
    parser.add_argument("--component-ridges", type=official._parse_float_tuple, default=DEFAULT_COMPONENT_RIDGES)
    parser.add_argument("--fit-rows-per-slice", type=int, default=denoise.FIT_ROWS_PER_SLICE)
    parser.add_argument("--select-rows-per-slice", type=int, default=denoise.SELECT_ROWS_PER_SLICE)
    parser.add_argument("--train-hidden-rows", type=int, default=512)
    parser.add_argument("--dev-fraction", type=float, default=0.25)
    parser.add_argument("--max-calibration-rows", type=int, default=None)
    parser.add_argument("--target-model", type=pathlib.Path, default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--target-device", type=str, default="mps")
    parser.add_argument("--target-dtype", type=str, default="float16")
    parser.add_argument("--target-max-length", type=int, default=256)
    parser.add_argument("--target-normalization", type=str, default="mean")
    parser.add_argument("--target-prompt-mode", type=str, default="continuation")
    parser.add_argument("--no-local-files-only", action="store_true")
    parser.add_argument("--budget-bytes", type=official._parse_int_tuple, default=DEFAULT_BUDGET_BYTES)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--run-date", type=str, default=None)
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        train_path=args.train_path,
        qwen_train_cache_dir=args.qwen_train_cache_dir,
        phi_train_score_cache=args.phi_train_score_cache,
        source_score_cache=args.source_score_cache,
        sample_seeds=args.sample_seeds,
        split_seeds=args.split_seeds,
        component_ridges=args.component_ridges,
        fit_rows_per_slice=args.fit_rows_per_slice,
        select_rows_per_slice=args.select_rows_per_slice,
        train_hidden_rows=args.train_hidden_rows,
        dev_fraction=args.dev_fraction,
        max_calibration_rows=args.max_calibration_rows,
        target_model=args.target_model,
        target_device=args.target_device,
        target_dtype=args.target_dtype,
        target_max_length=args.target_max_length,
        target_normalization=args.target_normalization,
        target_prompt_mode=args.target_prompt_mode,
        local_files_only=not args.no_local_files_only,
        budget_bytes=args.budget_bytes,
        bootstrap_samples=args.bootstrap_samples,
        run_date=args.run_date,
    )
    h = payload["headline"]
    print(
        json.dumps(
            {
                "pass_gate": payload["pass_gate"],
                "fixed_hybrid_accuracy": h["fixed_hybrid_accuracy"],
                "best_quantized_method": h["best_quantized_method"],
                "best_quantized_accuracy": h["best_quantized_accuracy"],
                "best_quantized_delta_vs_fixed_hybrid": h["best_quantized_delta_vs_fixed_hybrid"],
                "best_quantized_ci95_low_vs_fixed_hybrid": h["best_quantized_ci95_low_vs_fixed_hybrid"],
                "raw_source_score_logit_fusion_accuracy": h["raw_source_score_logit_fusion_accuracy"],
                "source_top1_or_top2_oracle_accuracy": h["source_top1_or_top2_oracle_accuracy"],
                "best_destructive_control_name": h["best_destructive_control_name"],
                "best_destructive_control_accuracy": h["best_destructive_control_accuracy"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
