from __future__ import annotations

"""Audit whether HellaSwag hidden-innovation packets need the debug byte."""

import argparse
import datetime as dt
import hashlib
import json
import math
import pathlib
import sys
import time
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_OUTPUT = pathlib.Path("results/source_private_hellaswag_minimal_packet_compaction_20260502")
DEFAULT_QWEN_ARTIFACT = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_global_stability_20260502/"
    "hellaswag_hidden_innovation_global_stability.json"
)
DEFAULT_QWEN_PREDICTIONS = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_global_stability_20260502/predictions.jsonl"
)
DEFAULT_TINY_ARTIFACT = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_eval_full_stress_20260502_tinyllama_train512_validation0_10042/"
    "hellaswag_hidden_innovation_eval_slice_stress.json"
)
DEFAULT_TINY_PREDICTIONS = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_eval_full_stress_20260502_tinyllama_train512_validation0_10042/"
    "bagged_gate/predictions.jsonl"
)
DEFAULT_CANDIDATE_COUNT = 4
FRAME_BYTES = 3
CACHELINE_BYTES = 64
DMA_BYTES = 128


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path | str) -> str:
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


def _read_json(path: pathlib.Path | str) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _read_jsonl(path: pathlib.Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with _resolve(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _minimal_payload_bits(candidate_count: int) -> int:
    if candidate_count < 2:
        raise ValueError("candidate_count must be at least 2")
    return int(math.ceil(math.log2(candidate_count)))


def _minimal_payload_bytes(candidate_count: int) -> int:
    return max(1, int(math.ceil(_minimal_payload_bits(candidate_count) / 8.0)))


def _encode_candidate(candidate_id: int, *, candidate_count: int = DEFAULT_CANDIDATE_COUNT) -> int:
    if not 0 <= int(candidate_id) < int(candidate_count):
        raise ValueError(f"candidate id {candidate_id} is outside [0, {candidate_count})")
    return int(candidate_id)


def _decode_candidate(packet_byte: int, *, candidate_count: int = DEFAULT_CANDIDATE_COUNT) -> int:
    decoded = int(packet_byte)
    if not 0 <= decoded < int(candidate_count):
        raise ValueError(f"packet byte {packet_byte} decodes outside [0, {candidate_count})")
    return decoded


def _round_up(value: int, granularity: int) -> int:
    return int(math.ceil(int(value) / float(granularity)) * granularity)


def _accuracy(predictions: np.ndarray, answers: np.ndarray) -> float:
    return float(np.mean(predictions.astype(np.int64) == answers.astype(np.int64)))


def _paired_ci(
    *,
    selected: np.ndarray,
    baseline: np.ndarray,
    answers: np.ndarray,
    seed: int,
    samples: int,
) -> dict[str, float]:
    delta = (selected == answers).astype(np.float64) - (baseline == answers).astype(np.float64)
    if samples <= 0:
        mean = float(np.mean(delta))
        return {"delta": mean, "ci95_low": mean, "ci95_high": mean}
    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, len(delta), size=(int(samples), len(delta)))
    boot = np.mean(delta[boot_indices], axis=1)
    return {
        "delta": float(np.mean(delta)),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
    }


def _compact_predictions(
    predictions: np.ndarray,
    *,
    candidate_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    encoded = np.asarray(
        [_encode_candidate(int(item), candidate_count=candidate_count) for item in predictions],
        dtype=np.uint8,
    )
    decoded = np.asarray(
        [_decode_candidate(int(item), candidate_count=candidate_count) for item in encoded],
        dtype=np.int64,
    )
    return encoded, decoded


def _packet_accounting(
    *,
    row_count: int,
    original_raw_bytes: int,
    candidate_count: int,
) -> dict[str, Any]:
    compact_raw = _minimal_payload_bytes(candidate_count)
    original_framed = int(original_raw_bytes) + FRAME_BYTES
    compact_framed = int(compact_raw) + FRAME_BYTES
    return {
        "candidate_count": int(candidate_count),
        "theoretical_payload_bits": _minimal_payload_bits(candidate_count),
        "original_raw_payload_bytes_per_request": int(original_raw_bytes),
        "compact_raw_payload_bytes_per_request": int(compact_raw),
        "original_framed_record_bytes_per_request": int(original_framed),
        "compact_framed_record_bytes_per_request": int(compact_framed),
        "raw_payload_reduction_fraction": float(
            (int(original_raw_bytes) - compact_raw) / float(original_raw_bytes)
        ),
        "framed_record_reduction_fraction": float(
            (original_framed - compact_framed) / float(original_framed)
        ),
        "logical_original_raw_payload_bytes_total": int(row_count * int(original_raw_bytes)),
        "logical_compact_raw_payload_bytes_total": int(row_count * compact_raw),
        "logical_original_framed_record_bytes_total": int(row_count * original_framed),
        "logical_compact_framed_record_bytes_total": int(row_count * compact_framed),
        "single_request_original_cacheline_bytes": _round_up(original_framed, CACHELINE_BYTES),
        "single_request_compact_cacheline_bytes": _round_up(compact_framed, CACHELINE_BYTES),
        "single_request_original_dma_bytes": _round_up(original_framed, DMA_BYTES),
        "single_request_compact_dma_bytes": _round_up(compact_framed, DMA_BYTES),
        "batch64_packed_original_framed_bytes": _round_up(64 * original_framed, CACHELINE_BYTES),
        "batch64_packed_compact_framed_bytes": _round_up(64 * compact_framed, CACHELINE_BYTES),
    }


def _source_row(
    *,
    source_name: str,
    artifact: dict[str, Any],
    predictions_rows: list[dict[str, Any]],
    prediction_field: str,
    baseline_accuracy: float,
    baseline_name: str,
    baseline_predictions: np.ndarray,
    candidate_count: int,
    positive_pass: bool,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    answers = np.asarray([int(row["answer_index"]) for row in predictions_rows], dtype=np.int64)
    predictions = np.asarray([int(row[prediction_field]) for row in predictions_rows], dtype=np.int64)
    encoded, decoded = _compact_predictions(predictions, candidate_count=candidate_count)
    if len(baseline_predictions) != len(predictions):
        raise ValueError(f"{source_name} baseline length does not match predictions")
    ci = _paired_ci(
        selected=decoded,
        baseline=baseline_predictions,
        answers=answers,
        seed=seed,
        samples=bootstrap_samples,
    )
    original_packet = artifact.get("packet_contract") or artifact.get("bagged_gate", {}).get("packet_contract")
    if not original_packet:
        raise ValueError(f"{source_name} artifact does not expose packet_contract")
    original_raw = int(original_packet["raw_payload_bytes"])
    accounting = _packet_accounting(
        row_count=len(predictions),
        original_raw_bytes=original_raw,
        candidate_count=candidate_count,
    )
    return {
        "source_name": source_name,
        "prediction_field": prediction_field,
        "row_count": int(len(predictions)),
        "original_accuracy": _accuracy(predictions, answers),
        "compact_accuracy": _accuracy(decoded, answers),
        "prediction_equivalence": bool(np.array_equal(predictions, decoded)),
        "encoded_min": int(np.min(encoded)),
        "encoded_max": int(np.max(encoded)),
        "baseline_name": baseline_name,
        "baseline_accuracy": float(baseline_accuracy),
        "compact_delta_vs_baseline": ci["delta"],
        "compact_ci95_low_vs_baseline": ci["ci95_low"],
        "compact_ci95_high_vs_baseline": ci["ci95_high"],
        "positive_source_gate_passed": bool(positive_pass),
        "packet_accounting": accounting,
        "source_text_exposed": bool(original_packet.get("source_text_exposed", False)),
        "source_kv_exposed": bool(original_packet.get("source_kv_exposed", False)),
        "raw_hidden_vector_transmitted": bool(original_packet.get("raw_hidden_vector_transmitted", False)),
        "raw_scores_transmitted": bool(original_packet.get("raw_scores_transmitted", False)),
    }


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Minimal Packet Compaction",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- rows covered: `{h['total_rows_covered']}`",
        f"- original packet: `{h['original_raw_payload_bytes']}B` raw / `{h['original_framed_record_bytes']}B` framed",
        f"- compact packet: `{h['compact_raw_payload_bytes']}B` raw / `{h['compact_framed_record_bytes']}B` framed",
        f"- theoretical payload bits for four candidates: `{h['theoretical_payload_bits']}`",
        f"- prediction-equivalent rows: `{h['prediction_equivalent_row_count']}/{h['total_rows_covered']}`",
        f"- qwen mean-zscore compact accuracy: `{h['qwen_mean_zscore_accuracy']:.6f}`",
        f"- qwen hybrid compact accuracy: `{h['qwen_hybrid_accuracy']:.6f}`",
        f"- tiny compact accuracy: `{h['tiny_accuracy']:.6f}`",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_gate(
    *,
    output_dir: pathlib.Path = DEFAULT_OUTPUT,
    qwen_artifact: pathlib.Path = DEFAULT_QWEN_ARTIFACT,
    qwen_predictions: pathlib.Path = DEFAULT_QWEN_PREDICTIONS,
    tiny_artifact: pathlib.Path = DEFAULT_TINY_ARTIFACT,
    tiny_predictions: pathlib.Path = DEFAULT_TINY_PREDICTIONS,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    bootstrap_samples: int = 500,
    run_date: str = "2026-05-02",
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    qwen_payload = _read_json(qwen_artifact)
    tiny_payload = _read_json(tiny_artifact)
    qwen_rows = _read_jsonl(qwen_predictions)
    tiny_rows = _read_jsonl(tiny_predictions)

    qwen_answers = np.asarray([int(row["answer_index"]) for row in qwen_rows], dtype=np.int64)
    qwen_best_label_predictions = np.asarray(
        [int(row["source_label_prediction"]) for row in qwen_rows],
        dtype=np.int64,
    )
    tiny_answers = np.asarray([int(row["answer_index"]) for row in tiny_rows], dtype=np.int64)
    tiny_best_label_predictions = np.asarray(
        [int(row["source_label_prediction"]) for row in tiny_rows],
        dtype=np.int64,
    )

    qwen_h = qwen_payload["headline"]
    tiny_h = tiny_payload["headline"]
    source_rows = [
        _source_row(
            source_name="qwen_mean_zscore",
            artifact=qwen_payload,
            predictions_rows=qwen_rows,
            prediction_field="mean_zscore_prediction",
            baseline_accuracy=float(qwen_h["best_label_copy_accuracy"]),
            baseline_name="best_label_copy",
            baseline_predictions=qwen_best_label_predictions,
            candidate_count=candidate_count,
            positive_pass=bool(qwen_payload["pass_gate"]),
            seed=14001,
            bootstrap_samples=bootstrap_samples,
        ),
        _source_row(
            source_name="qwen_hybrid_vote_on_score_agreement",
            artifact=qwen_payload,
            predictions_rows=qwen_rows,
            prediction_field="hybrid_vote_on_score_agreement_prediction",
            baseline_accuracy=float(qwen_h["best_label_copy_accuracy"]),
            baseline_name="best_label_copy",
            baseline_predictions=qwen_best_label_predictions,
            candidate_count=candidate_count,
            positive_pass=bool(qwen_payload["pass_gate"]),
            seed=14002,
            bootstrap_samples=bootstrap_samples,
        ),
        _source_row(
            source_name="tinyllama_mean_zscore",
            artifact=tiny_payload,
            predictions_rows=tiny_rows,
            prediction_field="selected_prediction",
            baseline_accuracy=float(tiny_h["best_label_copy_eval_accuracy"]),
            baseline_name="best_label_copy",
            baseline_predictions=tiny_best_label_predictions,
            candidate_count=candidate_count,
            positive_pass=bool(tiny_payload.get("bagged_gate", {}).get("pass_gate")),
            seed=14003,
            bootstrap_samples=bootstrap_samples,
        ),
    ]
    total_rows = int(sum(row["row_count"] for row in source_rows))
    equivalent_rows = int(sum(row["row_count"] for row in source_rows if row["prediction_equivalence"]))
    accounting = source_rows[0]["packet_accounting"]
    exposure_clear = all(
        not row["source_text_exposed"]
        and not row["source_kv_exposed"]
        and not row["raw_hidden_vector_transmitted"]
        and not row["raw_scores_transmitted"]
        for row in source_rows
    )
    pass_gate = bool(
        all(row["prediction_equivalence"] for row in source_rows)
        and all(row["positive_source_gate_passed"] for row in source_rows)
        and accounting["compact_raw_payload_bytes_per_request"] < accounting[
            "original_raw_payload_bytes_per_request"
        ]
        and accounting["compact_framed_record_bytes_per_request"] < accounting[
            "original_framed_record_bytes_per_request"
        ]
        and exposure_clear
    )
    headline = {
        "source_row_count": len(source_rows),
        "total_rows_covered": total_rows,
        "prediction_equivalent_row_count": equivalent_rows,
        "all_prediction_equivalent": bool(all(row["prediction_equivalence"] for row in source_rows)),
        "theoretical_payload_bits": accounting["theoretical_payload_bits"],
        "original_raw_payload_bytes": accounting["original_raw_payload_bytes_per_request"],
        "compact_raw_payload_bytes": accounting["compact_raw_payload_bytes_per_request"],
        "original_framed_record_bytes": accounting["original_framed_record_bytes_per_request"],
        "compact_framed_record_bytes": accounting["compact_framed_record_bytes_per_request"],
        "raw_payload_reduction_fraction": accounting["raw_payload_reduction_fraction"],
        "framed_record_reduction_fraction": accounting["framed_record_reduction_fraction"],
        "qwen_mean_zscore_accuracy": source_rows[0]["compact_accuracy"],
        "qwen_hybrid_accuracy": source_rows[1]["compact_accuracy"],
        "tiny_accuracy": source_rows[2]["compact_accuracy"],
        "native_gpu_claims_allowed": False,
        "source_exposure_clear": exposure_clear,
    }
    payload = {
        "gate": "source_private_hellaswag_minimal_packet_compaction",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": pass_gate,
        "pass_rule": (
            "Pass if every compact one-byte candidate-id packet decodes to exactly the original "
            "selected prediction, all inherited positive source gates remain passed, raw and framed "
            "packet bytes both decrease, and no source text, KV, raw hidden vector, or raw score "
            "vector is transmitted."
        ),
        "candidate_count": int(candidate_count),
        "frame_bytes_assumed": FRAME_BYTES,
        "headline": headline,
        "source_rows": source_rows,
        "systems_packet_sideband": {
            **accounting,
            "communication_object": "task_level_source_private_candidate_id_packet",
            "communication_objective": "downstream_candidate_decision_accuracy",
            "not_a_kv_reconstruction_method": True,
            "not_a_vector_fidelity_codec": True,
            "does_not_preserve_source_kv": True,
            "does_not_enable_general_source_state_reconstruction": True,
            "future_matched_rate_rows_required": [
                "c2c_native_task_accuracy",
                "kvcomm_native_task_accuracy",
                "qjl_matched_rate_task_accuracy",
                "turboquant_matched_rate_task_accuracy",
                "kivi_matched_rate_task_accuracy",
                "kvquant_matched_rate_task_accuracy",
                "same_byte_text",
                "full_source_kv_upper_bound",
            ],
            "kv_compression_boundary": {
                "c2c_status": "pending_native_cache_communication_row",
                "kvcomm_status": "pending_native_selective_kv_row",
                "qjl_status": "byte_floor_proxy_only_until_native_task_row",
                "turboquant_status": "byte_floor_proxy_only_until_native_task_row",
                "kivi_status": "byte_floor_proxy_only_until_native_task_row",
                "kvquant_status": "byte_floor_proxy_only_until_native_task_row",
            },
            "native_gpu_claims_allowed": False,
            "native_systems_complete": False,
            "claim_scope": (
                "Mac-local byte-accounting compaction audit for the promoted HellaSwag packet rows; "
                "no serving-throughput, HBM-traffic, or kernel-speedup claim."
            ),
        },
        "inputs": {
            "qwen_artifact": _display_path(qwen_artifact),
            "qwen_artifact_sha256": _sha256_file(qwen_artifact),
            "qwen_predictions": _display_path(qwen_predictions),
            "qwen_predictions_sha256": _sha256_file(qwen_predictions),
            "tiny_artifact": _display_path(tiny_artifact),
            "tiny_artifact_sha256": _sha256_file(tiny_artifact),
            "tiny_predictions": _display_path(tiny_predictions),
            "tiny_predictions_sha256": _sha256_file(tiny_predictions),
        },
        "interpretation": (
            "The HellaSwag hidden-innovation decoder only needs the selected candidate id at runtime. "
            "The previous second byte was a confidence/debug field and is not used to decode the "
            "answer. This compaction preserves the promoted Qwen and TinyLlama packet predictions "
            "exactly while reducing the logical packet from 2B raw / 5B framed to 1B raw / 4B framed. "
            "It strengthens the systems rate-frontier contribution, but it does not solve the "
            "cross-family receiver/common-language gap."
        ),
        "timing": {"total_seconds": float(time.perf_counter() - started)},
    }
    json_path = output_dir / "hellaswag_minimal_packet_compaction.json"
    md_path = output_dir / "hellaswag_minimal_packet_compaction.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload)
    manifest = {
        "gate": payload["gate"],
        "created_utc": payload["created_utc"],
        "headline": headline,
        "inputs": payload["inputs"],
        "files": [
            {"path": _display_path(path), "sha256": _sha256_file(path), "bytes": path.stat().st_size}
            for path in (json_path, md_path)
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qwen-artifact", type=pathlib.Path, default=DEFAULT_QWEN_ARTIFACT)
    parser.add_argument("--qwen-predictions", type=pathlib.Path, default=DEFAULT_QWEN_PREDICTIONS)
    parser.add_argument("--tiny-artifact", type=pathlib.Path, default=DEFAULT_TINY_ARTIFACT)
    parser.add_argument("--tiny-predictions", type=pathlib.Path, default=DEFAULT_TINY_PREDICTIONS)
    parser.add_argument("--candidate-count", type=int, default=DEFAULT_CANDIDATE_COUNT)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--run-date", default="2026-05-02")
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        qwen_artifact=args.qwen_artifact,
        qwen_predictions=args.qwen_predictions,
        tiny_artifact=args.tiny_artifact,
        tiny_predictions=args.tiny_predictions,
        candidate_count=args.candidate_count,
        bootstrap_samples=args.bootstrap_samples,
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    print(f"pass_gate={payload['pass_gate']}")


if __name__ == "__main__":
    main()
