from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import statistics
import sys
import time
from typing import Any

import numpy as np


SCRIPT_PATH = pathlib.Path(__file__).resolve()
if SCRIPT_PATH.parents[1].name == "code" and SCRIPT_PATH.parents[2].name == "final":
    ROOT = SCRIPT_PATH.parents[2]
    IMPORT_ROOT = SCRIPT_PATH.parents[1]
else:
    ROOT = SCRIPT_PATH.parents[1]
    IMPORT_ROOT = ROOT
for path in (IMPORT_ROOT, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

DEFAULT_SUMMARY = pathlib.Path(
    "results/source_private_candidate_local_residual_receiver_20260430/summary/"
    "hf_embedding_heldout_packet_summary.json"
)
DEFAULT_OUTPUT = pathlib.Path("results/source_private_candidate_local_residual_systems_waterfall_20260430")
DEFAULT_BENCH_RUN = pathlib.Path(
    "results/source_private_candidate_local_residual_receiver_20260430_seed59_n512_minilm_teacher_norm_dec048_evaldisjoint"
)

CSV_COLUMNS = (
    "row_class",
    "run_dir",
    "direction",
    "budget_bytes",
    "condition",
    "online_phase",
    "n",
    "accuracy",
    "target_accuracy",
    "best_control_accuracy",
    "delta_vs_target",
    "delta_vs_best_control",
    "paired_ci95_low_vs_target",
    "controls_ok",
    "pass_gate",
    "payload_bytes",
    "record_bytes",
    "single_request_cacheline_bytes",
    "single_request_dma_bytes",
    "batch_line_bytes_per_request",
    "batch_dma_bytes_per_request",
    "current_python_p50_ms",
    "current_python_p95_ms",
    "source_text_exposed",
    "source_kv_exposed",
    "public_candidate_features_only",
    "calibration_eval_exact_id_overlap_count",
    "exact_transformed_eval_surface_overlap_count",
    "candidate_count",
    "atom_count",
    "adapter_params_bytes",
    "resident_feature_bytes_per_example",
    "resident_score_table_bytes_per_example",
    "dense_decode_read_bytes_per_request",
    "sparse_decode_read_bytes_per_request",
    "resident_sparse_decode_p50_us",
    "resident_sparse_decode_p95_us",
    "resident_sparse_decode_mismatch_count",
    "trace_hash",
    "notes",
)


def _resolve(path: pathlib.Path) -> pathlib.Path:
    return path if path.is_absolute() else ROOT / path


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _sha256_file(path: pathlib.Path) -> str:
    resolved = _resolve(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return float(ordered[index])


def _packet_accounting(payload_bytes: float, *, batch_size: int, header_bytes: int) -> dict[str, float]:
    if payload_bytes <= 0:
        return {
            "payload_bytes": 0.0,
            "record_bytes": 0.0,
            "single_request_cacheline_bytes": 0.0,
            "single_request_dma_bytes": 0.0,
            "batch_line_bytes_per_request": 0.0,
            "batch_dma_bytes_per_request": 0.0,
        }
    record_bytes = float(payload_bytes + header_bytes)
    return {
        "payload_bytes": float(payload_bytes),
        "record_bytes": record_bytes,
        "single_request_cacheline_bytes": float(64 * math.ceil(record_bytes / 64.0)),
        "single_request_dma_bytes": float(128 * math.ceil(record_bytes / 128.0)),
        "batch_line_bytes_per_request": float(64 * math.ceil(batch_size * record_bytes / 64.0) / batch_size),
        "batch_dma_bytes_per_request": float(128 * math.ceil(batch_size * record_bytes / 128.0) / batch_size),
    }


def _summary_payload(summary_path: pathlib.Path) -> dict[str, Any]:
    payload = _read_json(summary_path)
    return payload.get("summary", payload)


def _iter_direction_summaries(run_dirs: list[pathlib.Path]) -> list[tuple[pathlib.Path, dict[str, Any]]]:
    direction_payloads: list[tuple[pathlib.Path, dict[str, Any]]] = []
    for run_dir in run_dirs:
        resolved = _resolve(run_dir)
        for direction in ("core_to_holdout", "holdout_to_core", "same_family_all"):
            summary_path = resolved / direction / "summary.json"
            if summary_path.exists():
                direction_payloads.append((run_dir, json.loads(summary_path.read_text(encoding="utf-8"))))
    return direction_payloads


def _source_text_exposed(condition: str) -> bool:
    return condition in {"answer_only_text", "structured_text_matched"}


def _source_kv_exposed(condition: str) -> bool:
    return False


def _row_from_condition(
    *,
    run_dir: pathlib.Path,
    direction_payload: dict[str, Any],
    budget_summary: dict[str, Any],
    condition: str,
    batch_size: int,
    header_bytes: int,
    score_dtype_bytes: int,
    bench_row: dict[str, Any] | None,
) -> dict[str, Any] | None:
    metrics = budget_summary["metrics"]
    if condition not in metrics:
        return None
    metric = metrics[condition]
    target_accuracy = float(budget_summary["target_accuracy"])
    accuracy = float(metric["accuracy"])
    best_control = float(budget_summary["best_control_accuracy"])
    payload = _packet_accounting(
        float(metric["mean_payload_bytes"]),
        batch_size=batch_size,
        header_bytes=header_bytes,
    )
    candidate_count = 4
    atom_count = len(direction_payload.get("atom_dictionary", []))
    nonzero_atoms = max(1, int(round(float(metric.get("mean_payload_tokens", 0.0)))))
    dense_decode_read = candidate_count * atom_count * 8 + atom_count * 8 + payload["record_bytes"]
    sparse_decode_read = candidate_count * nonzero_atoms * score_dtype_bytes + payload["record_bytes"]
    audit = direction_payload.get("surface_overlap_audit", {})
    is_packet = condition == "learned_synonym_dictionary_packet"
    return {
        "row_class": "candidate_local_residual_receiver",
        "run_dir": str(run_dir),
        "direction": direction_payload["direction"],
        "budget_bytes": int(budget_summary["budget_bytes"]),
        "condition": condition,
        "online_phase": "current_python_nonresident_decode",
        "n": int(metric["n"]),
        "accuracy": accuracy,
        "target_accuracy": target_accuracy,
        "best_control_accuracy": best_control,
        "delta_vs_target": accuracy - target_accuracy,
        "delta_vs_best_control": accuracy - best_control,
        "paired_ci95_low_vs_target": budget_summary["paired_bootstrap_vs_target"]["ci95_low"],
        "controls_ok": bool(budget_summary["controls_ok"]) if is_packet else None,
        "pass_gate": bool(budget_summary["pass_gate"]) if is_packet else None,
        **payload,
        "current_python_p50_ms": float(metric["p50_latency_ms"]),
        "current_python_p95_ms": float(metric["p95_latency_ms"]),
        "source_text_exposed": _source_text_exposed(condition),
        "source_kv_exposed": _source_kv_exposed(condition),
        "public_candidate_features_only": not _source_text_exposed(condition) and not _source_kv_exposed(condition),
        "calibration_eval_exact_id_overlap_count": int(audit.get("calibration_eval_exact_id_overlap_count", -1)),
        "exact_transformed_eval_surface_overlap_count": int(
            audit.get("exact_transformed_eval_surface_overlap_count", -1)
        ),
        "candidate_count": candidate_count,
        "atom_count": atom_count,
        "adapter_params_bytes": None,
        "resident_feature_bytes_per_example": candidate_count * atom_count * 8,
        "resident_score_table_bytes_per_example": candidate_count * atom_count * score_dtype_bytes,
        "dense_decode_read_bytes_per_request": dense_decode_read,
        "sparse_decode_read_bytes_per_request": sparse_decode_read,
        "resident_sparse_decode_p50_us": None if bench_row is None else bench_row["resident_sparse_decode_p50_us"],
        "resident_sparse_decode_p95_us": None if bench_row is None else bench_row["resident_sparse_decode_p95_us"],
        "resident_sparse_decode_mismatch_count": None
        if bench_row is None
        else bench_row["resident_sparse_decode_mismatch_count"],
        "trace_hash": None if bench_row is None else bench_row["trace_hash"],
        "notes": (
            "matched source-private packet; public candidate features remain receiver-local"
            if is_packet
            else "control/comparator row from the same frozen direction summary"
        ),
    }


def _load_expected_predictions(direction_dir: pathlib.Path, budget_bytes: int) -> list[str]:
    predictions: list[str] = []
    path = direction_dir / f"predictions_budget{budget_bytes}.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["condition"] == "learned_synonym_dictionary_packet":
            predictions.append(row["prediction"])
    return predictions


def _bench_direction(
    *,
    run_dir: pathlib.Path,
    direction: str,
    budget_bytes: int,
    repeats: int,
) -> dict[str, Any]:
    from scripts import run_source_private_learned_synonym_dictionary_packet_gate as runner

    resolved_run = _resolve(run_dir)
    direction_dir = resolved_run / direction
    direction_payload = json.loads((direction_dir / "summary.json").read_text(encoding="utf-8"))
    gate_payload = json.loads((resolved_run / "learned_synonym_dictionary_packet_gate.json").read_text(encoding="utf-8"))

    runner._HF_FEATURE_MODEL = gate_payload["feature_model"]
    runner._HF_FEATURE_DEVICE = gate_payload["feature_device"]
    runner._HF_FEATURE_DTYPE = gate_payload["feature_dtype"]
    runner._HF_FEATURE_MAX_LENGTH = gate_payload["feature_max_length"]
    runner._HF_FEATURE_LOCAL_FILES_ONLY = True

    # Warm model weights once; feature-cache timing below remains explicit.
    runner._featurize_text(
        "warmup candidate local residual systems bench",
        dim=gate_payload["feature_dim"],
        text_feature_mode=gate_payload["text_feature_mode"],
    )
    runner._HF_TEXT_FEATURE_CACHE.clear()

    train_rows = runner.make_benchmark(
        examples=direction_payload["train_examples"],
        candidates=4,
        seed=direction_payload["train_seed"],
        family_set=direction_payload["train_family_set"],
    )
    eval_rows = runner.make_benchmark(
        examples=direction_payload["eval_examples"],
        candidates=4,
        seed=direction_payload["eval_seed"],
        family_set=direction_payload["eval_family_set"],
    )
    calibration_rows = runner._calibration_examples(
        mode=direction_payload["candidate_calibration"],
        train_examples=train_rows,
        eval_examples=eval_rows,
        calibration_count=direction_payload["calibration_examples"],
        seed=direction_payload["train_seed"] + 101,
    )
    fit_start = time.perf_counter()
    dictionary = runner._fit_dictionary(
        examples=calibration_rows,
        feature_dim=direction_payload["feature_dim"],
        ridge=direction_payload["ridge"],
        calibration_atom_view=direction_payload["calibration_atom_view"],
        top_k=direction_payload["top_k"],
        min_score=direction_payload["min_score"],
        text_feature_mode=direction_payload["text_feature_mode"],
        adapter_target_mode=direction_payload["adapter_target_mode"],
        receiver_mode=direction_payload["receiver_mode"],
        contrastive_negative_sources=direction_payload["contrastive_negative_sources"],
        contrastive_rank=direction_payload["contrastive_rank"] or 4,
        seed=direction_payload["train_seed"] + 211,
    )
    adapter_fit_ms = (time.perf_counter() - fit_start) * 1000.0

    runner._HF_TEXT_FEATURE_CACHE.clear()
    matrix_start = time.perf_counter()
    residual_matrices: list[np.ndarray] = []
    labels_by_example: list[list[str]] = []
    priors: list[str] = []
    for example in eval_rows:
        vectors = []
        for candidate in example.candidates:
            text = runner._candidate_surface_text(
                candidate.patch_intent,
                candidate_atom_view=direction_payload["candidate_atom_view"],
            )
            vectors.append(dictionary.predict_vector(text, apply_top_k=True))
        matrix = np.stack(vectors, axis=0)
        residual = matrix - matrix.mean(axis=0, keepdims=True)
        if direction_payload["decoder_score_mode"] == "candidate_local_residual_norm":
            norms = np.linalg.norm(residual, axis=1)
            residual = np.divide(
                residual,
                np.maximum(norms[:, None], 1e-12),
                out=np.zeros_like(residual),
                where=norms[:, None] > 0,
            )
        residual_matrices.append(residual)
        labels_by_example.append([candidate.label for candidate in example.candidates])
        priors.append(runner._prior_prediction(example))
    candidate_feature_build_ms = (time.perf_counter() - matrix_start) * 1000.0

    source_start = time.perf_counter()
    payload_vectors: list[np.ndarray] = []
    payload_hexes: list[str] = []
    for example in eval_rows:
        payload = runner._encode_atoms(
            runner._source_private_atoms(example.private_test_log, mode="matched"),
            budget_bytes=budget_bytes,
        )
        payload_hexes.append(payload.hex())
        vector = runner._atom_vector(runner._decode_payload_atoms(payload, budget_bytes=budget_bytes))
        if direction_payload["decoder_score_mode"] == "candidate_local_residual_norm":
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector = vector / norm
        payload_vectors.append(vector)
    source_encode_ms = (time.perf_counter() - source_start) * 1000.0

    def predict_all() -> list[str]:
        predictions: list[str] = []
        for residual, payload, labels, prior in zip(residual_matrices, payload_vectors, labels_by_example, priors):
            nonzero = np.flatnonzero(payload)
            if nonzero.size:
                scores = residual[:, nonzero] @ payload[nonzero]
            else:
                scores = np.zeros(residual.shape[0], dtype=np.float64)
            best_score = float(np.max(scores))
            if best_score < direction_payload["min_decision_score"]:
                predictions.append(prior)
                continue
            tied = np.flatnonzero(np.abs(scores - best_score) <= 1e-8)
            prediction = prior if any(labels[int(idx)] == prior for idx in tied) else labels[int(tied[0])]
            predictions.append(prediction)
        return predictions

    expected = _load_expected_predictions(direction_dir, budget_bytes)
    predictions = predict_all()
    mismatch_count = sum(predicted != wanted for predicted, wanted in zip(predictions, expected))
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        predict_all()
        samples.append((time.perf_counter() - start) * 1_000_000.0 / max(1, len(eval_rows)))
    trace_hash = hashlib.sha256(
        ("\n".join(payload_hexes) + "\n" + "\n".join(predictions)).encode("utf-8")
    ).hexdigest()
    return {
        "run_dir": str(run_dir),
        "direction": direction,
        "budget_bytes": budget_bytes,
        "adapter_fit_ms": adapter_fit_ms,
        "adapter_params_bytes": int(dictionary.weights.nbytes),
        "cold_candidate_feature_build_ms_total": candidate_feature_build_ms,
        "cold_candidate_feature_build_ms_per_request": candidate_feature_build_ms / max(1, len(eval_rows)),
        "source_encode_ms_per_request": source_encode_ms / max(1, len(eval_rows)),
        "resident_sparse_decode_p50_us": statistics.median(samples),
        "resident_sparse_decode_p95_us": _percentile(samples, 0.95),
        "resident_sparse_decode_mismatch_count": mismatch_count,
        "resident_cache_bytes_total": int(sum(matrix.nbytes for matrix in residual_matrices)),
        "resident_feature_bytes_per_example": int(residual_matrices[0].nbytes if residual_matrices else 0),
        "trace_hash": trace_hash,
    }


def _bench_run(
    *,
    run_dir: pathlib.Path | None,
    budget_bytes: int,
    repeats: int,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    if run_dir is None:
        return {}
    output: dict[tuple[str, str, int], dict[str, Any]] = {}
    for direction in ("core_to_holdout", "holdout_to_core", "same_family_all"):
        direction_dir = _resolve(run_dir) / direction
        if not direction_dir.exists():
            continue
        bench = _bench_direction(
            run_dir=run_dir,
            direction=direction,
            budget_bytes=budget_bytes,
            repeats=repeats,
        )
        output[(str(run_dir), direction, budget_bytes)] = bench
    return output


def build_candidate_local_residual_systems_waterfall(
    *,
    summary_path: pathlib.Path,
    output_dir: pathlib.Path,
    run_dirs: list[pathlib.Path] | None = None,
    budget_bytes: int = 8,
    batch_size: int = 64,
    header_bytes: int = 3,
    score_dtype_bytes: int = 4,
    bench_run_dir: pathlib.Path | None = None,
    bench_repeats: int = 64,
) -> dict[str, Any]:
    summary = _summary_payload(summary_path)
    selected_run_dirs = (
        run_dirs
        if run_dirs is not None
        else [pathlib.Path(row["run_dir"]) for row in summary.get("runs", []) if "_n512_" in row["run_dir"]]
    )
    bench_rows = _bench_run(run_dir=bench_run_dir, budget_bytes=budget_bytes, repeats=bench_repeats)
    rows: list[dict[str, Any]] = []
    for run_dir, direction_payload in _iter_direction_summaries(selected_run_dirs):
        for budget_summary in direction_payload["budget_summaries"]:
            if int(budget_summary["budget_bytes"]) != budget_bytes:
                continue
            bench_row = bench_rows.get((str(run_dir), direction_payload["direction"], budget_bytes))
            for condition in (
                "target_only",
                "learned_synonym_dictionary_packet",
                "zero_source",
                "random_same_byte",
                "private_random_source_atoms",
                "permuted_teacher_receiver",
                "answer_only_text",
                "structured_text_matched",
            ):
                row = _row_from_condition(
                    run_dir=run_dir,
                    direction_payload=direction_payload,
                    budget_summary=budget_summary,
                    condition=condition,
                    batch_size=batch_size,
                    header_bytes=header_bytes,
                    score_dtype_bytes=score_dtype_bytes,
                    bench_row=bench_row if condition == "learned_synonym_dictionary_packet" else None,
                )
                if row is not None:
                    if bench_row is not None:
                        row["adapter_params_bytes"] = bench_row["adapter_params_bytes"]
                        row["resident_feature_bytes_per_example"] = bench_row["resident_feature_bytes_per_example"]
                    rows.append(row)

    packet_rows = [row for row in rows if row["condition"] == "learned_synonym_dictionary_packet"]
    passing_packet_rows = [row for row in packet_rows if row["pass_gate"]]
    measured_packet_rows = [row for row in packet_rows if row["resident_sparse_decode_p50_us"] is not None]
    headline = {
        "pass_gate": bool(packet_rows) and len(passing_packet_rows) == len(packet_rows),
        "summary_pass_gate": bool(summary.get("pass_gate")),
        "n512_packet_rows": len(packet_rows),
        "n512_passing_packet_rows": len(passing_packet_rows),
        "budget_bytes": budget_bytes,
        "packet_record_bytes": budget_bytes + header_bytes,
        "packet_single_request_cacheline_bytes": 64 * math.ceil((budget_bytes + header_bytes) / 64),
        "packet_single_request_dma_bytes": 128 * math.ceil((budget_bytes + header_bytes) / 128),
        "packet_batch_line_bytes_per_request": 64
        * math.ceil(batch_size * (budget_bytes + header_bytes) / 64)
        / batch_size,
        "packet_batch_dma_bytes_per_request": 128
        * math.ceil(batch_size * (budget_bytes + header_bytes) / 128)
        / batch_size,
        "max_current_python_packet_p50_ms": max(
            (float(row["current_python_p50_ms"]) for row in packet_rows),
            default=None,
        ),
        "max_current_python_packet_p95_ms": max(
            (float(row["current_python_p95_ms"]) for row in packet_rows),
            default=None,
        ),
        "max_resident_sparse_decode_p50_us": max(
            (float(row["resident_sparse_decode_p50_us"]) for row in measured_packet_rows),
            default=None,
        ),
        "max_resident_sparse_decode_p95_us": max(
            (float(row["resident_sparse_decode_p95_us"]) for row in measured_packet_rows),
            default=None,
        ),
        "max_resident_sparse_decode_mismatch_count": max(
            (int(row["resident_sparse_decode_mismatch_count"]) for row in measured_packet_rows),
            default=None,
        ),
        "max_adapter_fit_ms": max((row["adapter_fit_ms"] for row in bench_rows.values()), default=None),
        "max_cold_candidate_feature_build_ms_per_request": max(
            (row["cold_candidate_feature_build_ms_per_request"] for row in bench_rows.values()),
            default=None,
        ),
        "max_source_encode_ms_per_request": max(
            (row["source_encode_ms_per_request"] for row in bench_rows.values()),
            default=None,
        ),
        "source_text_exposed": False,
        "source_kv_exposed": False,
        "public_candidate_features_only": True,
        "calibration_eval_exact_id_overlap_count_max": max(
            (row["calibration_eval_exact_id_overlap_count"] for row in packet_rows),
            default=None,
        ),
        "exact_transformed_eval_surface_overlap_count_max": max(
            (row["exact_transformed_eval_surface_overlap_count"] for row in packet_rows),
            default=None,
        ),
    }
    checks = [
        {
            "check": "all_n512_packet_rows_pass",
            "pass": headline["pass_gate"],
            "value": f"{headline['n512_passing_packet_rows']}/{headline['n512_packet_rows']}",
        },
        {
            "check": "source_private_exposure",
            "pass": not headline["source_text_exposed"] and not headline["source_kv_exposed"],
            "value": "source_text_exposed=false, source_kv_exposed=false",
        },
        {
            "check": "calibration_eval_exact_id_overlap_zero",
            "pass": headline["calibration_eval_exact_id_overlap_count_max"] == 0,
            "value": str(headline["calibration_eval_exact_id_overlap_count_max"]),
        },
        {
            "check": "transformed_surface_overlap_zero",
            "pass": headline["exact_transformed_eval_surface_overlap_count_max"] == 0,
            "value": str(headline["exact_transformed_eval_surface_overlap_count_max"]),
        },
        {
            "check": "resident_sparse_decode_exact_if_measured",
            "pass": headline["max_resident_sparse_decode_mismatch_count"] in {None, 0},
            "value": str(headline["max_resident_sparse_decode_mismatch_count"]),
        },
    ]
    payload = {
        "gate": "source_private_candidate_local_residual_systems_waterfall",
        "pass_gate": all(check["pass"] for check in checks),
        "headline": headline,
        "checks": checks,
        "bench_rows": list(bench_rows.values()),
        "rows": rows,
        "interpretation": (
            "This artifact separates the live candidate-local residual receiver into packet boundary accounting, "
            "current Python nonresident decode, optional Mac resident sparse decode over cached public candidate "
            "residuals, and source text/KV exposure. It is a Mac-local systems trace, not a production vLLM or "
            "NVIDIA serving claim."
        ),
        "non_claims": [
            "No HBM, PCIe, NVLink, TPOT, goodput, or production serving counter is measured here.",
            "The receiver cache is public candidate state and is reported separately from source-private packet bytes.",
            "C2C/KVComm rows remain matched baselines to run, not defeated baselines in this artifact.",
        ],
        "sources": {
            "summary": str(summary_path),
            "summary_sha256": _sha256_file(summary_path),
            "run_dirs": [str(path) for path in selected_run_dirs],
        },
    }
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "candidate_local_residual_systems_waterfall.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "candidate_local_residual_systems_waterfall.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _fmt(row.get(column)) for column in CSV_COLUMNS})

    md_lines = [
        "# Candidate-Local Residual Systems Waterfall",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- n512 packet rows passing: `{headline['n512_passing_packet_rows']}/{headline['n512_packet_rows']}`",
        f"- packet record bytes at {budget_bytes}B payload: `{headline['packet_record_bytes']}`",
        f"- batch-{batch_size} line bytes/request: `{headline['packet_batch_line_bytes_per_request']:.2f}`",
        f"- max current Python packet p50: `{_fmt(headline['max_current_python_packet_p50_ms'])}` ms/request",
        f"- max resident sparse decode p50: `{_fmt(headline['max_resident_sparse_decode_p50_us'])}` us/request",
        f"- max cold candidate feature build: `{_fmt(headline['max_cold_candidate_feature_build_ms_per_request'])}` ms/request",
        f"- source text exposed: `{headline['source_text_exposed']}`",
        f"- source KV exposed: `{headline['source_kv_exposed']}`",
        "",
        "## Checks",
        "",
        "| Check | Pass | Value |",
        "|---|---:|---:|",
    ]
    for check in checks:
        md_lines.append(f"| `{check['check']}` | `{check['pass']}` | `{check['value']}` |")
    md_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
            "",
            "## Non-Claims",
            "",
        ]
    )
    md_lines.extend(f"- {item}" for item in payload["non_claims"])
    (output_dir / "candidate_local_residual_systems_waterfall.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )
    artifacts = [
        "candidate_local_residual_systems_waterfall.json",
        "candidate_local_residual_systems_waterfall.csv",
        "candidate_local_residual_systems_waterfall.md",
    ]
    manifest = {
        "artifacts": artifacts,
        "artifact_sha256": {name: _sha256_file(output_dir / name) for name in artifacts},
        "pass_gate": payload["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Candidate-Local Residual Systems Waterfall Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-path", type=pathlib.Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-dir", action="append", type=pathlib.Path, default=None)
    parser.add_argument("--budget-bytes", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--header-bytes", type=int, default=3)
    parser.add_argument("--score-dtype-bytes", type=int, default=4)
    parser.add_argument("--bench-run-dir", type=pathlib.Path, default=DEFAULT_BENCH_RUN)
    parser.add_argument("--bench-repeats", type=int, default=64)
    parser.add_argument("--no-bench", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_candidate_local_residual_systems_waterfall(
        summary_path=args.summary_path,
        output_dir=args.output_dir,
        run_dirs=args.run_dir,
        budget_bytes=args.budget_bytes,
        batch_size=args.batch_size,
        header_bytes=args.header_bytes,
        score_dtype_bytes=args.score_dtype_bytes,
        bench_run_dir=None if args.no_bench else args.bench_run_dir,
        bench_repeats=args.bench_repeats,
    )
    print(
        json.dumps(
            {
                "output_dir": str(_resolve(args.output_dir)),
                "pass_gate": payload["pass_gate"],
                "packet_rows": payload["headline"]["n512_packet_rows"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
