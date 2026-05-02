from __future__ import annotations

"""Build a Mac-local systems card for the TinyLlama HellaSwag full-validation gate."""

import argparse
import datetime as dt
import hashlib
import json
import math
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT = pathlib.Path("results/source_private_hellaswag_tinyllama_full_mac_systems_card_20260502")
DEFAULT_FULL_EVAL = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_eval_full_stress_20260502_tinyllama_train512_validation0_10042/"
    "hellaswag_hidden_innovation_eval_slice_stress.json"
)


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


def _rounded_cache_bytes(record_bytes: int, *, granularity: int, batch_size: int = 1) -> float:
    total = int(record_bytes) * int(batch_size)
    rounded = int(math.ceil(total / granularity) * granularity)
    return float(rounded / batch_size)


def _file_bytes(path: str | None, *, fallback: int | None = None) -> int | None:
    if not path:
        return fallback
    resolved = _resolve(path)
    if resolved.exists():
        return resolved.stat().st_size
    return fallback


def _hidden_cache_bytes(metadata: dict[str, Any]) -> int | None:
    omitted = metadata.get("hidden_cache_omitted_artifact") or {}
    return _file_bytes(metadata.get("hidden_cache"), fallback=omitted.get("bytes"))


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# TinyLlama HellaSwag Full-Validation Mac Systems Card",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- benchmark rows: `{h['row_count']}`",
        f"- packet: `{h['packet_raw_bytes']}B` raw / `{h['packet_framed_bytes']}B` framed",
        f"- selected accuracy: `{h['accuracy']:.6f}`",
        f"- delta vs best label-copy: `{h['delta_vs_best_label_copy']:.6f}`",
        f"- source scoring wall time: `{h['source_scoring_wall_time_s']:.3f}s`",
        f"- source hidden extraction wall time: `{h['source_hidden_extraction_wall_time_s']:.3f}s`",
        f"- total wall time: `{h['total_wall_time_s']:.3f}s`",
        f"- native GPU claims allowed: `{h['native_gpu_claims_allowed']}`",
        "",
        "## Claim Boundary",
        "",
        payload["claim_boundary"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_card(
    *,
    output_dir: pathlib.Path = DEFAULT_OUTPUT,
    full_eval: pathlib.Path = DEFAULT_FULL_EVAL,
    run_date: str = "2026-05-02",
) -> dict[str, Any]:
    payload = _read_json(full_eval)
    bagged = payload["bagged_gate"]
    bagged_headline = bagged["headline"]
    metadata = payload.get("eval_cache_metadata", {})
    score_model = metadata.get("score_model", {})
    hidden_model = metadata.get("hidden_model", {})
    packet = bagged["packet_contract"]
    raw_bytes = int(packet["raw_payload_bytes"])
    framed_bytes = int(packet["framed_record_bytes"])

    score_cache_bytes = _file_bytes(metadata.get("score_cache"))
    hidden_cache_bytes = _hidden_cache_bytes(metadata)
    hidden_meta_bytes = _file_bytes(metadata.get("hidden_cache_meta"))
    total_wall = float(payload.get("timing", {}).get("total_seconds", 0.0))
    source_scoring = float(score_model.get("latency_s") or 0.0)
    source_hidden = float(hidden_model.get("latency_s") or 0.0)
    packet_build = max(total_wall - source_scoring - source_hidden, 0.0)

    headline = {
        "benchmark": "HellaSwag",
        "split": "validation",
        "row_count": int(payload["headline"]["eval_rows"]),
        "source_model": score_model.get("model_path") or hidden_model.get("model_path"),
        "target_model": "frozen candidate selector/evaluator contract",
        "device": hidden_model.get("device") or score_model.get("device"),
        "backend": "PyTorch/MPS Mac-local artifact",
        "precision": hidden_model.get("dtype") or score_model.get("dtype"),
        "accuracy": bagged_headline["selected_eval_accuracy"],
        "best_label_copy_accuracy": bagged_headline["best_label_copy_eval_accuracy"],
        "source_label_copy_accuracy": bagged_headline["source_label_copy_eval_accuracy"],
        "score_only_accuracy": bagged_headline["score_only_bagged_control_accuracy"],
        "delta_vs_best_label_copy": bagged_headline["selected_minus_best_label_copy"],
        "delta_vs_score_only": bagged_headline["selected_minus_score_only_bagged_control"],
        "paired_ci95_low_vs_best_label_copy": bagged_headline["paired_ci95_low_vs_best_label_copy"],
        "packet_raw_bytes": raw_bytes,
        "packet_framed_bytes": framed_bytes,
        "single_request_cacheline_bytes": _rounded_cache_bytes(framed_bytes, granularity=64),
        "single_request_dma_bytes": _rounded_cache_bytes(framed_bytes, granularity=128),
        "batch64_cacheline_bytes_per_request": _rounded_cache_bytes(framed_bytes, granularity=64, batch_size=64),
        "batch64_dma_bytes_per_request": _rounded_cache_bytes(framed_bytes, granularity=128, batch_size=64),
        "source_score_cache_bytes": score_cache_bytes,
        "source_hidden_cache_bytes": hidden_cache_bytes,
        "source_hidden_meta_bytes": hidden_meta_bytes,
        "total_wall_time_s": total_wall,
        "source_scoring_wall_time_s": source_scoring,
        "source_hidden_extraction_wall_time_s": source_hidden,
        "packet_build_and_gate_wall_time_s": packet_build,
        "examples_per_second_end_to_end": (int(payload["headline"]["eval_rows"]) / total_wall) if total_wall else None,
        "source_text_exposed": bool(packet["source_text_exposed"]),
        "source_kv_exposed": bool(packet["source_kv_exposed"]),
        "raw_hidden_vector_transmitted": bool(packet["raw_hidden_vector_transmitted"]),
        "raw_scores_transmitted": bool(packet["raw_scores_transmitted"]),
        "native_gpu_claims_allowed": False,
    }
    pass_gate = bool(
        bagged["pass_gate"]
        and raw_bytes == 2
        and framed_bytes == 5
        and not headline["source_text_exposed"]
        and not headline["source_kv_exposed"]
        and not headline["raw_hidden_vector_transmitted"]
        and not headline["raw_scores_transmitted"]
    )
    systems_rows = [
        {
            "row_id": "latentwire_packet",
            "native_status": "mac_measured",
            "payload_bytes": framed_bytes,
            "source_exposure": "no text/KV/raw hidden/raw scores",
            "timing_status": "measured",
        },
        {
            "row_id": "prefix_prompt",
            "native_status": "byte_floor_proxy",
            "payload_bytes": None,
            "source_exposure": "continuous prompt or text-conditioned prompt state",
            "timing_status": "not native in this artifact",
        },
        {
            "row_id": "kv_cache_transport",
            "native_status": "byte_floor_proxy",
            "payload_bytes": None,
            "source_exposure": "KV or KV sketch/state",
            "timing_status": "not native in this artifact",
        },
    ]
    claim_boundary = (
        "This card supports Mac-local byte/exposure accounting and phase timing for the fixed TinyLlama "
        "source-private packet. It does not support GPU throughput, HBM traffic, vLLM/SGLang latency, "
        "or superiority over C2C, KVComm, QJL, TurboQuant, KIVI, or KVQuant."
    )
    result = {
        "gate": "source_private_hellaswag_tinyllama_full_mac_systems_card",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "pass_gate": pass_gate,
        "pass_rule": (
            "Pass if the full TinyLlama HellaSwag gate passes and the runtime packet remains 2B raw / "
            "5B framed with no source text, KV cache, raw hidden vector, or raw score exposure. "
            "This is a Mac-local systems card, not a native serving benchmark."
        ),
        "headline": headline,
        "systems_rows": systems_rows,
        "claim_boundary": claim_boundary,
        "input_artifact": {
            "path": _display_path(full_eval),
            "sha256": _sha256_file(full_eval),
        },
    }

    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "tinyllama_full_mac_systems_card.json"
    md_path = output_dir / "tinyllama_full_mac_systems_card.md"
    manifest_path = output_dir / "manifest.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, result)
    manifest = {
        "gate": result["gate"],
        "created_utc": result["created_utc"],
        "headline": headline,
        "files": [
            {"path": _display_path(path), "sha256": _sha256_file(path), "bytes": _resolve(path).stat().st_size}
            for path in (json_path, md_path)
        ],
        "inputs": [result["input_artifact"]],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--full-eval", type=pathlib.Path, default=DEFAULT_FULL_EVAL)
    parser.add_argument("--run-date", default="2026-05-02")
    args = parser.parse_args()
    payload = build_card(output_dir=args.output_dir, full_eval=args.full_eval, run_date=args.run_date)
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
