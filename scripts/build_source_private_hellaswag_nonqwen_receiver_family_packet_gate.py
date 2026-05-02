from __future__ import annotations

"""Run a strict non-Qwen receiver scout for HellaSwag fixed-byte packets.

This wrapper keeps the existing TinyLlama source-private packet contract, scores
the same HellaSwag rows with a non-Qwen target family, and then reuses the
receiver-family packet evaluator with additional source-destroying packet
controls. It is intentionally sliceable because Phi-3-mini scoring is expensive
on a Mac.
"""

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import random
import sys
import time
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_hellaswag_receiver_family_packet_gate as receiver_gate  # noqa: E402
from scripts import build_source_private_hellaswag_score_packet_headroom as headroom  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_nonqwen_receiver_family_packet_gate_20260502_validation1024_1536"
)
DEFAULT_EVAL_FULL = pathlib.Path(
    "results/source_private_hellaswag_bridge_contract_20260501/official_splits/hellaswag_validation.jsonl"
)
DEFAULT_SOURCE_PACKET_JSONL = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_eval_full_stress_20260502_tinyllama_train512_validation0_10042/"
    "bagged_gate/predictions.jsonl"
)
DEFAULT_SOURCE_PACKET_ARTIFACT = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_eval_full_stress_20260502_tinyllama_train512_validation0_10042/"
    "hellaswag_hidden_innovation_eval_slice_stress.json"
)
DEFAULT_PHI_MODEL = (
    "/Users/sujeethjinesh/.cache/huggingface/hub/"
    "models--microsoft--Phi-3-mini-4k-instruct/snapshots/"
    "f39ac1d28e925b323eae81227eaba4464caced4e"
)
DEFAULT_TARGET_FAMILY = "Phi-3-mini"
CONTROL_FIELDS = (
    "wrong_example_hidden_prediction",
    "candidate_roll_hidden_prediction",
    "zero_hidden_prediction",
    "source_label_prediction",
    "row_shuffle_packet",
    "random_same_byte_packet",
    "target_derived_packet",
    "candidate_derangement_packet",
)
STRICT_TARGET_LIFT = 0.05
STRICT_RECEIVER_PACKET_LIFT = 0.005


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


def _write_jsonl(path: pathlib.Path | str, rows: list[dict[str, Any]]) -> None:
    resolved = _resolve(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _content_digest(rows: list[arc_gate.ArcRow]) -> str:
    return hashlib.sha256("\n".join(row.content_id for row in rows).encode("utf-8")).hexdigest()


def _slice_rows(
    *,
    eval_full_path: pathlib.Path,
    source_packet_jsonl: pathlib.Path,
    output_dir: pathlib.Path,
    slice_start: int,
    slice_rows: int,
) -> tuple[list[arc_gate.ArcRow], list[dict[str, Any]], dict[str, Any]]:
    if slice_start < 0:
        raise ValueError("slice_start must be non-negative")
    if slice_rows <= 0:
        raise ValueError("slice_rows must be positive")
    all_eval_rows = arc_gate._load_rows(_resolve(eval_full_path))
    all_packet_rows = _read_jsonl(source_packet_jsonl)
    if slice_start + slice_rows > len(all_eval_rows) or slice_start + slice_rows > len(all_packet_rows):
        raise ValueError("requested slice exceeds available HellaSwag packet/eval rows")
    eval_rows = all_eval_rows[slice_start : slice_start + slice_rows]
    packet_rows = [dict(row) for row in all_packet_rows[slice_start : slice_start + slice_rows]]
    eval_ids = [row.row_id for row in eval_rows]
    packet_ids = [str(row["row_id"]) for row in packet_rows]
    if eval_ids != packet_ids:
        raise ValueError("eval rows and source packet rows are not aligned")
    slice_path = output_dir / f"hellaswag_validation_rows_{slice_start}_{slice_start + slice_rows}.jsonl"
    _write_jsonl(
        slice_path,
        [
            {
                "id": row.row_id,
                "content_id": row.content_id,
                "question": row.question,
                "choices": list(row.choices),
                "choice_labels": list(row.choice_labels),
                "answer_index": row.answer_index,
                "answer_label": row.answer_label,
                "source_name": row.source_name,
            }
            for row in eval_rows
        ],
    )
    metadata = {
        "eval_full_path": _display_path(eval_full_path),
        "source_packet_jsonl": _display_path(source_packet_jsonl),
        "slice_start": int(slice_start),
        "slice_end_exclusive": int(slice_start + slice_rows),
        "slice_rows": int(slice_rows),
        "slice_path": _display_path(slice_path),
        "slice_sha256": _sha256_file(slice_path),
        "content_digest": _content_digest(eval_rows),
    }
    return eval_rows, packet_rows, metadata


def _load_or_build_target_scores(
    *,
    rows: list[arc_gate.ArcRow],
    score_cache: pathlib.Path,
    target_lm_model: str,
    target_lm_device: str,
    target_lm_dtype: str,
    target_lm_max_length: int,
    target_lm_normalization: str,
    target_lm_prompt_mode: str,
    local_files_only: bool,
) -> tuple[list[list[float]], list[int], dict[str, Any], str]:
    scores, predictions, state, sha = headroom._source_scores(
        rows=rows,
        score_cache=score_cache,
        source_lm_model=target_lm_model,
        source_lm_device=target_lm_device,
        source_lm_dtype=target_lm_dtype,
        source_lm_max_length=target_lm_max_length,
        source_lm_normalization=target_lm_normalization,
        source_lm_prompt_mode=target_lm_prompt_mode,
        local_files_only=local_files_only,
    )
    if sha is None:
        sha = _sha256_file(score_cache)
    return scores, predictions, state, sha


def _augment_packet_controls(
    *,
    packet_rows: list[dict[str, Any]],
    target_predictions: list[int],
    seed: int,
) -> list[dict[str, Any]]:
    selected = [int(row["selected_prediction"]) for row in packet_rows]
    rng = random.Random(seed)
    shuffled = list(selected)
    rng.shuffle(shuffled)
    augmented: list[dict[str, Any]] = []
    for index, row in enumerate(packet_rows):
        source_candidate = int(row["selected_prediction"])
        deranged_offset = 1 + ((index + seed) % 3)
        item = dict(row)
        item["row_shuffle_packet"] = int(shuffled[index])
        item["random_same_byte_packet"] = int(rng.randrange(4))
        item["target_derived_packet"] = int(target_predictions[index])
        item["candidate_derangement_packet"] = int((source_candidate + deranged_offset) % 4)
        augmented.append(item)
    return augmented


def _write_target_global_artifact(
    *,
    path: pathlib.Path,
    score_cache: pathlib.Path,
    slice_metadata: dict[str, Any],
) -> None:
    path.write_text(
        json.dumps(
            {
                "eval_slices": [
                    {
                        "name": "nonqwen_receiver_family_slice",
                        "start": slice_metadata["slice_start"],
                        "end": slice_metadata["slice_end_exclusive"],
                        "rows": slice_metadata["slice_rows"],
                        "score_cache": _display_path(score_cache),
                    }
                ]
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Non-Qwen Receiver-Family Packet Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- target-family transfer gate: `{h['target_family_transfer_gate']}`",
        f"- receiver-improvement gate: `{h['receiver_improvement_gate']}`",
        f"- eval slice: `{h['slice_start']}:{h['slice_end_exclusive']}`",
        f"- rows: `{h['row_count']}`",
        f"- train/eval rows: `{h['train_rows']}/{h['eval_rows']}`",
        f"- target family: `{h['target_family']}`",
        f"- target-only eval accuracy: `{h['target_only_eval_accuracy']:.6f}`",
        f"- packet-only eval accuracy: `{h['packet_only_eval_accuracy']:.6f}`",
        f"- receiver eval accuracy: `{h['receiver_eval_accuracy']:.6f}`",
        f"- packet minus target-only: `{h['packet_minus_target_only']:.6f}`",
        f"- receiver minus packet-only: `{h['receiver_minus_packet_only']:.6f}`",
        f"- target score cache hit: `{h['target_score_cache_hit']}`",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_gate(
    *,
    output_dir: pathlib.Path = DEFAULT_OUTPUT,
    eval_full_path: pathlib.Path = DEFAULT_EVAL_FULL,
    source_packet_jsonl: pathlib.Path = DEFAULT_SOURCE_PACKET_JSONL,
    source_packet_artifact: pathlib.Path = DEFAULT_SOURCE_PACKET_ARTIFACT,
    slice_start: int = 1024,
    slice_rows: int = 512,
    train_prefix_rows: int = 128,
    bootstrap_samples: int = 500,
    target_score_cache: pathlib.Path | None = None,
    target_family: str = DEFAULT_TARGET_FAMILY,
    target_lm_model: str = DEFAULT_PHI_MODEL,
    target_lm_device: str = "mps",
    target_lm_dtype: str = "float16",
    target_lm_max_length: int = 256,
    target_lm_normalization: str = "mean",
    target_lm_prompt_mode: str = "continuation",
    local_files_only: bool = True,
    control_seed: int = 20260502,
    run_date: str = "2026-05-02",
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    eval_rows, packet_rows, slice_metadata = _slice_rows(
        eval_full_path=eval_full_path,
        source_packet_jsonl=source_packet_jsonl,
        output_dir=output_dir,
        slice_start=slice_start,
        slice_rows=slice_rows,
    )
    score_cache = _resolve(target_score_cache) if target_score_cache is not None else output_dir / "target_score_cache.json"
    target_scores, target_predictions, target_model_state, target_score_sha = _load_or_build_target_scores(
        rows=eval_rows,
        score_cache=score_cache,
        target_lm_model=target_lm_model,
        target_lm_device=target_lm_device,
        target_lm_dtype=target_lm_dtype,
        target_lm_max_length=target_lm_max_length,
        target_lm_normalization=target_lm_normalization,
        target_lm_prompt_mode=target_lm_prompt_mode,
        local_files_only=local_files_only,
    )
    augmented_packets = _augment_packet_controls(
        packet_rows=packet_rows,
        target_predictions=target_predictions,
        seed=control_seed,
    )
    sliced_packet_jsonl = output_dir / "tinyllama_source_packet_slice_augmented.jsonl"
    _write_jsonl(sliced_packet_jsonl, augmented_packets)
    target_global_artifact = output_dir / "target_global_artifact.json"
    _write_target_global_artifact(
        path=target_global_artifact,
        score_cache=score_cache,
        slice_metadata=slice_metadata,
    )
    receiver_payload = receiver_gate.build_gate(
        output_dir=output_dir / "receiver_gate",
        source_packet_jsonl=sliced_packet_jsonl,
        target_global_artifact=target_global_artifact,
        source_packet_artifact=source_packet_artifact,
        source_family="TinyLlama",
        target_family=target_family,
        control_fields=CONTROL_FIELDS,
        train_prefix_rows=train_prefix_rows,
        bootstrap_samples=bootstrap_samples,
        run_date=run_date,
    )
    rh = receiver_payload["headline"]
    packet_minus_target = float(rh["packet_only_eval_accuracy"] - rh["target_only_eval_accuracy"])
    strict_source_utility_gate = bool(packet_minus_target >= STRICT_TARGET_LIFT)
    pass_gate = bool(
        strict_source_utility_gate
        and receiver_payload["target_family_transfer_gate"]
        and receiver_payload["receiver_improvement_gate"]
    )
    headline = {
        "slice_start": int(slice_start),
        "slice_end_exclusive": int(slice_start + slice_rows),
        "row_count": int(slice_rows),
        "train_rows": int(train_prefix_rows),
        "eval_rows": int(slice_rows - train_prefix_rows),
        "source_family": "TinyLlama",
        "target_family": target_family,
        "target_only_eval_accuracy": rh["target_only_eval_accuracy"],
        "packet_only_eval_accuracy": rh["packet_only_eval_accuracy"],
        "receiver_eval_accuracy": rh["receiver_eval_accuracy"],
        "packet_minus_target_only": packet_minus_target,
        "receiver_minus_target_only": rh["receiver_minus_target_only"],
        "receiver_ci95_low_vs_target_only": rh["receiver_ci95_low_vs_target_only"],
        "receiver_minus_packet_only": rh["receiver_minus_packet_only"],
        "receiver_ci95_low_vs_packet_only": rh["receiver_ci95_low_vs_packet_only"],
        "target_or_packet_oracle_eval_accuracy": rh["target_or_packet_oracle_eval_accuracy"],
        "source_utility_gate": strict_source_utility_gate,
        "target_family_transfer_gate": receiver_payload["target_family_transfer_gate"],
        "receiver_improvement_gate": receiver_payload["receiver_improvement_gate"],
        "strict_target_lift_required": STRICT_TARGET_LIFT,
        "strict_receiver_packet_lift_required": STRICT_RECEIVER_PACKET_LIFT,
        "selected_receiver_kind": rh["selected_receiver_kind"],
        "target_score_cache_hit": bool(target_model_state.get("cache_hit")),
        "target_score_latency_s": target_model_state.get("latency_s"),
        "target_lm_device": target_model_state.get("device", target_lm_device),
        "target_lm_dtype": target_lm_dtype,
        "native_systems_complete": False,
    }
    interpretation = (
        "This is a strict receiver-family scout: the source packet comes from TinyLlama, while the "
        "target-side scores come from a non-Qwen model family. A pass would show that the same "
        "fixed-byte packet has utility for a different target family and that a target-aware receiver "
        "beats the packet-only baseline. A fail still helps the paper by narrowing the claim to "
        "source-private packet utility rather than learned cross-family latent reasoning."
    )
    payload = {
        "gate": "source_private_hellaswag_nonqwen_receiver_family_packet_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "pass_gate": pass_gate,
        "pass_rule": (
            "Strict pass requires packet-only to beat Phi target-only by >=0.05, the selected "
            "target-family receiver to beat Phi target-only with positive paired CI, and the selected "
            "receiver to beat packet-only by >=0.005 with positive paired CI. Controls include wrong "
            "example hidden, candidate-roll hidden, zero hidden, source-label copy, row-shuffled packet, "
            "random same-byte packet, target-derived packet, and candidate derangement."
        ),
        "headline": headline,
        "slice_metadata": slice_metadata,
        "source_packet": {
            "full_predictions_jsonl": _display_path(source_packet_jsonl),
            "full_predictions_sha256": _sha256_file(source_packet_jsonl),
            "sliced_augmented_jsonl": _display_path(sliced_packet_jsonl),
            "sliced_augmented_sha256": _sha256_file(sliced_packet_jsonl),
            "artifact_path": _display_path(source_packet_artifact),
            "artifact_sha256": _sha256_file(source_packet_artifact),
            "raw_payload_bytes": 2,
            "framed_record_bytes": 5,
            "exposes_source_text": False,
            "exposes_source_kv": False,
            "exposes_raw_hidden": False,
            "exposes_raw_scores": False,
        },
        "target_scores": {
            "score_cache": _display_path(score_cache),
            "score_cache_sha256": target_score_sha,
            "target_global_artifact": _display_path(target_global_artifact),
            "target_global_artifact_sha256": _sha256_file(target_global_artifact),
            "model_state": target_model_state,
        },
        "receiver_gate": receiver_payload,
        "interpretation": interpretation,
        "timing": {"total_seconds": float(time.perf_counter() - started)},
    }
    json_path = output_dir / "hellaswag_nonqwen_receiver_family_packet_gate.json"
    md_path = output_dir / "hellaswag_nonqwen_receiver_family_packet_gate.md"
    manifest_path = output_dir / "manifest.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload)
    manifest = {
        "gate": payload["gate"],
        "created_utc": payload["created_utc"],
        "headline": headline,
        "files": [
            {"path": _display_path(path), "sha256": _sha256_file(path), "bytes": _resolve(path).stat().st_size}
            for path in (
                json_path,
                md_path,
                manifest_path,
                sliced_packet_jsonl,
                target_global_artifact,
                score_cache,
                output_dir / "receiver_gate" / "hellaswag_receiver_family_packet_gate.json",
                output_dir / "receiver_gate" / "manifest.json",
            )
            if _resolve(path).exists()
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-full-path", type=pathlib.Path, default=DEFAULT_EVAL_FULL)
    parser.add_argument("--source-packet-jsonl", type=pathlib.Path, default=DEFAULT_SOURCE_PACKET_JSONL)
    parser.add_argument("--source-packet-artifact", type=pathlib.Path, default=DEFAULT_SOURCE_PACKET_ARTIFACT)
    parser.add_argument("--slice-start", type=int, default=1024)
    parser.add_argument("--slice-rows", type=int, default=512)
    parser.add_argument("--train-prefix-rows", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--target-score-cache", type=pathlib.Path, default=None)
    parser.add_argument("--target-family", default=DEFAULT_TARGET_FAMILY)
    parser.add_argument("--target-lm-model", default=DEFAULT_PHI_MODEL)
    parser.add_argument("--target-lm-device", default="mps")
    parser.add_argument("--target-lm-dtype", default="float16")
    parser.add_argument("--target-lm-max-length", type=int, default=256)
    parser.add_argument("--target-lm-normalization", choices=("mean", "sum"), default="mean")
    parser.add_argument("--target-lm-prompt-mode", choices=("qa", "continuation", "generic_mcq"), default="continuation")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--control-seed", type=int, default=20260502)
    parser.add_argument("--run-date", default="2026-05-02")
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        eval_full_path=args.eval_full_path,
        source_packet_jsonl=args.source_packet_jsonl,
        source_packet_artifact=args.source_packet_artifact,
        slice_start=args.slice_start,
        slice_rows=args.slice_rows,
        train_prefix_rows=args.train_prefix_rows,
        bootstrap_samples=args.bootstrap_samples,
        target_score_cache=args.target_score_cache,
        target_family=args.target_family,
        target_lm_model=args.target_lm_model,
        target_lm_device=args.target_lm_device,
        target_lm_dtype=args.target_lm_dtype,
        target_lm_max_length=args.target_lm_max_length,
        target_lm_normalization=args.target_lm_normalization,
        target_lm_prompt_mode=args.target_lm_prompt_mode,
        local_files_only=args.local_files_only,
        control_seed=args.control_seed,
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
