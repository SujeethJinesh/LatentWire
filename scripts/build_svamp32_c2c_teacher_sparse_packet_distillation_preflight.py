#!/usr/bin/env python3
"""Build the SVAMP32 C2C-teacher sparse-packet distillation preflight.

This is an evidence aggregator, not a new receiver. It checks whether the
frozen SVAMP32 C2C teacher surface currently supports a deployable sparse
packet claim, while preserving the earlier oracle-syndrome bound as headroom.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import random
import sys
from dataclasses import dataclass
from datetime import date
from typing import Any, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import harness_common as harness


@dataclass(frozen=True)
class RowSpec:
    label: str
    path: pathlib.Path
    method: str


def _resolve(path: str | pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_spec(spec: str) -> RowSpec:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"Expected label=path=...,method=..., got {spec!r}")
    label, raw_fields = spec.split("=", 1)
    fields: dict[str, str] = {}
    for item in raw_fields.split(","):
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Expected key=value in {spec!r}; got {item!r}")
        key, value = item.split("=", 1)
        fields[key.strip()] = value.strip()
    if not label or not fields.get("path") or not fields.get("method"):
        raise argparse.ArgumentTypeError(f"Spec needs label, path, and method: {spec!r}")
    return RowSpec(label=label, path=_resolve(fields["path"]), method=fields["method"])


def _records_for_method(spec: RowSpec) -> list[dict[str, Any]]:
    records = _read_jsonl(spec.path)
    raw_grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        raw_grouped.setdefault(str(row["method"]), []).append(row)
    if spec.method in raw_grouped:
        return [dict(row) for row in raw_grouped[spec.method]]
    grouped = harness.group_by_method(records)
    if spec.method not in grouped:
        raise KeyError(
            f"Method {spec.method!r} not found in {spec.path}; "
            f"raw_available={sorted(raw_grouped)}, normalized_available={sorted(grouped)}"
        )
    return [dict(row) for row in grouped[spec.method]]


def _by_id(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for row in records:
        example_id = str(row["example_id"])
        if example_id in out:
            duplicates.add(example_id)
        out[example_id] = dict(row)
    if duplicates:
        raise ValueError(f"Duplicate example_id values: {sorted(duplicates)}")
    return out


def _ordered(records: Sequence[dict[str, Any]], reference_ids: Sequence[str]) -> list[dict[str, Any]]:
    by_id = _by_id(records)
    missing = [example_id for example_id in reference_ids if example_id not in by_id]
    if missing:
        raise ValueError(f"Missing reference IDs: {missing}")
    return [by_id[example_id] for example_id in reference_ids]


def _accuracy(correct: Sequence[bool]) -> float:
    return float(sum(int(item) for item in correct) / max(len(correct), 1))


def _paired_ci_from_bools(
    selected_ok: Sequence[bool],
    baseline_ok: Sequence[bool],
    *,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    if len(selected_ok) != len(baseline_ok):
        raise ValueError("paired CI inputs must have the same length")
    diff = [float(sel) - float(base) for sel, base in zip(selected_ok, baseline_ok, strict=True)]
    rng = random.Random(seed)
    draws: list[float] = []
    n = len(diff)
    if n == 0:
        return {"delta": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "helps": 0, "harms": 0}
    for _ in range(samples):
        draws.append(sum(diff[rng.randrange(n)] for _ in range(n)) / n)
    draws.sort()
    return {
        "delta": float(sum(diff) / n),
        "ci95_low": float(draws[int(0.025 * (samples - 1))]),
        "ci95_high": float(draws[int(0.975 * (samples - 1))]),
        "helps": int(sum(int(sel and not base) for sel, base in zip(selected_ok, baseline_ok, strict=True))),
        "harms": int(sum(int(base and not sel) for sel, base in zip(selected_ok, baseline_ok, strict=True))),
    }


def _summary_from_records(
    *,
    label: str,
    records: Sequence[dict[str, Any]],
    target_records: Sequence[dict[str, Any]],
    teacher_only_ids: set[str],
    clean_residual_ids: set[str],
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    correct = [bool(row.get("correct", False)) for row in records]
    target_correct = [bool(row.get("correct", False)) for row in target_records]
    correct_ids = {str(row["example_id"]) for row, ok in zip(records, correct, strict=True) if ok}
    ci = _paired_ci_from_bools(correct, target_correct, seed=seed, samples=bootstrap_samples)
    return {
        "label": label,
        "kind": "observed_generation_baseline",
        "correct_count": int(sum(correct)),
        "accuracy": _accuracy(correct),
        "wins_vs_target": int(ci["helps"]),
        "losses_vs_target": int(ci["harms"]),
        "paired_vs_target": ci,
        "teacher_only_recovered_count": len(correct_ids & teacher_only_ids),
        "teacher_only_recovered_ids": sorted(correct_ids & teacher_only_ids),
        "clean_residual_recovered_count": len(correct_ids & clean_residual_ids),
        "clean_residual_recovered_ids": sorted(correct_ids & clean_residual_ids),
    }


def _union_summary(
    *,
    label: str,
    candidate_records: Sequence[Sequence[dict[str, Any]]],
    target_records: Sequence[dict[str, Any]],
    teacher_only_ids: set[str],
    clean_residual_ids: set[str],
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    selected_ok: list[bool] = []
    selected_ids: set[str] = set()
    for row_tuple in zip(*candidate_records, strict=True):
        row_ids = {str(row["example_id"]) for row in row_tuple}
        if len(row_ids) != 1:
            raise ValueError(f"Union candidate ID mismatch: {sorted(row_ids)}")
        ok = any(bool(row.get("correct", False)) for row in row_tuple)
        selected_ok.append(ok)
        if ok:
            selected_ids.add(next(iter(row_ids)))
    target_ok = [bool(row.get("correct", False)) for row in target_records]
    ci = _paired_ci_from_bools(selected_ok, target_ok, seed=seed, samples=bootstrap_samples)
    return {
        "label": label,
        "kind": "oracle_candidate_union_bound_not_deployable",
        "correct_count": int(sum(selected_ok)),
        "accuracy": _accuracy(selected_ok),
        "wins_vs_target": int(ci["helps"]),
        "losses_vs_target": int(ci["harms"]),
        "paired_vs_target": ci,
        "teacher_only_recovered_count": len(selected_ids & teacher_only_ids),
        "teacher_only_recovered_ids": sorted(selected_ids & teacher_only_ids),
        "clean_residual_recovered_count": len(selected_ids & clean_residual_ids),
        "clean_residual_recovered_ids": sorted(selected_ids & clean_residual_ids),
    }


def _condition_summary(run: dict[str, Any], condition: str = "matched") -> dict[str, Any]:
    conditions = run.get("condition_summaries", {})
    if condition not in conditions:
        raise KeyError(f"Condition {condition!r} missing from run")
    return conditions[condition]


def _run_record(
    *,
    label: str,
    kind: str,
    artifact_path: pathlib.Path,
    run: dict[str, Any],
) -> dict[str, Any]:
    matched = _condition_summary(run, "matched")
    control_clean_union_ids = sorted(str(item) for item in run.get("control_clean_union_ids", []))
    payload_bytes = int(run.get("syndrome_bytes", 0) or 0)
    return {
        "label": label,
        "kind": kind,
        "artifact_path": _display_path(artifact_path),
        "status": str(run.get("status", "")),
        "correct_count": int(matched.get("correct_count", 0)),
        "accuracy": float(int(matched.get("correct_count", 0)) / 32.0),
        "teacher_only_recovered_count": int(matched.get("teacher_only_correct_count", 0)),
        "teacher_only_recovered_ids": list(matched.get("teacher_only_correct_ids", [])),
        "clean_residual_recovered_count": int(matched.get("clean_correct_count", 0)),
        "clean_residual_recovered_ids": list(matched.get("clean_correct_ids", [])),
        "source_necessary_clean_count": len(run.get("source_necessary_clean_ids", [])),
        "source_necessary_clean_ids": list(run.get("source_necessary_clean_ids", [])),
        "control_clean_union_count": len(control_clean_union_ids),
        "control_clean_union_ids": control_clean_union_ids,
        "target_self_correct_count": int(matched.get("target_self_correct_count", 0)),
        "payload_bytes": payload_bytes,
        "framed_bytes": max(1, payload_bytes),
        "cacheline_rounded_bytes": 64 if payload_bytes else 0,
    }


def _best_run_from_json(path: pathlib.Path) -> dict[str, Any]:
    payload = _read_json(path)
    runs = payload.get("runs")
    if runs is None:
        run = payload.get("run")
        if run is None:
            raise ValueError(f"No run/runs in {path}")
        return dict(run)
    if not runs:
        raise ValueError(f"Empty runs in {path}")
    return max(
        (dict(run) for run in runs),
        key=lambda run: (
            len(run.get("source_necessary_clean_ids", [])),
            int(_condition_summary(run, "matched").get("correct_count", 0)),
            -int(run.get("syndrome_bits", 999999)),
        ),
    )


def _artifact_summary(label: str, kind: str, path: pathlib.Path) -> dict[str, Any]:
    return _run_record(
        label=label,
        kind=kind,
        artifact_path=path,
        run=_best_run_from_json(path),
    )


def _load_target_sets(path: pathlib.Path, teacher_probe_path: pathlib.Path) -> tuple[set[str], set[str]]:
    target_set = _read_json(path)
    ids = target_set.get("ids", {})
    teacher_only = {str(item) for item in ids.get("teacher_only", [])}
    clean_residual = {str(item) for item in ids.get("clean_residual_targets", [])}
    if not teacher_only:
        teacher_probe = _read_json(teacher_probe_path)
        teacher_only = {str(item) for item in teacher_probe.get("teacher_only_ids", [])}
    if not clean_residual:
        clean_residual = set(teacher_only)
    return teacher_only, clean_residual


def _build_manifest(output_json: pathlib.Path, output_md: pathlib.Path, payload: dict[str, Any]) -> dict[str, Any]:
    artifacts = {
        _display_path(output_json): {"sha256": _sha256(output_json)},
        _display_path(output_md): {"sha256": _sha256(output_md)},
    }
    for key, raw_path in payload["artifacts"].items():
        path = _resolve(raw_path)
        if path.exists():
            artifacts[_display_path(path)] = {"sha256": _sha256(path)}
    return {
        "date": payload["date"],
        "status": payload["status"],
        "artifacts": artifacts,
    }


def _markdown(payload: dict[str, Any]) -> str:
    h = payload["headline"]
    lines = [
        "# SVAMP32 C2C Teacher Sparse-Packet Distillation Preflight",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- deployable distillation pass: `{h['deployable_distillation_pass']}`",
        f"- oracle sparse sidecar alive: `{h['oracle_sparse_sidecar_alive']}`",
        f"- target: `{h['target_correct']}/{h['n']}`",
        f"- C2C teacher: `{h['teacher_correct']}/{h['n']}`",
        f"- clean residual targets: `{h['clean_residual_count']}`",
        "",
        "## Evidence Table",
        "",
        "| Row | Kind | Correct | Teacher-only | Clean residual | Source-necessary clean | Control clean | Bytes | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["evidence_rows"]:
        lines.append(
            f"| `{row['label']}` | `{row['kind']}` | {row['correct_count']}/{h['n']} | "
            f"{row['teacher_only_recovered_count']} | {row['clean_residual_recovered_count']} | "
            f"{row.get('source_necessary_clean_count', 0)} | "
            f"{row.get('control_clean_union_count', 0)} | {row.get('framed_bytes', 0)} | "
            f"`{row.get('status', '')}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- The dense C2C teacher remains the only strong complementary surface on this frozen slice.",
            "- The 1-byte C2C-derived syndrome sidecar remains useful as an oracle bound, not as a deployable method.",
            "- Existing deployable predictors from source final answers, source hidden summaries, source-token query bottlenecks, and C2C prefill traces do not recover the clean C2C residual IDs.",
            "- Do not claim sparse packets beat or solve C2C from this evidence. The next method must predict the C2C residual from a genuinely source-causal signal or collect a richer dense-teacher trace.",
            "",
            "## Next Gate",
            "",
            payload["next_gate"],
            "",
        ]
    )
    return "\n".join(lines)


def analyze(
    *,
    target_spec: RowSpec,
    source_spec: RowSpec,
    text_spec: RowSpec,
    teacher_spec: RowSpec,
    teacher_probe_path: pathlib.Path,
    target_set_path: pathlib.Path,
    targetpool_oracle_path: pathlib.Path,
    augmented_oracle_path: pathlib.Path,
    source_latent_paths: Sequence[pathlib.Path],
    learned_syndrome_paths: Sequence[pathlib.Path],
    c2c_mechanism_paths: Sequence[pathlib.Path],
    output_json: pathlib.Path,
    output_md: pathlib.Path,
    manifest_path: pathlib.Path,
    run_date: str,
    bootstrap_samples: int,
) -> dict[str, Any]:
    target_records = _records_for_method(target_spec)
    reference_ids = [str(row["example_id"]) for row in target_records]
    source_records = _ordered(_records_for_method(source_spec), reference_ids)
    text_records = _ordered(_records_for_method(text_spec), reference_ids)
    teacher_records = _ordered(_records_for_method(teacher_spec), reference_ids)
    target_records = _ordered(target_records, reference_ids)
    n = len(reference_ids)
    if n != 32:
        raise ValueError(f"This preflight expects the frozen SVAMP32 surface; got n={n}")

    teacher_only_ids, clean_residual_ids = _load_target_sets(target_set_path, teacher_probe_path)
    base_seed = 52060505
    evidence_rows: list[dict[str, Any]] = [
        _summary_from_records(
            label="target_only",
            records=target_records,
            target_records=target_records,
            teacher_only_ids=teacher_only_ids,
            clean_residual_ids=clean_residual_ids,
            seed=base_seed,
            bootstrap_samples=bootstrap_samples,
        ),
        _summary_from_records(
            label="source_alone",
            records=source_records,
            target_records=target_records,
            teacher_only_ids=teacher_only_ids,
            clean_residual_ids=clean_residual_ids,
            seed=base_seed + 1,
            bootstrap_samples=bootstrap_samples,
        ),
        _summary_from_records(
            label="same_byte_text_to_text",
            records=text_records,
            target_records=target_records,
            teacher_only_ids=teacher_only_ids,
            clean_residual_ids=clean_residual_ids,
            seed=base_seed + 2,
            bootstrap_samples=bootstrap_samples,
        ),
        _summary_from_records(
            label="dense_c2c_teacher",
            records=teacher_records,
            target_records=target_records,
            teacher_only_ids=teacher_only_ids,
            clean_residual_ids=clean_residual_ids,
            seed=base_seed + 3,
            bootstrap_samples=bootstrap_samples,
        ),
        _union_summary(
            label="target_source_text_oracle_union",
            candidate_records=[target_records, source_records, text_records],
            target_records=target_records,
            teacher_only_ids=teacher_only_ids,
            clean_residual_ids=clean_residual_ids,
            seed=base_seed + 4,
            bootstrap_samples=bootstrap_samples,
        ),
        _artifact_summary(
            "oracle_c2c_syndrome_targetpool",
            "oracle_sparse_packet_bound_not_deployable",
            targetpool_oracle_path,
        ),
        _artifact_summary(
            "oracle_c2c_syndrome_augmentedpool",
            "oracle_sparse_packet_bound_not_deployable",
            augmented_oracle_path,
        ),
    ]

    for idx, path in enumerate(source_latent_paths):
        evidence_rows.append(
            _artifact_summary(f"source_latent_syndrome_{idx}", "deployable_source_hidden_probe", path)
        )
    for idx, path in enumerate(learned_syndrome_paths):
        evidence_rows.append(
            _artifact_summary(f"learned_query_syndrome_{idx}", "deployable_source_token_probe", path)
        )
    for idx, path in enumerate(c2c_mechanism_paths):
        evidence_rows.append(
            _artifact_summary(f"c2c_prefill_trace_syndrome_{idx}", "deployable_c2c_trace_probe", path)
        )

    deployable_rows = [
        row
        for row in evidence_rows
        if row["kind"]
        in {
            "observed_generation_baseline",
            "oracle_candidate_union_bound_not_deployable",
            "deployable_source_hidden_probe",
            "deployable_source_token_probe",
            "deployable_c2c_trace_probe",
        }
        and row["label"] not in {"target_only", "dense_c2c_teacher", "target_source_text_oracle_union"}
    ]
    deployable_distillation_pass = any(
        int(row.get("source_necessary_clean_count", 0)) >= 2
        and int(row.get("correct_count", 0)) >= 14
        and int(row.get("control_clean_union_count", 0)) == 0
        for row in deployable_rows
    )
    oracle_sparse_sidecar_alive = any(
        row["kind"] == "oracle_sparse_packet_bound_not_deployable"
        and int(row.get("source_necessary_clean_count", 0)) >= 2
        and int(row.get("correct_count", 0)) >= 14
        for row in evidence_rows
    )
    target_correct = int(sum(bool(row.get("correct", False)) for row in target_records))
    teacher_correct = int(sum(bool(row.get("correct", False)) for row in teacher_records))
    status = (
        "c2c_teacher_sparse_packet_distillation_deployable_pass"
        if deployable_distillation_pass
        else "c2c_teacher_sparse_packet_distillation_preflight_fails_deployable_method_oracle_bound_alive"
        if oracle_sparse_sidecar_alive
        else "c2c_teacher_sparse_packet_distillation_preflight_fails_no_oracle_bound"
    )

    artifacts = {
        "target": _display_path(target_spec.path),
        "source": _display_path(source_spec.path),
        "text": _display_path(text_spec.path),
        "teacher": _display_path(teacher_spec.path),
        "teacher_probe": _display_path(teacher_probe_path),
        "target_set": _display_path(target_set_path),
        "targetpool_oracle": _display_path(targetpool_oracle_path),
        "augmented_oracle": _display_path(augmented_oracle_path),
    }
    payload: dict[str, Any] = {
        "date": run_date,
        "status": status,
        "artifacts": artifacts,
        "reference_ids": reference_ids,
        "headline": {
            "n": n,
            "target_correct": target_correct,
            "teacher_correct": teacher_correct,
            "teacher_only_count": len(teacher_only_ids),
            "clean_residual_count": len(clean_residual_ids),
            "deployable_distillation_pass": deployable_distillation_pass,
            "oracle_sparse_sidecar_alive": oracle_sparse_sidecar_alive,
        },
        "evidence_rows": evidence_rows,
        "claim_boundary": {
            "can_claim": [
                "dense C2C teacher has target-complementary headroom on the frozen SVAMP32 surface",
                "a compact C2C-derived residue syndrome is an oracle sparse-packet bound",
                "current deployable source-derived predictors do not recover the oracle bound",
            ],
            "cannot_claim": [
                "Sparse packets beat C2C",
                "Sparse packets solve latent model-to-model communication",
                "C2C superiority on latency, HBM, energy, or throughput",
            ],
        },
        "next_gate": (
            "Collect or generate richer generation-time dense-teacher traces, then train a "
            "source-causal sparse residual packet that must recover at least 2/6 clean "
            "C2C residual IDs while preserving target-self wins and passing zero/shuffle/"
            "label-shuffle/target-only/slots-only controls."
        ),
    }

    _write_json(output_json, payload)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_markdown(payload), encoding="utf-8")
    manifest = _build_manifest(output_json, output_md, payload)
    _write_json(manifest_path, manifest)
    return payload


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=_parse_spec,
        default=_parse_spec(
            "target=path=results/svamp_exactid_baselines32_20260423/target_alone.jsonl,method=target_alone"
        ),
    )
    parser.add_argument(
        "--source",
        type=_parse_spec,
        default=_parse_spec(
            "source=path=results/svamp_exactid_baselines32_20260423/source_alone.jsonl,method=source_alone"
        ),
    )
    parser.add_argument(
        "--text",
        type=_parse_spec,
        default=_parse_spec(
            "t2t=path=results/svamp_exactid_baselines32_20260423/text_to_text.jsonl,method=text_to_text"
        ),
    )
    parser.add_argument(
        "--teacher",
        type=_parse_spec,
        default=_parse_spec(
            "c2c=path=results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl,method=c2c_generate"
        ),
    )
    parser.add_argument(
        "--teacher-probe-json",
        type=_resolve,
        default=_resolve("results/svamp_exactid_baselines32_20260423/c2c_teacher_innovation_probe.json"),
    )
    parser.add_argument(
        "--target-set-json",
        type=_resolve,
        default=_resolve(
            "results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json"
        ),
    )
    parser.add_argument(
        "--targetpool-oracle-json",
        type=_resolve,
        default=_resolve("results/svamp32_syndrome_sidecar_probe_20260424/targetpool_syndrome_probe.json"),
    )
    parser.add_argument(
        "--augmented-oracle-json",
        type=_resolve,
        default=_resolve("results/svamp32_syndrome_sidecar_probe_20260424/augmentedpool_syndrome_probe.json"),
    )
    parser.add_argument(
        "--source-latent-json",
        type=_resolve,
        action="append",
        default=[
            _resolve("results/svamp32_source_latent_syndrome_probe_20260424/qwen25_05b_last_targetpool_probe.json"),
            _resolve("results/svamp32_source_latent_syndrome_probe_20260424/qwen25_05b_mid_last_targetpool_probe.json"),
        ],
    )
    parser.add_argument(
        "--learned-syndrome-json",
        type=_resolve,
        action="append",
        default=[
            _resolve("results/svamp32_learned_syndrome_probe_20260424/qbottleneck_q4_h16_f8_seed1_targetpool_probe.json"),
            _resolve("results/svamp32_learned_syndrome_probe_20260424/qbottleneck_q8_h64_f8_seed1_targetpool_probe.json"),
        ],
    )
    parser.add_argument(
        "--c2c-mechanism-json",
        type=_resolve,
        action="append",
        default=[
            _resolve("results/svamp32_c2c_mechanism_syndrome_probe_20260426/prefill_scalar_trace_targetpool_probe.json"),
            _resolve("results/svamp32_c2c_mechanism_syndrome_probe_20260426/prefill_residual_trace_targetpool_probe.json"),
        ],
    )
    parser.add_argument(
        "--output-json",
        type=_resolve,
        default=_resolve(
            f"results/svamp32_c2c_teacher_sparse_packet_distillation_preflight_{date.today():%Y%m%d}/summary.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        type=_resolve,
        default=_resolve(
            f"results/svamp32_c2c_teacher_sparse_packet_distillation_preflight_{date.today():%Y%m%d}/summary.md"
        ),
    )
    parser.add_argument(
        "--manifest-json",
        type=_resolve,
        default=_resolve(
            f"results/svamp32_c2c_teacher_sparse_packet_distillation_preflight_{date.today():%Y%m%d}/manifest.json"
        ),
    )
    parser.add_argument("--date", default=f"{date.today():%Y-%m-%d}")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args(argv)

    return analyze(
        target_spec=args.target,
        source_spec=args.source,
        text_spec=args.text,
        teacher_spec=args.teacher,
        teacher_probe_path=args.teacher_probe_json,
        target_set_path=args.target_set_json,
        targetpool_oracle_path=args.targetpool_oracle_json,
        augmented_oracle_path=args.augmented_oracle_json,
        source_latent_paths=args.source_latent_json,
        learned_syndrome_paths=args.learned_syndrome_json,
        c2c_mechanism_paths=args.c2c_mechanism_json,
        output_json=args.output_json,
        output_md=args.output_md,
        manifest_path=args.manifest_json,
        run_date=args.date,
        bootstrap_samples=args.bootstrap_samples,
    )


if __name__ == "__main__":
    main()
