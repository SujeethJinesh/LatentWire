"""Build capture-manifest templates from the active hybrid trace plans.

These manifests are not model evidence. They remove one manual step before
SSQ-LR/HORN/HBSM real packets by enumerating the exact metadata entries that a
tensor or sensitivity dump must fill.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACE_PLAN_DIR = ROOT / "experimental/shared/results/hybrid_trace_plan_20260507"
DEFAULT_OUTPUT_DIR = ROOT / "experimental/shared/results/hybrid_capture_manifests_20260507"
PROJECTS = ("ssq_lr", "horn", "hbsm")
TEMPLATE_MARKER = "TO_FILL_BEFORE_CAPTURE"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _slug(text: str) -> str:
    return (
        text.replace("/", "--")
        .replace(" ", "_")
        .replace(".", "-")
        .replace("_", "-")
        .lower()
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path} line {line_number} must be a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def _group_by_model(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        model_id = str(row.get("model_id", "")).strip()
        if not model_id:
            raise ValueError("trace-plan row missing model_id")
        grouped[model_id].append(row)
    return dict(sorted(grouped.items()))


def _single_architecture_hash(rows: list[dict[str, Any]], *, model_id: str) -> str:
    hashes = {str(row.get("architecture_map_hash", "")).strip() for row in rows}
    if len(hashes) != 1 or not next(iter(hashes)):
        raise ValueError(f"{model_id} rows must have exactly one architecture_map_hash")
    return next(iter(hashes))


def _base_metadata(
    *,
    project: str,
    model_id: str,
    rows: list[dict[str, Any]],
    trace_plan_dir: Path,
    trace_plan_hash: str,
    trace_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "_template_only": True,
        "fill_before_use": [
            "served_model_id",
            "model_revision",
            "tokenizer_revision",
            "context_lengths",
            "dtype",
            "device",
            "command",
        ],
        "model_id": model_id,
        "canonical_model_id": model_id,
        "served_model_id": TEMPLATE_MARKER,
        "model_revision": TEMPLATE_MARKER,
        "tokenizer_revision": TEMPLATE_MARKER,
        "prompt_source": str(trace_config["prompt_source"]),
        "prompt_ids_hash": str(trace_config["prompt_ids_hash"]),
        "seed_list": [1],
        "context_lengths": [2048],
        "dtype": TEMPLATE_MARKER,
        "device": TEMPLATE_MARKER,
        "command": TEMPLATE_MARKER,
        "architecture_map_hash": _single_architecture_hash(rows, model_id=model_id),
        "trace_plan_hash": trace_plan_hash,
        "trace_plan_config_path": _repo_label(trace_plan_dir / "config.json"),
        "trace_plan_path": _repo_label(trace_plan_dir / f"{project}_trace_plan.jsonl"),
        "capture_manifest_claim_boundary": [
            "capture-manifest-template",
            "not model evidence",
            "not GPU evidence",
        ],
        "planned_entry_count": len(rows),
    }


def _ssq_lr_entries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "tensor": str(row["tensor_name"]),
            "prompt_id": str(row["prompt_id"]),
            "layer": int(row["layer"]),
            "layer_kind": str(row["layer_kind"]),
            "position_bucket": str(row["position_bucket"]),
            "state_tensor_kind": str(row["state_tensor_kind"]),
            "control_type": str(row["control_type"]),
        }
        for row in rows
    ]


def _horn_entries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries = []
    for row in rows:
        entry = {
            "tensor": str(row["tensor_name"]),
            "prompt_id": str(row["prompt_id"]),
            "prompt_cluster_id": str(row["prompt_cluster_id"]),
            "layer_left": int(row["layer_left"]),
            "layer_right": int(row["layer_right"]),
            "direction": str(row["direction"]),
            "matched_boundary_direction": str(row["matched_boundary_direction"]),
            "boundary_index": int(row["boundary_index"]),
            "pre_norm_position": str(row["pre_norm_position"]),
            "post_norm_position": str(row["post_norm_position"]),
            "control_type": str(row["control_type"]),
        }
        if str(row["control_type"]) == "permuted_direction":
            entry["tensor_alias_of"] = str(row["tensor_name"]).replace(
                "/permuted_label", "/observed"
            )
        entries.append(entry)
    return entries


def _hbsm_entry_templates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    templates = []
    for row in rows:
        templates.append(
            {
                "prompt_id": str(row["prompt_id"]),
                "layer": int(row["layer"]),
                "boundary_flag": bool(row["boundary_flag"]),
                "precision_perturbation": str(row["precision_perturbation"]),
                "kl_or_nll_drift": TEMPLATE_MARKER,
                "cheap_predictor": TEMPLATE_MARKER,
                "parameter_count": TEMPLATE_MARKER,
                "weight_norm": TEMPLATE_MARKER,
                "top_decile_flag": TEMPLATE_MARKER,
                "random_top_decile": TEMPLATE_MARKER,
                "train_test_split": str(row["train_test_split"]),
                "control_type": str(row["control_type"]),
                "required_metric_fields": [
                    "kl_or_nll_drift",
                    "cheap_predictor",
                    "parameter_count",
                    "weight_norm",
                    "top_decile_flag",
                    "random_top_decile",
                ],
            }
        )
    return templates


def build_capture_manifests(
    *,
    trace_plan_dir: Path = DEFAULT_TRACE_PLAN_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    trace_config = json.loads((trace_plan_dir / "config.json").read_text(encoding="utf-8"))
    trace_hashes = trace_config.get("trace_plan_hashes")
    if not isinstance(trace_hashes, dict):
        raise ValueError("trace plan config missing trace_plan_hashes")

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {project: {} for project in PROJECTS}

    for project in PROJECTS:
        plan_path = trace_plan_dir / f"{project}_trace_plan.jsonl"
        trace_plan_hash = f"sha256:{_sha256(plan_path)}"
        if trace_hashes.get(project) != trace_plan_hash:
            raise ValueError(f"{project} trace-plan hash mismatch against config.json")
        for model_id, rows in _group_by_model(_load_jsonl(plan_path)).items():
            metadata = _base_metadata(
                project=project,
                model_id=model_id,
                rows=rows,
                trace_plan_dir=trace_plan_dir,
                trace_plan_hash=trace_plan_hash,
                trace_config=trace_config,
            )
            model_slug = _slug(model_id)
            if project == "ssq_lr":
                payload = {**metadata, "ssq_lr_entries": _ssq_lr_entries(rows)}
                filename = f"ssq_lr__{model_slug}__metadata_template.json"
            elif project == "horn":
                payload = {**metadata, "horn_entries": _horn_entries(rows)}
                filename = f"horn__{model_slug}__metadata_template.json"
            else:
                payload = {
                    "_template_only": True,
                    "metadata": metadata,
                    "hbsm_entry_templates": _hbsm_entry_templates(rows),
                    "fill_before_use": [
                        "convert hbsm_entry_templates to hbsm_entries",
                        "fill all required_metric_fields",
                    ],
                }
                filename = f"hbsm__{model_slug}__row_packet_template.json"
            path = output_dir / filename
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            counts[project][model_id] = len(rows)
            generated.append(
                {
                    "project": project,
                    "model_id": model_id,
                    "path": _repo_label(path),
                    "sha256": f"sha256:{_sha256(path)}",
                    "planned_entry_count": len(rows),
                    "trace_plan_hash": trace_plan_hash,
                }
            )

    summary = {
        "surface": "hybrid_trace_capture_manifest_templates",
        "decision": "CAPTURE_MANIFEST_READY_NOT_MODEL_EVIDENCE",
        "trace_plan_dir": _repo_label(trace_plan_dir),
        "generated": generated,
        "counts": counts,
        "claim_boundary": [
            "capture-manifest-template",
            "not model evidence",
            "not GPU evidence",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_dir / "decision.md").write_text(
        "# Hybrid Capture Manifest Decision\n\n"
        "`CAPTURE_MANIFEST_READY_NOT_MODEL_EVIDENCE`\n\n"
        "These files are templates for tensor/sensitivity capture metadata. They contain no tensors, "
        "no model outputs, no sensitivity metrics, and no GPU evidence.\n",
        encoding="utf-8",
    )
    lines = [
        "# Hybrid Capture Manifest Templates",
        "",
        "These templates are generated from the frozen trace plans. Fill the marked fields,",
        "write the requested tensors or sensitivity metrics, then run the packet builder.",
        "They are not model evidence and not GPU evidence.",
        "",
        "| Project | Model | Entries | Template |",
        "|---|---|---:|---|",
    ]
    for item in generated:
        lines.append(
            f"| `{item['project']}` | `{item['model_id']}` | {item['planned_entry_count']} | "
            f"`{item['path']}` |"
        )
    lines.extend(
        [
            "",
            "Do not pass these templates directly to a packet builder. Builders reject",
            "`_template_only: true` and `TO_FILL_BEFORE_CAPTURE` markers.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-plan-dir", type=Path, default=DEFAULT_TRACE_PLAN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = build_capture_manifests(trace_plan_dir=args.trace_plan_dir, output_dir=args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), **summary}, sort_keys=True))


if __name__ == "__main__":
    main()
