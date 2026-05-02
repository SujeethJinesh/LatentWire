#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_SMOKE_JSON = ROOT / "results/gsm8k_smoke_contract_20260421/gsm8k_smoke_contract_20260421.json"
DEFAULT_LIVE_JSON = ROOT / "results/gsm8k_contract_residual_rank16_dynalign_20260421/gsm8k_contract_residual_sweep_20260421.json"
DEFAULT_CONTROL_JSON = ROOT / "results/gsm8k_contract_residual_rank16_tokenbasis_20260421/gsm8k_contract_residual_sweep_20260421.json"
DEFAULT_CAMPAIGN_JSON = ROOT / "results/gsm8k_contract_campaign_slice128_seed0_20260422/gsm8k_contract_campaign.json"
DEFAULT_HEALTH_JSON = ROOT / (
    "checkpoints/gsm8k_contract_residual_sweep_20260421/"
    "dynalign_module_replace/"
    "qwen25_to_qwen3_grouped_subspace_transport_w010_r16_dynalign_module_replace_cal64_chat_seed1.pt.health.json"
)
DEFAULT_OUTPUT_JSON = ROOT / "paper/gsm8k_contract_artifact_manifest_20260422.json"
DEFAULT_OUTPUT_MD = ROOT / "paper/gsm8k_contract_artifact_manifest_20260422.md"
LIVE_LABEL = "dynalign_module_replace_residrank16"
CONTROL_LABEL = "tokenbasis_replace_residrank16"


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _relative(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _row_by_label(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    matches = [row for row in rows if row.get("label") == label]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one row with label {label!r}, found {len(matches)}")
    return matches[0]


def _source_control_artifacts(row: dict[str, Any]) -> dict[str, Any] | None:
    source_controls = row.get("source_controls")
    if not source_controls:
        return None
    return {
        "status": source_controls.get("status"),
        "passed": bool(source_controls.get("passed", False)),
        "readout_json": source_controls.get("readout_json"),
        "readout_md": source_controls.get("readout_md"),
        "prediction_outputs": dict(source_controls.get("prediction_outputs", {})),
    }


def _build_manifest(
    *,
    smoke_payload: dict[str, Any],
    live_payload: dict[str, Any],
    control_payload: dict[str, Any],
    campaign_payload: dict[str, Any],
    health_payload: dict[str, Any],
    smoke_json: pathlib.Path,
    live_json: pathlib.Path,
    control_json: pathlib.Path,
    campaign_json: pathlib.Path,
    health_json: pathlib.Path,
) -> dict[str, Any]:
    smoke_rows = smoke_payload["rows"]
    live_row = _row_by_label(live_payload["rows"], LIVE_LABEL)
    control_row = _row_by_label(control_payload["rows"], CONTROL_LABEL)
    campaign_row = campaign_payload["aggregate_rows"][LIVE_LABEL]
    campaign_baseline = campaign_payload["baseline_summary"]

    return {
        "date": "2026-04-22",
        "current_story": {
            "status": "same_pair_live_clue",
            "live_label": LIVE_LABEL,
            "main_blocker": "seed_stability_and_cross_family_falsification",
        },
        "artifacts": {
            "smoke_contract_json": _relative(smoke_json),
            "live_contract_json": _relative(live_json),
            "matched_control_json": _relative(control_json),
            "larger_slice_campaign_json": _relative(campaign_json),
            "seed1_health_json": _relative(health_json),
        },
        "smoke_contract": {
            "slice_size": int(smoke_payload["config"]["slice_size"]),
            "target_accuracy": float(smoke_rows["target_alone"]["accuracy"]),
            "rotalign_accuracy": float(smoke_rows["rotalign_kv"]["accuracy"]),
            "rotalign_numeric_coverage": int(smoke_rows["rotalign_kv"]["numeric_extraction_coverage"]),
            "c2c_accuracy": float(smoke_rows["c2c_generate"]["accuracy"]),
            "all_checks_pass": bool(all(bool(info["passed"]) for info in smoke_payload["checks"].values())),
        },
        "same_pair_live_row": {
            "label": str(live_row["label"]),
            "accuracy": float(live_row["accuracy"]),
            "paired_vs_target": {
                "win": int(live_row["paired_vs_target"]["win"]),
                "loss": int(live_row["paired_vs_target"]["loss"]),
                "tie": int(live_row["paired_vs_target"]["tie"]),
            },
            "numeric_coverage": int(live_row["numeric_extraction_coverage"]),
            "checkpoint_path": str(live_row["checkpoint_path"]),
            "source_controls": _source_control_artifacts(live_row),
        },
        "matched_control": {
            "label": str(control_row["label"]),
            "accuracy": float(control_row["accuracy"]),
            "paired_vs_target": {
                "win": int(control_row["paired_vs_target"]["win"]),
                "loss": int(control_row["paired_vs_target"]["loss"]),
                "tie": int(control_row["paired_vs_target"]["tie"]),
            },
            "numeric_coverage": int(control_row["numeric_extraction_coverage"]),
            "checkpoint_path": str(control_row["checkpoint_path"]),
            "source_controls": _source_control_artifacts(control_row),
        },
        "larger_slice_campaign": {
            "slice_size": int(campaign_payload["aggregate_rows"][LIVE_LABEL]["paired_n"]),
            "seeds_completed": list(campaign_row["seeds"]),
            "candidate_accuracy_mean": float(campaign_row["accuracy_mean"]),
            "candidate_accuracy_min": float(campaign_row["accuracy_min"]),
            "candidate_accuracy_max": float(campaign_row["accuracy_max"]),
            "delta_mean_vs_target": float(campaign_row["delta_mean"]),
            "delta_ci_low_mean": float(campaign_row["delta_ci_low_mean"]),
            "delta_ci_high_mean": float(campaign_row["delta_ci_high_mean"]),
            "wins_mean": float(campaign_row["wins_mean"]),
            "losses_mean": float(campaign_row["losses_mean"]),
            "target_accuracy": float(campaign_baseline["target_alone"]["accuracy"]),
            "c2c_accuracy": float(campaign_baseline["c2c_generate"]["accuracy"]),
        },
        "seed1_health": {
            "seed": int(health_payload["seed"]),
            "checkpoint_path": str(health_payload["checkpoint_path"]),
            "nonfinite_numel": int(health_payload["nonfinite_numel"]),
            "first_bad_key": health_payload["first_bad_key"],
            "nonfinite_keys": list(health_payload["nonfinite_keys"]),
            "top_abs_tensors": list(health_payload["top_abs_tensors"]),
        },
        "next_exact_gates": [
            "finish the larger frozen same-pair seed-repeat campaign without non-finite checkpoints",
            "preserve or beat the old 0.0938 ceiling while keeping full numeric coverage",
            "run one strict matched cross-family falsification pair before widening benchmarks",
        ],
    }


def _write_markdown(path: pathlib.Path, manifest: dict[str, Any]) -> None:
    smoke = manifest["smoke_contract"]
    live = manifest["same_pair_live_row"]
    control = manifest["matched_control"]
    campaign = manifest["larger_slice_campaign"]
    health = manifest["seed1_health"]
    top_tensor = health["top_abs_tensors"][0]["key"] if health["top_abs_tensors"] else "-"

    lines = [
        "# GSM8K Contract Artifact Manifest",
        "",
        f"- date: `{manifest['date']}`",
        f"- live label: `{manifest['current_story']['live_label']}`",
        f"- main blocker: `{manifest['current_story']['main_blocker']}`",
        "",
        "## Artifact Paths",
        "",
    ]
    for label, artifact in manifest["artifacts"].items():
        lines.append(f"- `{label}`: `{artifact}`")

    lines.extend(
        [
            "",
            "## Evidence Summary",
            "",
            f"- smoke contract: target=`{smoke['target_accuracy']:.4f}`, rotalign=`{smoke['rotalign_accuracy']:.4f}`, c2c=`{smoke['c2c_accuracy']:.4f}`, rotalign numeric coverage=`{smoke['rotalign_numeric_coverage']}`",
            f"- live 32-example row: `{live['label']}` accuracy=`{live['accuracy']:.4f}`, wins=`{live['paired_vs_target']['win']}`, losses=`{live['paired_vs_target']['loss']}`, coverage=`{live['numeric_coverage']}`",
            f"- matched control: `{control['label']}` accuracy=`{control['accuracy']:.4f}`, wins=`{control['paired_vs_target']['win']}`, losses=`{control['paired_vs_target']['loss']}`, coverage=`{control['numeric_coverage']}`",
            f"- larger frozen slice: candidate mean=`{campaign['candidate_accuracy_mean']:.4f}`, target=`{campaign['target_accuracy']:.4f}`, c2c=`{campaign['c2c_accuracy']:.4f}`, delta=`{campaign['delta_mean_vs_target']:.4f}` [{campaign['delta_ci_low_mean']:.4f}, {campaign['delta_ci_high_mean']:.4f}]",
            f"- seed-{health['seed']} health: nonfinite=`{health['nonfinite_numel']}`, first_bad_key=`{health['first_bad_key']}`, top_tensor=`{top_tensor}`",
            "",
            "## Next Exact Gates",
            "",
        ]
    )
    for gate in manifest["next_exact_gates"]:
        lines.append(f"- {gate}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def build_manifest(
    *,
    smoke_json: pathlib.Path = DEFAULT_SMOKE_JSON,
    live_json: pathlib.Path = DEFAULT_LIVE_JSON,
    control_json: pathlib.Path = DEFAULT_CONTROL_JSON,
    campaign_json: pathlib.Path = DEFAULT_CAMPAIGN_JSON,
    health_json: pathlib.Path = DEFAULT_HEALTH_JSON,
    output_json: pathlib.Path = DEFAULT_OUTPUT_JSON,
    output_md: pathlib.Path = DEFAULT_OUTPUT_MD,
) -> dict[str, Any]:
    manifest = _build_manifest(
        smoke_payload=_read_json(smoke_json),
        live_payload=_read_json(live_json),
        control_payload=_read_json(control_json),
        campaign_payload=_read_json(campaign_json),
        health_payload=_read_json(health_json),
        smoke_json=smoke_json,
        live_json=live_json,
        control_json=control_json,
        campaign_json=campaign_json,
        health_json=health_json,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _write_markdown(output_md, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a tracked manifest for the current GSM8K contract evidence chain.")
    parser.add_argument("--smoke-json", type=pathlib.Path, default=DEFAULT_SMOKE_JSON)
    parser.add_argument("--live-json", type=pathlib.Path, default=DEFAULT_LIVE_JSON)
    parser.add_argument("--control-json", type=pathlib.Path, default=DEFAULT_CONTROL_JSON)
    parser.add_argument("--campaign-json", type=pathlib.Path, default=DEFAULT_CAMPAIGN_JSON)
    parser.add_argument("--health-json", type=pathlib.Path, default=DEFAULT_HEALTH_JSON)
    parser.add_argument("--output-json", type=pathlib.Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=pathlib.Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    return build_manifest(
        smoke_json=args.smoke_json,
        live_json=args.live_json,
        control_json=args.control_json,
        campaign_json=args.campaign_json,
        health_json=args.health_json,
        output_json=args.output_json,
        output_md=args.output_md,
    )


if __name__ == "__main__":
    main()
