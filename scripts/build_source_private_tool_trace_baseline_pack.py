from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import statistics
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _best_budget_summary(sweep: dict[str, Any]) -> dict[str, Any]:
    best_budget = sweep["best_budget_bytes"]
    for summary in sweep["budget_summaries"]:
        if summary["budget_bytes"] == best_budget:
            return summary
    raise ValueError(f"best budget {best_budget!r} not present")


def _condition_row(
    *,
    surface: str,
    summary: dict[str, Any],
    condition: str,
    label: str,
    kind: str,
) -> dict[str, Any]:
    metrics = summary["metrics"][condition]
    return {
        "surface": surface,
        "condition": condition,
        "label": label,
        "kind": kind,
        "accuracy": metrics["accuracy"],
        "correct": metrics["correct"],
        "n": summary["n"],
        "mean_bytes": metrics["mean_payload_bytes"],
        "max_bytes": metrics["max_payload_bytes"],
        "mean_tokens": metrics["mean_payload_tokens"],
        "p50_latency_ms": metrics["p50_latency_ms"],
    }


def deterministic_baseline_rows(surface: str, sweep: dict[str, Any]) -> list[dict[str, Any]]:
    summary = _best_budget_summary(sweep)
    return [
        _condition_row(surface=surface, summary=summary, condition="target_only", label="target-only", kind="no-source"),
        _condition_row(surface=surface, summary=summary, condition="target_wrapper", label="target wrapper/no-source", kind="no-source"),
        _condition_row(
            surface=surface,
            summary=summary,
            condition="matched_repair_packet",
            label=f"matched deterministic trace packet ({summary['budget_bytes']} bytes)",
            kind="method-oracle",
        ),
        _condition_row(surface=surface, summary=summary, condition="zero_source", label="zero-source", kind="source-destroying control"),
        _condition_row(surface=surface, summary=summary, condition="shuffled_source", label="shuffled-source", kind="source-destroying control"),
        _condition_row(surface=surface, summary=summary, condition="random_same_byte", label="random same-byte", kind="source-destroying control"),
        _condition_row(surface=surface, summary=summary, condition="answer_only", label="answer-only sidecar", kind="leakage control"),
        _condition_row(surface=surface, summary=summary, condition="answer_masked", label="answer-masked", kind="leakage control"),
        _condition_row(surface=surface, summary=summary, condition="target_derived_sidecar", label="target-derived sidecar", kind="target-prior control"),
        _condition_row(surface=surface, summary=summary, condition="structured_text_matched", label="matched-byte hidden-log text", kind="matched-byte text baseline"),
        _condition_row(surface=surface, summary=summary, condition="full_hidden_log", label="full hidden-log relay", kind="oracle/text relay"),
        _condition_row(surface=surface, summary=summary, condition="full_diag_text", label="full diagnostic text", kind="oracle"),
    ]


def _model_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "surface": row["surface"],
            "family_set": row["family_set"],
            "seed": row["seed"],
            "model": row["model"],
            "prompt_mode": row["prompt_mode"],
            "pass_gate": row["pass_gate"],
            "matched_accuracy": row["matched_accuracy"],
            "target_accuracy": row["target_only_accuracy"],
            "best_control_accuracy": row["best_control_accuracy"],
            "packet_valid_rate": row["packet_valid_rate"],
            "mean_bytes": row["mean_packet_bytes"],
            "delta_target_low": row["delta_target_low"],
            "delta_target_high": row["delta_target_high"],
            "delta_control_low": row["delta_control_low"],
            "delta_control_high": row["delta_control_high"],
        }
        for row in summary["rows"]
    ]


def aggregate_baseline_pack(*, deterministic: list[tuple[str, dict[str, Any]]], seed_repeat: dict[str, Any]) -> dict[str, Any]:
    deterministic_rows = [row for surface, sweep in deterministic for row in deterministic_baseline_rows(surface, sweep)]
    model_rows = _model_rows(seed_repeat)
    primary = [row for row in model_rows if row["prompt_mode"] == "trace_no_hint"]
    destruction = [row for row in model_rows if row["prompt_mode"] == "raw_log_no_trace"]
    systems = {
        "primary_mean_bytes_min": min(row["mean_bytes"] for row in primary),
        "primary_mean_bytes_max": max(row["mean_bytes"] for row in primary),
        "primary_valid_rate_min": min(row["packet_valid_rate"] for row in primary),
        "primary_valid_rate_mean": statistics.fmean(row["packet_valid_rate"] for row in primary),
        "deterministic_text_relay_mean_accuracy": statistics.fmean(
            row["accuracy"] for row in deterministic_rows if row["condition"] == "structured_text_matched"
        ),
        "deterministic_full_log_mean_accuracy": statistics.fmean(
            row["accuracy"] for row in deterministic_rows if row["condition"] == "full_hidden_log"
        ),
    }
    threat_model = [
        {
            "risk": "target prior or wrapper solves the task",
            "control": "target-only and target-wrapper/no-source rows",
            "result": "target rows stay at 0.250 on all 500-example surfaces",
        },
        {
            "risk": "packet works without matched private source evidence",
            "control": "zero-source, shuffled-source, and random same-byte rows",
            "result": "best controls remain 0.252-0.258 while matched rows are 0.808-1.000",
        },
        {
            "risk": "answer-label leakage explains the gain",
            "control": "answer-only and answer-masked sidecars",
            "result": "answer controls stay at target-only",
        },
        {
            "risk": "target-derived metadata explains the gain",
            "control": "target-derived sidecar",
            "result": "target-derived sidecar stays at target-only",
        },
        {
            "risk": "matched-byte text relay explains the gain",
            "control": "matched-byte hidden-log text baseline",
            "result": "matched-byte text stays at 0.250 while 2-byte trace packets reach 1.000 deterministically",
        },
        {
            "risk": "raw hidden logs are enough without the trace protocol",
            "control": "raw_log_no_trace model rows",
            "result": "trace removal returns Qwen3 to 0.250 with 0 valid packets",
        },
        {
            "risk": "template-family overfitting",
            "control": "disjoint held-out repair families",
            "result": "Qwen3 reaches 0.922/0.924 and Phi-3 reaches 1.000 on held-out seeds",
        },
        {
            "risk": "seed instability",
            "control": "four frozen 500-example surfaces",
            "result": "8/8 primary rows pass; min paired lower bound over target-only is 0.516",
        },
    ]
    remaining_gaps = [
        {
            "gap": "matched-byte structured JSON/free-text relay",
            "status": "partially covered by truncated hidden-log text; JSON relay still needed",
        },
        {
            "gap": "target helper-only/no-log oracle",
            "status": "target-only and target-wrapper covered; stronger helper-only no-log baselines still needed",
        },
        {
            "gap": "masked trace component ablations",
            "status": "raw_log_no_trace is covered; expected/actual, line-number, test-name masking remain future reviewer-risk rows",
        },
        {
            "gap": "candidate/selector separation",
            "status": "candidate pool recall is deterministic 1.0; paper table should still separate pool recall and selector accuracy",
        },
        {
            "gap": "second target-family pair",
            "status": "source emitters are cross-family; target decoder is deterministic protocol decoder, so a learned/LLM target-family row is not yet claimed",
        },
    ]
    return {
        "gate": "source_private_tool_trace_baseline_pack_20260429",
        "status": "reviewer-facing evidence package",
        "claim": "Explicit source-private tool-trace packets communicate hidden execution evidence to a target-side candidate decoder.",
        "claim_boundary": "The method is not raw-log repair inference and not unstructured latent transfer; the explicit private REPAIR_DIAG trace field is the communication interface.",
        "pass_gate": seed_repeat["pass_gate"] and seed_repeat["min_primary_delta_target_low"] > 0.15,
        "deterministic_rows": deterministic_rows,
        "model_rows": model_rows,
        "systems": systems,
        "threat_model": threat_model,
        "remaining_gaps": remaining_gaps,
        "seed_repeat_summary": {
            "n_surfaces": seed_repeat["n_surfaces"],
            "n_primary_rows": seed_repeat["n_primary_rows"],
            "n_destruction_rows": seed_repeat["n_destruction_rows"],
            "min_primary_delta_target_low": seed_repeat["min_primary_delta_target_low"],
            "min_primary_delta_control_low": seed_repeat["min_primary_delta_control_low"],
            "max_destruction_matched_accuracy": seed_repeat["max_destruction_matched_accuracy"],
            "by_model": seed_repeat["by_model"],
        },
    }


def _write_markdown(path: pathlib.Path, pack: dict[str, Any]) -> None:
    lines = [
        "# Source-Private Tool-Trace Baseline Pack",
        "",
        f"- gate: `{pack['gate']}`",
        f"- pass gate: `{pack['pass_gate']}`",
        f"- claim: {pack['claim']}",
        f"- boundary: {pack['claim_boundary']}",
        "",
        "## Model Rows",
        "",
        "| Surface | Family | Seed | Model | Mode | Matched | Target | Best control | Valid | Bytes | Delta target 95% CI |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pack["model_rows"]:
        lines.append(
            f"| {row['surface']} | {row['family_set']} | {row['seed']} | {row['model']} | {row['prompt_mode']} | "
            f"{row['matched_accuracy']:.3f} | {row['target_accuracy']:.3f} | {row['best_control_accuracy']:.3f} | "
            f"{row['packet_valid_rate']:.3f} | {row['mean_bytes']:.2f} | "
            f"[{row['delta_target_low']:.3f}, {row['delta_target_high']:.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Deterministic Baselines",
            "",
            "| Surface | Kind | Label | Accuracy | Mean bytes | Mean tokens |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in pack["deterministic_rows"]:
        lines.append(
            f"| {row['surface']} | {row['kind']} | {row['label']} | "
            f"{row['accuracy']:.3f} | {row['mean_bytes']:.2f} | {row['mean_tokens']:.2f} |"
        )
    systems = pack["systems"]
    lines.extend(
        [
            "",
            "## Systems",
            "",
            f"- primary packet mean bytes range: `{systems['primary_mean_bytes_min']:.2f}-{systems['primary_mean_bytes_max']:.2f}`",
            f"- primary packet validity range: `{systems['primary_valid_rate_min']:.3f}-{max(row['packet_valid_rate'] for row in pack['model_rows'] if row['prompt_mode'] == 'trace_no_hint'):.3f}`",
            f"- deterministic matched-byte text mean accuracy: `{systems['deterministic_text_relay_mean_accuracy']:.3f}`",
            f"- deterministic full hidden-log relay mean accuracy: `{systems['deterministic_full_log_mean_accuracy']:.3f}`",
            "",
            "## Threat Model",
            "",
            "| Risk | Control | Result |",
            "|---|---|---|",
        ]
    )
    for row in pack["threat_model"]:
        lines.append(f"| {row['risk']} | {row['control']} | {row['result']} |")
    lines.extend(["", "## Remaining Reviewer Gaps", "", "| Gap | Status |", "|---|---|"])
    for row in pack["remaining_gaps"]:
        lines.append(f"| {row['gap']} | {row['status']} |")
    lines.extend(["", "## Next Reviewer-Risk Gate", "", "`source_private_tool_trace_paper_claim_draft_20260429`: convert this pack into method, benchmark, baseline, and limitation sections with exact claim language.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        "core_medium_sweep": ROOT / "results/source_private_hidden_repair_packet_medium_20260429/sweep_summary.json",
        "holdout_sweep": ROOT / "results/source_private_hidden_repair_packet_holdout_families_20260429/sweep_summary.json",
        "seed_repeat": ROOT / "results/source_private_hidden_repair_packet_seed_repeat_20260429/seed_repeat_summary.json",
    }
    pack = aggregate_baseline_pack(
        deterministic=[
            ("core_medium_seed29", _read_json(inputs["core_medium_sweep"])),
            ("holdout_seed30", _read_json(inputs["holdout_sweep"])),
        ],
        seed_repeat=_read_json(inputs["seed_repeat"]),
    )
    (output_dir / "baseline_pack.json").write_text(json.dumps(pack, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_dir / "baseline_pack.md", pack)
    manifest = {
        "gate": pack["gate"],
        "pass_gate": pack["pass_gate"],
        "artifacts": ["baseline_pack.json", "baseline_pack.md", "manifest.json", "manifest.md"],
        "input_sha256": {name: _sha256_file(path) for name, path in inputs.items()},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest_lines = [
        "# Source-Private Tool-Trace Baseline Pack Manifest",
        "",
        f"- gate: `{manifest['gate']}`",
        f"- pass gate: `{manifest['pass_gate']}`",
        "",
        "## Artifacts",
        "",
    ]
    manifest_lines.extend(f"- `{artifact}`" for artifact in manifest["artifacts"])
    (output_dir / "manifest.md").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
