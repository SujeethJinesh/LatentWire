#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_gsm8k_contract_diagnostics as diagnostics
from scripts import harness_common as harness
from scripts import run_gsm8k_smoke_contract as smoke


SMOKE_PAYLOAD_NAME = "gsm8k_smoke_contract_20260421.json"
RESIDUAL_PAYLOAD_NAME = "gsm8k_contract_residual_sweep_20260421.json"


@dataclass(frozen=True)
class CampaignConfig:
    results_root: str
    source_model: str = smoke.DEFAULT_SOURCE_MODEL
    target_model: str = smoke.DEFAULT_TARGET_MODEL
    checkpoint_path: str = smoke.DEFAULT_CHECKPOINT
    eval_file: str = smoke.DEFAULT_EVAL_FILE
    slice_size: int = 32
    device: str = "mps"
    max_new_tokens: int = 64
    gate: float = 0.10
    kv_transport: str = "k_only"
    position_selection_ratio: float = 0.5
    position_selection_metric: str = "attention"
    source_reasoning_mode: str = "brief_analysis"
    use_chat_template: bool = True
    enable_thinking: bool = False
    seeds: tuple[int, ...] = (0,)
    bases: tuple[str, ...] = ("dynalign_module_replace",)
    ranks: tuple[int, ...] = (16,)
    candidate_labels: tuple[str, ...] = ("dynalign_module_replace_residrank16",)
    baseline_results_dir: str | None = None
    skip_smoke: bool = False
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 0


def _run(cmd: list[str]) -> None:
    harness.run(cmd, cwd=ROOT)


def _campaign_baseline_dir(config: CampaignConfig) -> pathlib.Path:
    if config.baseline_results_dir is not None:
        return ROOT / config.baseline_results_dir
    return ROOT / config.results_root / "smoke"


def _seed_results_dir(config: CampaignConfig, seed: int) -> pathlib.Path:
    return ROOT / config.results_root / f"seed{int(seed)}"


def _materialized_eval_path(config: CampaignConfig) -> str:
    return str(
        harness.resolve_materialized_eval_file(
            None,
            results_dir=ROOT / config.results_root,
            slice_size=config.slice_size,
        )
    )


def _run_smoke_contract(config: CampaignConfig, baseline_dir: pathlib.Path) -> None:
    if config.skip_smoke and baseline_dir.joinpath(SMOKE_PAYLOAD_NAME).exists():
        return
    cmd = [
        harness.python_executable(ROOT),
        str(ROOT / "scripts" / "run_gsm8k_smoke_contract.py"),
        "--source-model",
        config.source_model,
        "--target-model",
        config.target_model,
        "--checkpoint-path",
        str(ROOT / config.checkpoint_path),
        "--eval-file",
        config.eval_file,
        "--slice-size",
        str(config.slice_size),
        "--materialized-eval-file",
        _materialized_eval_path(config),
        "--results-dir",
        str(baseline_dir.relative_to(ROOT)),
        "--device",
        config.device,
        "--max-new-tokens",
        str(config.max_new_tokens),
        "--gate",
        str(config.gate),
        "--kv-transport",
        config.kv_transport,
        "--position-selection-ratio",
        str(config.position_selection_ratio),
        "--position-selection-metric",
        config.position_selection_metric,
        "--source-reasoning-mode",
        config.source_reasoning_mode,
    ]
    if not config.use_chat_template:
        cmd.append("--no-chat-template")
    if config.enable_thinking:
        cmd.append("--enable-thinking")
    _run(cmd)


def _run_residual_sweep(config: CampaignConfig, baseline_dir: pathlib.Path, seed: int) -> pathlib.Path:
    results_dir = _seed_results_dir(config, seed)
    cmd = [
        harness.python_executable(ROOT),
        str(ROOT / "scripts" / "run_gsm8k_contract_residual_sweep.py"),
        "--source-model",
        config.source_model,
        "--target-model",
        config.target_model,
        "--eval-file",
        config.eval_file,
        "--slice-size",
        str(config.slice_size),
        "--materialized-eval-file",
        _materialized_eval_path(config),
        "--baseline-results-dir",
        str(baseline_dir.relative_to(ROOT)),
        "--results-dir",
        str(results_dir.relative_to(ROOT)),
        "--device",
        config.device,
        "--dtype",
        "float32",
        "--max-new-tokens",
        str(config.max_new_tokens),
        "--gate",
        str(config.gate),
        "--kv-transport",
        config.kv_transport,
        "--position-selection-ratio",
        str(config.position_selection_ratio),
        "--position-selection-metric",
        config.position_selection_metric,
        "--source-reasoning-mode",
        config.source_reasoning_mode,
        "--seed",
        str(seed),
    ]
    if not config.use_chat_template:
        cmd.append("--no-chat-template")
    if config.enable_thinking:
        cmd.append("--enable-thinking")
    for rank in config.ranks:
        cmd.extend(["--rank", str(rank)])
    for base in config.bases:
        cmd.extend(["--base", base])
    _run(cmd)
    return results_dir


def _run_diagnostics(
    *,
    config: CampaignConfig,
    baseline_dir: pathlib.Path,
    seed_results_dir: pathlib.Path,
    candidate_label: str,
) -> pathlib.Path:
    candidate_output = seed_results_dir / f"{candidate_label}.jsonl"
    if not candidate_output.exists():
        raise FileNotFoundError(f"Candidate output not found: {candidate_output}")
    cmd = [
        harness.python_executable(ROOT),
        str(ROOT / "scripts" / "analyze_gsm8k_contract_diagnostics.py"),
        "--candidate-prediction-output",
        str(candidate_output.relative_to(ROOT)),
        "--candidate-label",
        candidate_label,
        "--baseline-results-dir",
        str(baseline_dir.relative_to(ROOT)),
        "--results-dir",
        str(seed_results_dir.relative_to(ROOT)),
    ]
    _run(cmd)
    return seed_results_dir / f"{candidate_label}_diagnostics_20260422.json"


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _bootstrap_mean_ci(
    values: list[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float | None, float | None]:
    if not values or samples <= 0:
        return None, None
    rng = random.Random(int(seed))
    n = len(values)
    means: list[float] = []
    for _ in range(samples):
        total = 0.0
        for _ in range(n):
            total += float(values[rng.randrange(n)])
        means.append(total / n)
    means.sort()
    low_idx = max(0, min(len(means) - 1, int(0.025 * (len(means) - 1))))
    high_idx = max(0, min(len(means) - 1, int(0.975 * (len(means) - 1))))
    return means[low_idx], means[high_idx]


def _candidate_paired_stats(
    *,
    baseline_dir: pathlib.Path,
    materialized_eval_file: pathlib.Path,
    candidate_output: pathlib.Path,
    candidate_method: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    baseline_output = baseline_dir / "gsm8k32_latentwire.jsonl"
    baseline_records = harness.attach_prompts(harness.read_jsonl(baseline_output), materialized_eval_file)
    target_records = harness.group_by_method(baseline_records)["target_alone"]

    candidate_records = harness.attach_prompts(harness.read_jsonl(candidate_output), materialized_eval_file)
    method_records = diagnostics._resolve_candidate_records(candidate_records, method_name=candidate_method)
    target_by_id = {str(row["example_id"]): row for row in target_records}
    deltas = [float(bool(row["correct"])) - float(bool(target_by_id[str(row["example_id"])]["correct"])) for row in method_records]
    ci_low, ci_high = _bootstrap_mean_ci(
        deltas,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    paired = harness.paired_vs_baseline(method_records, target_records)
    return {
        "paired_n": len(deltas),
        "delta_vs_target": float(sum(deltas) / max(len(deltas), 1)),
        "delta_ci_low": ci_low,
        "delta_ci_high": ci_high,
        "win": int(paired["win"]),
        "loss": int(paired["loss"]),
        "tie": int(paired["tie"]),
    }


def _aggregate_rows(seed_payloads: dict[int, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = {}
    for seed, payload in seed_payloads.items():
        for row in payload["rows"]:
            by_label.setdefault(str(row["label"]), []).append({"seed": seed, **row})
    summary: dict[str, dict[str, Any]] = {}
    for label, rows in sorted(by_label.items()):
        accuracies = [float(row["accuracy"]) for row in rows]
        wins = [int(row["paired_vs_target"]["win"]) for row in rows]
        losses = [int(row["paired_vs_target"]["loss"]) for row in rows]
        summary[label] = {
            "n_seeds": len(rows),
            "seeds": [int(row["seed"]) for row in rows],
            "accuracy_mean": float(sum(accuracies) / max(len(accuracies), 1)),
            "accuracy_min": float(min(accuracies)),
            "accuracy_max": float(max(accuracies)),
            "wins_mean": float(sum(wins) / max(len(wins), 1)),
            "losses_mean": float(sum(losses) / max(len(losses), 1)),
            "base_label": str(rows[0].get("base_label", "")),
            "residual_rank": int(rows[0].get("residual_rank", -1)),
        }
    return summary


def _candidate_row(seed_payload: dict[str, Any], candidate_label: str) -> dict[str, Any] | None:
    for row in seed_payload.get("rows", []):
        if str(row.get("label")) == candidate_label:
            return row
    return None


def _aggregate_diagnostics(diagnostics_by_label: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for label, entries in sorted(diagnostics_by_label.items()):
        oracle_accs = [float(entry["summary_metrics"]["oracle_accuracy"]) for entry in entries]
        candidate_accs = [float(entry["summary_metrics"]["candidate_accuracy"]) for entry in entries]
        oracle_headroom = [float(oracle - cand) for oracle, cand in zip(oracle_accs, candidate_accs)]
        win_support_n = sum(int(entry["candidate_only_win_support"]["n"]) for entry in entries)
        win_support_source = sum(int(entry["candidate_only_win_support"]["source_correct"]) for entry in entries)
        win_support_text = sum(int(entry["candidate_only_win_support"]["text_correct"]) for entry in entries)
        text_loss_n = sum(int(entry["text_to_text_loss_support"]["n"]) for entry in entries)
        text_loss_source = sum(int(entry["text_to_text_loss_support"]["source_correct"]) for entry in entries)
        summary[label] = {
            "n_seeds": len(entries),
            "oracle_accuracy_mean": float(sum(oracle_accs) / max(len(oracle_accs), 1)),
            "oracle_accuracy_min": float(min(oracle_accs)),
            "oracle_accuracy_max": float(max(oracle_accs)),
            "oracle_headroom_mean": float(sum(oracle_headroom) / max(len(oracle_headroom), 1)),
            "candidate_only_win_n": int(win_support_n),
            "candidate_only_win_source_correct": int(win_support_source),
            "candidate_only_win_text_correct": int(win_support_text),
            "text_only_loss_n": int(text_loss_n),
            "text_only_loss_source_correct": int(text_loss_source),
        }
    return summary


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# GSM8K Contract Campaign",
        "",
        f"- date: `{payload['date']}`",
        f"- source -> target: `{payload['config']['source_model']} -> {payload['config']['target_model']}`",
        f"- slice size: `{payload['config']['slice_size']}`",
        f"- seeds: `{', '.join(str(seed) for seed in payload['config']['seeds'])}`",
        f"- bases: `{', '.join(payload['config']['bases'])}`",
        f"- ranks: `{', '.join(str(rank) for rank in payload['config']['ranks'])}`",
        f"- baseline dir: `{payload['artifacts']['baseline_results_dir']}`",
        "",
        "## Aggregated Rows",
        "",
        "| Label | Seeds | Accuracy mean | Accuracy min | Accuracy max | Delta mean | Delta CI | Win mean | Loss mean | Positive seeds |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for label, row in payload["aggregate_rows"].items():
        ci = (
            f"[{row['delta_ci_low_mean']:.4f}, {row['delta_ci_high_mean']:.4f}]"
            if row.get("delta_ci_low_mean") is not None and row.get("delta_ci_high_mean") is not None
            else "-"
        )
        lines.append(
            f"| {label} | {row['n_seeds']} | {row['accuracy_mean']:.4f} | {row['accuracy_min']:.4f} | "
            f"{row['accuracy_max']:.4f} | {row.get('delta_mean', 0.0):.4f} | {ci} | "
            f"{row['wins_mean']:.2f} | {row['losses_mean']:.2f} | {row.get('positive_seed_count', 0)} |"
        )
    lines.extend(["", "## Seed Artifacts", ""])
    for seed, info in sorted(payload["seed_artifacts"].items()):
        lines.append(
            f"- seed `{seed}`: residual=`{info['residual_payload']}`"
            + (
                f", diagnostics=`{', '.join(info['diagnostics'])}`"
                if info["diagnostics"]
                else ""
            )
        )
    if payload.get("diagnostic_rows"):
        lines.extend(
            [
                "",
                "## Diagnostic Summary",
                "",
                "| Label | Oracle mean | Oracle min | Oracle max | Headroom mean | Candidate-only wins | Source correct on wins | Text correct on wins | Text-only losses | Source correct on text losses |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for label, row in payload["diagnostic_rows"].items():
            lines.append(
                f"| {label} | {row['oracle_accuracy_mean']:.4f} | {row['oracle_accuracy_min']:.4f} | "
                f"{row['oracle_accuracy_max']:.4f} | {row['oracle_headroom_mean']:.4f} | "
                f"{row['candidate_only_win_n']} | {row['candidate_only_win_source_correct']} | "
                f"{row['candidate_only_win_text_correct']} | {row['text_only_loss_n']} | "
                f"{row['text_only_loss_source_correct']} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run_campaign(config: CampaignConfig) -> dict[str, Any]:
    baseline_dir = _campaign_baseline_dir(config)
    _run_smoke_contract(config, baseline_dir)
    baseline_payload = _load_json(baseline_dir / SMOKE_PAYLOAD_NAME)
    materialized_eval_file = pathlib.Path(_materialized_eval_path(config))
    harness.materialize_slice(ROOT / config.eval_file, materialized_eval_file, config.slice_size)

    seed_payloads: dict[int, dict[str, Any]] = {}
    seed_artifacts: dict[int, dict[str, Any]] = {}
    paired_stats_by_label: dict[str, list[dict[str, Any]]] = {}
    diagnostics_by_label: dict[str, list[dict[str, Any]]] = {}
    for seed in config.seeds:
        seed_results_dir = _run_residual_sweep(config, baseline_dir, seed)
        residual_payload = _load_json(seed_results_dir / RESIDUAL_PAYLOAD_NAME)
        seed_payloads[int(seed)] = residual_payload
        diagnostics_paths: list[str] = []
        for candidate_label in config.candidate_labels:
            row = _candidate_row(residual_payload, candidate_label)
            if row is None or str(row.get("status", "ok")) != "ok":
                continue
            candidate_output = seed_results_dir / f"{candidate_label}.jsonl"
            if not candidate_output.exists():
                continue
            paired_stats = _candidate_paired_stats(
                baseline_dir=baseline_dir,
                materialized_eval_file=materialized_eval_file,
                candidate_output=candidate_output,
                candidate_method="rotalign_kv",
                bootstrap_samples=config.bootstrap_samples,
                bootstrap_seed=config.bootstrap_seed + int(seed) * 10_000 + sum(ord(ch) for ch in candidate_label),
            )
            paired_stats_by_label.setdefault(candidate_label, []).append({"seed": int(seed), **paired_stats})
            diagnostics_json = _run_diagnostics(
                config=config,
                baseline_dir=baseline_dir,
                seed_results_dir=seed_results_dir,
                candidate_label=candidate_label,
            )
            diagnostics_by_label.setdefault(candidate_label, []).append(_load_json(diagnostics_json))
            diagnostics_paths.append(str(diagnostics_json))
        seed_artifacts[int(seed)] = {
            "residual_payload": str(seed_results_dir / RESIDUAL_PAYLOAD_NAME),
            "diagnostics": diagnostics_paths,
        }

    aggregate_rows = _aggregate_rows(seed_payloads)
    for label, entries in paired_stats_by_label.items():
        deltas = [float(entry["delta_vs_target"]) for entry in entries]
        lows = [entry["delta_ci_low"] for entry in entries if entry["delta_ci_low"] is not None]
        highs = [entry["delta_ci_high"] for entry in entries if entry["delta_ci_high"] is not None]
        row = aggregate_rows.setdefault(label, {})
        row["delta_mean"] = float(sum(deltas) / max(len(deltas), 1))
        row["delta_min"] = float(min(deltas))
        row["delta_max"] = float(max(deltas))
        row["delta_ci_low_mean"] = float(sum(lows) / max(len(lows), 1)) if lows else None
        row["delta_ci_high_mean"] = float(sum(highs) / max(len(highs), 1)) if highs else None
        row["positive_seed_count"] = int(sum(int(delta > 0.0) for delta in deltas))
        row["paired_n"] = int(entries[0]["paired_n"])
    diagnostic_rows = _aggregate_diagnostics(diagnostics_by_label)

    payload = {
        "date": str(date.today()),
        "config": asdict(config),
        "artifacts": {
            "baseline_results_dir": str(baseline_dir),
            "baseline_payload": str(baseline_dir / SMOKE_PAYLOAD_NAME),
        },
        "baseline_summary": baseline_payload["rows"],
        "aggregate_rows": aggregate_rows,
        "diagnostic_rows": diagnostic_rows,
        "paired_stats_by_label": paired_stats_by_label,
        "seed_artifacts": seed_artifacts,
    }
    results_root = ROOT / config.results_root
    results_root.mkdir(parents=True, exist_ok=True)
    harness.write_json(results_root / "gsm8k_contract_campaign.json", payload)
    _write_markdown(results_root / "gsm8k_contract_campaign.md", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a reviewer-driven GSM8K contract campaign with larger slices, seed repeats, and diagnostics.")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--source-model", default=smoke.DEFAULT_SOURCE_MODEL)
    parser.add_argument("--target-model", default=smoke.DEFAULT_TARGET_MODEL)
    parser.add_argument("--checkpoint-path", default=smoke.DEFAULT_CHECKPOINT)
    parser.add_argument("--eval-file", default=smoke.DEFAULT_EVAL_FILE)
    parser.add_argument("--slice-size", type=int, default=32)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--gate", type=float, default=0.10)
    parser.add_argument("--kv-transport", default="k_only")
    parser.add_argument("--position-selection-ratio", type=float, default=0.5)
    parser.add_argument("--position-selection-metric", default="attention")
    parser.add_argument("--source-reasoning-mode", default="brief_analysis")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--base", action="append", dest="bases")
    parser.add_argument("--rank", action="append", type=int, dest="ranks")
    parser.add_argument("--candidate-label", action="append", dest="candidate_labels")
    parser.add_argument("--baseline-results-dir", default=None)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    config = CampaignConfig(
        results_root=args.results_root,
        source_model=args.source_model,
        target_model=args.target_model,
        checkpoint_path=args.checkpoint_path,
        eval_file=args.eval_file,
        slice_size=args.slice_size,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        gate=args.gate,
        kv_transport=args.kv_transport,
        position_selection_ratio=args.position_selection_ratio,
        position_selection_metric=args.position_selection_metric,
        source_reasoning_mode=args.source_reasoning_mode,
        use_chat_template=not args.no_chat_template,
        enable_thinking=args.enable_thinking,
        seeds=tuple(args.seeds) if args.seeds else CampaignConfig.seeds,
        bases=tuple(args.bases) if args.bases else CampaignConfig.bases,
        ranks=tuple(args.ranks) if args.ranks else CampaignConfig.ranks,
        candidate_labels=tuple(args.candidate_labels) if args.candidate_labels else CampaignConfig.candidate_labels,
        baseline_results_dir=args.baseline_results_dir,
        skip_smoke=bool(args.skip_smoke),
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    return run_campaign(config)


if __name__ == "__main__":
    main()
