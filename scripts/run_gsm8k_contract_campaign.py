#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def _run(cmd: list[str]) -> None:
    smoke._run(cmd, cwd=ROOT)


def _campaign_baseline_dir(config: CampaignConfig) -> pathlib.Path:
    if config.baseline_results_dir is not None:
        return ROOT / config.baseline_results_dir
    return ROOT / config.results_root / "smoke"


def _seed_results_dir(config: CampaignConfig, seed: int) -> pathlib.Path:
    return ROOT / config.results_root / f"seed{int(seed)}"


def _materialized_eval_path(config: CampaignConfig) -> str:
    root_tag = pathlib.Path(config.results_root).name.replace("/", "_")
    return f"/tmp/{root_tag}_gsm8k_eval_{int(config.slice_size)}.jsonl"


def _run_smoke_contract(config: CampaignConfig, baseline_dir: pathlib.Path) -> None:
    if config.skip_smoke and baseline_dir.joinpath(SMOKE_PAYLOAD_NAME).exists():
        return
    cmd = [
        str(ROOT / ".venv" / "bin" / "python"),
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
        str(ROOT / ".venv" / "bin" / "python"),
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
        str(ROOT / ".venv" / "bin" / "python"),
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
        "| Label | Seeds | Accuracy mean | Accuracy min | Accuracy max | Win mean | Loss mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in payload["aggregate_rows"].items():
        lines.append(
            f"| {label} | {row['n_seeds']} | {row['accuracy_mean']:.4f} | {row['accuracy_min']:.4f} | "
            f"{row['accuracy_max']:.4f} | {row['wins_mean']:.2f} | {row['losses_mean']:.2f} |"
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run_campaign(config: CampaignConfig) -> dict[str, Any]:
    baseline_dir = _campaign_baseline_dir(config)
    _run_smoke_contract(config, baseline_dir)
    baseline_payload = _load_json(baseline_dir / SMOKE_PAYLOAD_NAME)

    seed_payloads: dict[int, dict[str, Any]] = {}
    seed_artifacts: dict[int, dict[str, Any]] = {}
    for seed in config.seeds:
        seed_results_dir = _run_residual_sweep(config, baseline_dir, seed)
        residual_payload = _load_json(seed_results_dir / RESIDUAL_PAYLOAD_NAME)
        seed_payloads[int(seed)] = residual_payload
        diagnostics_paths: list[str] = []
        for candidate_label in config.candidate_labels:
            available_labels = {str(row["label"]) for row in residual_payload["rows"]}
            if candidate_label in available_labels:
                diagnostics_json = _run_diagnostics(
                    config=config,
                    baseline_dir=baseline_dir,
                    seed_results_dir=seed_results_dir,
                    candidate_label=candidate_label,
                )
                diagnostics_paths.append(str(diagnostics_json))
        seed_artifacts[int(seed)] = {
            "residual_payload": str(seed_results_dir / RESIDUAL_PAYLOAD_NAME),
            "diagnostics": diagnostics_paths,
        }

    payload = {
        "date": str(date.today()),
        "config": asdict(config),
        "artifacts": {
            "baseline_results_dir": str(baseline_dir),
            "baseline_payload": str(baseline_dir / SMOKE_PAYLOAD_NAME),
        },
        "baseline_summary": baseline_payload["rows"],
        "aggregate_rows": _aggregate_rows(seed_payloads),
        "seed_artifacts": seed_artifacts,
    }
    results_root = ROOT / config.results_root
    results_root.mkdir(parents=True, exist_ok=True)
    smoke._write_json(results_root / "gsm8k_contract_campaign.json", payload)
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
    )
    return run_campaign(config)


if __name__ == "__main__":
    main()
