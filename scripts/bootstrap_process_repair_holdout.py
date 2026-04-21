"""Bootstrap held-out stochastic route pools for process-repair evaluation."""

from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import subprocess
from dataclasses import asdict, dataclass
from typing import Sequence


DEFAULT_CHECKPOINT = (
    "checkpoints/bridge_ridge_qk_dynalign_prefdist_module_replace_20260420_diag/"
    "qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_prefdist_module_replace_cal16_chat.pt"
)
DEFAULT_SOURCE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_TARGET_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_METHOD = "rotalign_kv_gate_0.10"

SPLITS = {
    "gsm70": ("data/gsm8k_eval_70.jsonl", "qwen_gsm70"),
    "svamp70": ("data/svamp_eval_70.jsonl", "qwen_svamp70"),
}


@dataclass(frozen=True)
class CommandSpec:
    name: str
    output: str
    command: list[str]


@dataclass(frozen=True)
class SplitPlan:
    split: str
    eval_file: str
    route_outputs: list[str]
    route_commands: list[CommandSpec]
    repair_output_jsonl: str
    repair_output_md: str
    repair_command: CommandSpec


def _shell(command: Sequence[str]) -> str:
    return shlex.join([str(part) for part in command])


def _read_prefix(eval_file: pathlib.Path, limit: int) -> list[str]:
    lines = eval_file.read_text(encoding="utf-8").splitlines()
    if limit <= 0:
        raise ValueError("--limit must be positive when provided")
    if limit > len(lines):
        raise ValueError(f"--limit={limit} exceeds {eval_file} length {len(lines)}")
    return lines[:limit]


def _eval_file_for_split(
    *,
    split: str,
    output_dir: pathlib.Path,
    limit: int | None,
) -> tuple[str, str]:
    eval_file, tag = SPLITS[split]
    if limit is None:
        return eval_file, tag
    limited_tag = f"{tag}_n{limit}"
    limited_path = output_dir / f"{limited_tag}.jsonl"
    lines = _read_prefix(pathlib.Path(eval_file), limit)
    limited_path.parent.mkdir(parents=True, exist_ok=True)
    limited_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(limited_path), limited_tag


def _route_output(output_dir: pathlib.Path, tag: str, salt: int) -> str:
    return str(
        output_dir
        / f"{tag}_dynalign_prefdist_asym_kv_random_r025_v075_cal16_chat_salt{salt}_telemetry.jsonl"
    )


def build_route_command(
    *,
    python: str,
    checkpoint: str,
    source_model: str,
    target_model: str,
    eval_file: str,
    device: str,
    dtype: str,
    max_new_tokens: int,
    fixed_gate: float,
    salt: int,
    output: str,
) -> list[str]:
    return [
        python,
        "scripts/evaluate.py",
        "--translator",
        checkpoint,
        "--source-model",
        source_model,
        "--target-model",
        target_model,
        "--eval-file",
        eval_file,
        "--task-type",
        "generation",
        "--device",
        device,
        "--dtype",
        dtype,
        "--max-new-tokens",
        str(max_new_tokens),
        "--methods",
        "target",
        "rotalign",
        "--gate-mode",
        "fixed",
        "--fixed-gate",
        str(fixed_gate),
        "--fusion-rule",
        "static",
        "--kv-transport",
        "both",
        "--position-selection-metric",
        "attention",
        "--position-selection-ratio",
        "0.50",
        "--kv-route-selection-ratio",
        "0.25",
        "--kv-value-selection-ratio",
        "0.75",
        "--kv-route-selection-metric",
        "random",
        "--kv-value-selection-metric",
        "random",
        "--runtime-head-selection-metric",
        "attention_peak",
        "--runtime-head-selection-ratio",
        "1.0",
        "--source-reasoning-mode",
        "brief_analysis",
        "--source-use-chat-template",
        "--target-use-chat-template",
        "--source-enable-thinking",
        "false",
        "--target-enable-thinking",
        "false",
        "--random-salt",
        str(salt),
        "--prediction-output",
        output,
    ]


def build_repair_command(
    *,
    python: str,
    inputs: Sequence[str],
    eval_file: str,
    method: str,
    target_model: str,
    device: str,
    dtype: str,
    max_new_tokens: int,
    output_jsonl: str,
    output_md: str,
) -> list[str]:
    return [
        python,
        "scripts/process_repair_routes.py",
        "--inputs",
        *inputs,
        "--eval-file",
        eval_file,
        "--method",
        method,
        "--baseline-method",
        "target_alone",
        "--selection-policy",
        "target_on_strict_format",
        "--model",
        target_model,
        "--device",
        device,
        "--dtype",
        dtype,
        "--max-new-tokens",
        str(max_new_tokens),
        "--use-chat-template",
        "--no-enable-thinking",
        "--output-jsonl",
        output_jsonl,
        "--output-md",
        output_md,
    ]


def build_split_plan(
    *,
    split: str,
    output_dir: pathlib.Path,
    salts: Sequence[int],
    limit: int | None,
    python: str,
    checkpoint: str,
    source_model: str,
    target_model: str,
    device: str,
    dtype: str,
    max_new_tokens: int,
    repair_dtype: str,
    repair_max_new_tokens: int,
    fixed_gate: float,
    method: str = DEFAULT_METHOD,
) -> SplitPlan:
    eval_file, tag = _eval_file_for_split(split=split, output_dir=output_dir, limit=limit)
    route_outputs = [_route_output(output_dir, tag, int(salt)) for salt in salts]
    route_commands = [
        CommandSpec(
            name=f"{split}_route_salt{salt}",
            output=route_output,
            command=build_route_command(
                python=python,
                checkpoint=checkpoint,
                source_model=source_model,
                target_model=target_model,
                eval_file=eval_file,
                device=device,
                dtype=dtype,
                max_new_tokens=max_new_tokens,
                fixed_gate=fixed_gate,
                salt=int(salt),
                output=route_output,
            ),
        )
        for salt, route_output in zip(salts, route_outputs)
    ]
    repair_output_jsonl = str(output_dir / f"{tag}_process_repair_strict_selector_telemetry.jsonl")
    repair_output_md = str(output_dir / f"{tag}_process_repair_strict_selector_summary.md")
    repair_command = CommandSpec(
        name=f"{split}_process_repair",
        output=repair_output_jsonl,
        command=build_repair_command(
            python=python,
            inputs=route_outputs,
            eval_file=eval_file,
            method=method,
            target_model=target_model,
            device=device,
            dtype=repair_dtype,
            max_new_tokens=repair_max_new_tokens,
            output_jsonl=repair_output_jsonl,
            output_md=repair_output_md,
        ),
    )
    return SplitPlan(
        split=split,
        eval_file=eval_file,
        route_outputs=route_outputs,
        route_commands=route_commands,
        repair_output_jsonl=repair_output_jsonl,
        repair_output_md=repair_output_md,
        repair_command=repair_command,
    )


def write_shell_script(path: pathlib.Path, *, repo_root: pathlib.Path, plans: Sequence[SplitPlan]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {_shell([str(repo_root)])}",
        "export PYTHONUNBUFFERED=1",
        "",
    ]
    for plan in plans:
        lines.append(f"# {plan.split}: generate stochastic route pools")
        for spec in plan.route_commands:
            lines.append(_shell(spec.command))
        lines.append("")
        lines.append(f"# {plan.split}: repair selected route")
        lines.append(_shell(plan.repair_command.command))
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def write_manifest(path: pathlib.Path, *, plans: Sequence[SplitPlan], settings: dict[str, object]) -> None:
    payload = {
        "settings": settings,
        "plans": [asdict(plan) for plan in plans],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def execute_plans(plans: Sequence[SplitPlan]) -> None:
    for plan in plans:
        for spec in [*plan.route_commands, plan.repair_command]:
            subprocess.run(spec.command, check=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write or execute held-out process-repair bootstrap commands.")
    parser.add_argument("--splits", nargs="+", choices=sorted(SPLITS), default=["gsm70", "svamp70"])
    parser.add_argument("--salts", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output-dir", default="results/process_repair_holdout_20260421")
    parser.add_argument("--python", default="./venv_arm64/bin/python")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source-model", default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--repair-dtype", default="float32")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repair-max-new-tokens", type=int, default=128)
    parser.add_argument("--fixed-gate", type=float, default=0.10)
    parser.add_argument("--limit", type=int, default=None, help="Optional first-N subset for quick smoke manifests.")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--shell-script", default=None)
    parser.add_argument("--execute", action="store_true", help="Run generated commands sequentially after writing files.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plans = [
        build_split_plan(
            split=split,
            output_dir=output_dir,
            salts=args.salts,
            limit=args.limit,
            python=args.python,
            checkpoint=args.checkpoint,
            source_model=args.source_model,
            target_model=args.target_model,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            repair_dtype=args.repair_dtype,
            repair_max_new_tokens=args.repair_max_new_tokens,
            fixed_gate=args.fixed_gate,
        )
        for split in args.splits
    ]
    manifest = pathlib.Path(args.manifest) if args.manifest else output_dir / "process_repair_holdout_manifest.json"
    shell_script = pathlib.Path(args.shell_script) if args.shell_script else output_dir / "run_process_repair_holdout.sh"
    settings = {
        "splits": args.splits,
        "salts": args.salts,
        "limit": args.limit,
        "checkpoint": args.checkpoint,
        "source_model": args.source_model,
        "target_model": args.target_model,
        "method": DEFAULT_METHOD,
        "claim_policy": "dev-smoke if --limit is set; held-out only when limit is unset and protocol is frozen",
    }
    write_manifest(manifest, plans=plans, settings=settings)
    write_shell_script(shell_script, repo_root=pathlib.Path.cwd().resolve(), plans=plans)
    print(f"Wrote manifest: {manifest}")
    print(f"Wrote shell script: {shell_script}")
    if args.execute:
        execute_plans(plans)


if __name__ == "__main__":
    main()
