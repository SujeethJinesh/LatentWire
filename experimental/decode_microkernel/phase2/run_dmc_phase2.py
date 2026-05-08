#!/usr/bin/env python3
"""Prepare a Decode Microkernel Consolidation Phase 2 serving packet.

Phase 2 is intentionally stricter than Phase 1: it requires real serving
rows from a vLLM decode path with DMC integrated into inference, not replay
timing. This runner does not fake those rows. Until a supported serving hook is
implemented, it emits an auditable FAIL_INFRA packet with fixed-input,
environment, prompt, model, and load diagnostics.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PHASE1_PACKET_REL = (
    "experimental/decode_microkernel/phase1/results/"
    "dmc_phase1_20260508T000525Z"
)
DEFAULT_PHASE1_PACKET = ROOT / PHASE1_PACKET_REL
DEFAULT_RESULTS_DIR = ROOT / "experimental/decode_microkernel/phase2/results"
DEFAULT_PROMPT_FILE = ROOT / "experimental/shared/prompts/aime_2025_indices_0_11.jsonl"
PHASE1_PACKET_ID = "dmc_phase1_20260508T000525Z"
PHASE1_PASS_DECISION = "PASS_DMC_PHASE1_CONSOLIDATED_REPLAY"
SCHEMA_VERSION = "dmc_phase2_metrics_v1"
INFRA_DECISION = "FAIL_INFRA_DMC_PHASE2"

ROLE_MODELS = {
    "primary": "ibm-granite/granite-4.0-h-tiny",
    "same_family": "ibm-granite/granite-4.0-h-small",
    "cross_family": "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
}

THRESHOLDS: dict[str, Any] = {
    "min_rows_by_role": {"primary": 12, "same_family": 12, "cross_family": 12},
    "min_row_launch_reduction_fraction": 0.10,
    "min_role_median_decode_latency_reduction": {
        "primary": 0.05,
        "same_family": 0.05,
        "cross_family": 0.03,
    },
    "bootstrap_ci95_lower_bound_min_exclusive": 0.0,
    "require_positive_median_tokens_per_second_gain": True,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def command_output(command: list[str], *, timeout: int = 60) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=ROOT,
        )
    except Exception as exc:  # pragma: no cover - environment-specific.
        return {"command": command, "returncode": None, "stdout": "", "stderr": repr(exc)}
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def git_sha() -> str | None:
    result = command_output(["git", "rev-parse", "HEAD"], timeout=30)
    if result["returncode"] == 0:
        return str(result["stdout"]).strip()
    return None


def ensure_fixed_packet(actual: Path) -> None:
    if actual.resolve() != DEFAULT_PHASE1_PACKET.resolve():
        raise ValueError(
            f"Phase 2 must use fixed Phase 1 PASS packet {DEFAULT_PHASE1_PACKET}; got {actual}"
        )


def packet_files(packet: Path, rel_paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rel in sorted(set(rel_paths)):
        path = packet / rel
        rows.append(
            {
                "path": rel,
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else None,
                "sha256": file_sha256(path) if path.is_file() else None,
            }
        )
    return rows


def build_input_manifest(phase1_packet: Path, prompt_file: Path) -> dict[str, Any]:
    phase1_files = [
        "checker_result.json",
        "command_metadata.json",
        "environment.json",
        "input_artifact_manifest.json",
        "logs/stdout.log",
        "logs/stderr.log",
        "metrics.json",
        "replay_schedule.json",
    ]
    prompt_file_entry = {
        "path": str(prompt_file.relative_to(ROOT)) if prompt_file.exists() else str(prompt_file),
        "exists": prompt_file.exists(),
        "bytes": prompt_file.stat().st_size if prompt_file.exists() else None,
        "sha256": file_sha256(prompt_file) if prompt_file.is_file() else None,
    }
    return {
        "schema_version": "dmc_phase2_input_manifest_v1",
        "created_at_utc": utc_now(),
        "packets": [
            {
                "packet_role": "phase1_pass_packet",
                "packet_id": PHASE1_PACKET_ID,
                "packet_path": str(phase1_packet.relative_to(ROOT)),
                "files": packet_files(phase1_packet, phase1_files),
            }
        ],
        "prompt_source_file": prompt_file_entry,
    }


def parse_prompt_file(prompt_file: Path) -> tuple[list[dict[str, Any]], list[str]]:
    reasons: list[str] = []
    prompts: list[dict[str, Any]] = []
    if not prompt_file.is_file():
        return [], [f"canonical AIME-2025 prompt file is missing: {prompt_file}"]
    try:
        lines = [line for line in prompt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        for index, line in enumerate(lines):
            item = json.loads(line)
            if not isinstance(item, dict):
                reasons.append(f"prompt row {index} is not a JSON object")
                continue
            prompt = item.get("prompt") or item.get("problem") or item.get("question")
            if not isinstance(prompt, str) or not prompt.strip():
                reasons.append(f"prompt row {index} has no prompt/problem/question text")
                continue
            prompts.append(
                {
                    "index": int(item.get("index", index)),
                    "prompt_id": str(item.get("prompt_id", item.get("id", index))),
                    "prompt": prompt,
                    "answer": item.get("answer"),
                }
            )
    except Exception as exc:
        return [], [f"cannot parse canonical AIME-2025 prompt file: {exc!r}"]
    if [row["index"] for row in prompts] != list(range(12)):
        reasons.append("prompt indices are not exactly deterministic indices 0-11")
    if len(prompts) != 12:
        reasons.append(f"prompt count {len(prompts)} is not the preregistered count 12")
    return prompts, reasons


def prompt_payload_sha256(prompts: list[dict[str, Any]]) -> str | None:
    if not prompts:
        return None
    payload = json.dumps(prompts, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return bytes_sha256(payload)


def build_prompt_manifest(prompt_file: Path) -> tuple[dict[str, Any], list[str]]:
    prompts, reasons = parse_prompt_file(prompt_file)
    prompt_sha = prompt_payload_sha256(prompts)
    return (
        {
            "schema_version": "dmc_phase2_prompt_manifest_v1",
            "created_at_utc": utc_now(),
            "source": "AIME-2025",
            "selection": "deterministic_indices_0_11",
            "prompt_file": str(prompt_file),
            "prompt_file_sha256": file_sha256(prompt_file) if prompt_file.is_file() else None,
            "prompt_count": len(prompts),
            "prompt_sha256": prompt_sha,
            "prompts": prompts,
        },
        reasons,
    )


def package_version(name: str) -> dict[str, Any]:
    try:
        module = __import__(name)
        return {"available": True, "version": getattr(module, "__version__", None)}
    except Exception as exc:
        return {"available": False, "import_error": repr(exc)}


def build_environment() -> dict[str, Any]:
    torch_info: dict[str, Any]
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": torch.cuda.get_device_capability(index),
                    "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else [],
        }
    except Exception as exc:  # pragma: no cover - environment-specific.
        torch_info = {"import_error": repr(exc)}
    return {
        "schema_version": "dmc_phase2_environment_v1",
        "created_at_utc": utc_now(),
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "cwd": str(Path.cwd()),
        "git_sha": git_sha(),
        "torch": torch_info,
        "packages": {
            "vllm": package_version("vllm"),
            "transformers": package_version("transformers"),
            "huggingface_hub": package_version("huggingface_hub"),
            "triton": package_version("triton"),
        },
        "commands": {
            "pip_freeze": command_output([sys.executable, "-m", "pip", "freeze"], timeout=120),
            "nvidia_smi": command_output(["nvidia-smi"], timeout=30),
            "nvcc_version": command_output(["nvcc", "--version"], timeout=30),
            "nsys_version": command_output(["nsys", "--version"], timeout=30),
        },
        "environment_variables": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
            "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
            "VLLM_ATTENTION_BACKEND": os.environ.get("VLLM_ATTENTION_BACKEND"),
            "VLLM_USE_V1": os.environ.get("VLLM_USE_V1"),
            "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
        },
    }


def model_diagnostics(*, attempt_load: bool) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "schema_version": "dmc_phase2_model_diagnostics_v1",
        "attempted_vllm_model_load": attempt_load,
        "models": {},
    }
    try:
        from huggingface_hub import HfApi, scan_cache_dir

        api = HfApi()
        try:
            cache = scan_cache_dir()
            cached_repos = {
                repo.repo_id: {
                    "repo_path": str(repo.repo_path),
                    "revisions": sorted(str(rev.commit_hash) for rev in repo.revisions),
                }
                for repo in cache.repos
            }
        except Exception as exc:
            cached_repos = {"scan_error": repr(exc)}
    except Exception as exc:
        api = None
        cached_repos = {"import_error": repr(exc)}

    for role, model_id in ROLE_MODELS.items():
        row: dict[str, Any] = {"role": role, "model_id": model_id, "hf_cache": cached_repos.get(model_id)}
        if api is not None:
            try:
                info = api.model_info(model_id)
                row["huggingface_model_info"] = {
                    "sha": getattr(info, "sha", None),
                    "last_modified": str(getattr(info, "last_modified", None)),
                    "private": getattr(info, "private", None),
                    "gated": getattr(info, "gated", None),
                }
            except Exception as exc:
                row["huggingface_model_info_error"] = repr(exc)
        try:
            from transformers import AutoConfig, AutoTokenizer

            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            row["transformers_config"] = {
                "ok": True,
                "model_type": getattr(config, "model_type", None),
                "architectures": getattr(config, "architectures", None),
                "torch_dtype": str(getattr(config, "torch_dtype", None)),
            }
            row["transformers_tokenizer"] = {
                "ok": True,
                "class": tokenizer.__class__.__name__,
                "vocab_size": getattr(tokenizer, "vocab_size", None),
            }
        except Exception as exc:
            row["transformers_load_error"] = repr(exc)
        if attempt_load:
            try:
                from vllm import LLM

                llm = LLM(
                    model=model_id,
                    trust_remote_code=True,
                    dtype="bfloat16",
                    enforce_eager=True,
                    max_model_len=2048,
                    gpu_memory_utilization=0.80,
                )
                row["vllm_load"] = {"ok": True, "llm_class": llm.__class__.__name__}
                del llm
            except Exception as exc:
                row["vllm_load"] = {"ok": False, "error": repr(exc)}
        diagnostics["models"][role] = row
    return diagnostics


def choose_run_dir(results_dir: Path, run_id: str | None) -> Path:
    if run_id is None:
        run_id = "dmc_phase2_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return results_dir / run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-packet", type=Path, default=DEFAULT_PHASE1_PACKET)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument(
        "--attempt-vllm-model-load",
        action="store_true",
        help="Attempt full vLLM model construction for diagnostics only; no serving rows are run.",
    )
    args = parser.parse_args(argv)

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    infra_reasons: list[str] = []
    started_at = utc_now()
    phase1_packet = args.phase1_packet.resolve()
    prompt_file = args.prompt_file.resolve()
    run_dir = choose_run_dir(args.results_dir.resolve(), args.run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    phase1_checker_decision: str | None = None
    try:
        ensure_fixed_packet(phase1_packet)
        phase1_checker = load_json(phase1_packet / "checker_result.json")
        phase1_checker_decision = str(phase1_checker.get("decision"))
        if phase1_checker_decision != PHASE1_PASS_DECISION:
            infra_reasons.append(
                f"fixed Phase 1 checker decision is {phase1_checker_decision}, expected {PHASE1_PASS_DECISION}"
            )
    except Exception as exc:
        infra_reasons.append(f"cannot validate fixed Phase 1 packet: {exc!r}")
        stderr_lines.append(repr(exc))

    prompt_manifest, prompt_reasons = build_prompt_manifest(prompt_file)
    infra_reasons.extend(prompt_reasons)
    write_json(run_dir / "prompt_manifest.json", prompt_manifest)

    diagnostic_stdout = io.StringIO()
    diagnostic_stderr = io.StringIO()
    with contextlib.redirect_stdout(diagnostic_stdout), contextlib.redirect_stderr(diagnostic_stderr):
        model_diag = model_diagnostics(attempt_load=args.attempt_vllm_model_load)
    captured_stdout = diagnostic_stdout.getvalue().strip()
    captured_stderr = diagnostic_stderr.getvalue().strip()
    if captured_stdout:
        stdout_lines.append(captured_stdout)
    if captured_stderr:
        stderr_lines.append(captured_stderr)
    write_json(run_dir / "model_diagnostics.json", model_diag)

    infra_reasons.append(
        "no supported vLLM decode micro-operation replacement hook is implemented in this repo; "
        "Phase 2 requires real serving rows, so replay latency is not used as a proxy"
    )

    metrics = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "decision_hint": INFRA_DECISION,
        "phase1_packet": PHASE1_PACKET_REL,
        "phase1_packet_id": PHASE1_PACKET_ID,
        "phase1_checker_decision": phase1_checker_decision,
        "phase1_checker_result_sha256": file_sha256(phase1_packet / "checker_result.json")
        if (phase1_packet / "checker_result.json").is_file()
        else None,
        "phase1_metrics_sha256": file_sha256(phase1_packet / "metrics.json")
        if (phase1_packet / "metrics.json").is_file()
        else None,
        "required_models": ROLE_MODELS,
        "prompt_source": "AIME-2025",
        "prompt_selection": "deterministic_indices_0_11",
        "prompt_sha256": prompt_manifest.get("prompt_sha256"),
        "thresholds": THRESHOLDS,
        "method": {
            "name": "dmc_vllm_serving_integration",
            "serving_backend": "vllm",
            "serving_integration_implemented": False,
            "serving_rows_are_real": False,
            "phase1_replay_latency_used_as_proxy": False,
            "cpu_only_timing": False,
            "model_substitution": False,
            "boundary_fusion_claim": False,
        },
        "rows": [],
        "role_summary": {},
        "bootstrap_samples": args.bootstrap_samples,
        "infra_reasons": infra_reasons,
        "interpretation": (
            "This is an explicit infrastructure packet. It is not a serving benchmark "
            "and cannot support a DMC serving-speedup claim."
        ),
    }
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "input_artifact_manifest.json", build_input_manifest(phase1_packet, prompt_file))
    environment_stdout = io.StringIO()
    environment_stderr = io.StringIO()
    with contextlib.redirect_stdout(environment_stdout), contextlib.redirect_stderr(environment_stderr):
        environment = build_environment()
    captured_stdout = environment_stdout.getvalue().strip()
    captured_stderr = environment_stderr.getvalue().strip()
    if captured_stdout:
        stdout_lines.append(captured_stdout)
    if captured_stderr:
        stderr_lines.append(captured_stderr)
    write_json(run_dir / "environment.json", environment)
    write_json(
        run_dir / "command_metadata.json",
        {
            "schema_version": "dmc_phase2_command_v1",
            "started_at_utc": started_at,
            "ended_at_utc": utc_now(),
            "argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            "cwd": str(Path.cwd()),
            "git_sha": git_sha(),
            "script_path": str(Path(__file__).resolve().relative_to(ROOT)),
            "script_sha256": file_sha256(Path(__file__).resolve()),
            "phase1_packet": str(phase1_packet),
            "prompt_file": str(prompt_file),
            "run_dir": str(run_dir),
            "bootstrap_samples": args.bootstrap_samples,
            "attempt_vllm_model_load": args.attempt_vllm_model_load,
            "path_python": shutil.which("python"),
            "environment_variables": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "HF_HOME": os.environ.get("HF_HOME"),
                "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
                "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
                "VLLM_ATTENTION_BACKEND": os.environ.get("VLLM_ATTENTION_BACKEND"),
                "VLLM_USE_V1": os.environ.get("VLLM_USE_V1"),
            },
        },
    )
    stdout_lines.append(
        json.dumps(
            {
                "decision_hint": INFRA_DECISION,
                "run_dir": str(run_dir.relative_to(ROOT)),
                "infra_reason_count": len(infra_reasons),
            },
            sort_keys=True,
        )
    )
    (run_dir / "logs/stdout.log").write_text("\n".join(stdout_lines) + "\n", encoding="utf-8")
    (run_dir / "logs/stderr.log").write_text("\n".join(stderr_lines) + "\n", encoding="utf-8")
    print(stdout_lines[-1])
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
