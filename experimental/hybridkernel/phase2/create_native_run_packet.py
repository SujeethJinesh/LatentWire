"""Create a native HybridKernel profiler run-packet skeleton.

The generated directory is a handoff scaffold for a user-operated NVIDIA host.
It is intentionally not admissible evidence until the TODO markers are replaced
with real server-side profiler artifacts and reduced native metrics.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experimental.hybridkernel.phase2.analyze_profiler_metrics import analyze, _write_markdown
from experimental.hybridkernel.phase2.check_profiler_run_artifacts import (
    READOUT_MARKERS,
    SKELETON_TODO_MARKER,
)


PHASE2_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = PHASE2_DIR / "profiler_runs"
DEFAULT_MODEL = "ibm-granite/granite-4.0-h-tiny"
DEFAULT_LABEL = "granite_boundary"
CONTROL_MATRIX_PATH = PHASE2_DIR / "native_control_matrix.json"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    if not slug:
        raise ValueError("label must contain at least one filename-safe character")
    return slug.lower()


def _write_new(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing run-packet file: {path}")
    path.write_text(content, encoding="utf-8")


def _packet_readme(model: str) -> str:
    return f"""# HybridKernel Native Run Packet

Status: skeleton only; not native profiler evidence.

Fill this directory on the NVIDIA host and return the whole directory. Do not
return screenshots, notebooks, or client-only traces.

Required final command:

```bash
python "$HWK_ROOT/phase2/check_profiler_run_artifacts.py" \\
  --run-dir "$HWK_RUN" \\
  | tee "$HWK_RUN/artifact_check.json"
```

The checker must pass before any HybridKernel result is interpreted. A pass
only means the packet is complete enough for review; it is not a speed claim.

Initial model target: `{model}`.
"""


def _environment_template() -> str:
    return f"""# HybridKernel Native Environment Metadata
# {SKELETON_TODO_MARKER}: replace this whole file with the command output below.

date -u
hostname
nvidia-smi
nsys --version
ncu --version
python -VV
python -m pip freeze

python - <<'PY'
import importlib.metadata as m
for name in ["vllm", "torch", "triton", "transformers"]:
    try:
        print(f"{{name}}=={{m.version(name)}}")
    except Exception as exc:
        print(f"{{name}}: unavailable ({{exc}})")
PY
"""


def _profile_scope(model: str) -> dict[str, object]:
    qwen_model = "Qwen/Qwen3-Next-80B-A3B-Instruct"
    return {
        "profiled_process": "vllm_server",
        "nsys_profiled_process": "vllm_server",
        "ncu_profiled_process": "vllm_server",
        "trace_scope": "server-side CUDA kernels under fixed request replay",
        "nsys_trace_scope": "server-side CUDA kernels under fixed request replay",
        "ncu_trace_scope": "server-side CUDA kernels under suspicious-kernel replay",
        "request_driver_process": "profiler_driver_http_client",
        "model": model,
        "vllm_command": (
            "python -m vllm.entrypoints.openai.api_server "
            f"--model {model} --dtype bfloat16 --max-model-len 2048 --disable-log-requests"
        ),
        "model_scopes": [
            {
                "row_role": "primary_hybrid,same_family_control",
                "model": model,
                "vllm_command": (
                    "python -m vllm.entrypoints.openai.api_server "
                    f"--model {model} --dtype bfloat16 --max-model-len 2048 --disable-log-requests"
                ),
                "scope": "Granite primary boundary and same-model non-boundary control rows",
            },
            {
                "row_role": "cross_family_falsification",
                "model": qwen_model,
                "vllm_command": (
                    "python -m vllm.entrypoints.openai.api_server "
                    f"--model {qwen_model} --dtype bfloat16 --max-model-len 2048 --disable-log-requests"
                ),
                "scope": "Qwen3-Next cross-family falsification rows",
            },
        ],
        "notes": "Update this JSON if the actual server command or profiling scope differs.",
    }


def _readout_template() -> str:
    rows = [
        f"| {marker} | {SKELETON_TODO_MARKER}: add trace/counter evidence | {SKELETON_TODO_MARKER}: yes/no |"
        for marker in READOUT_MARKERS
    ]
    return (
        "| Question | Evidence | Decision |\n"
        "|---|---|---|\n"
        + "\n".join(rows)
        + "\n\nReplace every TODO marker before returning this packet.\n"
    )


def _metrics_template(model: str, min_runs: int) -> dict[str, object]:
    row_specs = [
        {
            "label": "primary",
            "model": model,
            "row_role": "primary_hybrid",
            "control_family": "same_family_matched_segment",
            "control_model_or_segment": "granite_hybrid_attention_ssm_boundary_windows",
            "boundary_direction": "mixed_attention_ssm",
            "boundary_indices": [],
        },
        {
            "label": "same-family-control",
            "model": model,
            "row_role": "same_family_control",
            "control_family": "same_model_non_boundary_segment_control",
            "control_model_or_segment": (
                "granite_same_model_non_boundary_ssm_to_ssm_or_attention_internal_windows"
            ),
            "boundary_direction": "non_boundary_same_family",
            "boundary_indices": [],
        },
        {
            "label": "cross-family-falsification",
            "model": "Qwen/Qwen3-Next-80B-A3B-Instruct",
            "row_role": "cross_family_falsification",
            "control_family": "cross_family_hybrid_control",
            "control_model_or_segment": "qwen3_next_hybrid_boundary_windows",
            "boundary_direction": "linear_attention_gated_delta_boundary",
            "boundary_indices": [],
        },
    ]
    return {
        "description": (
            f"{SKELETON_TODO_MARKER}: fill one row per independent native "
            "NVIDIA/vLLM trace reduced from Nsight artifacts. Promotion requires "
            f"{min_runs} primary rows, {min_runs} same-family controls, and "
            f"{min_runs} cross-family falsification rows on the same request/runtime shape."
        ),
        "rows": [
            {
                "model": spec["model"],
                "run_id": f"{spec['label']}-repeat-{idx}",
                "total_step_ms": None,
                "attention_ssm_boundary_ms": None,
                "matched_non_boundary_ms": None,
                "recoverable_fraction": None,
                "dtype": "bfloat16",
                "profiled_process": "vllm_server",
                "trace_scope": "server-side CUDA kernels, not client-only HTTP replay",
                "cuda_graph_enabled": None,
                "batch_shape": {
                    "batch_size": 1,
                    "prefill_tokens": 128,
                    "decode_tokens": 64,
                    "requests": 16,
                },
                "control_model_or_segment": spec["control_model_or_segment"],
                "row_role": spec["row_role"],
                "control_family": spec["control_family"],
                "boundary_direction": spec["boundary_direction"],
                "nsys_artifact": None,
                "nsys_artifact_sha256": None,
                "ncu_artifact": None,
                "ncu_artifact_sha256": None,
                "kernel_names": [],
                "boundary_indices": spec["boundary_indices"],
                "time_window_ms": {"start": None, "end": None},
                "recoverable_fraction_basis": None,
                "reduction_command": None,
                "reduction_notes": None,
                "notes": f"{SKELETON_TODO_MARKER}: replace nulls after native profiling.",
            }
            for spec in row_specs
            for idx in range(min_runs)
        ],
    }


def _directory_readme(kind: str, required_artifact: str) -> str:
    return f"""# {kind}

Store real server-side `{required_artifact}` artifacts here. README files are
not accepted as profiler evidence by the artifact checker.
"""


def create_run_packet(
    *,
    output_dir: Path | None = None,
    run_root: Path = DEFAULT_RUN_ROOT,
    label: str = DEFAULT_LABEL,
    timestamp: str | None = None,
    model: str = DEFAULT_MODEL,
    min_runs: int = 3,
) -> Path:
    """Create and return a native run-packet skeleton directory."""

    if min_runs < 3:
        raise ValueError("min_runs must be at least 3 for the native review gate")

    if output_dir is None:
        output_dir = run_root / f"{timestamp or _utc_timestamp()}_{_slugify(label)}"
    output_dir = output_dir.resolve()

    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"run path exists and is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty run directory: {output_dir}")

    for relative in ["metadata", "logs", "nsys", "ncu"]:
        (output_dir / relative).mkdir(parents=True, exist_ok=True)

    _write_new(output_dir / "README.md", _packet_readme(model))
    _write_new(output_dir / "metadata/environment.txt", _environment_template())
    _write_new(
        output_dir / "metadata/profile_scope.json",
        json.dumps(_profile_scope(model), indent=2) + "\n",
    )
    architecture_map_path = PHASE2_DIR / "architecture_map.json"
    _write_new(
        output_dir / "metadata/architecture_map.json",
        architecture_map_path.read_text(encoding="utf-8"),
    )
    _write_new(
        output_dir / "metadata/native_control_matrix.json",
        CONTROL_MATRIX_PATH.read_text(encoding="utf-8"),
    )
    _write_new(
        output_dir / "logs/README.md",
        _directory_readme("Profiler Logs", "server and client replay logs"),
    )
    _write_new(
        output_dir / "nsys/README.md",
        _directory_readme("Nsight Systems", "*.nsys-rep, *.sqlite, or *.qdrep"),
    )
    _write_new(
        output_dir / "ncu/README.md",
        _directory_readme("Nsight Compute", "*.ncu-rep"),
    )
    _write_new(output_dir / "readout.md", _readout_template())

    metrics_payload = _metrics_template(model, min_runs)
    _write_new(
        output_dir / "profiler_metrics.json",
        json.dumps(metrics_payload, indent=2) + "\n",
    )
    analysis = analyze(metrics_payload)
    _write_new(
        output_dir / "profiler_analysis_gate.json",
        json.dumps(analysis, indent=2) + "\n",
    )
    _write_markdown(analysis, output_dir / "profiler_analysis_gate.md")

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--timestamp")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--min-runs", type=int, default=3)
    args = parser.parse_args()

    run_dir = create_run_packet(
        output_dir=args.output_dir,
        run_root=args.run_root,
        label=args.label,
        timestamp=args.timestamp,
        model=args.model,
        min_runs=args.min_runs,
    )
    print(
        json.dumps(
            {
                "status": "SKELETON_CREATED",
                "run_dir": str(run_dir),
                "next": "Fill native artifacts and run check_profiler_run_artifacts.py before review.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
