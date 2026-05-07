"""Build resource-limited local SSQ-LR/HORN packets from cached Granite Tiny.

This runner is deliberately non-promoting. It consumes the generated capture
manifest templates, runs one tiny local `transformers` forward, writes saved
tensor packets, builds real gate packets, and validates them as
`RESOURCE_LIMITED_NOT_PROMOTABLE_*` evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experimental.shared.activation_dumper import save_tensor_packet
from experimental.shared.check_gate_packet import validate_gate_packet
from experimental.shared.hybrid_trace_packet_builder import build_horn_packet, build_ssq_lr_packet


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ID = "ibm-granite/granite-4.0-h-tiny"
DEFAULT_CANONICAL_MODEL_ID = "ibm-granite-4.0-h-tiny"
GRANITE_TINY_REVISION = "791e0d3d28c86e106c9b6e0b4cecdee0375b6124"
DEFAULT_HF_HOME = ROOT / ".debug/hf_home"
DEFAULT_PROMPTS = ROOT / "experimental/shared/prompts/hybrid_reasoning_smoke_12_20260506.jsonl"
DEFAULT_MANIFEST_DIR = ROOT / "experimental/shared/results/hybrid_capture_manifests_20260507"
DEFAULT_OUTPUT_DIR = ROOT / "experimental/shared/results/hybrid_manifest_local_capture_20260507"
SSQ_BUCKETS = ("prefill_end", "2k_or_end", "8k_or_end", "final_minus_128")


def _load_prompt(prompt_path: Path, prompt_id: str) -> dict[str, Any]:
    with prompt_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("prompt_id")) == prompt_id:
                return row
    raise ValueError(f"prompt_id {prompt_id!r} not found in {prompt_path}")


def _load_template(manifest_dir: Path, *, project: str, canonical_model_id: str) -> dict[str, Any]:
    filename = f"{project}__{canonical_model_id.replace('.', '-')}__metadata_template.json"
    path = manifest_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"missing capture manifest template: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _filled_metadata(
    template: dict[str, Any],
    *,
    project: str,
    entries: list[dict[str, Any]],
    max_input_tokens: int,
) -> dict[str, Any]:
    metadata = {
        key: value
        for key, value in template.items()
        if key
        not in {
            "_template_only",
            "fill_before_use",
            "planned_entry_count",
            "ssq_lr_entries",
            "horn_entries",
        }
    }
    metadata.update(
        {
            "model_id": DEFAULT_CANONICAL_MODEL_ID,
            "canonical_model_id": DEFAULT_CANONICAL_MODEL_ID,
            "served_model_id": DEFAULT_MODEL_ID,
            "model_revision": GRANITE_TINY_REVISION,
            "tokenizer_revision": GRANITE_TINY_REVISION,
            "context_lengths": [max_input_tokens],
            "dtype": "float16",
            "device": "cpu",
            "command": "python -m experimental.shared.hybrid_manifest_local_capture_runner",
            "resource_limit_note": (
                f"{project} local runner used one prompt and max_input_tokens={max_input_tokens}; "
                "this packet is execution-plumbing evidence only and cannot promote a gate."
            ),
        }
    )
    metadata[f"{project}_entries"] = entries
    return metadata


def select_ssq_entries(
    template: dict[str, Any],
    *,
    prompt_id: str,
    layer: int = 0,
) -> list[dict[str, Any]]:
    entries = [
        entry
        for entry in template["ssq_lr_entries"]
        if str(entry["prompt_id"]) == prompt_id and int(entry["layer"]) == layer
    ]
    by_bucket = {str(entry["position_bucket"]): entry for entry in entries}
    missing = set(SSQ_BUCKETS) - set(by_bucket)
    if missing:
        raise ValueError(f"SSQ-LR template missing buckets for layer {layer}: {sorted(missing)}")
    return [dict(by_bucket[bucket]) for bucket in SSQ_BUCKETS]


def select_horn_entries(
    template: dict[str, Any],
    *,
    prompt_id: str,
    boundary_indices: tuple[int, int] = (0, 1),
) -> list[dict[str, Any]]:
    wanted = set(boundary_indices)
    entries = [
        dict(entry)
        for entry in template["horn_entries"]
        if str(entry["prompt_id"]) == prompt_id and int(entry["boundary_index"]) in wanted
    ]
    directions = {
        str(entry["direction"])
        for entry in entries
        if str(entry["control_type"]) == "boundary"
    }
    if directions != {"attention->ssm", "ssm->attention"}:
        raise ValueError(f"HORN template selection must include both boundary directions, got {directions}")
    controls_by_boundary = {
        boundary: {
            str(entry["control_type"])
            for entry in entries
            if int(entry["boundary_index"]) == boundary
        }
        for boundary in wanted
    }
    for boundary, controls in controls_by_boundary.items():
        if controls != {"boundary", "non_boundary", "permuted_direction"}:
            raise ValueError(f"HORN boundary {boundary} missing required controls: {controls}")
    return entries


def _run_tiny_forward(
    *,
    model_id: str,
    prompt: str,
    hf_home: Path,
    max_input_tokens: int,
) -> tuple[Any, dict[str, Any]]:
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True, trust_remote_code=False)
    tokenized = tokenizer(prompt, return_tensors="pt")
    if tokenized["input_ids"].shape[1] > max_input_tokens:
        tokenized = {key: value[:, :max_input_tokens] for key, value in tokenized.items()}
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    model.eval()
    load_seconds = time.perf_counter() - load_started
    forward_started = time.perf_counter()
    with torch.no_grad():
        output = model(**tokenized, use_cache=True, output_hidden_states=True)
    return output, {
        "input_tokens": int(tokenized["input_ids"].shape[1]),
        "load_seconds": load_seconds,
        "forward_seconds": time.perf_counter() - forward_started,
        "logits_shape": [int(dim) for dim in output.logits.shape],
    }


def _cache_layers(output: Any) -> list[Any]:
    return list(getattr(output.past_key_values, "layers", []))


def _ssq_tensors(output: Any, entries: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    layers = _cache_layers(output)
    tensors: dict[str, torch.Tensor] = {}
    for entry in entries:
        layer = int(entry["layer"])
        state = getattr(layers[layer], "recurrent_states", None)
        if state is None:
            state = getattr(layers[layer], "ssm_states", None)
        if state is None:
            raise ValueError(f"cache layer {layer} lacks recurrent_states/ssm_states")
        tensors[str(entry["tensor"])] = state.detach().float()
    return tensors


def _hidden_tensor(output: Any, layer_index: int) -> torch.Tensor:
    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states is None:
        raise ValueError("model output did not include hidden_states")
    index = min(max(layer_index, 0) + 1, len(hidden_states) - 1)
    return hidden_states[index].detach().float()


def _horn_tensors(output: Any, entries: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for entry in entries:
        if str(entry["control_type"]) == "permuted_direction":
            continue
        tensors[str(entry["tensor"])] = _hidden_tensor(output, int(entry["layer_right"]))
    return tensors


def run_capture(
    *,
    projects: tuple[str, ...] = ("ssq_lr", "horn"),
    model_id: str = DEFAULT_MODEL_ID,
    prompt_id: str = "hrsmoke_0001",
    prompt_path: Path = DEFAULT_PROMPTS,
    manifest_dir: Path = DEFAULT_MANIFEST_DIR,
    hf_home: Path = DEFAULT_HF_HOME,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_input_tokens: int = 8,
) -> dict[str, Any]:
    unsupported = set(projects) - {"ssq_lr", "horn"}
    if unsupported:
        raise ValueError(f"unsupported local runner projects: {sorted(unsupported)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = _load_prompt(prompt_path, prompt_id)
    output, execution = _run_tiny_forward(
        model_id=model_id,
        prompt=str(prompt["prompt"]),
        hf_home=hf_home,
        max_input_tokens=max_input_tokens,
    )
    project_summaries: dict[str, dict[str, Any]] = {}

    if "ssq_lr" in projects:
        template = _load_template(manifest_dir, project="ssq_lr", canonical_model_id=DEFAULT_CANONICAL_MODEL_ID)
        entries = select_ssq_entries(template, prompt_id=prompt_id)
        tensor_packet = output_dir / "ssq_lr_tensor_packet"
        gate_packet = output_dir / "ssq_lr_gate_packet"
        save_tensor_packet(
            tensor_packet,
            tensors=_ssq_tensors(output, entries),
            metadata=_filled_metadata(
                template,
                project="ssq_lr",
                entries=entries,
                max_input_tokens=max_input_tokens,
            ),
        )
        build_ssq_lr_packet(tensor_packet, gate_packet)
        report = validate_gate_packet(gate_packet, mode="real", project="ssq_lr")
        project_summaries["ssq_lr"] = {
            "tensor_packet": str(tensor_packet),
            "gate_packet": str(gate_packet),
            "checker_ok": bool(report["ok"]),
            "checker_decision": report["decision"],
            "errors": report["errors"],
        }

    if "horn" in projects:
        template = _load_template(manifest_dir, project="horn", canonical_model_id=DEFAULT_CANONICAL_MODEL_ID)
        entries = select_horn_entries(template, prompt_id=prompt_id)
        tensor_packet = output_dir / "horn_tensor_packet"
        gate_packet = output_dir / "horn_gate_packet"
        save_tensor_packet(
            tensor_packet,
            tensors=_horn_tensors(output, entries),
            metadata=_filled_metadata(
                template,
                project="horn",
                entries=entries,
                max_input_tokens=max_input_tokens,
            ),
        )
        build_horn_packet(tensor_packet, gate_packet)
        report = validate_gate_packet(gate_packet, mode="real", project="horn")
        project_summaries["horn"] = {
            "tensor_packet": str(tensor_packet),
            "gate_packet": str(gate_packet),
            "checker_ok": bool(report["ok"]),
            "checker_decision": report["decision"],
            "errors": report["errors"],
        }

    summary = {
        "surface": "hybrid_manifest_local_capture_runner",
        "decision": "RESOURCE_LIMITED_CAPTURE_PACKETS_WRITTEN_NOT_PROMOTABLE",
        "model_id": model_id,
        "prompt_id": prompt_id,
        "projects": list(projects),
        "execution": execution,
        "project_summaries": project_summaries,
        "claim_boundary": [
            "resource-limited local capture",
            "not SSQ-LR/HORN/HBSM gate promotion",
            "not GPU evidence",
            "not quality evidence",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_dir / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "project_summaries": project_summaries}, sort_keys=True))
    return summary


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Hybrid Manifest Local Capture Runner",
        "",
        f"Decision: `{summary['decision']}`",
        "",
        "This is a resource-limited local capture packet. It cannot promote SSQ-LR, HORN, or HBSM.",
        "",
        f"- Model: `{summary['model_id']}`",
        f"- Prompt: `{summary['prompt_id']}`",
        f"- Input tokens: `{summary['execution']['input_tokens']}`",
        f"- Load seconds: `{summary['execution']['load_seconds']:.2f}`",
        f"- Forward seconds: `{summary['execution']['forward_seconds']:.2f}`",
        "",
        "| Project | Checker | Gate packet |",
        "|---|---|---|",
    ]
    for project, row in summary["project_summaries"].items():
        status = "PASS" if row["checker_ok"] else "FAIL"
        lines.append(f"| `{project}` | `{status}` | `{row['gate_packet']}` |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", action="append", choices=("ssq_lr", "horn"))
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--prompt-id", default="hrsmoke_0001")
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-input-tokens", type=int, default=8)
    args = parser.parse_args()
    run_capture(
        projects=tuple(args.project) if args.project else ("ssq_lr", "horn"),
        model_id=args.model_id,
        prompt_id=args.prompt_id,
        prompt_path=args.prompt_path,
        manifest_dir=args.manifest_dir,
        hf_home=args.hf_home,
        output_dir=args.output_dir,
        max_input_tokens=args.max_input_tokens,
    )


if __name__ == "__main__":
    main()
