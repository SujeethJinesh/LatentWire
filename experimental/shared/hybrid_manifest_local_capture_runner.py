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
import shutil
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


def _reset_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _load_prompt(prompt_path: Path, prompt_id: str) -> dict[str, Any]:
    with prompt_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("prompt_id")) == prompt_id:
                return row
    raise ValueError(f"prompt_id {prompt_id!r} not found in {prompt_path}")


def _first_prompt_ids(prompt_path: Path, limit: int) -> tuple[str, ...]:
    if limit < 1:
        raise ValueError("--prompt-limit must be at least 1")
    prompt_ids: list[str] = []
    with prompt_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt_ids.append(str(row["prompt_id"]))
            if len(prompt_ids) >= limit:
                break
    if len(prompt_ids) < limit:
        raise ValueError(f"{prompt_path} only contains {len(prompt_ids)} prompts, requested {limit}")
    return tuple(prompt_ids)


def _first_horn_boundary_indices(template: dict[str, Any], limit: int) -> tuple[int, ...]:
    if limit < 1:
        raise ValueError("--horn-boundary-limit must be at least 1")
    boundary_indices = sorted({int(entry["boundary_index"]) for entry in template["horn_entries"]})
    if len(boundary_indices) < limit:
        raise ValueError(f"HORN template only contains {len(boundary_indices)} boundaries, requested {limit}")
    return tuple(boundary_indices[:limit])


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
    prompt_count: int,
) -> dict[str, Any]:
    capture_note = {
        "horn": " HORN tensors are captured from right-layer forward pre-hooks, not hidden-state proxies.",
        "ssq_lr": " SSQ-LR tensors are captured from the returned recurrent SSM cache state.",
    }.get(project, "")
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
                f"{project} local runner used {prompt_count} prompt(s) and max_input_tokens={max_input_tokens}; "
                "this packet is execution-plumbing evidence only and cannot promote a gate."
                f"{capture_note}"
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


def _load_tiny_model_and_tokenizer(
    *,
    model_id: str,
    hf_home: Path,
) -> tuple[Any, Any, float]:
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True, trust_remote_code=False)
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
    return tokenizer, model, time.perf_counter() - load_started


def _tokenize_prompt(tokenizer: Any, prompt: str, max_input_tokens: int) -> dict[str, torch.Tensor]:
    tokenized = tokenizer(prompt, return_tensors="pt")
    if tokenized["input_ids"].shape[1] > max_input_tokens:
        tokenized = {key: value[:, :max_input_tokens] for key, value in tokenized.items()}
    return tokenized


def _run_loaded_forward(
    *,
    model: Any,
    tokenized: dict[str, torch.Tensor],
    horn_entries: list[dict[str, Any]] | None = None,
) -> tuple[Any, dict[str, Any], dict[str, torch.Tensor]]:
    horn_captures: dict[str, torch.Tensor] = {}
    hook_handles: list[Any] = []
    if horn_entries:
        horn_captures, hook_handles = _install_horn_right_input_hooks(model, horn_entries)
    forward_started = time.perf_counter()
    try:
        with torch.no_grad():
            output = model(**tokenized, use_cache=True, output_hidden_states=True)
    finally:
        for handle in hook_handles:
            handle.remove()
    return (
        output,
        {
            "input_tokens": int(tokenized["input_ids"].shape[1]),
            "forward_seconds": time.perf_counter() - forward_started,
            "logits_shape": [int(dim) for dim in output.logits.shape],
            "horn_capture_mode": "right_layer_forward_pre_hook" if horn_entries else "none",
            "horn_tensor_count": len(horn_captures),
        },
        horn_captures,
    )


def _run_tiny_forward(
    *,
    model_id: str,
    prompt: str,
    hf_home: Path,
    max_input_tokens: int,
    horn_entries: list[dict[str, Any]] | None = None,
) -> tuple[Any, dict[str, Any], dict[str, torch.Tensor]]:
    tokenizer, model, load_seconds = _load_tiny_model_and_tokenizer(model_id=model_id, hf_home=hf_home)
    output, execution, horn_captures = _run_loaded_forward(
        model=model,
        tokenized=_tokenize_prompt(tokenizer, prompt, max_input_tokens),
        horn_entries=horn_entries,
    )
    execution["load_seconds"] = load_seconds
    return output, execution, horn_captures


def _cache_layers(output: Any) -> list[Any]:
    return list(getattr(output.past_key_values, "layers", []))


def _decoder_layers(model: Any) -> Any:
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(model, "layers", None),
        getattr(getattr(model, "decoder", None), "layers", None),
        getattr(getattr(model, "backbone", None), "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "__len__") and hasattr(candidate, "__getitem__"):
            return candidate
    raise ValueError("could not locate decoder layers for HORN right-input hooks")


def _first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for key in sorted(value):
            tensor = _first_tensor(value[key])
            if tensor is not None:
                return tensor
    return None


def _install_horn_right_input_hooks(
    model: Any,
    entries: list[dict[str, Any]],
) -> tuple[dict[str, torch.Tensor], list[Any]]:
    layers = _decoder_layers(model)
    names_by_right_layer: dict[int, list[str]] = {}
    for entry in entries:
        if str(entry["control_type"]) == "permuted_direction":
            continue
        names_by_right_layer.setdefault(int(entry["layer_right"]), []).append(str(entry["tensor"]))
    captures: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    for layer_index, tensor_names in sorted(names_by_right_layer.items()):
        if layer_index < 0 or layer_index >= len(layers):
            raise ValueError(f"HORN entry requested layer_right={layer_index}, but model has {len(layers)} layers")

        def hook(_module: Any, inputs: tuple[Any, ...], *, names: tuple[str, ...] = tuple(tensor_names)) -> None:
            tensor = _first_tensor(inputs)
            if tensor is None:
                raise ValueError("HORN right-input hook could not find a tensor in layer inputs")
            captured = tensor.detach().float().cpu()
            for name in names:
                captures[name] = captured.clone()

        handles.append(layers[layer_index].register_forward_pre_hook(hook))

    return captures, handles


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


def _horn_tensors(captures: dict[str, torch.Tensor], entries: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for entry in entries:
        if str(entry["control_type"]) == "permuted_direction":
            continue
        tensor_name = str(entry["tensor"])
        if tensor_name not in captures:
            raise ValueError(f"HORN right-input capture missing tensor {tensor_name!r}")
        tensors[tensor_name] = captures[tensor_name]
    return tensors


def run_capture(
    *,
    projects: tuple[str, ...] = ("ssq_lr", "horn"),
    model_id: str = DEFAULT_MODEL_ID,
    prompt_id: str = "hrsmoke_0001",
    prompt_ids: tuple[str, ...] | None = None,
    prompt_limit: int | None = None,
    ssq_prompt_limit: int | None = None,
    horn_prompt_limit: int | None = None,
    horn_boundary_limit: int | None = None,
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
    def resolve_project_prompt_ids(limit: int | None) -> tuple[str, ...]:
        if prompt_ids is not None:
            return prompt_ids
        effective_limit = limit if limit is not None else prompt_limit
        return _first_prompt_ids(prompt_path, effective_limit) if effective_limit else (prompt_id,)

    project_prompt_ids: dict[str, tuple[str, ...]] = {
        project: resolve_project_prompt_ids(ssq_prompt_limit if project == "ssq_lr" else horn_prompt_limit)
        for project in projects
    }
    resolved_prompt_ids = tuple(
        dict.fromkeys(prompt for ids in project_prompt_ids.values() for prompt in ids)
    )
    project_entries: dict[str, list[dict[str, Any]]] = {}
    if "ssq_lr" in projects:
        ssq_template = _load_template(manifest_dir, project="ssq_lr", canonical_model_id=DEFAULT_CANONICAL_MODEL_ID)
        project_entries["ssq_lr"] = [
            entry
            for selected_prompt_id in project_prompt_ids["ssq_lr"]
            for entry in select_ssq_entries(ssq_template, prompt_id=selected_prompt_id)
        ]
    if "horn" in projects:
        horn_template = _load_template(manifest_dir, project="horn", canonical_model_id=DEFAULT_CANONICAL_MODEL_ID)
        horn_boundary_indices = (
            _first_horn_boundary_indices(horn_template, horn_boundary_limit)
            if horn_boundary_limit
            else (0, 1)
        )
        project_entries["horn"] = [
            entry
            for selected_prompt_id in project_prompt_ids["horn"]
            for entry in select_horn_entries(
                horn_template,
                prompt_id=selected_prompt_id,
                boundary_indices=horn_boundary_indices,
            )
        ]
    outputs: dict[str, Any] = {}
    horn_captures: dict[str, torch.Tensor] = {}
    execution_runs: list[dict[str, Any]] = []
    tokenizer, model, load_seconds = _load_tiny_model_and_tokenizer(model_id=model_id, hf_home=hf_home)
    for selected_prompt_id in resolved_prompt_ids:
        prompt = _load_prompt(prompt_path, selected_prompt_id)
        prompt_horn_entries = [
            entry
            for entry in project_entries.get("horn", [])
            if str(entry["prompt_id"]) == selected_prompt_id
        ]
        output, run_execution, run_horn_captures = _run_loaded_forward(
            model=model,
            tokenized=_tokenize_prompt(tokenizer, str(prompt["prompt"]), max_input_tokens),
            horn_entries=prompt_horn_entries or None,
        )
        outputs[selected_prompt_id] = output
        horn_captures.update(run_horn_captures)
        execution_runs.append({"prompt_id": selected_prompt_id, **run_execution})
    execution = {
        "prompt_count": len(resolved_prompt_ids),
        "prompt_ids": list(resolved_prompt_ids),
        "input_tokens": max(int(row["input_tokens"]) for row in execution_runs),
        "load_seconds": load_seconds,
        "forward_seconds": sum(float(row["forward_seconds"]) for row in execution_runs),
        "runs": execution_runs,
    }
    project_summaries: dict[str, dict[str, Any]] = {}

    if "ssq_lr" in projects:
        template = _load_template(manifest_dir, project="ssq_lr", canonical_model_id=DEFAULT_CANONICAL_MODEL_ID)
        entries = project_entries["ssq_lr"]
        ssq_tensors: dict[str, torch.Tensor] = {}
        for selected_prompt_id in resolved_prompt_ids:
            prompt_entries = [entry for entry in entries if str(entry["prompt_id"]) == selected_prompt_id]
            ssq_tensors.update(_ssq_tensors(outputs[selected_prompt_id], prompt_entries))
        tensor_packet = output_dir / "ssq_lr_tensor_packet"
        gate_packet = output_dir / "ssq_lr_gate_packet"
        _reset_output_dir(tensor_packet)
        _reset_output_dir(gate_packet)
        save_tensor_packet(
            tensor_packet,
            tensors=ssq_tensors,
            metadata=_filled_metadata(
                template,
                project="ssq_lr",
                entries=entries,
                max_input_tokens=max_input_tokens,
                prompt_count=len(project_prompt_ids["ssq_lr"]),
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
        entries = project_entries["horn"]
        tensor_packet = output_dir / "horn_tensor_packet"
        gate_packet = output_dir / "horn_gate_packet"
        _reset_output_dir(tensor_packet)
        _reset_output_dir(gate_packet)
        save_tensor_packet(
            tensor_packet,
            tensors=_horn_tensors(horn_captures, entries),
            metadata=_filled_metadata(
                template,
                project="horn",
                entries=entries,
                max_input_tokens=max_input_tokens,
                prompt_count=len(project_prompt_ids["horn"]),
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
        "prompt_ids": list(resolved_prompt_ids),
        "project_prompt_ids": {project: list(ids) for project, ids in project_prompt_ids.items()},
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
        f"- Prompts: `{', '.join(summary['prompt_ids'])}`",
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
    parser.add_argument("--prompt-limit", type=int)
    parser.add_argument("--ssq-prompt-limit", type=int)
    parser.add_argument("--horn-prompt-limit", type=int)
    parser.add_argument("--horn-boundary-limit", type=int)
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
        prompt_limit=args.prompt_limit,
        ssq_prompt_limit=args.ssq_prompt_limit,
        horn_prompt_limit=args.horn_prompt_limit,
        horn_boundary_limit=args.horn_boundary_limit,
        prompt_path=args.prompt_path,
        manifest_dir=args.manifest_dir,
        hf_home=args.hf_home,
        output_dir=args.output_dir,
        max_input_tokens=args.max_input_tokens,
    )


if __name__ == "__main__":
    main()
