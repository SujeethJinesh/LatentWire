"""Resource-limited HORN H2 noise-propagation scout.

This is a Mac-local decision scout after the Granite Tiny H1a magnitude screen
failed weakly. It perturbs the right-layer input at one boundary in each
direction, measures short-prompt NLL drift, and writes a follow-up H2 packet.
The packet can pass the H2 schema contract, but its claim boundary is
resource-limited scout evidence only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from experimental.shared.followup_gate_contracts import evaluate_horn_h2
from experimental.shared.hybrid_manifest_local_capture_runner import (
    DEFAULT_CANONICAL_MODEL_ID,
    DEFAULT_HF_HOME,
    DEFAULT_MODEL_ID,
    GRANITE_TINY_REVISION,
    _decoder_layers,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS = ROOT / "experimental/shared/prompts/hybrid_reasoning_smoke_12_20260506.jsonl"
DEFAULT_TEMPLATE = (
    ROOT
    / "experimental/shared/results/hybrid_capture_manifests_20260507/"
    "horn__ibm-granite-4-0-h-tiny__metadata_template.json"
)
DEFAULT_H1_PACKET = ROOT / "experimental/shared/results/hybrid_manifest_local_capture_20260507/horn_gate_packet"
DEFAULT_OUTPUT_DIR = ROOT / "experimental/shared/results/horn_h2_noise_replay_scout_20260507"
DEFAULT_PREREG = ROOT / "experimental/horn/phase2/preregister_horn_20260506.md"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _packet_hash(packet_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in packet_dir.rglob("*") if item.is_file()):
        if path.suffix == ".pt":
            continue
        digest.update(str(path.relative_to(packet_dir)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_prompts(path: Path, limit: int) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            prompts.append(json.loads(line))
            if len(prompts) >= limit:
                break
    if len(prompts) < limit:
        raise ValueError(f"{path} has {len(prompts)} prompts, requested {limit}")
    return prompts


def _load_boundaries(template_path: Path) -> dict[str, dict[str, Any]]:
    template = json.loads(template_path.read_text(encoding="utf-8"))
    boundaries: dict[str, dict[str, Any]] = {}
    for entry in template["horn_entries"]:
        if str(entry.get("control_type")) != "boundary":
            continue
        direction = str(entry["direction"])
        if direction not in boundaries:
            boundaries[direction] = dict(entry)
    required = {"attention->ssm", "ssm->attention"}
    missing = required - set(boundaries)
    if missing:
        raise ValueError(f"template missing boundary direction(s): {sorted(missing)}")
    return boundaries


def _tokenize(tokenizer: Any, text: str, max_input_tokens: int) -> dict[str, torch.Tensor]:
    tokenized = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_tokens)
    return {key: value.to("cpu") for key, value in tokenized.items()}


def _nll_from_logits(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    if input_ids.shape[1] < 2:
        return 0.0
    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = input_ids[:, 1:]
    return float(
        F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.shape[-1]),
            shifted_labels.reshape(-1),
            reduction="mean",
        ).item()
    )


def _run_nll(
    model: Any,
    tokenized: dict[str, torch.Tensor],
    *,
    layer_right: int | None = None,
    seed: int = 0,
    noise_scale: float = 0.0,
    mode: str = "clean",
) -> float:
    handles: list[Any] = []
    if layer_right is not None:
        layers = _decoder_layers(model)
        if layer_right < 0 or layer_right >= len(layers):
            raise ValueError(f"layer_right={layer_right} outside model layer count {len(layers)}")

        def hook(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
            if not inputs or not torch.is_tensor(inputs[0]):
                raise ValueError("HORN H2 hook expected first layer input to be a tensor")
            hidden = inputs[0]
            if mode == "identity":
                perturbed = hidden
            elif mode == "gaussian":
                generator = torch.Generator(device=hidden.device)
                generator.manual_seed(seed)
                std = hidden.detach().float().std(unbiased=False).to(hidden.device).to(hidden.dtype)
                noise = torch.randn(
                    hidden.shape,
                    dtype=hidden.dtype,
                    device=hidden.device,
                    generator=generator,
                ) * (std * noise_scale)
                perturbed = hidden + noise
            else:
                raise ValueError(f"unknown hook mode {mode!r}")
            return (perturbed, *inputs[1:])

        handles.append(layers[layer_right].register_forward_pre_hook(hook))
    try:
        with torch.no_grad():
            output = model(**tokenized)
        return _nll_from_logits(output.logits, tokenized["input_ids"])
    finally:
        for handle in handles:
            handle.remove()


def _write_packet(
    output_dir: Path,
    *,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluated = evaluate_horn_h2(rows)
    directional_rows = [row for row in rows if str(row.get("control_type")) == "directional_noise"]
    hook_off_rows = [row for row in rows if str(row.get("control_type")) == "hook_off"]
    flipped_rows = [row for row in rows if str(row.get("control_type")) == "flipped_direction_label"]
    expected_pairs = len(
        {
            (
                str(row.get("prompt_cluster_id") or row.get("prompt_id")),
                str(row.get("seed")),
                str(row.get("noise_side")),
            )
            for row in directional_rows
        }
    )
    demotion = (
        "DEMOTE_HORN_STANDALONE_WEAK_H2"
        if float(evaluated["directional_drift_ratio"]) < 1.2
        else "KEEP_HORN_ONLY_IF_FULL_H2_REPLAY_CLEARS"
    )
    summary = {
        "decision": evaluated["gate_status"],
        "gate_name": "horn_h2",
        "gate_status": evaluated["gate_status"],
        "gate_pass": evaluated["gate_pass"],
        "row_count": len(rows),
        "rows": rows,
        "claim_boundary": [
            "resource-limited HORN H2 scout",
            "not H2 promotion",
            "not GPU evidence",
            "not a precision-allocation claim",
        ],
        "source_h1a_packet_sha256": config["source_gate_packet_sha256"],
        "source_h1a_decision": config["source_h1a_decision"],
        "source_h1a_selected_ratio": config["source_h1a_selected_ratio"],
        "prompt_count": len({str(row["prompt_id"]) for row in rows}),
        "directional_noise_row_count": len(directional_rows),
        "hook_off_row_count": len(hook_off_rows),
        "flipped_direction_label_row_count": len(flipped_rows),
        "expected_paired_unit_count": expected_pairs,
        "pairing_complete": int(evaluated["paired_unit_count"]) == expected_pairs,
        "selected_direction_matches_h1": evaluated["selected_h2_direction"]
        == evaluated["selected_direction_from_h1"],
        "demotion_recommendation": demotion,
        **evaluated,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (output_dir / "raw_rows.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_dir / "summary.md").write_text(
        "# HORN H2 Noise Replay Scout\n\n"
        f"Decision: `{summary['decision']}`\n\n"
        "This is a resource-limited Mac-local scout. It cannot promote HORN.\n\n"
        f"- Directional drift ratio: `{summary['directional_drift_ratio']:.6g}`\n"
        f"- Paired lower bound: `{summary['directional_ratio_ci_low']:.6g}`\n"
        f"- Hook-off max delta: `{summary['hook_off_max_delta']:.6g}`\n"
        f"- Seeds: `{summary['seed_count']}`\n"
        f"- Demotion recommendation: `{summary['demotion_recommendation']}`\n",
        encoding="utf-8",
    )
    (output_dir / "decision.md").write_text(f"`{summary['decision']}`\n", encoding="utf-8")
    return summary


def run_scout(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    prompt_path: Path = DEFAULT_PROMPTS,
    template_path: Path = DEFAULT_TEMPLATE,
    h1_packet_dir: Path = DEFAULT_H1_PACKET,
    prereg_path: Path = DEFAULT_PREREG,
    model_id: str = DEFAULT_MODEL_ID,
    hf_home: Path = DEFAULT_HF_HOME,
    prompt_limit: int = 4,
    max_input_tokens: int = 8,
    seeds: tuple[int, ...] = (1, 2, 3),
    noise_scale: float = 0.05,
) -> dict[str, Any]:
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True, trust_remote_code=False)
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float16,
    )
    model.eval()
    load_seconds = time.perf_counter() - started

    h1_summary = json.loads((h1_packet_dir / "summary.json").read_text(encoding="utf-8"))
    selected_from_h1 = str(h1_summary["selected_h1_direction"])
    boundaries = _load_boundaries(template_path)
    prompts = _load_prompts(prompt_path, prompt_limit)

    rows: list[dict[str, Any]] = []
    forward_seconds = 0.0
    for prompt in prompts:
        tokenized = _tokenize(tokenizer, str(prompt["prompt"]), max_input_tokens)
        prompt_id = str(prompt["prompt_id"])
        cluster_id = f"{prompt.get('task', 'prompt')}:{prompt_id}"
        clean_started = time.perf_counter()
        clean_nll = _run_nll(model, tokenized)
        forward_seconds += time.perf_counter() - clean_started

        hook_off_by_direction: dict[str, float] = {}
        for direction, boundary in sorted(boundaries.items()):
            hook_started = time.perf_counter()
            hook_nll = _run_nll(
                model,
                tokenized,
                layer_right=int(boundary["layer_right"]),
                mode="identity",
            )
            forward_seconds += time.perf_counter() - hook_started
            hook_off_by_direction[direction] = abs(hook_nll - clean_nll)
            base_row = {
                "model_id": DEFAULT_CANONICAL_MODEL_ID,
                "prompt_id": prompt_id,
                "prompt_cluster_id": cluster_id,
                "selected_direction_from_h1": selected_from_h1,
                "boundary_direction": direction,
                "noise_side": "right_layer_input",
                "noise_std_basis": "gaussian_right_input_std_resource_limited",
                "noise_scale": noise_scale,
                "seed": 0,
                "clean_nll": clean_nll,
                "noisy_nll": hook_nll,
                "delta_nll": abs(hook_nll - clean_nll),
                "hook_off_delta": abs(hook_nll - clean_nll),
            }
            rows.append({**base_row, "control_type": "hook_off"})
            rows.append({**base_row, "control_type": "flipped_direction_label", "boundary_direction": _flip(direction)})

        for seed in seeds:
            for direction, boundary in sorted(boundaries.items()):
                noisy_started = time.perf_counter()
                noisy_nll = _run_nll(
                    model,
                    tokenized,
                    layer_right=int(boundary["layer_right"]),
                    seed=seed,
                    noise_scale=noise_scale,
                    mode="gaussian",
                )
                forward_seconds += time.perf_counter() - noisy_started
                rows.append(
                    {
                        "model_id": DEFAULT_CANONICAL_MODEL_ID,
                        "prompt_id": prompt_id,
                        "prompt_cluster_id": cluster_id,
                        "selected_direction_from_h1": selected_from_h1,
                        "boundary_direction": direction,
                        "noise_side": "right_layer_input",
                        "noise_std_basis": "gaussian_right_input_std_resource_limited",
                        "noise_scale": noise_scale,
                        "seed": seed,
                        "clean_nll": clean_nll,
                        "noisy_nll": noisy_nll,
                        "delta_nll": abs(noisy_nll - clean_nll),
                        "hook_off_delta": hook_off_by_direction[direction],
                        "control_type": "directional_noise",
                    }
                )

    config = {
        "gate_name": "horn_h2",
        "project": "horn",
        "source_gate_packet_sha256": f"sha256:{_packet_hash(h1_packet_dir)}",
        "source_h1a_decision": str(h1_summary["decision"]),
        "source_h1a_selected_ratio": float(h1_summary["selected_h1_ratio"]),
        "preregistration_sha256": f"sha256:{_sha256(prereg_path)}",
        "seed_list": list(seeds),
        "command": "python -m experimental.shared.horn_h2_noise_replay_scout",
        "model_id": DEFAULT_CANONICAL_MODEL_ID,
        "served_model_id": model_id,
        "model_revision": GRANITE_TINY_REVISION,
        "tokenizer_revision": GRANITE_TINY_REVISION,
        "prompt_source": str(prompt_path),
        "prompt_limit": prompt_limit,
        "max_input_tokens": max_input_tokens,
        "noise_scale": noise_scale,
        "load_seconds": load_seconds,
        "forward_seconds": forward_seconds,
        "resource_limit_note": (
            "Mac-local HORN H2 scout over short Granite Tiny prompts; useful for demotion "
            "or follow-up planning only, not a promotable H2 packet."
        ),
    }
    summary = _write_packet(output_dir, rows=rows, config=config)
    print(json.dumps({"output_dir": str(output_dir), **summary}, sort_keys=True))
    return summary


def _flip(direction: str) -> str:
    if direction == "attention->ssm":
        return "ssm->attention"
    if direction == "ssm->attention":
        return "attention->ssm"
    return direction


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return tuple(dict.fromkeys(seeds))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--template-path", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--h1-packet-dir", type=Path, default=DEFAULT_H1_PACKET)
    parser.add_argument("--prereg-path", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--prompt-limit", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=8)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--noise-scale", type=float, default=0.05)
    args = parser.parse_args()
    run_scout(
        output_dir=args.output_dir,
        prompt_path=args.prompt_path,
        template_path=args.template_path,
        h1_packet_dir=args.h1_packet_dir,
        prereg_path=args.prereg_path,
        model_id=args.model_id,
        hf_home=args.hf_home,
        prompt_limit=args.prompt_limit,
        max_input_tokens=args.max_input_tokens,
        seeds=_parse_seeds(args.seeds),
        noise_scale=args.noise_scale,
    )


if __name__ == "__main__":
    main()
