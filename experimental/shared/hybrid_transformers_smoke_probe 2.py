"""Resource-limited local execution smoke probe for active hybrid models.

The probe checks whether a cached hybrid model can run through native
`transformers` and expose recurrent/attention cache tensors. It is intentionally
small and non-promoting: one short prompt, no quality claim, no GPU claim, and no
SSQ-LR/HORN/HBSM gate promotion.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ID = "ibm-granite/granite-4.0-h-tiny"
DEFAULT_HF_HOME = ROOT / ".debug/hf_home"
DEFAULT_PROMPTS = ROOT / "experimental/shared/prompts/hybrid_reasoning_smoke_12_20260506.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "experimental/shared/results/hybrid_transformers_smoke_probe_20260507"


def _load_first_prompt(prompt_path: Path) -> dict[str, Any]:
    with prompt_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict) or "prompt" not in payload:
                    raise ValueError(f"invalid prompt row in {prompt_path}")
                return payload
    raise ValueError(f"no prompts found in {prompt_path}")


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def summarize_cache(cache: Any, *, max_layers: int = 12) -> dict[str, Any]:
    layers = list(getattr(cache, "layers", []))
    rows: list[dict[str, Any]] = []
    recurrent_count = 0
    attention_count = 0
    for layer_index, layer in enumerate(layers):
        fields: dict[str, list[int]] = {}
        for field_name in ("recurrent_states", "ssm_states", "conv_states", "keys", "values"):
            value = getattr(layer, field_name, None)
            shape = _shape(value)
            if shape is not None:
                fields[field_name] = shape
        if fields:
            if "recurrent_states" in fields or "ssm_states" in fields:
                recurrent_count += 1
            if "keys" in fields or "values" in fields:
                attention_count += 1
            if len(rows) < max_layers:
                rows.append({"layer": layer_index, "fields": fields})
    return {
        "cache_type": type(cache).__name__,
        "cache_layer_count": len(layers),
        "recurrent_state_layer_count": recurrent_count,
        "attention_cache_layer_count": attention_count,
        "sampled_layers": rows,
    }


def run_probe(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    hf_home: Path = DEFAULT_HF_HOME,
    prompt_path: Path = DEFAULT_PROMPTS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_input_tokens: int = 16,
    device_map: str = "cpu",
    dtype: str = "float16",
) -> dict[str, Any]:
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
    output_dir.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    prompt = _load_first_prompt(prompt_path)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True, trust_remote_code=False)
    tokenized = tokenizer(prompt["prompt"], return_tensors="pt")
    if tokenized["input_ids"].shape[1] > max_input_tokens:
        tokenized = {key: value[:, :max_input_tokens] for key, value in tokenized.items()}
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=True,
        trust_remote_code=False,
        dtype=getattr(torch, dtype),
        low_cpu_mem_usage=True,
        device_map=device_map,
    )
    model.eval()
    load_seconds = time.perf_counter() - load_started
    forward_started = time.perf_counter()
    with torch.no_grad():
        output = model(**tokenized, use_cache=True)
    forward_seconds = time.perf_counter() - forward_started
    cache_summary = summarize_cache(output.past_key_values)
    summary = {
        "surface": "hybrid_transformers_smoke_probe",
        "decision": "RESOURCE_LIMITED_EXECUTION_SMOKE_NOT_PROMOTABLE",
        "model_id": model_id,
        "prompt_id": prompt.get("prompt_id"),
        "input_tokens": int(tokenized["input_ids"].shape[1]),
        "dtype": dtype,
        "device_map": device_map,
        "load_seconds": load_seconds,
        "forward_seconds": forward_seconds,
        "rss_gb_after_forward": process.memory_info().rss / (1024**3),
        "logits_shape": [int(dim) for dim in output.logits.shape],
        **cache_summary,
        "claim_boundary": [
            "resource-limited execution smoke",
            "not SSQ-LR/HORN/HBSM gate evidence",
            "not GPU evidence",
            "not quality evidence",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_dir / "summary.md").write_text(_summary_markdown(summary))
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "hf_home": str(hf_home),
                "prompt_path": str(prompt_path),
                "max_input_tokens": max_input_tokens,
                "device_map": device_map,
                "dtype": dtype,
                "command": "python -m experimental.shared.hybrid_transformers_smoke_probe",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (output_dir / "decision.md").write_text(
        "# Hybrid Transformers Smoke Probe Decision\n\n"
        "`RESOURCE_LIMITED_EXECUTION_SMOKE_NOT_PROMOTABLE`\n\n"
        "This proves only that the cached model can execute a tiny local forward and expose cache tensors.\n"
    )
    print(json.dumps({"decision": summary["decision"], "output_dir": str(output_dir)}, sort_keys=True))
    return summary


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Hybrid Transformers Smoke Probe",
        "",
        f"Decision: `{summary['decision']}`",
        "",
        "This is resource-limited execution evidence only. It cannot promote SSQ-LR, HORN, or HBSM.",
        "",
        f"- Model: `{summary['model_id']}`",
        f"- Prompt: `{summary['prompt_id']}`",
        f"- Input tokens: `{summary['input_tokens']}`",
        f"- Load seconds: `{summary['load_seconds']:.2f}`",
        f"- Forward seconds: `{summary['forward_seconds']:.2f}`",
        f"- Recurrent-state layers: `{summary['recurrent_state_layer_count']}`",
        f"- Attention-cache layers: `{summary['attention_cache_layer_count']}`",
        "",
        "| Layer | Fields |",
        "|---:|---|",
    ]
    for row in summary["sampled_layers"]:
        fields = ", ".join(f"{name}={shape}" for name, shape in row["fields"].items())
        lines.append(f"| {row['layer']} | `{fields}` |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-input-tokens", type=int, default=16)
    parser.add_argument("--device-map", default="cpu")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()
    run_probe(
        model_id=args.model_id,
        hf_home=args.hf_home,
        prompt_path=args.prompt_path,
        output_dir=args.output_dir,
        max_input_tokens=args.max_input_tokens,
        device_map=args.device_map,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
