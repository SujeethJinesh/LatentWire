"""Mac-local real-query probe for approximate SinkAware revival.

This uses a small Hugging Face causal LM on saved LatentWire generation text and
tests whether sink attention mass is predictable from hidden-query features.
It is not a kernel benchmark and does not revive the exact static-prior branch.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "experimental/sinkaware/phase2"
DEFAULT_TRACES = [
    ROOT / "results/surface_scout_qwen25math_instruct_qwen3_svamp32_20260426/source_alone.jsonl",
    ROOT / "results/surface_scout_qwen25math_instruct_qwen3_svamp32_20260426/text_to_text.jsonl",
    ROOT / "results/prompt_control_20260419/qwen_gsm10_target_alone_chat_thinking_false.jsonl",
]


def _r2(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot.clamp_min(1e-12))


def _linear_r2(train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor, test_y: torch.Tensor) -> float:
    train_x = torch.cat([train_x, torch.ones(train_x.shape[0], 1)], dim=1)
    test_x = torch.cat([test_x, torch.ones(test_x.shape[0], 1)], dim=1)
    ridge = 1e-3 * torch.eye(train_x.shape[1])
    weights = torch.linalg.solve(train_x.T @ train_x + ridge, train_x.T @ train_y[:, None])
    return _r2(test_y, (test_x @ weights).squeeze(-1))


def _rank_features(train_x: torch.Tensor, test_x: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    mean_x = train_x.mean(dim=0, keepdim=True)
    centered_train = train_x - mean_x
    centered_test = test_x - mean_x
    _, _, vh = torch.linalg.svd(centered_train, full_matrices=False)
    basis = vh[: min(rank, vh.shape[0])].T
    return centered_train @ basis, centered_test @ basis


def _load_texts(paths: list[Path], max_traces: int) -> list[str]:
    texts: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                text = str(row.get("prediction") or row.get("text") or "")
                if len(text.split()) >= 8:
                    texts.append(text)
                if len(texts) >= max_traces:
                    return texts
    return texts


def _extract_samples(
    model_name: str,
    texts: list[str],
    max_length: int,
    sink_tokens: int,
) -> list[dict[str, object]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
    model.eval()
    samples: list[dict[str, object]] = []
    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            if encoded["input_ids"].shape[1] <= sink_tokens + 2:
                continue
            outputs = model(**encoded, output_attentions=True, output_hidden_states=True)
            seq_len = encoded["input_ids"].shape[1]
            pos = torch.arange(seq_len, dtype=torch.float32)
            query_positions = torch.arange(sink_tokens, seq_len)
            denom = max(1, seq_len - 1)
            for layer_idx, attention in enumerate(outputs.attentions):
                hidden = outputs.hidden_states[layer_idx][0, query_positions].detach().float()
                sink_mass = attention[0, :, query_positions, :sink_tokens].sum(dim=-1).mean(dim=0).detach().float()
                norm_pos = (pos[query_positions] / denom).unsqueeze(1)
                for h, y, p in zip(hidden, sink_mass, norm_pos):
                    samples.append({"layer": layer_idx, "hidden": h, "position": p, "target": y})
    return samples


def _evaluate(samples: list[dict[str, object]], ranks: tuple[int, ...]) -> dict[str, object]:
    rows = []
    for layer in sorted({int(sample["layer"]) for sample in samples}):
        layer_samples = [sample for sample in samples if int(sample["layer"]) == layer]
        n = len(layer_samples)
        split = int(0.67 * n)
        hidden = torch.stack([sample["hidden"] for sample in layer_samples])
        position = torch.stack([sample["position"] for sample in layer_samples])
        target = torch.stack([sample["target"] for sample in layer_samples]).float()
        train_h, test_h = hidden[:split], hidden[split:]
        train_p, test_p = position[:split], position[split:]
        train_y, test_y = target[:split], target[split:]
        static_pred = train_y.mean().expand_as(test_y)
        row: dict[str, object] = {
            "layer": layer,
            "n_samples": n,
            "static_r2": _r2(test_y, static_pred),
            "position_r2": _linear_r2(train_p, train_y, test_p, test_y),
        }
        for rank in ranks:
            train_rank, test_rank = _rank_features(train_h, test_h, rank)
            row[f"rank{rank}_hidden_r2"] = _linear_r2(train_rank, train_y, test_rank, test_y)
            row[f"rank{rank}_hidden_plus_pos_r2"] = _linear_r2(
                torch.cat([train_rank, train_p], dim=1),
                train_y,
                torch.cat([test_rank, test_p], dim=1),
                test_y,
            )
        rows.append(row)
    metrics = [key for key in rows[0] if key.endswith("_r2")]
    summary = {metric: mean(float(row[metric]) for row in rows) for metric in metrics}
    return {"rows": rows, "summary": summary, "status": _status(summary)}


def _status(summary: dict[str, object]) -> str:
    best_hidden = max(value for key, value in summary.items() if key.endswith("hidden_r2"))
    best_hidden_pos = max(value for key, value in summary.items() if key.endswith("hidden_plus_pos_r2"))
    position = float(summary["position_r2"])
    if best_hidden_pos >= position + 0.05 and best_hidden_pos > 0.25:
        return "ALIVE as approximate real-query predictor; next gate is real Q/K or kernel-side profiling."
    if best_hidden > 0.10:
        return "WEAKLY ALIVE; hidden-query signal exists but does not beat simple position strongly enough."
    return "WEAKENED; real-query proxy does not justify approximate SinkAware before better tensors."


def _write_markdown(result: dict[str, object]) -> None:
    lines = [
        "# SinkAware Real-Query Approximation Probe",
        "",
        f"Status: **{result['status']}**",
        "",
        f"- model: `{result['model_name']}`",
        f"- traces: {result['n_traces']}",
        f"- token samples: {result['n_samples']}",
        f"- sink tokens: {result['sink_tokens']}",
        "",
        "This probes real model attention sink mass from saved LatentWire text traces.",
        "It is not a kernel benchmark, and it does not revive exact static sink reuse.",
        "",
        "## Mean Across Layers",
        "",
        "| Model | Static R2 | Position R2 | Rank-1 hidden R2 | Rank-2 hidden R2 | Rank-4 hidden R2 | Rank-8 hidden R2 | Best hidden+pos R2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    summary = result["summary"]
    lines.append(
        "| {model_name} | {static_r2:.3f} | {position_r2:.3f} | {rank1_hidden_r2:.3f} | {rank2_hidden_r2:.3f} | {rank4_hidden_r2:.3f} | {rank8_hidden_r2:.3f} | {best_hidden_pos:.3f} |".format(
            model_name=result["model_name"],
            best_hidden_pos=max(value for key, value in summary.items() if key.endswith("hidden_plus_pos_r2")),
            **summary,
        )
    )
    lines.extend(
        [
            "",
            "## Per Layer",
            "",
            "| Layer | Samples | Static R2 | Position R2 | Rank-8 hidden R2 | Rank-8 hidden+pos R2 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["rows"]:
        lines.append(
            "| {layer} | {n_samples} | {static_r2:.3f} | {position_r2:.3f} | {rank8_hidden_r2:.3f} | {rank8_hidden_plus_pos_r2:.3f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "The only useful pre-GPU SinkAware path is approximate. A static prior remains killed.",
            "Advance only if real query features predict sink mass materially better than position-only structure.",
        ]
    )
    (OUT_DIR / "real_query_sink_probe.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="sshleifer/tiny-gpt2")
    parser.add_argument("--max-traces", type=int, default=48)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--input-jsonl", action="append", type=Path, default=[])
    args = parser.parse_args()

    cache_dir = ROOT / ".debug/hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))
    paths = args.input_jsonl or DEFAULT_TRACES
    texts = _load_texts(paths, args.max_traces)
    samples = _extract_samples(args.model_name, texts, max_length=args.max_length, sink_tokens=args.sink_tokens)
    result = _evaluate(samples, ranks=(1, 2, 4, 8))
    result.update(
        {
            "model_name": args.model_name,
            "n_traces": len(texts),
            "n_samples": len(samples),
            "sink_tokens": args.sink_tokens,
        }
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "real_query_sink_probe.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_markdown(result)
    print(json.dumps({"status": result["status"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
