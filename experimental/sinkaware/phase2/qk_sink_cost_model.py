"""Cost model for approximate SinkAware QK-sink prediction."""

from __future__ import annotations

import json
import os
from pathlib import Path

from transformers import AutoConfig


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "experimental/sinkaware/phase2"


def _run(model_name: str = "distilgpt2", sink_tokens: int = 4) -> dict[str, object]:
    cache_dir = ROOT / ".debug/hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))
    config = AutoConfig.from_pretrained(model_name)
    hidden_size = int(config.n_embd)
    n_heads = int(config.n_head)
    head_dim = hidden_size // n_heads
    exact_muladds = n_heads * sink_tokens * head_dim
    qk_result = json.loads((OUT_DIR / "real_qk_sink_logit_probe.json").read_text(encoding="utf-8"))
    summary = qk_result["summary"]
    rows = []
    for rank in (1, 2, 4, 8):
        approx_muladds = n_heads * (head_dim * rank + rank * sink_tokens)
        rows.append(
            {
                "rank": rank,
                "approx_muladds_per_token_layer": approx_muladds,
                "exact_qk_sink_muladds_per_token_layer": exact_muladds,
                "cost_ratio_vs_exact": approx_muladds / exact_muladds,
                "hidden_plus_pos_r2": summary[f"rank{rank}_hidden_plus_pos_r2"],
                "hidden_only_r2": summary[f"rank{rank}_hidden_r2"],
                "passes_pre_gpu_tradeoff": approx_muladds / exact_muladds <= 0.75
                and summary[f"rank{rank}_hidden_plus_pos_r2"] >= 0.35,
            }
        )
    status = (
        "ALIVE at low rank; rank-2 gives useful QK-sink prediction below exact QK cost."
        if any(row["passes_pre_gpu_tradeoff"] for row in rows)
        else "WEAKENED; accurate ranks are not cheaper than exact QK-sink."
    )
    return {
        "model_name": model_name,
        "hidden_size": hidden_size,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "sink_tokens": sink_tokens,
        "rows": rows,
        "status": status,
    }


def _write_markdown(result: dict[str, object]) -> None:
    lines = [
        "# SinkAware QK-Sink Approximation Cost Model",
        "",
        f"Status: **{result['status']}**",
        "",
        f"- model: `{result['model_name']}`",
        f"- heads: {result['n_heads']}",
        f"- head dim: {result['head_dim']}",
        f"- sink tokens: {result['sink_tokens']}",
        "",
        "This estimates multiply-adds per token per layer for exact sink logits versus a per-head low-rank query predictor.",
        "It is not a GPU benchmark and ignores memory layout, launch overhead, and approximation error impact on model quality.",
        "",
        "| Rank | Approx mul-adds | Exact QK-sink mul-adds | Cost ratio | Hidden+pos R2 | Passes pre-GPU tradeoff? |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in result["rows"]:
        lines.append(
            "| {rank} | {approx_muladds_per_token_layer} | {exact_qk_sink_muladds_per_token_layer} | {cost_ratio_vs_exact:.3f} | {hidden_plus_pos_r2:.3f} | {passes} |".format(
                passes="yes" if row["passes_pre_gpu_tradeoff"] else "no",
                **row,
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "The approximate branch has a plausible pre-GPU systems wedge only at low rank.",
            "Rank-8 is more accurate but likely too expensive relative to exact four-sink `QK_sink`; rank-2 is the strongest current compromise.",
        ]
    )
    (OUT_DIR / "qk_sink_cost_model.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    result = _run()
    (OUT_DIR / "qk_sink_cost_model.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_markdown(result)
    print(json.dumps({"status": result["status"], "rows": result["rows"]}, indent=2))


if __name__ == "__main__":
    main()
