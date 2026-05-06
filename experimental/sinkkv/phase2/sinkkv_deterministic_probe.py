"""Run the deterministic no-download SinkKV policy probe.

The probe validates byte accounting and output-drift metrics before any real
model activation dump is used. It is synthetic-only and cannot promote SinkKV to
GPU work by itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experimental.shared.sensitivity_metrics import rel_l2
from experimental.sinkkv.phase2.reference.sinkkv_policy import (
    budget_matched_protected_kv,
    uniform_mxfp4_kv,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "experimental/sinkkv/phase2/results/sinkkv_deterministic_probe"


def _make_sink_heavy_tensors(
    *,
    seed: int,
    batch: int = 1,
    heads: int = 4,
    seq_len: int = 96,
    head_dim: int = 32,
    sink_count: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    sink_direction = torch.randn(heads, head_dim, generator=generator)
    sink_direction = sink_direction / sink_direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    key = 0.35 * torch.randn(batch, heads, seq_len, head_dim, generator=generator)
    value = 0.45 * torch.randn(batch, heads, seq_len, head_dim, generator=generator)
    query = 0.35 * torch.randn(batch, heads, seq_len, head_dim, generator=generator)

    key[:, :, :sink_count, :] = 2.5 * sink_direction[None, :, None, :] + 0.08 * torch.randn(
        batch, heads, sink_count, head_dim, generator=generator
    )
    value[:, :, :sink_count, :] = 2.0 * torch.randn(batch, heads, sink_count, head_dim, generator=generator)
    query = query + 1.8 * sink_direction[None, :, None, :]

    return query.float(), key.float(), value.float()


def _attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = query.shape[-1] ** -0.5
    scores = torch.matmul(query, key.transpose(-1, -2)) * scale
    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, value)
    return output, weights


def _row(
    *,
    name: str,
    key: torch.Tensor,
    value: torch.Tensor,
    query: torch.Tensor,
    reference_output: torch.Tensor,
    reference_weights: torch.Tensor,
    budget_bits_per_element: float,
    protected_positions: tuple[int, ...],
) -> dict[str, object]:
    output, weights = _attention(query, key, value)
    return {
        "row": name,
        "budget_bits_per_element": budget_bits_per_element,
        "protected_positions": list(protected_positions),
        "attention_output_rel_l2": rel_l2(reference_output, output),
        "softmax_attention_l1": torch.mean(torch.abs(reference_weights - weights)).item(),
        "sink_mass": weights[..., :4].sum(dim=-1).mean().item(),
    }


def run_probe(*, seed: int = 20260506, output_dir: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    query, key, value = _make_sink_heavy_tensors(seed=seed)
    reference_output, reference_weights = _attention(query, key, value)

    uniform = uniform_mxfp4_kv(key, value)
    sink = budget_matched_protected_kv(key, value, protected_positions=(0, 1, 2, 3))
    recent = budget_matched_protected_kv(key, value, protected_positions=(92, 93, 94, 95))

    rows = [
        _row(
            name="full_precision_kv",
            key=key,
            value=value,
            query=query,
            reference_output=reference_output,
            reference_weights=reference_weights,
            budget_bits_per_element=16.0,
            protected_positions=(),
        ),
        _row(
            name=uniform.policy_name,
            key=uniform.key,
            value=uniform.value,
            query=query,
            reference_output=reference_output,
            reference_weights=reference_weights,
            budget_bits_per_element=uniform.budget_bits_per_element,
            protected_positions=uniform.protected_positions,
        ),
        _row(
            name="sink_" + sink.policy_name,
            key=sink.key,
            value=sink.value,
            query=query,
            reference_output=reference_output,
            reference_weights=reference_weights,
            budget_bits_per_element=sink.budget_bits_per_element,
            protected_positions=sink.protected_positions,
        ),
        _row(
            name="recent_" + recent.policy_name,
            key=recent.key,
            value=recent.value,
            query=query,
            reference_output=reference_output,
            reference_weights=reference_weights,
            budget_bits_per_element=recent.budget_bits_per_element,
            protected_positions=recent.protected_positions,
        ),
    ]

    row_by_name = {row["row"]: row for row in rows}
    uniform_rel_l2 = float(row_by_name["uniform_mxfp4_kv"]["attention_output_rel_l2"])
    sink_rel_l2 = float(row_by_name["sink_protected_budget_matched_kv"]["attention_output_rel_l2"])
    recent_rel_l2 = float(row_by_name["recent_protected_budget_matched_kv"]["attention_output_rel_l2"])
    sink_vs_uniform_recovery = 1.0 - (sink_rel_l2 / uniform_rel_l2)
    recent_vs_uniform_recovery = 1.0 - (recent_rel_l2 / uniform_rel_l2)

    decision = "SYNTHETIC_VALIDATION_ONLY"
    if sink_vs_uniform_recovery >= 0.10 and sink_rel_l2 < recent_rel_l2:
        decision = "SYNTHETIC_PASS_REAL_DUMPS_NEXT"
    elif sink_rel_l2 >= uniform_rel_l2:
        decision = "SYNTHETIC_FAIL_KILL_IF_REAL_DUMPS_MATCH"

    summary = {
        "seed": seed,
        "surface": "deterministic_sink_heavy_synthetic",
        "decision": decision,
        "rows": rows,
        "sink_vs_uniform_recovery": sink_vs_uniform_recovery,
        "recent_vs_uniform_recovery": recent_vs_uniform_recovery,
        "claim_boundary": [
            "synthetic-only validation",
            "not GPU speed",
            "not benchmark accuracy",
            "does not skip QK_sink",
        ],
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (output_dir / "raw_rows.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    markdown_rows = "\n".join(
        f"| {row['row']} | {row['budget_bits_per_element']:.3f} | {row['attention_output_rel_l2']:.6f} | "
        f"{row['softmax_attention_l1']:.6f} | {row['sink_mass']:.6f} |"
        for row in rows
    )
    (output_dir / "summary.md").write_text(
        "# SinkKV Deterministic Probe\n\n"
        f"- seed: `{seed}`\n"
        f"- decision: `{decision}`\n"
        f"- sink-vs-uniform recovery: `{sink_vs_uniform_recovery:.3f}`\n"
        f"- recent-vs-uniform recovery: `{recent_vs_uniform_recovery:.3f}`\n\n"
        "This is a synthetic-only validation. It is not GPU speed evidence, not benchmark accuracy, "
        "and it does not skip QK_sink.\n\n"
        "| row | bits/element | output rel-L2 | softmax L1 | sink mass |\n"
        "|---|---:|---:|---:|---:|\n"
        f"{markdown_rows}\n"
    )
    (output_dir / "decision.md").write_text(
        "# Decision\n\n"
        f"`{decision}`\n\n"
        "A synthetic pass only authorizes the first real activation dump. It does not authorize GPU work.\n"
    )
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "batch": 1,
                "heads": 4,
                "seq_len": 96,
                "head_dim": 32,
                "sink_count": 4,
                "uniform_budget_bits_per_element": 4.0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_probe(seed=args.seed, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
