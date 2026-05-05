"""Mac-local ThoughtFlow Phase 2 retention simulation."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "experimental/thoughtflow_fp8/phase2"


@dataclass(frozen=True)
class Token:
    text: str
    label: str
    importance: float


def _make_trace(seed: int, n_steps: int = 9) -> list[Token]:
    rng = random.Random(seed)
    trace = [
        Token("<bos>", "anchor", 1.0),
        Token("problem", "anchor", 0.8),
        Token("context", "anchor", 0.7),
        Token("question", "anchor", 0.7),
    ]
    for step in range(n_steps):
        if step in {0, 3, 6}:
            trace.append(Token(f"phase_{step}", "phase", 0.35 + 0.05 * rng.random()))
        trace.extend(
            [
                Token(f"reason_{step}_a", "reason", 0.45 + 0.25 * rng.random()),
                Token(f"value_{step}", "math_state", 0.55 + 0.35 * rng.random()),
                Token(f"reason_{step}_b", "reason", 0.35 + 0.25 * rng.random()),
            ]
        )
        if step in {2, 5, 8}:
            trace.append(Token(f"check_{step}", "phase", 0.30 + 0.10 * rng.random()))
    trace.append(Token("answer", "phase", 0.9))
    return trace


def _topk(indices: list[int], scores: list[float], k: int) -> set[int]:
    return set(sorted(indices, key=lambda idx: (-scores[idx], idx))[:k])


def longflow_like(trace: list[Token], budget: int) -> set[int]:
    scores = [token.importance for token in trace]
    return _topk(list(range(len(trace))), scores, budget)


def thin_kv_like(trace: list[Token], budget: int) -> set[int]:
    recent = set(range(max(0, len(trace) - max(2, budget // 4)), len(trace)))
    remaining = max(0, budget - len(recent))
    scores = [token.importance + (0.1 if token.label == "math_state" else 0.0) for token in trace]
    return recent | _topk([idx for idx in range(len(trace)) if idx not in recent], scores, remaining)


def rkv_like(trace: list[Token], budget: int) -> set[int]:
    anchors = set(range(min(4, len(trace))))
    recent_count = max(0, budget - len(anchors))
    recent = set(range(max(0, len(trace) - recent_count), len(trace)))
    return anchors | recent


def thoughtflow(trace: list[Token], budget: int) -> set[int]:
    protected = {idx for idx, token in enumerate(trace) if token.label in {"anchor", "phase"}}
    if len(protected) >= budget:
        scores = [token.importance for token in trace]
        return _topk(sorted(protected), scores, budget)
    remaining = budget - len(protected)
    scores = [token.importance for token in trace]
    filler = _topk([idx for idx in range(len(trace)) if idx not in protected], scores, remaining)
    return protected | filler


def _recall(trace: list[Token], kept: set[int], label: str) -> float:
    indices = [idx for idx, token in enumerate(trace) if token.label == label]
    if not indices:
        return 1.0
    return sum(1 for idx in indices if idx in kept) / len(indices)


def _run() -> dict[str, object]:
    policies = {
        "longflow_like": longflow_like,
        "thin_kv_like": thin_kv_like,
        "rkv_like": rkv_like,
        "thoughtflow": thoughtflow,
    }
    rows = []
    for seed in range(24):
        trace = _make_trace(seed)
        budget = math.ceil(len(trace) * 0.35)
        for name, policy in policies.items():
            kept = policy(trace, budget)
            rows.append(
                {
                    "seed": seed,
                    "policy": name,
                    "tokens": len(trace),
                    "budget": budget,
                    "keep_rate": len(kept) / len(trace),
                    "anchor_recall": _recall(trace, kept, "anchor"),
                    "phase_recall": _recall(trace, kept, "phase"),
                    "math_state_recall": _recall(trace, kept, "math_state"),
                }
            )
    summary = {}
    for name in policies:
        policy_rows = [row for row in rows if row["policy"] == name]
        summary[name] = {
            metric: sum(float(row[metric]) for row in policy_rows) / len(policy_rows)
            for metric in ("keep_rate", "anchor_recall", "phase_recall", "math_state_recall")
        }
    return {"rows": rows, "summary": summary}


def _write_markdown(result: dict[str, object]) -> None:
    summary = result["summary"]
    lines = [
        "# ThoughtFlow-FP8 Phase 2 Retention Simulation",
        "",
        "Status: **ALIVE, but only as a synthetic Mac-local gate.** Real current-model traces are still required before a reviewer pack.",
        "",
        "| Policy | Keep rate | Anchor recall | Phase recall | Math-state recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy, metrics in summary.items():
        lines.append(
            "| {policy} | {keep_rate:.3f} | {anchor_recall:.3f} | {phase_recall:.3f} | {math_state_recall:.3f} |".format(
                policy=policy, **metrics
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "The anchor/phase policy preserves phase markers and anchors at the same keep budget where LongFlow-like, ThinKV-like, and R-KV-like proxies drop at least one protected class.",
            "This is not accuracy evidence and not a GPU systems result. The next gate is to rerun the same policy simulator on real cached/current-model reasoning traces.",
        ]
    )
    (OUT_DIR / "phase_eviction_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = _run()
    (OUT_DIR / "phase_eviction_analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_markdown(result)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
