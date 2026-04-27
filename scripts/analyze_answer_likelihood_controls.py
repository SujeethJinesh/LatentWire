from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / float(len(values))


def _summary(records: list[dict[str, Any]], score_field: str) -> dict[str, Any]:
    scores = [
        float(record[score_field])
        for record in records
        if record.get(score_field) is not None and math.isfinite(float(record[score_field]))
    ]
    return {
        "n": len(records),
        "finite_scores": len(scores),
        "correct": sum(1 for record in records if bool(record.get("correct"))),
        "mean_score": _mean(scores),
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
    }


def analyze(
    *,
    live_records: list[dict[str, Any]],
    controls: dict[str, list[dict[str, Any]]],
    unavailable_controls: list[str],
    score_field: str,
    min_mean_delta: float,
    min_best_control_wins: int,
) -> dict[str, Any]:
    live_by_id = {str(record["example_id"]): record for record in live_records}
    control_by_name = {
        name: {str(record["example_id"]): record for record in records}
        for name, records in controls.items()
    }
    live_summary = _summary(live_records, score_field)
    control_summaries = {
        name: _summary(records, score_field)
        for name, records in controls.items()
    }
    checks: list[dict[str, Any]] = []
    n = len(live_records)
    all_counts_match = all(summary["n"] == n for summary in control_summaries.values())
    checks.append(
        {
            "name": "row_count_parity",
            "pass": bool(all_counts_match),
            "detail": {"live_n": n, **{name: summary["n"] for name, summary in control_summaries.items()}},
        }
    )
    all_finite = live_summary["finite_scores"] == n and all(
        summary["finite_scores"] == n for summary in control_summaries.values()
    )
    checks.append(
        {
            "name": "finite_scores",
            "pass": bool(all_finite),
            "detail": {
                "live": live_summary["finite_scores"],
                **{name: summary["finite_scores"] for name, summary in control_summaries.items()},
            },
        }
    )

    paired_controls: dict[str, Any] = {}
    mean_control_pass = True
    for name, by_id in control_by_name.items():
        deltas: list[float] = []
        wins = losses = ties = 0
        for example_id, live in live_by_id.items():
            control = by_id.get(example_id)
            if control is None or live.get(score_field) is None or control.get(score_field) is None:
                continue
            delta = float(live[score_field]) - float(control[score_field])
            deltas.append(delta)
            if delta > 1e-9:
                wins += 1
            elif delta < -1e-9:
                losses += 1
            else:
                ties += 1
        mean_delta = _mean(deltas)
        pass_mean = mean_delta is not None and mean_delta >= min_mean_delta
        mean_control_pass = mean_control_pass and pass_mean
        paired_controls[name] = {
            "mean_delta_live_minus_control": mean_delta,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "deltas": deltas,
            "pass_min_mean_delta": pass_mean,
        }

    checks.append(
        {
            "name": "mean_delta_vs_each_control",
            "pass": bool(mean_control_pass),
            "detail": {
                name: row["mean_delta_live_minus_control"]
                for name, row in paired_controls.items()
            },
        }
    )

    best_control_wins = 0
    best_control_losses = 0
    best_control_ties = 0
    best_control_deltas: list[float] = []
    for example_id, live in live_by_id.items():
        if live.get(score_field) is None:
            continue
        control_scores = []
        for by_id in control_by_name.values():
            control = by_id.get(example_id)
            if control is not None and control.get(score_field) is not None:
                control_scores.append(float(control[score_field]))
        if not control_scores:
            continue
        delta = float(live[score_field]) - max(control_scores)
        best_control_deltas.append(delta)
        if delta > 1e-9:
            best_control_wins += 1
        elif delta < -1e-9:
            best_control_losses += 1
        else:
            best_control_ties += 1
    best_control_pass = best_control_wins >= min_best_control_wins
    checks.append(
        {
            "name": "per_example_best_control_wins",
            "pass": bool(best_control_pass),
            "detail": {
                "wins": best_control_wins,
                "losses": best_control_losses,
                "ties": best_control_ties,
                "required_wins": min_best_control_wins,
            },
        }
    )

    status = "answer_likelihood_controls_pass" if all(check["pass"] for check in checks) else "answer_likelihood_controls_fail"
    return {
        "gate": {
            "status": status,
            "checks": checks,
            "score_field": score_field,
            "min_mean_delta": min_mean_delta,
            "min_best_control_wins": min_best_control_wins,
            "unavailable_controls": unavailable_controls,
        },
        "live_summary": live_summary,
        "control_summaries": control_summaries,
        "paired_controls": paired_controls,
        "best_control": {
            "wins": best_control_wins,
            "losses": best_control_losses,
            "ties": best_control_ties,
            "deltas": best_control_deltas,
            "mean_delta_live_minus_best_control": _mean(best_control_deltas),
        },
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Answer-Likelihood Control Analysis",
        "",
        f"- status: `{payload['gate']['status']}`",
        f"- score field: `{payload['gate']['score_field']}`",
        f"- unavailable controls: `{', '.join(payload['gate']['unavailable_controls']) or '-'}`",
        "",
        "## Summaries",
        "",
        "| row | n | finite | correct | mean score |",
        "|---|---:|---:|---:|---:|",
    ]
    live = payload["live_summary"]
    lines.append(
        f"| live | {live['n']} | {live['finite_scores']} | {live['correct']} | {live['mean_score']:.6f} |"
    )
    for name, summary in payload["control_summaries"].items():
        mean_score = summary["mean_score"]
        mean_text = "-" if mean_score is None else f"{mean_score:.6f}"
        lines.append(
            f"| {name} | {summary['n']} | {summary['finite_scores']} | {summary['correct']} | {mean_text} |"
        )
    lines.extend(["", "## Paired Controls", "", "| control | mean live-control | wins | losses | ties | pass |", "|---|---:|---:|---:|---:|---|"])
    for name, row in payload["paired_controls"].items():
        mean_delta = row["mean_delta_live_minus_control"]
        mean_text = "-" if mean_delta is None else f"{mean_delta:.6f}"
        lines.append(
            f"| {name} | {mean_text} | {row['wins']} | {row['losses']} | {row['ties']} | {row['pass_min_mean_delta']} |"
        )
    best = payload["best_control"]
    lines.extend(
        [
            "",
            "## Best Control",
            "",
            f"- wins/losses/ties: `{best['wins']}/{best['losses']}/{best['ties']}`",
            f"- mean live-best-control delta: `{best['mean_delta_live_minus_best_control']:.6f}`",
            "",
            "## Checks",
            "",
        ]
    )
    for check in payload["gate"]["checks"]:
        lines.append(f"- `{check['name']}`: `{check['pass']}`")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", required=True)
    parser.add_argument("--control", action="append", default=[], help="name=path JSONL control")
    parser.add_argument("--unavailable-control", action="append", default=[])
    parser.add_argument("--score-field", default="answer_mean_logprob")
    parser.add_argument("--min-mean-delta", type=float, default=0.05)
    parser.add_argument("--min-best-control-wins", type=int, default=3)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    controls: dict[str, list[dict[str, Any]]] = {}
    for spec in args.control:
        if "=" not in spec:
            raise ValueError(f"--control must be name=path, got {spec!r}")
        name, raw_path = spec.split("=", 1)
        controls[name] = _load_jsonl(Path(raw_path))
    payload = analyze(
        live_records=_load_jsonl(Path(args.live)),
        controls=controls,
        unavailable_controls=list(args.unavailable_control),
        score_field=args.score_field,
        min_mean_delta=args.min_mean_delta,
        min_best_control_wins=args.min_best_control_wins,
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_markdown(payload, Path(args.output_md))


if __name__ == "__main__":
    main()
