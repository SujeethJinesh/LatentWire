#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass
from datetime import date
from math import comb
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import harness_common as harness
from scripts import run_gsm8k_smoke_contract as smoke


DEFAULT_BASELINE_RESULTS_DIR = "results/gsm8k_smoke_contract_20260421"


@dataclass(frozen=True)
class DiagnosticsConfig:
    candidate_prediction_output: str
    candidate_label: str
    baseline_results_dir: str = DEFAULT_BASELINE_RESULTS_DIR
    candidate_method: str = "rotalign_kv"
    results_dir: str | None = None
    source_output_name: str = "gsm8k32_source_alone.jsonl"
    output_tag: str | None = None


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _prompt_excerpt(prompt: str, max_chars: int = 96) -> str:
    clean = " ".join(str(prompt).split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."


def _records_by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record["example_id"]): record for record in records}


def _ordered_example_ids(records: list[dict[str, Any]]) -> list[str]:
    return [str(record["example_id"]) for record in records]


def _record_numeric_prediction(row: dict[str, Any]) -> str | None:
    return harness._extract_prediction_numeric_answer(str(row.get("prediction", "")))


def _record_reference_numeric(row: dict[str, Any]) -> str | None:
    answers = row.get("answer", [])
    if isinstance(answers, str):
        answers = [answers]
    for answer in answers:
        numeric = harness._extract_reference_numeric_answer(str(answer))
        if numeric is not None:
            return numeric
    return None


def _same_numeric_prediction(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_numeric = _record_numeric_prediction(left)
    right_numeric = _record_numeric_prediction(right)
    return left_numeric is not None and left_numeric == right_numeric


def _row_validity(
    *,
    label: str,
    records: list[dict[str, Any]],
    reference_ids: list[str],
) -> dict[str, Any]:
    ids = _ordered_example_ids(records)
    numeric_coverage = sum(
        int(harness._has_numeric_extraction(str(row.get("prediction", "")))) for row in records
    )
    return {
        "label": label,
        "row_count": len(records),
        "expected_count": len(reference_ids),
        "row_count_matches": len(records) == len(reference_ids),
        "ordered_id_parity": ids == reference_ids,
        "set_id_parity": set(ids) == set(reference_ids),
        "duplicate_ids": sorted({example_id for example_id in ids if ids.count(example_id) > 1}),
        "numeric_extraction_coverage": int(numeric_coverage),
        "numeric_extraction_coverage_rate": float(numeric_coverage / max(len(records), 1)),
        "empty_predictions": int(
            sum(int(not str(row.get("prediction", "")).strip()) for row in records)
        ),
    }


def _flip_matrix(
    *,
    candidate_records: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
) -> dict[str, int]:
    target_by_id = _records_by_id(target_records)
    counts = {
        "target_wrong_candidate_wrong": 0,
        "target_wrong_candidate_right": 0,
        "target_right_candidate_wrong": 0,
        "target_right_candidate_right": 0,
    }
    for candidate_row in candidate_records:
        target_row = target_by_id[str(candidate_row["example_id"])]
        target_correct = bool(target_row["correct"])
        candidate_correct = bool(candidate_row["correct"])
        if target_correct and candidate_correct:
            counts["target_right_candidate_right"] += 1
        elif target_correct and not candidate_correct:
            counts["target_right_candidate_wrong"] += 1
        elif not target_correct and candidate_correct:
            counts["target_wrong_candidate_right"] += 1
        else:
            counts["target_wrong_candidate_wrong"] += 1
    return counts


def _exact_sign_p_value(help_count: int, harm_count: int) -> float | None:
    n = int(help_count) + int(harm_count)
    if n == 0:
        return None
    k = min(int(help_count), int(harm_count))
    tail = sum(comb(n, i) for i in range(k + 1)) / float(2**n)
    return min(1.0, 2.0 * tail)


def _resolve_candidate_records(
    candidate_records: list[dict[str, Any]],
    *,
    method_name: str,
) -> list[dict[str, Any]]:
    grouped = harness.group_by_method(candidate_records)
    if method_name in grouped:
        return grouped[method_name]
    if len(grouped) == 1:
        return next(iter(grouped.values()))
    raise KeyError(f"Candidate method {method_name!r} not found in {sorted(grouped)}")


def _paired_counts(
    method_records: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]],
) -> dict[str, int]:
    return harness.paired_vs_baseline(method_records, baseline_records)


def _oracle_counts(
    left_records: list[dict[str, Any]],
    right_records: list[dict[str, Any]],
) -> dict[str, int]:
    right_by_id = _records_by_id(right_records)
    correct = 0
    left_only = 0
    right_only = 0
    both = 0
    neither = 0
    for left_row in left_records:
        example_id = str(left_row["example_id"])
        right_row = right_by_id[example_id]
        left_correct = bool(left_row["correct"])
        right_correct = bool(right_row["correct"])
        correct += int(left_correct or right_correct)
        if left_correct and right_correct:
            both += 1
        elif left_correct:
            left_only += 1
        elif right_correct:
            right_only += 1
        else:
            neither += 1
    return {
        "correct": correct,
        "left_only": left_only,
        "right_only": right_only,
        "both": both,
        "neither": neither,
    }


def _detail_rows(
    *,
    examples: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
    source_records: list[dict[str, Any]],
    text_records: list[dict[str, Any]],
    selector: str,
) -> list[dict[str, Any]]:
    target_by_id = _records_by_id(target_records)
    source_by_id = _records_by_id(source_records)
    text_by_id = _records_by_id(text_records)
    example_by_id = {
        str(example["example_id"]): example
        for example in examples
        if "example_id" in example
    }
    rows: list[dict[str, Any]] = []
    for idx, candidate_row in enumerate(candidate_records):
        example_id = str(candidate_row["example_id"])
        target_row = target_by_id[example_id]
        source_row = source_by_id[example_id]
        text_row = text_by_id[example_id]
        candidate_correct = bool(candidate_row["correct"])
        target_correct = bool(target_row["correct"])
        text_correct = bool(text_row["correct"])
        if selector == "candidate_only_wins" and not (candidate_correct and not target_correct):
            continue
        if selector == "candidate_only_losses" and not (target_correct and not candidate_correct):
            continue
        if selector == "text_only_losses" and not (target_correct and not text_correct):
            continue
        example = example_by_id.get(example_id, examples[idx])
        candidate_numeric = _record_numeric_prediction(candidate_row)
        target_numeric = _record_numeric_prediction(target_row)
        source_numeric = _record_numeric_prediction(source_row)
        text_numeric = _record_numeric_prediction(text_row)
        reference_numeric = _record_reference_numeric(candidate_row)
        rows.append(
            {
                "index": idx,
                "example_id": example_id,
                "prompt_excerpt": _prompt_excerpt(example["prompt"]),
                "reference_answer": example["answers"][0],
                "reference_numeric": reference_numeric,
                "candidate_correct": candidate_correct,
                "target_correct": target_correct,
                "source_correct": bool(source_row["correct"]),
                "text_correct": text_correct,
                "candidate_numeric_prediction": candidate_numeric,
                "target_numeric_prediction": target_numeric,
                "source_numeric_prediction": source_numeric,
                "text_numeric_prediction": text_numeric,
                "candidate_same_numeric_as_source": _same_numeric_prediction(candidate_row, source_row),
                "candidate_same_numeric_as_text": _same_numeric_prediction(candidate_row, text_row),
                "candidate_same_numeric_as_target": _same_numeric_prediction(candidate_row, target_row),
                "source_copy_risk": bool(candidate_correct and not target_correct and bool(source_row["correct"])),
                "latent_noncopy_help": bool(
                    candidate_correct
                    and not target_correct
                    and not bool(source_row["correct"])
                    and not text_correct
                ),
                "text_poison": bool(target_correct and not text_correct and not bool(source_row["correct"])),
                "candidate_prediction": str(candidate_row["prediction"]),
                "target_prediction": str(target_row["prediction"]),
                "source_prediction": str(source_row["prediction"]),
                "text_prediction": str(text_row["prediction"]),
            }
        )
    return rows


def _source_win_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    total = len(rows)
    source_correct = sum(int(row["source_correct"]) for row in rows)
    text_correct = sum(int(row["text_correct"]) for row in rows)
    return {
        "n": total,
        "source_correct": source_correct,
        "source_wrong": total - source_correct,
        "text_correct": text_correct,
        "text_wrong": total - text_correct,
        "candidate_same_numeric_as_source": sum(
            int(row.get("candidate_same_numeric_as_source", False)) for row in rows
        ),
        "candidate_same_numeric_as_text": sum(
            int(row.get("candidate_same_numeric_as_text", False)) for row in rows
        ),
        "latent_noncopy_help": sum(int(row.get("latent_noncopy_help", False)) for row in rows),
        "text_poison": sum(int(row.get("text_poison", False)) for row in rows),
    }


def _validity_passes(validity: list[dict[str, Any]]) -> bool:
    return all(
        bool(row["row_count_matches"])
        and bool(row["ordered_id_parity"])
        and int(row["empty_predictions"]) == 0
        and float(row["numeric_extraction_coverage_rate"]) >= 0.99
        for row in validity
    )


def _diagnostic_gate(
    *,
    validity: list[dict[str, Any]],
    candidate_pairs: dict[str, int],
    win_support: dict[str, int],
    oracle: dict[str, int],
    candidate_correct: int,
) -> dict[str, Any]:
    wins = int(candidate_pairs["win"])
    losses = int(candidate_pairs["loss"])
    win_n = int(win_support["n"])
    source_copy_rate = (
        float(int(win_support["source_correct"]) / win_n) if win_n else None
    )
    noncopy_help_rate = (
        float(int(win_support["latent_noncopy_help"]) / win_n) if win_n else None
    )
    validity_ok = _validity_passes(validity)
    oracle_headroom = int(oracle["correct"]) - int(candidate_correct)
    if not validity_ok:
        status = "invalid_artifact"
    elif wins <= losses:
        status = "target_parity_or_negative"
    elif source_copy_rate is not None and source_copy_rate > 0.2:
        status = "copy_risk"
    elif oracle_headroom <= 0:
        status = "positive_noncopy_but_oracle_saturated"
    else:
        status = "positive_noncopy_with_headroom"
    return {
        "status": status,
        "validity_ok": validity_ok,
        "wins": wins,
        "losses": losses,
        "help_harm_ratio": None if losses == 0 else float(wins / losses),
        "source_copy_rate_on_candidate_only_wins": source_copy_rate,
        "latent_noncopy_help_rate_on_candidate_only_wins": noncopy_help_rate,
        "oracle_headroom_examples": oracle_headroom,
        "promote_if": [
            "validity_ok",
            "wins > losses on the larger frozen slice",
            "source_copy_rate_on_candidate_only_wins <= 0.2",
            "controls with shuffled/zero source collapse to target level",
            "the same pattern survives seed repeats and one strict cross-family pair",
        ],
        "demote_if": [
            "artifact validity fails",
            "wins <= losses",
            "candidate-only wins mostly have source/text already correct",
            "shuffled/zero-source controls preserve the win pattern",
            "finite seed repeats are target-parity or negative",
        ],
    }


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    metrics = payload["summary_metrics"]
    oracle = payload["oracle_bound"]
    candidate_pairs = payload["paired_vs_target"]
    win_support = payload["candidate_only_win_support"]
    t2t_poison = payload["text_to_text_loss_support"]
    flip_matrix = payload.get("flip_matrix", {})
    diagnostic_gate = payload.get("diagnostic_gate", {})
    lines = [
        "# GSM8K Contract Diagnostics",
        "",
        f"- date: `{payload['date']}`",
        f"- candidate: `{payload['candidate_label']}`",
        f"- source -> target: `{payload['config']['source_model']} -> {payload['config']['target_model']}`",
        f"- eval file: `{payload['config']['eval_file']}`",
        f"- slice: `{payload['config']['slice_size']}`",
        "",
        "## Summary",
        "",
        "| Row | Accuracy | Correct |",
        "|---|---:|---:|",
        f"| source_alone | {metrics['source_alone_accuracy']:.4f} | {metrics['source_alone_correct']} |",
        f"| target_alone | {metrics['target_alone_accuracy']:.4f} | {metrics['target_alone_correct']} |",
        f"| text_to_text | {metrics['text_to_text_accuracy']:.4f} | {metrics['text_to_text_correct']} |",
        f"| {payload['candidate_label']} | {metrics['candidate_accuracy']:.4f} | {metrics['candidate_correct']} |",
        f"| oracle(target, candidate) | {metrics['oracle_accuracy']:.4f} | {oracle['correct']} |",
        "",
        "## Reviewer Diagnostics",
        "",
        f"- candidate vs target: wins=`{candidate_pairs['win']}`, losses=`{candidate_pairs['loss']}`, ties=`{candidate_pairs['tie']}`",
        f"- oracle headroom above candidate: `{oracle['correct'] - metrics['candidate_correct']}` examples",
        f"- source correctness on candidate-only wins: `{win_support['source_correct']} / {win_support['n']}`",
        f"- text correctness on candidate-only wins: `{win_support['text_correct']} / {win_support['n']}`",
        f"- candidate numeric equals source numeric on candidate-only wins: `{win_support.get('candidate_same_numeric_as_source', 0)} / {win_support['n']}`",
        f"- latent non-copy helps: `{win_support.get('latent_noncopy_help', 0)} / {win_support['n']}`",
        f"- source correctness on target-only text-to-text losses: `{t2t_poison['source_correct']} / {t2t_poison['n']}`",
        f"- text wrong on target-only losses: `{t2t_poison['text_wrong']} / {t2t_poison['n']}`",
        "",
    ]
    if payload.get("validity"):
        lines.extend(
            [
                "## Artifact Validity",
                "",
                "| Row | Count | ID parity | Numeric coverage | Empty predictions |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in payload["validity"]:
            lines.append(
                f"| {row['label']} | {row['row_count']}/{row['expected_count']} | "
                f"{'T' if row['ordered_id_parity'] else 'F'} | "
                f"{row['numeric_extraction_coverage']}/{row['row_count']} | "
                f"{row['empty_predictions']} |"
            )
        lines.append("")
    if flip_matrix:
        lines.extend(
            [
                "## Paired Flip Matrix",
                "",
                "| Target | Candidate | Count |",
                "|---|---|---:|",
                f"| wrong | wrong | {flip_matrix['target_wrong_candidate_wrong']} |",
                f"| wrong | right | {flip_matrix['target_wrong_candidate_right']} |",
                f"| right | wrong | {flip_matrix['target_right_candidate_wrong']} |",
                f"| right | right | {flip_matrix['target_right_candidate_right']} |",
                "",
                f"- exact two-sided sign p-value on helps vs harms: `{payload.get('paired_sign_p_value')}`",
                "",
            ]
        )
    if diagnostic_gate:
        lines.extend(
            [
                "## Diagnostic Gate",
                "",
                f"- status: `{diagnostic_gate['status']}`",
                f"- validity ok: `{diagnostic_gate['validity_ok']}`",
                f"- source-copy rate on candidate-only wins: `{diagnostic_gate['source_copy_rate_on_candidate_only_wins']}`",
                f"- oracle headroom examples: `{diagnostic_gate['oracle_headroom_examples']}`",
                "",
            ]
        )

    def _append_table(title: str, rows: list[dict[str, Any]]) -> None:
        lines.extend([f"## {title}", ""])
        if not rows:
            lines.append("_None._")
            lines.append("")
            return
        lines.extend(
            [
                "| Idx | Example ID | Source | Text | Target | Candidate | Cand=Src | Gold | Prompt |",
                "|---:|---|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['index']} | {row['example_id']} | "
                f"{'T' if row['source_correct'] else 'F'} | "
                f"{'T' if row['text_correct'] else 'F'} | "
                f"{'T' if row['target_correct'] else 'F'} | "
                f"{'T' if row['candidate_correct'] else 'F'} | "
                f"{'T' if row.get('candidate_same_numeric_as_source') else 'F'} | "
                f"{row['reference_answer']!r} | {row['prompt_excerpt']} |"
            )
        lines.append("")

    _append_table("Candidate-Only Wins", payload["candidate_only_wins"])
    _append_table("Candidate-Only Losses", payload["candidate_only_losses"])
    _append_table("Text-To-Text Target-Only Losses", payload["text_to_text_target_only_losses"])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _ensure_materialized_eval(
    *,
    baseline_config: dict[str, Any],
) -> pathlib.Path:
    materialized = pathlib.Path(str(baseline_config["materialized_eval_file"]))
    harness.materialize_slice(ROOT / str(baseline_config["eval_file"]), materialized, int(baseline_config["slice_size"]))
    return materialized


def _ensure_source_output(
    *,
    baseline_payload: dict[str, Any],
    materialized_eval_file: pathlib.Path,
    output_path: pathlib.Path,
) -> pathlib.Path:
    if output_path.exists():
        return output_path
    config = baseline_payload["config"]
    cmd = [
        harness.python_executable(ROOT),
        str(ROOT / "latent_bridge" / "evaluate.py"),
        "--source-model",
        str(config["source_model"]),
        "--target-model",
        str(config["target_model"]),
        "--translator",
        str(ROOT / str(config["checkpoint_path"])),
        "--eval-file",
        str(materialized_eval_file),
        "--task-type",
        "generation",
        "--device",
        str(config["device"]),
        "--max-new-tokens",
        str(config["max_new_tokens"]),
        "--source-reasoning-mode",
        str(config["source_reasoning_mode"]),
        "--methods",
        "source",
        "--prediction-output",
        str(output_path),
    ]
    cmd.extend(
        harness.chat_template_cli_args(
            enabled=bool(config.get("use_chat_template", False)),
            thinking=bool(config.get("enable_thinking", False)),
        )
    )
    smoke._run(cmd, cwd=ROOT)
    return output_path


def run_diagnostics(config: DiagnosticsConfig) -> dict[str, Any]:
    baseline_json = ROOT / config.baseline_results_dir / "gsm8k_smoke_contract_20260421.json"
    baseline_payload = _read_json(baseline_json)
    baseline_config = baseline_payload["config"]
    materialized_eval_file = _ensure_materialized_eval(baseline_config=baseline_config)
    examples = harness.load_generation(str(materialized_eval_file))

    artifact_paths = baseline_payload["artifacts"]
    baseline_output = pathlib.Path(
        str(
            artifact_paths.get("bridge_prediction_output")
            or artifact_paths["latentwire_prediction_output"]
        )
    )
    baseline_records = harness.attach_prompts(harness.read_jsonl(baseline_output), materialized_eval_file)
    baseline_groups = harness.group_by_method(baseline_records)
    target_records = baseline_groups["target_alone"]
    text_records = baseline_groups["text_to_text"]

    candidate_output = ROOT / config.candidate_prediction_output
    candidate_records = harness.attach_prompts(harness.read_jsonl(candidate_output), materialized_eval_file)
    candidate_method_records = _resolve_candidate_records(candidate_records, method_name=config.candidate_method)

    diagnostics_dir = ROOT / (config.results_dir or str(candidate_output.parent.relative_to(ROOT)))
    source_output = _ensure_source_output(
        baseline_payload=baseline_payload,
        materialized_eval_file=materialized_eval_file,
        output_path=diagnostics_dir / config.source_output_name,
    )
    source_records = harness.attach_prompts(harness.read_jsonl(source_output), materialized_eval_file)
    source_method_records = harness.group_by_method(source_records)["source_alone"]

    reference_ids = _ordered_example_ids(target_records)
    validity = [
        _row_validity(label="source_alone", records=source_method_records, reference_ids=reference_ids),
        _row_validity(label="target_alone", records=target_records, reference_ids=reference_ids),
        _row_validity(label="text_to_text", records=text_records, reference_ids=reference_ids),
        _row_validity(label=config.candidate_label, records=candidate_method_records, reference_ids=reference_ids),
    ]
    candidate_pairs = _paired_counts(candidate_method_records, target_records)
    oracle = _oracle_counts(candidate_method_records, target_records)
    flip_matrix = _flip_matrix(candidate_records=candidate_method_records, target_records=target_records)

    candidate_only_wins = _detail_rows(
        examples=examples,
        candidate_records=candidate_method_records,
        target_records=target_records,
        source_records=source_method_records,
        text_records=text_records,
        selector="candidate_only_wins",
    )
    candidate_only_losses = _detail_rows(
        examples=examples,
        candidate_records=candidate_method_records,
        target_records=target_records,
        source_records=source_method_records,
        text_records=text_records,
        selector="candidate_only_losses",
    )
    text_target_only_losses = _detail_rows(
        examples=examples,
        candidate_records=candidate_method_records,
        target_records=target_records,
        source_records=source_method_records,
        text_records=text_records,
        selector="text_only_losses",
    )
    candidate_correct = int(sum(int(row["correct"]) for row in candidate_method_records))
    candidate_only_win_support = _source_win_summary(candidate_only_wins)
    candidate_only_loss_support = _source_win_summary(candidate_only_losses)
    text_to_text_loss_support = _source_win_summary(text_target_only_losses)
    output_tag = config.output_tag or "20260422"

    payload = {
        "date": str(date.today()),
        "candidate_label": config.candidate_label,
        "config": {
            "source_model": baseline_config["source_model"],
            "target_model": baseline_config["target_model"],
            "eval_file": baseline_config["eval_file"],
            "slice_size": baseline_config["slice_size"],
            "materialized_eval_file": str(materialized_eval_file),
            "candidate_method": config.candidate_method,
            "output_tag": output_tag,
        },
        "artifacts": {
            "baseline_contract": str(baseline_json),
            "candidate_prediction_output": str(candidate_output),
            "source_prediction_output": str(source_output),
        },
        "validity": validity,
        "summary_metrics": {
            "source_alone_accuracy": float(sum(int(row["correct"]) for row in source_method_records) / max(len(source_method_records), 1)),
            "source_alone_correct": int(sum(int(row["correct"]) for row in source_method_records)),
            "target_alone_accuracy": float(sum(int(row["correct"]) for row in target_records) / max(len(target_records), 1)),
            "target_alone_correct": int(sum(int(row["correct"]) for row in target_records)),
            "text_to_text_accuracy": float(sum(int(row["correct"]) for row in text_records) / max(len(text_records), 1)),
            "text_to_text_correct": int(sum(int(row["correct"]) for row in text_records)),
            "candidate_accuracy": float(sum(int(row["correct"]) for row in candidate_method_records) / max(len(candidate_method_records), 1)),
            "candidate_correct": candidate_correct,
            "oracle_accuracy": float(oracle["correct"] / max(len(candidate_method_records), 1)),
        },
        "paired_vs_target": candidate_pairs,
        "flip_matrix": flip_matrix,
        "paired_sign_p_value": _exact_sign_p_value(
            flip_matrix["target_wrong_candidate_right"],
            flip_matrix["target_right_candidate_wrong"],
        ),
        "oracle_bound": oracle,
        "candidate_only_win_support": candidate_only_win_support,
        "candidate_only_loss_support": candidate_only_loss_support,
        "text_to_text_loss_support": text_to_text_loss_support,
        "diagnostic_gate": _diagnostic_gate(
            validity=validity,
            candidate_pairs=candidate_pairs,
            win_support=candidate_only_win_support,
            oracle=oracle,
            candidate_correct=candidate_correct,
        ),
        "candidate_only_wins": candidate_only_wins,
        "candidate_only_losses": candidate_only_losses,
        "text_to_text_target_only_losses": text_target_only_losses,
    }
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    harness.write_json(diagnostics_dir / f"{config.candidate_label}_diagnostics_{output_tag}.json", payload)
    _write_markdown(diagnostics_dir / f"{config.candidate_label}_diagnostics_{output_tag}.md", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze reviewer-requested GSM8K contract diagnostics for a candidate row.")
    parser.add_argument("--candidate-prediction-output", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--baseline-results-dir", default=DEFAULT_BASELINE_RESULTS_DIR)
    parser.add_argument("--candidate-method", default="rotalign_kv")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument(
        "--source-output-name",
        default="gsm8k32_source_alone.jsonl",
        help="Source-alone JSONL filename to reuse or materialize in the diagnostics results dir.",
    )
    parser.add_argument(
        "--output-tag",
        default=None,
        help="Suffix tag for diagnostics artifacts; defaults to the legacy 20260422 tag.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    config = DiagnosticsConfig(
        candidate_prediction_output=args.candidate_prediction_output,
        candidate_label=args.candidate_label,
        baseline_results_dir=args.baseline_results_dir,
        candidate_method=args.candidate_method,
        results_dir=args.results_dir,
        source_output_name=args.source_output_name,
        output_tag=args.output_tag,
    )
    return run_diagnostics(config)


if __name__ == "__main__":
    main()
