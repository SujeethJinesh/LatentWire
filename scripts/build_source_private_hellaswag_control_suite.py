from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import pathlib
import random
import statistics
import sys
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_hellaswag_bridge_contract as hs_contract  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path("results/source_private_hellaswag_control_suite_20260501")
DEFAULT_TRAIN = pathlib.Path("results/source_private_hellaswag_bridge_contract_20260501/official_splits/hellaswag_train.jsonl")
DEFAULT_EVAL = pathlib.Path(
    "results/source_private_hellaswag_bridge_contract_20260501/official_splits/hellaswag_validation_first1024.jsonl"
)
DEFAULT_ANCHOR = pathlib.Path(
    "results/source_private_hellaswag_fixed_packet_gate_20260501_qwen05_hashed_validation1024_2b/predictions.jsonl"
)
DEFAULT_FIXED_RESULT = pathlib.Path(
    "results/source_private_hellaswag_fixed_packet_gate_20260501_qwen05_hashed_validation1024_2b/arc_challenge_fixed_packet_gate.json"
)

CONTROL_CONDITIONS = (
    "matched_source_private_packet",
    "source_label_text_copy",
    "same_activity_shuffled_source_packet",
    "same_split_type_shuffled_source_packet",
    "activity_label_train_majority_prior",
    "split_type_train_majority_prior",
)


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_anchor_source_choices(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    choices: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("condition") != arc_gate.MATCHED_CONDITION:
                continue
            metadata = dict(row.get("metadata", {}))
            choices[str(row["content_id"])] = {
                "row_id": str(row["row_id"]),
                "source_selected_index": int(metadata["source_selected_index"]),
                "source_selected_label": str(metadata.get("source_selected_label", "")),
                "source_selected_choice_sha256": str(metadata.get("source_selected_choice_sha256", "")),
            }
    if not choices:
        raise ValueError(f"{path} contained no matched source-private packet rows")
    return choices


def _source_predictions_from_anchor(
    eval_rows: list[arc_gate.ArcRow],
    anchor_choices: dict[str, dict[str, Any]],
) -> list[int]:
    predictions: list[int] = []
    missing: list[str] = []
    mismatch: list[str] = []
    for row in eval_rows:
        entry = anchor_choices.get(row.content_id)
        if entry is None:
            missing.append(row.content_id)
            continue
        selected_index = int(entry["source_selected_index"])
        if selected_index < 0 or selected_index >= len(row.choices):
            mismatch.append(row.content_id)
            continue
        selected_hash = arc_gate._sha256_text(row.choices[selected_index])
        if entry.get("source_selected_choice_sha256") and entry["source_selected_choice_sha256"] != selected_hash:
            mismatch.append(row.content_id)
            continue
        predictions.append(selected_index)
    if missing or mismatch:
        raise ValueError(
            "anchor source choices did not match eval rows: "
            f"missing={len(missing)} mismatch={len(mismatch)}"
        )
    return predictions


def _content_id_from_raw(raw: dict[str, Any]) -> str:
    context = str(raw["ctx"]).strip()
    endings = [str(ending).strip() for ending in raw["endings"]]
    return hs_contract._content_id(context, endings)


def _load_hf_metadata(*, split: str, hf_dataset: str, hf_cache_dir: pathlib.Path) -> dict[str, dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(hf_dataset, split=split, cache_dir=str(hf_cache_dir))
    metadata: dict[str, dict[str, Any]] = {}
    for index, raw_row in enumerate(dataset, start=1):
        raw = dict(raw_row)
        content_id = _content_id_from_raw(raw)
        metadata[content_id] = {
            "row_index": index,
            "id": str(raw.get("ind") or content_id[:16]),
            "activity_label": str(raw.get("activity_label", "")),
            "split": str(raw.get("split", split)),
            "split_type": str(raw.get("split_type", "")),
            "source_id": str(raw.get("source_id", "")),
        }
    return metadata


def _accuracy(rows: list[arc_gate.ArcRow], predictions: list[int]) -> float:
    if not rows:
        return 0.0
    return float(sum(prediction == row.answer_index for row, prediction in zip(rows, predictions, strict=True)) / len(rows))


def _prediction_rows(
    *,
    condition: str,
    eval_rows: list[arc_gate.ArcRow],
    predictions: list[int],
    payload_bytes: int,
    metadata: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    metadata = metadata or [{} for _ in eval_rows]
    for row, prediction, meta in zip(eval_rows, predictions, metadata, strict=True):
        out.append(
            {
                "condition": condition,
                "row_id": row.row_id,
                "content_id": row.content_id,
                "answer_index": row.answer_index,
                "answer_label": row.answer_label,
                "prediction_index": int(prediction),
                "prediction_label": row.choice_labels[int(prediction)],
                "correct": bool(int(prediction) == row.answer_index),
                "payload_bytes": int(payload_bytes),
                "metadata": meta,
            }
        )
    return out


def _same_group_nonself_indices(groups: list[str], *, seed_text: str) -> tuple[list[int], int]:
    by_group: dict[str, list[int]] = collections.defaultdict(list)
    for index, group in enumerate(groups):
        by_group[group].append(index)
    fallback_count = 0
    other_indices: list[int] = []
    for index, group in enumerate(groups):
        candidates = [candidate for candidate in by_group[group] if candidate != index]
        if not candidates:
            candidates = [candidate for candidate in range(len(groups)) if candidate != index]
            fallback_count += 1
        if not candidates:
            other_indices.append(index)
            continue
        seed = int(hashlib.blake2b(f"{seed_text}|{index}|{group}".encode("utf-8"), digest_size=8).hexdigest(), 16)
        other_indices.append(candidates[seed % len(candidates)])
    return other_indices, fallback_count


def _majority_index_by_group(
    train_rows: list[arc_gate.ArcRow],
    train_metadata: dict[str, dict[str, Any]],
    *,
    group_key: str,
    default_index: int,
) -> dict[str, int]:
    counts: dict[str, collections.Counter[int]] = collections.defaultdict(collections.Counter)
    for row in train_rows:
        group = str(train_metadata.get(row.content_id, {}).get(group_key, ""))
        if not group:
            continue
        counts[group][row.answer_index] += 1
    return {
        group: max(counter, key=lambda index: (counter[index], -index))
        for group, counter in counts.items()
    } | {"": default_index}


def _summarize_condition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "correct": 0, "accuracy": 0.0, "mean_payload_bytes": 0.0}
    return {
        "n": len(rows),
        "correct": int(sum(1 for row in rows if row["correct"])),
        "accuracy": float(sum(1 for row in rows if row["correct"]) / len(rows)),
        "mean_payload_bytes": float(statistics.fmean(float(row["payload_bytes"]) for row in rows)),
    }


def _paired_bootstrap(rows: list[dict[str, Any]], *, condition: str, baseline: str, seed: int, samples: int) -> dict[str, float]:
    by_id: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_id.setdefault(str(row["content_id"]), {})[str(row["condition"])] = row
    deltas = [
        float(conditions[condition]["correct"]) - float(conditions[baseline]["correct"])
        for _, conditions in sorted(by_id.items())
        if condition in conditions and baseline in conditions
    ]
    if not deltas:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    rng = random.Random(seed)
    n = len(deltas)
    means = [statistics.fmean(deltas[rng.randrange(n)] for _ in range(n)) for _ in range(samples)]
    return {
        "mean": float(statistics.fmean(deltas)),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
    }


def build_control_suite(
    *,
    output_dir: pathlib.Path,
    train_path: pathlib.Path,
    eval_path: pathlib.Path,
    anchor_predictions: pathlib.Path,
    fixed_result: pathlib.Path,
    hf_dataset: str,
    hf_cache_dir: pathlib.Path,
    bootstrap_samples: int,
    run_date: str,
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    train_path = _resolve(train_path)
    eval_path = _resolve(eval_path)
    anchor_predictions = _resolve(anchor_predictions)
    fixed_result = _resolve(fixed_result)
    hf_cache_dir = _resolve(hf_cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = arc_gate._load_rows(train_path)
    eval_rows = arc_gate._load_rows(eval_path)
    fixed_payload = json.loads(fixed_result.read_text(encoding="utf-8"))
    budget_bytes = int(fixed_payload["budget_bytes"])
    feature_dim = int(fixed_payload["feature_dim"])
    code_dim = int(fixed_payload["code_dim"])
    seed = int(fixed_payload["seed"])
    feature_mode = str(fixed_payload["feature_mode"])
    if feature_mode != "hashed":
        raise ValueError("HellaSwag control suite currently expects feature_mode='hashed'")

    train_metadata = _load_hf_metadata(split="train", hf_dataset=hf_dataset, hf_cache_dir=hf_cache_dir)
    validation_metadata = _load_hf_metadata(split="validation", hf_dataset=hf_dataset, hf_cache_dir=hf_cache_dir)
    eval_metadata = [validation_metadata[row.content_id] for row in eval_rows]
    source_predictions = _source_predictions_from_anchor(eval_rows, _load_anchor_source_choices(anchor_predictions))

    eval_pair_features = arc_gate._features(
        arc_gate._choice_pair_texts(eval_rows),
        dim=feature_dim,
        feature_mode=feature_mode,
        feature_model="",
        feature_device="auto",
        feature_dtype="float32",
        feature_max_length=128,
        local_files_only=True,
        anchor_texts=arc_gate._choice_pair_texts(train_rows),
    )
    residuals = arc_gate._candidate_residuals(eval_rows, eval_pair_features)
    projection = arc_gate._projection_matrix(feature_dim, code_dim, seed=seed + 171)
    priors = arc_gate._index_prior(train_rows)
    packets = [
        arc_gate._encode_packet(row_residuals[source_index], projection, budget_bytes=budget_bytes)
        for row_residuals, source_index in zip(residuals, source_predictions, strict=True)
    ]

    base_rows = arc_gate._rows_for_predictions(
        eval_rows=eval_rows,
        residuals=residuals,
        source_predictions=source_predictions,
        projection=projection,
        budget_bytes=budget_bytes,
        index_prior=priors,
        seed=seed + 911,
    )
    matched_rows = [row for row in base_rows if row["condition"] == arc_gate.MATCHED_CONDITION]

    source_label_rows = _prediction_rows(
        condition="source_label_text_copy",
        eval_rows=eval_rows,
        predictions=source_predictions,
        payload_bytes=min(1, budget_bytes),
        metadata=[{"diagnostic": "forbidden top-label text copy control"} for _ in eval_rows],
    )

    activity_groups = [str(meta.get("activity_label", "")) for meta in eval_metadata]
    split_type_groups = [str(meta.get("split_type", "")) for meta in eval_metadata]
    same_activity_indices, activity_fallbacks = _same_group_nonself_indices(
        activity_groups,
        seed_text="hellaswag-same-activity",
    )
    same_split_type_indices, split_type_fallbacks = _same_group_nonself_indices(
        split_type_groups,
        seed_text="hellaswag-same-split-type",
    )

    shuffled_rows: list[dict[str, Any]] = []
    for condition, other_indices, fallback_count in [
        ("same_activity_shuffled_source_packet", same_activity_indices, activity_fallbacks),
        ("same_split_type_shuffled_source_packet", same_split_type_indices, split_type_fallbacks),
    ]:
        for row_index, row in enumerate(eval_rows):
            other_index = other_indices[row_index]
            payload, packet_meta = packets[other_index]
            prediction, decode_meta = arc_gate._predict_from_code(
                row=row,
                residuals=residuals[row_index],
                payload=payload,
                projection=projection,
                index_prior=priors,
            )
            shuffled_rows.append(
                {
                    "condition": condition,
                    "row_id": row.row_id,
                    "content_id": row.content_id,
                    "answer_index": row.answer_index,
                    "answer_label": row.answer_label,
                    "prediction_index": int(prediction),
                    "prediction_label": row.choice_labels[int(prediction)],
                    "correct": bool(int(prediction) == row.answer_index),
                    "payload_bytes": budget_bytes,
                    "metadata": {
                        **packet_meta,
                        **decode_meta,
                        "source_row_id": eval_rows[other_index].row_id,
                        "fallback_count_for_condition": fallback_count,
                    },
                }
            )

    default_index = max(range(len(priors)), key=lambda index: (priors[index], -index))
    activity_majority = _majority_index_by_group(
        train_rows,
        train_metadata,
        group_key="activity_label",
        default_index=default_index,
    )
    split_type_majority = _majority_index_by_group(
        train_rows,
        train_metadata,
        group_key="split_type",
        default_index=default_index,
    )
    activity_prior_predictions = [
        activity_majority.get(str(meta.get("activity_label", "")), default_index)
        for meta in eval_metadata
    ]
    split_type_prior_predictions = [
        split_type_majority.get(str(meta.get("split_type", "")), default_index)
        for meta in eval_metadata
    ]
    metadata_prior_rows = [
        *_prediction_rows(
            condition="activity_label_train_majority_prior",
            eval_rows=eval_rows,
            predictions=activity_prior_predictions,
            payload_bytes=0,
            metadata=[{"diagnostic": "forbidden metadata train-label prior"} for _ in eval_rows],
        ),
        *_prediction_rows(
            condition="split_type_train_majority_prior",
            eval_rows=eval_rows,
            predictions=split_type_prior_predictions,
            payload_bytes=0,
            metadata=[{"diagnostic": "forbidden metadata train-label prior"} for _ in eval_rows],
        ),
    ]

    prediction_rows = [*matched_rows, *source_label_rows, *shuffled_rows, *metadata_prior_rows]
    (output_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    metrics = {
        condition: _summarize_condition([row for row in prediction_rows if row["condition"] == condition])
        for condition in CONTROL_CONDITIONS
    }
    matched = metrics[arc_gate.MATCHED_CONDITION]["accuracy"]
    source_label = metrics["source_label_text_copy"]["accuracy"]
    best_metadata_control_name = max(
        [
            "same_activity_shuffled_source_packet",
            "same_split_type_shuffled_source_packet",
            "activity_label_train_majority_prior",
            "split_type_train_majority_prior",
        ],
        key=lambda name: metrics[name]["accuracy"],
    )
    best_metadata_control = metrics[best_metadata_control_name]["accuracy"]
    source_label_ci = _paired_bootstrap(
        prediction_rows,
        condition=arc_gate.MATCHED_CONDITION,
        baseline="source_label_text_copy",
        seed=seed + 2001,
        samples=bootstrap_samples,
    )
    metadata_ci = _paired_bootstrap(
        prediction_rows,
        condition=arc_gate.MATCHED_CONDITION,
        baseline=best_metadata_control_name,
        seed=seed + 2002,
        samples=bootstrap_samples,
    )
    payload = {
        "gate": "source_private_hellaswag_control_suite",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "train_path": _display_path(train_path),
        "eval_path": _display_path(eval_path),
        "anchor_predictions": _display_path(anchor_predictions),
        "fixed_result": _display_path(fixed_result),
        "train_sha256": _sha256_file(train_path),
        "eval_sha256": _sha256_file(eval_path),
        "anchor_predictions_sha256": _sha256_file(anchor_predictions),
        "fixed_result_sha256": _sha256_file(fixed_result),
        "eval_rows": len(eval_rows),
        "budget_bytes": budget_bytes,
        "condition_metrics": metrics,
        "headline": {
            "matched_accuracy": matched,
            "source_label_text_copy_accuracy": source_label,
            "matched_minus_source_label_text_copy": matched - source_label,
            "best_metadata_or_activity_control": best_metadata_control_name,
            "best_metadata_or_activity_control_accuracy": best_metadata_control,
            "matched_minus_best_metadata_or_activity_control": matched - best_metadata_control,
            "paired_ci95_vs_source_label_text_copy": source_label_ci,
            "paired_ci95_vs_best_metadata_or_activity_control": metadata_ci,
            "same_activity_fallback_count": activity_fallbacks,
            "same_split_type_fallback_count": split_type_fallbacks,
            "label_copy_threat_present": (matched - source_label) < 0.02,
            "metadata_controls_clean": (matched - best_metadata_control) >= 0.03,
        },
        "pass_gate": bool((matched - best_metadata_control) >= 0.03),
        "strict_non_label_copy_pass_gate": bool((matched - source_label) >= 0.02),
        "claim_boundary": (
            "This suite tests HellaSwag metadata/activity leakage and the stronger source-label-copy threat. "
            "A metadata pass does not by itself prove a non-label-copy communication contribution."
        ),
    }
    (output_dir / "hellaswag_control_suite.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# HellaSwag Control Suite",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- strict non-label-copy pass gate: `{payload['strict_non_label_copy_pass_gate']}`",
        f"- matched/source-label-copy: `{matched:.3f}` / `{source_label:.3f}`",
        f"- best metadata/activity control: `{best_metadata_control_name}` at `{best_metadata_control:.3f}`",
        f"- label-copy threat present: `{payload['headline']['label_copy_threat_present']}`",
        "",
    ]
    (output_dir / "hellaswag_control_suite.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HellaSwag-specific source-private control suite.")
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-path", type=pathlib.Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval-path", type=pathlib.Path, default=DEFAULT_EVAL)
    parser.add_argument("--anchor-predictions", type=pathlib.Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--fixed-result", type=pathlib.Path, default=DEFAULT_FIXED_RESULT)
    parser.add_argument("--hf-dataset", default=hs_contract.DEFAULT_HF_DATASET)
    parser.add_argument("--hf-cache-dir", type=pathlib.Path, default=hs_contract.DEFAULT_HF_CACHE)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--run-date", default=str(dt.datetime.now(dt.UTC).date()))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = build_control_suite(
        output_dir=args.output_dir,
        train_path=args.train_path,
        eval_path=args.eval_path,
        anchor_predictions=args.anchor_predictions,
        fixed_result=args.fixed_result,
        hf_dataset=args.hf_dataset,
        hf_cache_dir=args.hf_cache_dir,
        bootstrap_samples=args.bootstrap_samples,
        run_date=args.run_date,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
