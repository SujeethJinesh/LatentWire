from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pathlib
import sys
import time
from collections import Counter
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_hellaswag_hidden_innovation_repair_probe as repair  # noqa: E402
from scripts import build_source_private_hellaswag_score_packet_headroom as headroom  # noqa: E402
from scripts import build_source_private_hellaswag_top2_contrastive_repair_probe as top2  # noqa: E402
from scripts import build_source_private_hellaswag_train_source_score_repair_probe as score_repair  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_stability_gate_20260501_qwen05_train512_validation1024"
)
DEFAULT_SPLIT_SEEDS = (1729, 1731, 1733, 1735, 1737)
STRICT_DELTA = 0.02


def _read_hidden_cache_rows(
    train_path: pathlib.Path,
    hidden_cache: pathlib.Path,
    *,
    train_hidden_rows: int,
    selection_seed: int,
) -> list[arc_gate.ArcRow]:
    metadata = json.loads(hidden_cache.with_suffix(".json").read_text(encoding="utf-8"))
    all_rows = arc_gate._load_rows(train_path)
    selected_rows = top2._select_train_rows(all_rows, count=train_hidden_rows, seed=selection_seed)
    if metadata.get("row_count") != len(selected_rows):
        raise ValueError(
            f"hidden cache row count does not match selected train rows: "
            f"{metadata.get('row_count')} vs {len(selected_rows)}"
        )
    if metadata.get("row_ids") != [row.row_id for row in selected_rows]:
        raise ValueError("hidden cache row id order does not match the selected train rows")
    if metadata.get("content_digest") != headroom._content_digest(selected_rows):
        raise ValueError("hidden cache content digest does not match the selected train rows")
    return selected_rows


def _candidate_readout_summary(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "view": item["view"],
            "ridge": item["ridge"],
            "feature_dim": item["feature_dim"],
            "fit_accuracy": item["fit"]["accuracy"],
            "internal_dev_accuracy": item["internal_dev"]["accuracy"],
            "eval_accuracy": item["eval"]["accuracy"],
        }
        for item in items
    ]


def _fit_split(
    *,
    selection_policy: str,
    allowed_views: tuple[str, ...] | None,
    split_seed: int,
    train_rows: list[arc_gate.ArcRow],
    eval_rows: list[arc_gate.ArcRow],
    train_scores: list[list[float]],
    eval_scores: list[list[float]],
    train_features: dict[str, np.ndarray],
    eval_features: dict[str, np.ndarray],
    zero_eval_features: dict[str, np.ndarray],
    wrong_eval_features: dict[str, np.ndarray],
    candidate_roll_eval_features: dict[str, np.ndarray],
    dev_fraction: float,
    ridges: tuple[float, ...],
    bootstrap_samples: int,
) -> dict[str, Any]:
    fit_indices, dev_indices = top2._split_indices(
        len(train_rows), dev_fraction=dev_fraction, seed=split_seed + 17
    )
    candidate_readouts: list[dict[str, Any]] = []
    for view in train_features:
        for ridge in ridges:
            candidate_readouts.append(
                repair._fit_and_eval_view(
                    view=view,
                    ridge=ridge,
                    train_features=train_features[view],
                    eval_features=eval_features[view],
                    fit_indices=fit_indices,
                    dev_indices=dev_indices,
                    train_rows=train_rows,
                    train_scores=train_scores,
                    eval_rows=eval_rows,
                )
            )

    selectable_readouts = [
        item for item in candidate_readouts if allowed_views is None or item["view"] in allowed_views
    ]
    if not selectable_readouts:
        raise ValueError(f"selection policy {selection_policy} has no selectable views")
    selected = max(
        selectable_readouts,
        key=lambda item: (
            item["internal_dev"]["accuracy"],
            item["fit"]["accuracy"],
            item["view"] == "score_hidden_residual",
            -item["feature_dim"],
            -item["ridge"],
        ),
    )
    source_label_eval = top2._source_label_predictions(eval_scores)
    offsets = score_repair._fit_choice_bias_offsets(
        top2._take_rows(train_rows, fit_indices),
        top2._take_scores(train_scores, fit_indices),
    )
    trained_label_eval = [score_repair._predict_calibrated_label(row_scores, offsets) for row_scores in eval_scores]
    source_label_accuracy = top2._accuracy(eval_rows, source_label_eval)
    trained_label_accuracy = top2._accuracy(eval_rows, trained_label_eval)
    best_label_accuracy = max(source_label_accuracy, trained_label_accuracy)
    best_label_predictions = source_label_eval if source_label_accuracy >= trained_label_accuracy else trained_label_eval
    selected_predictions = [int(value) for value in selected["eval_predictions"]]
    best_score_control = max(
        (item for item in candidate_readouts if item["view"] == "score_only"),
        key=lambda item: (item["internal_dev"]["accuracy"], item["fit"]["accuracy"], item["eval"]["accuracy"]),
    )

    refit_model = repair._fit_candidate_ridge(
        features=train_features[selected["view"]],
        rows=train_rows,
        score_matrix=train_scores,
        fit_indices=fit_indices,
        ridge=selected["ridge"],
    )
    zero_predictions, _ = repair._predict_candidate_ridge(zero_eval_features[selected["view"]], refit_model)
    wrong_predictions, _ = repair._predict_candidate_ridge(wrong_eval_features[selected["view"]], refit_model)
    candidate_roll_predictions, _ = repair._predict_candidate_ridge(
        candidate_roll_eval_features[selected["view"]], refit_model
    )
    zero_accuracy = top2._accuracy(eval_rows, zero_predictions)
    wrong_accuracy = top2._accuracy(eval_rows, wrong_predictions)
    candidate_roll_accuracy = top2._accuracy(eval_rows, candidate_roll_predictions)
    paired_ci_label = top2._paired_ci_predictions(
        eval_rows,
        selected_predictions,
        best_label_predictions,
        seed=split_seed + 5001,
        samples=bootstrap_samples,
    )
    paired_ci_score = top2._paired_ci_predictions(
        eval_rows,
        selected_predictions,
        [int(value) for value in best_score_control["eval_predictions"]],
        seed=split_seed + 5002,
        samples=bootstrap_samples,
    )
    selected_eval_accuracy = selected["eval"]["accuracy"]
    row = {
        "selection_policy": selection_policy,
        "allowed_views": ",".join(allowed_views) if allowed_views is not None else "all",
        "split_seed": split_seed,
        "fit_rows": len(fit_indices),
        "internal_dev_rows": len(dev_indices),
        "selected_view": selected["view"],
        "selected_ridge": selected["ridge"],
        "selected_feature_dim": selected["feature_dim"],
        "selected_fit_accuracy": selected["fit"]["accuracy"],
        "selected_internal_dev_accuracy": selected["internal_dev"]["accuracy"],
        "selected_eval_accuracy": selected_eval_accuracy,
        "source_label_copy_eval_accuracy": source_label_accuracy,
        "trained_choice_bias_label_copy_eval_accuracy": trained_label_accuracy,
        "best_label_copy_eval_accuracy": best_label_accuracy,
        "selected_minus_best_label_copy": selected_eval_accuracy - best_label_accuracy,
        "score_only_control_accuracy": best_score_control["eval"]["accuracy"],
        "selected_minus_score_only_control": selected_eval_accuracy - best_score_control["eval"]["accuracy"],
        "zero_hidden_control_accuracy": zero_accuracy,
        "selected_minus_zero_hidden_control": selected_eval_accuracy - zero_accuracy,
        "wrong_example_hidden_control_accuracy": wrong_accuracy,
        "selected_minus_wrong_example_hidden_control": selected_eval_accuracy - wrong_accuracy,
        "candidate_roll_hidden_control_accuracy": candidate_roll_accuracy,
        "selected_minus_candidate_roll_hidden_control": selected_eval_accuracy - candidate_roll_accuracy,
        "paired_ci95_low_vs_best_label_copy": paired_ci_label["ci95_low"],
        "paired_ci95_high_vs_best_label_copy": paired_ci_label["ci95_high"],
        "paired_ci95_low_vs_score_only_control": paired_ci_score["ci95_low"],
        "paired_ci95_high_vs_score_only_control": paired_ci_score["ci95_high"],
        "pass_gate": bool(
            selected_eval_accuracy - best_label_accuracy >= STRICT_DELTA
            and paired_ci_label["ci95_low"] > 0.0
            and selected_eval_accuracy - zero_accuracy >= STRICT_DELTA
            and wrong_accuracy <= best_label_accuracy
            and candidate_roll_accuracy <= best_label_accuracy
        ),
    }
    return {"row": row, "candidate_readouts": _candidate_readout_summary(candidate_readouts)}


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected_view_counts = dict(Counter(row["selected_view"] for row in rows))
    selected_ridge_counts = {str(key): value for key, value in Counter(row["selected_ridge"] for row in rows).items()}
    return {
        "split_seed_count": len(rows),
        "pass_count": sum(1 for row in rows if row["pass_gate"]),
        "all_split_seeds_pass": all(row["pass_gate"] for row in rows),
        "selected_view_counts": selected_view_counts,
        "selected_ridge_counts": selected_ridge_counts,
        "selected_eval_accuracy_mean": float(np.mean([row["selected_eval_accuracy"] for row in rows])),
        "selected_eval_accuracy_min": min(row["selected_eval_accuracy"] for row in rows),
        "selected_eval_accuracy_max": max(row["selected_eval_accuracy"] for row in rows),
        "delta_vs_best_label_copy_mean": float(np.mean([row["selected_minus_best_label_copy"] for row in rows])),
        "delta_vs_best_label_copy_min": min(row["selected_minus_best_label_copy"] for row in rows),
        "paired_ci95_low_vs_best_label_copy_min": min(row["paired_ci95_low_vs_best_label_copy"] for row in rows),
        "selected_minus_zero_hidden_control_min": min(row["selected_minus_zero_hidden_control"] for row in rows),
        "wrong_example_hidden_control_accuracy_max": max(row["wrong_example_hidden_control_accuracy"] for row in rows),
        "candidate_roll_hidden_control_accuracy_max": max(row["candidate_roll_hidden_control_accuracy"] for row in rows),
        "strict_delta_required": STRICT_DELTA,
    }


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    unrestricted = payload["unrestricted_model_selection_diagnostic"]
    lines = [
        "# HellaSwag Hidden-Innovation Stability Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- split seeds passing: `{h['pass_count']}/{h['split_seed_count']}`",
        f"- eval accuracy mean/min/max: `{h['selected_eval_accuracy_mean']:.6f}` / `{h['selected_eval_accuracy_min']:.6f}` / `{h['selected_eval_accuracy_max']:.6f}`",
        f"- delta vs best label-copy mean/min: `{h['delta_vs_best_label_copy_mean']:.6f}` / `{h['delta_vs_best_label_copy_min']:.6f}`",
        f"- min CI95 low vs best label-copy: `{h['paired_ci95_low_vs_best_label_copy_min']:.6f}`",
        f"- min delta vs zero-hidden: `{h['selected_minus_zero_hidden_control_min']:.6f}`",
        f"- selected view counts: `{h['selected_view_counts']}`",
        f"- unrestricted selector diagnostic pass count: `{unrestricted['pass_count']}/{unrestricted['split_seed_count']}`",
        f"- unrestricted selector min delta vs best label-copy: `{unrestricted['delta_vs_best_label_copy_min']:.6f}`",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_gate(
    *,
    output_dir: pathlib.Path = DEFAULT_OUTPUT,
    train_path: pathlib.Path = repair.DEFAULT_TRAIN,
    eval_path: pathlib.Path = repair.DEFAULT_EVAL,
    eval_score_cache: pathlib.Path = repair.DEFAULT_EVAL_SCORE_CACHE,
    train_score_cache: pathlib.Path = repair.DEFAULT_TRAIN_SCORE_CACHE,
    train_hidden_cache: pathlib.Path = repair.DEFAULT_TRAIN_HIDDEN_CACHE,
    eval_hidden_cache: pathlib.Path = repair.DEFAULT_EVAL_HIDDEN_CACHE,
    train_hidden_rows: int = 512,
    selection_seed: int = 1729,
    split_seeds: tuple[int, ...] = DEFAULT_SPLIT_SEEDS,
    dev_fraction: float = 0.25,
    ridges: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0),
    bootstrap_samples: int = 500,
    run_date: str = "2026-05-01",
) -> dict[str, Any]:
    output_dir = top2._resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    train_path = top2._resolve(train_path)
    eval_path = top2._resolve(eval_path)
    eval_score_cache = top2._resolve(eval_score_cache)
    train_score_cache = top2._resolve(train_score_cache)
    train_hidden_cache = top2._resolve(train_hidden_cache)
    eval_hidden_cache = top2._resolve(eval_hidden_cache)

    train_rows = _read_hidden_cache_rows(
        train_path,
        train_hidden_cache,
        train_hidden_rows=train_hidden_rows,
        selection_seed=selection_seed,
    )
    eval_rows = arc_gate._load_rows(eval_path)
    train_scores, _, train_source_model = headroom._load_score_cache(train_score_cache, rows=train_rows)
    eval_scores, _, eval_source_model = headroom._load_score_cache(eval_score_cache, rows=eval_rows)
    train_hidden, train_hidden_meta = top2._load_hidden_cache(train_hidden_cache, rows=train_rows)
    eval_hidden, eval_hidden_meta = top2._load_hidden_cache(eval_hidden_cache, rows=eval_rows)

    views = (
        "score_only",
        "hidden_residual_only",
        "score_hidden_residual",
        "score_hidden_absolute",
        "score_hidden_absolute_residual",
    )
    train_features = {
        view: repair._candidate_feature_tensor(scores=train_scores, hidden=train_hidden, view=view) for view in views
    }
    eval_features = {
        view: repair._candidate_feature_tensor(scores=eval_scores, hidden=eval_hidden, view=view) for view in views
    }
    zero_hidden = np.zeros_like(eval_hidden)
    wrong_hidden = np.roll(eval_hidden, 1, axis=0)
    candidate_roll_hidden = np.roll(eval_hidden, 1, axis=1)
    zero_eval_features = {
        view: repair._candidate_feature_tensor(scores=eval_scores, hidden=zero_hidden, view=view) for view in views
    }
    wrong_eval_features = {
        view: repair._candidate_feature_tensor(scores=eval_scores, hidden=wrong_hidden, view=view) for view in views
    }
    candidate_roll_eval_features = {
        view: repair._candidate_feature_tensor(scores=eval_scores, hidden=candidate_roll_hidden, view=view)
        for view in views
    }

    anchored_rows: list[dict[str, Any]] = []
    unrestricted_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for split_seed in split_seeds:
        unrestricted_result = _fit_split(
            selection_policy="unrestricted_model_selection_diagnostic",
            allowed_views=None,
            split_seed=split_seed,
            train_rows=train_rows,
            eval_rows=eval_rows,
            train_scores=train_scores,
            eval_scores=eval_scores,
            train_features=train_features,
            eval_features=eval_features,
            zero_eval_features=zero_eval_features,
            wrong_eval_features=wrong_eval_features,
            candidate_roll_eval_features=candidate_roll_eval_features,
            dev_fraction=dev_fraction,
            ridges=ridges,
            bootstrap_samples=bootstrap_samples,
        )
        unrestricted_rows.append(unrestricted_result["row"])
        anchored_result = _fit_split(
            selection_policy="anchored_score_hidden_residual",
            allowed_views=("score_hidden_residual",),
            split_seed=split_seed,
            train_rows=train_rows,
            eval_rows=eval_rows,
            train_scores=train_scores,
            eval_scores=eval_scores,
            train_features=train_features,
            eval_features=eval_features,
            zero_eval_features=zero_eval_features,
            wrong_eval_features=wrong_eval_features,
            candidate_roll_eval_features=candidate_roll_eval_features,
            dev_fraction=dev_fraction,
            ridges=ridges,
            bootstrap_samples=bootstrap_samples,
        )
        anchored_rows.append(anchored_result["row"])
        for candidate in unrestricted_result["candidate_readouts"]:
            candidate_rows.append({"selection_policy": "shared_candidate_grid", "split_seed": split_seed, **candidate})

    headline = _aggregate(anchored_rows)
    unrestricted_headline = _aggregate(unrestricted_rows)
    pass_gate = bool(
        headline["all_split_seeds_pass"]
        and headline["delta_vs_best_label_copy_min"] >= STRICT_DELTA
        and headline["paired_ci95_low_vs_best_label_copy_min"] > 0.0
        and headline["selected_minus_zero_hidden_control_min"] >= STRICT_DELTA
    )
    payload = {
        "gate": "source_private_hellaswag_hidden_innovation_stability_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": pass_gate,
        "pass_rule": (
            "Pass if every cached-train split seed clears the anchored score+hidden-residual repair gate: "
            "delta >= 0.02 versus best source/trained label-copy, paired CI95 low > 0, zero-hidden "
            "delta >= 0.02, and wrong-example/candidate-roll hidden controls do not beat label-copy. "
            "An unrestricted selector is reported separately as a shortcut diagnostic."
        ),
        "train_path": top2._display_path(train_path),
        "train_sha256": top2._sha256_file(train_path),
        "eval_path": top2._display_path(eval_path),
        "eval_sha256": top2._sha256_file(eval_path),
        "train_score_cache": top2._display_path(train_score_cache),
        "train_score_cache_sha256": top2._sha256_file(train_score_cache),
        "eval_score_cache": top2._display_path(eval_score_cache),
        "eval_score_cache_sha256": top2._sha256_file(eval_score_cache),
        "train_hidden_cache": top2._display_path(train_hidden_cache),
        "train_hidden_cache_sha256": top2._sha256_file(train_hidden_cache),
        "eval_hidden_cache": top2._display_path(eval_hidden_cache),
        "eval_hidden_cache_sha256": top2._sha256_file(eval_hidden_cache),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_hidden_rows": train_hidden_rows,
        "selection_seed": selection_seed,
        "dev_fraction": dev_fraction,
        "split_seeds": list(split_seeds),
        "source_model": {
            "score_train": train_source_model,
            "score_eval": eval_source_model,
            "hidden_train": train_hidden_meta,
            "hidden_eval": eval_hidden_meta,
        },
        "packet_contract": {
            "packet_name": "hidden_innovation_candidate_selector_packet",
            "raw_payload_bytes": 2,
            "framed_record_bytes": 5,
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_vector_transmitted": False,
            "raw_scores_transmitted": False,
        },
        "headline": headline,
        "unrestricted_model_selection_diagnostic": unrestricted_headline,
        "stability_rows": anchored_rows,
        "unrestricted_stability_rows": unrestricted_rows,
        "interpretation": (
            "This gate repeats model selection over independent train/dev splits of the cached 512-row "
            "HellaSwag train-hidden slice while evaluating once on the frozen 1024-row validation slice. "
            "The promoted method is anchored to the score+hidden-residual view because the unrestricted "
            "selector can drift into score-only or hidden-only shortcuts. It does not yet test new "
            "train-hidden row samples or full validation."
        ),
        "timing": {"total_seconds": float(time.perf_counter() - started)},
    }
    json_path = output_dir / "hellaswag_hidden_innovation_stability_gate.json"
    csv_path = output_dir / "stability_rows.csv"
    candidate_path = output_dir / "candidate_readouts.jsonl"
    md_path = output_dir / "hellaswag_hidden_innovation_stability_gate.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(csv_path, anchored_rows)
    _write_jsonl(candidate_path, candidate_rows)
    _write_markdown(md_path, payload)
    manifest = {
        "gate": payload["gate"],
        "created_utc": payload["created_utc"],
        "headline": headline,
        "files": [
            {"path": top2._display_path(path), "sha256": top2._sha256_file(path), "bytes": path.stat().st_size}
            for path in (json_path, md_path, csv_path, candidate_path)
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("at least one split seed is required")
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-path", type=pathlib.Path, default=repair.DEFAULT_TRAIN)
    parser.add_argument("--eval-path", type=pathlib.Path, default=repair.DEFAULT_EVAL)
    parser.add_argument("--eval-score-cache", type=pathlib.Path, default=repair.DEFAULT_EVAL_SCORE_CACHE)
    parser.add_argument("--train-score-cache", type=pathlib.Path, default=repair.DEFAULT_TRAIN_SCORE_CACHE)
    parser.add_argument("--train-hidden-cache", type=pathlib.Path, default=repair.DEFAULT_TRAIN_HIDDEN_CACHE)
    parser.add_argument("--eval-hidden-cache", type=pathlib.Path, default=repair.DEFAULT_EVAL_HIDDEN_CACHE)
    parser.add_argument("--train-hidden-rows", type=int, default=512)
    parser.add_argument("--selection-seed", type=int, default=1729)
    parser.add_argument("--split-seeds", type=_parse_seeds, default=DEFAULT_SPLIT_SEEDS)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--run-date", default="2026-05-01")
    args = parser.parse_args()

    payload = build_gate(
        output_dir=args.output_dir,
        train_path=args.train_path,
        eval_path=args.eval_path,
        eval_score_cache=args.eval_score_cache,
        train_score_cache=args.train_score_cache,
        train_hidden_cache=args.train_hidden_cache,
        eval_hidden_cache=args.eval_hidden_cache,
        train_hidden_rows=args.train_hidden_rows,
        selection_seed=args.selection_seed,
        split_seeds=args.split_seeds,
        bootstrap_samples=args.bootstrap_samples,
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
