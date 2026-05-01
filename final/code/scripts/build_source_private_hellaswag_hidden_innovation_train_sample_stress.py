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
from scripts import build_source_private_hellaswag_hidden_innovation_stability_gate as stability  # noqa: E402
from scripts import build_source_private_hellaswag_hidden_summary_repair_probe as hidden_summary  # noqa: E402
from scripts import build_source_private_hellaswag_score_packet_headroom as headroom  # noqa: E402
from scripts import build_source_private_hellaswag_top2_contrastive_repair_probe as top2  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_train_sample_stress_20260501_qwen05_train512_validation1024"
)
DEFAULT_TRAIN_SAMPLE_SEEDS = (1729, 2027)
DEFAULT_SPLIT_SEEDS = (1729, 1731, 1733)
STRICT_DELTA = 0.02


def _cache_paths(output_dir: pathlib.Path, sample_seed: int) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    cache_dir = output_dir / "caches" / f"train_sample_seed_{sample_seed}"
    return (
        cache_dir / "source_train_score_cache.json",
        cache_dir / "source_train_hidden_cache.npz",
        cache_dir / "source_train_hidden_cache.json",
    )


def _sample_caches(
    *,
    output_dir: pathlib.Path,
    sample_seed: int,
    train_rows: list[arc_gate.ArcRow],
    source_lm_model: str,
    source_lm_device: str,
    source_lm_dtype: str,
    source_lm_max_length: int,
    source_lm_normalization: str,
    source_lm_prompt_mode: str,
    hidden_layers: tuple[int, ...],
    local_files_only: bool,
) -> tuple[list[list[float]], np.ndarray, dict[str, Any], dict[str, Any], pathlib.Path, pathlib.Path]:
    if sample_seed == 1729:
        default_score_cache = top2._resolve(repair.DEFAULT_TRAIN_SCORE_CACHE)
        default_hidden_cache = top2._resolve(repair.DEFAULT_TRAIN_HIDDEN_CACHE)
        try:
            train_scores, _, train_score_model = headroom._load_score_cache(default_score_cache, rows=train_rows)
            train_hidden, train_hidden_model = top2._load_hidden_cache(default_hidden_cache, rows=train_rows)
            return (
                train_scores,
                train_hidden,
                train_score_model | {"cache_path": top2._display_path(default_score_cache), "cache_hit": True},
                train_hidden_model | {"cache_path": top2._display_path(default_hidden_cache), "cache_hit": True},
                default_score_cache,
                default_hidden_cache,
            )
        except (FileNotFoundError, ValueError):
            pass

    score_cache, hidden_npz, hidden_meta = _cache_paths(output_dir, sample_seed)
    train_scores, _, train_score_model, _ = headroom._source_scores(
        rows=train_rows,
        score_cache=score_cache,
        source_lm_model=source_lm_model,
        source_lm_device=source_lm_device,
        source_lm_dtype=source_lm_dtype,
        source_lm_max_length=source_lm_max_length,
        source_lm_normalization=source_lm_normalization,
        source_lm_prompt_mode=source_lm_prompt_mode,
        local_files_only=local_files_only,
    )
    train_hidden, train_hidden_model = hidden_summary._source_hidden_features(
        train_rows,
        npz_path=hidden_npz,
        meta_path=hidden_meta,
        model_path=source_lm_model,
        device=source_lm_device,
        dtype=source_lm_dtype,
        max_length=source_lm_max_length,
        prompt_mode=source_lm_prompt_mode,
        layers=hidden_layers,
        local_files_only=local_files_only,
    )
    return train_scores, train_hidden, train_score_model, train_hidden_model, score_cache, hidden_npz


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    new_sample_rows = [row for row in rows if row["train_sample_seed"] != 1729]
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["train_sample_seed"]), []).append(row)
    sample_pass = {
        str(seed): all(row["pass_gate"] for row in sample_rows) for seed, sample_rows in grouped.items()
    }
    return {
        "train_sample_seed_count": len(grouped),
        "new_train_sample_seed_count": len({row["train_sample_seed"] for row in new_sample_rows}),
        "split_rows": len(rows),
        "pass_count": sum(1 for row in rows if row["pass_gate"]),
        "all_split_rows_pass": all(row["pass_gate"] for row in rows),
        "sample_pass": sample_pass,
        "new_sample_pass_count": sum(1 for key, value in sample_pass.items() if key != "1729" and value),
        "selected_view_counts": dict(Counter(row["selected_view"] for row in rows)),
        "selected_ridge_counts": {str(key): value for key, value in Counter(row["selected_ridge"] for row in rows).items()},
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
    lines = [
        "# HellaSwag Hidden-Innovation Train-Sample Stress",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- train sample seeds: `{h['train_sample_seed_count']}`",
        f"- new train sample seeds: `{h['new_train_sample_seed_count']}`",
        f"- split rows passing: `{h['pass_count']}/{h['split_rows']}`",
        f"- sample pass map: `{h['sample_pass']}`",
        f"- eval accuracy mean/min/max: `{h['selected_eval_accuracy_mean']:.6f}` / `{h['selected_eval_accuracy_min']:.6f}` / `{h['selected_eval_accuracy_max']:.6f}`",
        f"- delta vs best label-copy mean/min: `{h['delta_vs_best_label_copy_mean']:.6f}` / `{h['delta_vs_best_label_copy_min']:.6f}`",
        f"- min CI95 low vs best label-copy: `{h['paired_ci95_low_vs_best_label_copy_min']:.6f}`",
        f"- min delta vs zero-hidden: `{h['selected_minus_zero_hidden_control_min']:.6f}`",
        f"- selected view counts: `{h['selected_view_counts']}`",
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
    eval_hidden_cache: pathlib.Path = repair.DEFAULT_EVAL_HIDDEN_CACHE,
    train_hidden_rows: int = 512,
    train_sample_seeds: tuple[int, ...] = DEFAULT_TRAIN_SAMPLE_SEEDS,
    split_seeds: tuple[int, ...] = DEFAULT_SPLIT_SEEDS,
    dev_fraction: float = 0.25,
    ridges: tuple[float, ...] = (100.0, 1000.0, 10000.0, 100000.0),
    bootstrap_samples: int = 500,
    source_lm_model: str = "/Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
    source_lm_device: str = "auto_cpu",
    source_lm_dtype: str = "float32",
    source_lm_max_length: int = 256,
    source_lm_normalization: str = "mean",
    source_lm_prompt_mode: str = "continuation",
    hidden_layers: tuple[int, ...] = (-1,),
    local_files_only: bool = True,
    run_date: str = "2026-05-01",
) -> dict[str, Any]:
    output_dir = top2._resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    train_path = top2._resolve(train_path)
    eval_path = top2._resolve(eval_path)
    eval_score_cache = top2._resolve(eval_score_cache)
    eval_hidden_cache = top2._resolve(eval_hidden_cache)

    all_train_rows = arc_gate._load_rows(train_path)
    eval_rows = arc_gate._load_rows(eval_path)
    eval_scores, _, eval_source_model = headroom._load_score_cache(eval_score_cache, rows=eval_rows)
    eval_hidden, eval_hidden_model = top2._load_hidden_cache(eval_hidden_cache, rows=eval_rows)
    views = ("score_only", "score_hidden_residual")
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

    stress_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    sample_cache_rows: list[dict[str, Any]] = []
    for sample_seed in train_sample_seeds:
        train_rows = top2._select_train_rows(all_train_rows, count=train_hidden_rows, seed=sample_seed)
        train_scores, train_hidden, train_score_model, train_hidden_model, score_cache, hidden_cache = _sample_caches(
            output_dir=output_dir,
            sample_seed=sample_seed,
            train_rows=train_rows,
            source_lm_model=source_lm_model,
            source_lm_device=source_lm_device,
            source_lm_dtype=source_lm_dtype,
            source_lm_max_length=source_lm_max_length,
            source_lm_normalization=source_lm_normalization,
            source_lm_prompt_mode=source_lm_prompt_mode,
            hidden_layers=hidden_layers,
            local_files_only=local_files_only,
        )
        sample_cache_rows.append(
            {
                "train_sample_seed": sample_seed,
                "train_rows": len(train_rows),
                "content_digest": headroom._content_digest(train_rows),
                "train_score_cache": top2._display_path(score_cache),
                "train_hidden_cache": top2._display_path(hidden_cache),
                "train_score_cache_sha256": top2._sha256_file(score_cache),
                "train_hidden_cache_sha256": top2._sha256_file(hidden_cache),
                "train_score_cache_hit": bool(train_score_model.get("cache_hit")),
                "train_hidden_cache_hit": bool(train_hidden_model.get("cache_hit")),
            }
        )
        train_features = {
            view: repair._candidate_feature_tensor(scores=train_scores, hidden=train_hidden, view=view)
            for view in views
        }
        for split_seed in split_seeds:
            result = stability._fit_split(
                selection_policy="anchored_score_hidden_residual_train_sample_stress",
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
            stress_rows.append({"train_sample_seed": sample_seed, **result["row"]})
            for candidate in result["candidate_readouts"]:
                candidate_rows.append({"train_sample_seed": sample_seed, "split_seed": split_seed, **candidate})

    headline = _aggregate(stress_rows)
    pass_gate = bool(
        headline["new_train_sample_seed_count"] >= 1
        and headline["all_split_rows_pass"]
        and headline["delta_vs_best_label_copy_min"] >= STRICT_DELTA
        and headline["paired_ci95_low_vs_best_label_copy_min"] > 0.0
        and headline["selected_minus_zero_hidden_control_min"] >= STRICT_DELTA
    )
    payload = {
        "gate": "source_private_hellaswag_hidden_innovation_train_sample_stress",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": pass_gate,
        "pass_rule": (
            "Pass if at least one new train-row sample seed is included and every anchored score+hidden-residual "
            "sample/split row clears delta >= 0.02 versus best source/trained label-copy, paired CI95 low > 0, "
            "zero-hidden delta >= 0.02, and wrong-example/candidate-roll controls do not beat label-copy."
        ),
        "train_path": top2._display_path(train_path),
        "train_sha256": top2._sha256_file(train_path),
        "eval_path": top2._display_path(eval_path),
        "eval_sha256": top2._sha256_file(eval_path),
        "eval_score_cache": top2._display_path(eval_score_cache),
        "eval_score_cache_sha256": top2._sha256_file(eval_score_cache),
        "eval_hidden_cache": top2._display_path(eval_hidden_cache),
        "eval_hidden_cache_sha256": top2._sha256_file(eval_hidden_cache),
        "train_hidden_rows": train_hidden_rows,
        "train_sample_seeds": list(train_sample_seeds),
        "split_seeds": list(split_seeds),
        "dev_fraction": dev_fraction,
        "ridges": list(ridges),
        "source_model": {
            "score_eval": eval_source_model,
            "hidden_eval": eval_hidden_model,
            "source_lm_model": source_lm_model,
            "source_lm_device": source_lm_device,
            "source_lm_dtype": source_lm_dtype,
            "source_lm_max_length": source_lm_max_length,
            "source_lm_normalization": source_lm_normalization,
            "source_lm_prompt_mode": source_lm_prompt_mode,
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
        "sample_caches": sample_cache_rows,
        "stress_rows": stress_rows,
        "interpretation": (
            "This gate redraws the HellaSwag train rows used to fit the anchored source-side hidden-innovation "
            "denoiser, while keeping the frozen validation-first1024 readout fixed. It tests whether the "
            "score+hidden-residual packet survives beyond the original cached 512-row train-hidden slice. "
            "It is still a Mac-local train-sample stress, not a full-validation or NVIDIA serving result."
        ),
        "timing": {"total_seconds": float(time.perf_counter() - started)},
    }
    json_path = output_dir / "hellaswag_hidden_innovation_train_sample_stress.json"
    csv_path = output_dir / "stress_rows.csv"
    cache_path = output_dir / "sample_caches.jsonl"
    candidate_path = output_dir / "candidate_readouts.jsonl"
    md_path = output_dir / "hellaswag_hidden_innovation_train_sample_stress.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(csv_path, stress_rows)
    _write_jsonl(cache_path, sample_cache_rows)
    _write_jsonl(candidate_path, candidate_rows)
    _write_markdown(md_path, payload)
    manifest = {
        "gate": payload["gate"],
        "created_utc": payload["created_utc"],
        "headline": headline,
        "files": [
            {"path": top2._display_path(path), "sha256": top2._sha256_file(path), "bytes": path.stat().st_size}
            for path in (json_path, md_path, csv_path, cache_path, candidate_path)
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return result


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    result = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one float is required")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-path", type=pathlib.Path, default=repair.DEFAULT_TRAIN)
    parser.add_argument("--eval-path", type=pathlib.Path, default=repair.DEFAULT_EVAL)
    parser.add_argument("--eval-score-cache", type=pathlib.Path, default=repair.DEFAULT_EVAL_SCORE_CACHE)
    parser.add_argument("--eval-hidden-cache", type=pathlib.Path, default=repair.DEFAULT_EVAL_HIDDEN_CACHE)
    parser.add_argument("--train-hidden-rows", type=int, default=512)
    parser.add_argument("--train-sample-seeds", type=_parse_int_tuple, default=DEFAULT_TRAIN_SAMPLE_SEEDS)
    parser.add_argument("--split-seeds", type=_parse_int_tuple, default=DEFAULT_SPLIT_SEEDS)
    parser.add_argument("--ridges", type=_parse_float_tuple, default=(100.0, 1000.0, 10000.0, 100000.0))
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--source-lm-model", default="/Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775")
    parser.add_argument("--source-lm-device", default="auto_cpu")
    parser.add_argument("--source-lm-dtype", default="float32")
    parser.add_argument("--source-lm-max-length", type=int, default=256)
    parser.add_argument("--source-lm-normalization", choices=("mean", "sum"), default="mean")
    parser.add_argument("--source-lm-prompt-mode", choices=("qa", "continuation", "generic_mcq"), default="continuation")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--run-date", default="2026-05-01")
    args = parser.parse_args()

    payload = build_gate(
        output_dir=args.output_dir,
        train_path=args.train_path,
        eval_path=args.eval_path,
        eval_score_cache=args.eval_score_cache,
        eval_hidden_cache=args.eval_hidden_cache,
        train_hidden_rows=args.train_hidden_rows,
        train_sample_seeds=args.train_sample_seeds,
        split_seeds=args.split_seeds,
        ridges=args.ridges,
        bootstrap_samples=args.bootstrap_samples,
        source_lm_model=args.source_lm_model,
        source_lm_device=args.source_lm_device,
        source_lm_dtype=args.source_lm_dtype,
        source_lm_max_length=args.source_lm_max_length,
        source_lm_normalization=args.source_lm_normalization,
        source_lm_prompt_mode=args.source_lm_prompt_mode,
        local_files_only=args.local_files_only,
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
