from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
import time
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_hellaswag_hidden_innovation_bagged_gate as bagged  # noqa: E402
from scripts import build_source_private_hellaswag_hidden_summary_repair_probe as hidden_summary  # noqa: E402
from scripts import build_source_private_hellaswag_score_packet_headroom as headroom  # noqa: E402
from scripts import build_source_private_hellaswag_top2_contrastive_repair_probe as top2  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation1024_2048"
)
DEFAULT_EVAL_FULL = pathlib.Path(
    "results/source_private_hellaswag_bridge_contract_20260501/official_splits/hellaswag_validation.jsonl"
)
DEFAULT_TRAIN_SAMPLE_CACHE_DIR = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_train_sample_stress_20260501_qwen05_train512_validation1024"
)
DEFAULT_TRAIN_SAMPLE_SEEDS = (1729, 2027, 2039)
DEFAULT_SPLIT_SEEDS = (1729, 1731, 1733)
STRICT_DELTA = 0.02


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _slice_jsonl(
    *,
    source_path: pathlib.Path,
    output_path: pathlib.Path,
    start: int,
    count: int,
) -> dict[str, Any]:
    if start < 0:
        raise ValueError("slice start must be non-negative")
    if count <= 0:
        raise ValueError("slice count must be positive")
    source_path = _resolve(source_path)
    output_path = _resolve(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected: list[str] = []
    total_rows = 0
    with source_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if line.strip():
                total_rows += 1
            if start <= index < start + count:
                if line.strip():
                    selected.append(line.rstrip("\n"))
    if len(selected) != count:
        raise ValueError(
            f"requested {count} rows from {source_path} at start {start}, got {len(selected)}"
        )
    output_path.write_text("\n".join(selected) + "\n", encoding="utf-8")
    return {
        "source_path": _display_path(source_path),
        "slice_path": _display_path(output_path),
        "source_total_rows": total_rows,
        "slice_start": start,
        "slice_end_exclusive": start + count,
        "slice_rows": len(selected),
        "slice_sha256": top2._sha256_file(output_path),
    }


def _cache_eval_source_state(
    *,
    rows: list[arc_gate.ArcRow],
    score_cache: pathlib.Path,
    hidden_npz: pathlib.Path,
    hidden_meta: pathlib.Path,
    source_lm_model: str,
    source_lm_device: str,
    source_lm_dtype: str,
    source_lm_max_length: int,
    source_lm_normalization: str,
    source_lm_prompt_mode: str,
    hidden_layers: tuple[int, ...],
    local_files_only: bool,
) -> dict[str, Any]:
    _, _, score_model, score_cache_sha256 = headroom._source_scores(
        rows=rows,
        score_cache=score_cache,
        source_lm_model=source_lm_model,
        source_lm_device=source_lm_device,
        source_lm_dtype=source_lm_dtype,
        source_lm_max_length=source_lm_max_length,
        source_lm_normalization=source_lm_normalization,
        source_lm_prompt_mode=source_lm_prompt_mode,
        local_files_only=local_files_only,
    )
    _, hidden_model = hidden_summary._source_hidden_features(
        rows,
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
    return {
        "score_cache": _display_path(score_cache),
        "score_cache_sha256": score_cache_sha256 or top2._sha256_file(score_cache),
        "score_cache_hit": bool(score_model.get("cache_hit")),
        "hidden_cache": _display_path(hidden_npz),
        "hidden_cache_meta": _display_path(hidden_meta),
        "hidden_cache_sha256": top2._sha256_file(hidden_npz),
        "hidden_cache_hit": bool(hidden_model.get("cache_hit")),
        "score_model": score_model,
        "hidden_model": hidden_model,
    }


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Hidden-Innovation Heldout Eval-Slice Stress",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- eval slice: `{h['eval_slice_start']}:{h['eval_slice_end_exclusive']}`",
        f"- eval rows: `{h['eval_rows']}`",
        f"- selected accuracy: `{h['selected_eval_accuracy']:.6f}`",
        f"- best label-copy accuracy: `{h['best_label_copy_eval_accuracy']:.6f}`",
        f"- delta vs best label-copy: `{h['selected_minus_best_label_copy']:.6f}`",
        f"- CI95 vs best label-copy: `[{h['paired_ci95_low_vs_best_label_copy']:.6f}, {h['paired_ci95_high_vs_best_label_copy']:.6f}]`",
        f"- score-only bagged control: `{h['score_only_bagged_control_accuracy']:.6f}`",
        f"- zero-hidden control: `{h['zero_hidden_control_accuracy']:.6f}`",
        f"- wrong-example hidden control: `{h['wrong_example_hidden_control_accuracy']:.6f}`",
        f"- candidate-roll hidden control: `{h['candidate_roll_hidden_control_accuracy']:.6f}`",
        f"- jackknife subbags passing: `{h['jackknife_pass_count']}/{h['jackknife_row_count']}`",
        f"- packet: `{h['raw_payload_bytes']}B` raw / `{h['framed_record_bytes']}B` framed",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_gate(
    *,
    output_dir: pathlib.Path = DEFAULT_OUTPUT,
    eval_full_path: pathlib.Path = DEFAULT_EVAL_FULL,
    eval_slice_start: int = 1024,
    eval_slice_rows: int = 1024,
    eval_score_cache: pathlib.Path | None = None,
    eval_hidden_cache: pathlib.Path | None = None,
    train_sample_cache_dir: pathlib.Path = DEFAULT_TRAIN_SAMPLE_CACHE_DIR,
    train_sample_seeds: tuple[int, ...] = DEFAULT_TRAIN_SAMPLE_SEEDS,
    split_seeds: tuple[int, ...] = DEFAULT_SPLIT_SEEDS,
    bootstrap_samples: int = 500,
    aggregation_policy: str = "mean_zscore",
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
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    slice_path = output_dir / (
        f"hellaswag_validation_rows_{eval_slice_start}_{eval_slice_start + eval_slice_rows}.jsonl"
    )
    slice_metadata = _slice_jsonl(
        source_path=eval_full_path,
        output_path=slice_path,
        start=eval_slice_start,
        count=eval_slice_rows,
    )
    eval_rows = arc_gate._load_rows(slice_path)
    score_cache = (
        _resolve(eval_score_cache)
        if eval_score_cache is not None
        else output_dir / "source_eval_score_cache.json"
    )
    hidden_npz = (
        _resolve(eval_hidden_cache)
        if eval_hidden_cache is not None
        else output_dir / "source_eval_hidden_cache.npz"
    )
    hidden_meta = hidden_npz.with_suffix(".json")
    eval_cache_metadata = _cache_eval_source_state(
        rows=eval_rows,
        score_cache=score_cache,
        hidden_npz=hidden_npz,
        hidden_meta=hidden_meta,
        source_lm_model=source_lm_model,
        source_lm_device=source_lm_device,
        source_lm_dtype=source_lm_dtype,
        source_lm_max_length=source_lm_max_length,
        source_lm_normalization=source_lm_normalization,
        source_lm_prompt_mode=source_lm_prompt_mode,
        hidden_layers=hidden_layers,
        local_files_only=local_files_only,
    )
    bagged_payload = bagged.build_gate(
        output_dir=output_dir / "bagged_gate",
        train_sample_cache_dir=train_sample_cache_dir,
        eval_path=slice_path,
        eval_score_cache=score_cache,
        eval_hidden_cache=hidden_npz,
        train_sample_seeds=train_sample_seeds,
        split_seeds=split_seeds,
        bootstrap_samples=bootstrap_samples,
        aggregation_policy=aggregation_policy,
        source_lm_model=source_lm_model,
        source_lm_device=source_lm_device,
        source_lm_dtype=source_lm_dtype,
        source_lm_max_length=source_lm_max_length,
        source_lm_normalization=source_lm_normalization,
        source_lm_prompt_mode=source_lm_prompt_mode,
        hidden_layers=hidden_layers,
        local_files_only=local_files_only,
        run_date=run_date,
    )
    h = bagged_payload["headline"]
    j = bagged_payload["jackknife_summary"]
    eval_slice_end = eval_slice_start + eval_slice_rows
    standard_sized_slice = eval_slice_start >= 1024 and eval_slice_rows >= 1024
    terminal_tail_slice = (
        eval_slice_start >= 1024
        and eval_slice_rows >= 512
        and eval_slice_end == int(slice_metadata["source_total_rows"])
    )
    pass_gate = bool(
        (standard_sized_slice or terminal_tail_slice)
        and bagged_payload["pass_gate"]
        and h["selected_minus_best_label_copy"] >= STRICT_DELTA
        and h["paired_ci95_low_vs_best_label_copy"] > 0.0
        and h["selected_minus_score_only_bagged_control"] >= STRICT_DELTA
        and h["paired_ci95_low_vs_score_only_bagged"] > 0.0
        and h["selected_minus_zero_hidden_control"] >= STRICT_DELTA
        and j["all_pass"]
    )
    headline = {
        "eval_slice_start": eval_slice_start,
        "eval_slice_end_exclusive": eval_slice_end,
        "eval_rows": eval_slice_rows,
        "standard_sized_slice": standard_sized_slice,
        "terminal_tail_slice": terminal_tail_slice,
        "selected_eval_accuracy": h["selected_eval_accuracy"],
        "source_label_copy_eval_accuracy": h["source_label_copy_eval_accuracy"],
        "trained_choice_bias_label_copy_eval_accuracy": h["trained_choice_bias_label_copy_eval_accuracy"],
        "best_label_copy_eval_accuracy": h["best_label_copy_eval_accuracy"],
        "selected_minus_best_label_copy": h["selected_minus_best_label_copy"],
        "paired_ci95_low_vs_best_label_copy": h["paired_ci95_low_vs_best_label_copy"],
        "paired_ci95_high_vs_best_label_copy": h["paired_ci95_high_vs_best_label_copy"],
        "score_only_bagged_control_accuracy": h["score_only_bagged_control_accuracy"],
        "selected_minus_score_only_bagged_control": h["selected_minus_score_only_bagged_control"],
        "paired_ci95_low_vs_score_only_bagged": h["paired_ci95_low_vs_score_only_bagged"],
        "zero_hidden_control_accuracy": h["zero_hidden_control_accuracy"],
        "selected_minus_zero_hidden_control": h["selected_minus_zero_hidden_control"],
        "wrong_example_hidden_control_accuracy": h["wrong_example_hidden_control_accuracy"],
        "candidate_roll_hidden_control_accuracy": h["candidate_roll_hidden_control_accuracy"],
        "jackknife_row_count": j["row_count"],
        "jackknife_pass_count": j["pass_count"],
        "jackknife_min_delta_vs_best_label_copy": j["selected_minus_best_label_copy_min"],
        "jackknife_min_ci95_low_vs_best_label_copy": j["paired_ci95_low_vs_best_label_copy_min"],
        "train_sample_seed_count": h["train_sample_seed_count"],
        "component_model_count": h["component_model_count"],
        "raw_payload_bytes": bagged_payload["packet_contract"]["raw_payload_bytes"],
        "framed_record_bytes": bagged_payload["packet_contract"]["framed_record_bytes"],
        "strict_delta_required": STRICT_DELTA,
    }
    payload = {
        "gate": "source_private_hellaswag_hidden_innovation_eval_slice_stress",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": pass_gate,
        "pass_rule": (
            "Pass if a frozen post-first1024 validation slice is either a standard >=1024-row "
            "slice or the terminal validation tail with >=512 rows ending at the validation split "
            "boundary, and passes the same bagged hidden-innovation gate: >=0.02 over best "
            "label-copy, score-only bag, and zero-hidden controls, paired CI95 lows > 0, all "
            "jackknife subbags pass, and no source text/KV/raw hidden/raw score payload is exposed."
        ),
        "slice_metadata": slice_metadata,
        "eval_cache_metadata": eval_cache_metadata,
        "bagged_gate_path": _display_path(output_dir / "bagged_gate" / "hellaswag_hidden_innovation_bagged_gate.json"),
        "headline": headline,
        "packet_contract": bagged_payload["packet_contract"],
        "bagged_gate": bagged_payload,
        "interpretation": (
            "This gate freezes the three-sample bagged hidden-innovation method and moves the evaluation "
            "off the repeatedly inspected validation-first1024 slice. It is the cheapest Mac-local "
            "falsification of slice overfitting before spending full-validation or NVIDIA systems compute."
        ),
        "timing": {"total_seconds": float(time.perf_counter() - started)},
    }
    json_path = output_dir / "hellaswag_hidden_innovation_eval_slice_stress.json"
    md_path = output_dir / "hellaswag_hidden_innovation_eval_slice_stress.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload)
    manifest_files = [
        json_path,
        md_path,
        slice_path,
        score_cache,
        hidden_npz,
        hidden_meta,
        output_dir / "bagged_gate" / "hellaswag_hidden_innovation_bagged_gate.json",
        output_dir / "bagged_gate" / "jackknife_rows.csv",
        output_dir / "bagged_gate" / "manifest.json",
    ]
    manifest = {
        "gate": payload["gate"],
        "created_utc": payload["created_utc"],
        "headline": headline,
        "files": [
            {"path": _display_path(path), "sha256": top2._sha256_file(path), "bytes": path.stat().st_size}
            for path in manifest_files
            if path.exists()
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-full-path", type=pathlib.Path, default=DEFAULT_EVAL_FULL)
    parser.add_argument("--eval-slice-start", type=int, default=1024)
    parser.add_argument("--eval-slice-rows", type=int, default=1024)
    parser.add_argument("--eval-score-cache", type=pathlib.Path, default=None)
    parser.add_argument("--eval-hidden-cache", type=pathlib.Path, default=None)
    parser.add_argument("--train-sample-cache-dir", type=pathlib.Path, default=DEFAULT_TRAIN_SAMPLE_CACHE_DIR)
    parser.add_argument("--train-sample-seeds", type=_parse_int_tuple, default=DEFAULT_TRAIN_SAMPLE_SEEDS)
    parser.add_argument("--split-seeds", type=_parse_int_tuple, default=DEFAULT_SPLIT_SEEDS)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument(
        "--aggregation-policy",
        choices=("mean_zscore", "vote", "mean_zscore_vote_on_score_agreement"),
        default="mean_zscore",
    )
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
        eval_full_path=args.eval_full_path,
        eval_slice_start=args.eval_slice_start,
        eval_slice_rows=args.eval_slice_rows,
        eval_score_cache=args.eval_score_cache,
        eval_hidden_cache=args.eval_hidden_cache,
        train_sample_cache_dir=args.train_sample_cache_dir,
        train_sample_seeds=args.train_sample_seeds,
        split_seeds=args.split_seeds,
        bootstrap_samples=args.bootstrap_samples,
        aggregation_policy=args.aggregation_policy,
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
