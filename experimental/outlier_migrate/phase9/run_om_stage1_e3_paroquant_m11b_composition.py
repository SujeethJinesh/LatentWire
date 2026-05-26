#!/usr/bin/env python3
"""Run Stage 1 E3 ParoQuant plus M11b composition."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experimental.outlier_migrate.phase4 import run_om_phase4_intervention as phase4_runner
from experimental.outlier_migrate.phase9 import check_om_stage1_e3_paroquant_m11b_composition as checker
from experimental.outlier_migrate.phase9 import run_om_paroquant_baseline as paro_runner
from experimental.outlier_migrate.phase9 import run_om_phase9_m11_ema_drift as m11_runner
from experimental.shared import run_phase0_branch as shared


SCHEMA_VERSION = checker.SCHEMA_VERSION
DEFAULT_RESULTS_DIR = checker.RESULTS_DIR
GRANITE_M11B_DIR = ROOT / "experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z"
GRANITE_PARO_DIR = ROOT / "experimental/outlier_migrate/phase9/results/om_paroquant_granite_small_20260520T1555Z"
NEMOTRON_M11B_DIR = ROOT / "experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z"


def write_failure_packet(run_dir: Path, run_events_path: Path, exc: BaseException | str) -> None:
    payload: dict[str, Any] = {
        "schema_version": f"{SCHEMA_VERSION}_infra_error",
        "created_at_utc": shared.utc_now(),
        "decision": checker.FAIL_INFRA,
        "reason": str(exc),
    }
    if isinstance(exc, BaseException):
        payload["exception_type"] = type(exc).__name__
        payload["traceback"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    try:
        run_events_path.open("a", encoding="utf-8").write(
            json.dumps({"created_at_utc": shared.utc_now(), "event": "run_failed", "reason": str(exc)}, sort_keys=True) + "\n"
        )
        shared.write_json(run_dir / "infra_error.json", payload)
        shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
        checker.evaluate(run_dir)
    except Exception:
        pass


def release_model_memory(*objects: Any) -> None:
    for obj in objects:
        del obj
    phase4_runner.release_model_memory()


def read_score_cache(run_dir: Path, regime: str, expected_prompt_indices: set[int]) -> dict[int, dict[str, float]]:
    scores = m11_runner.read_score_cache_any(run_dir, regime, expected_prompt_indices=expected_prompt_indices)
    if scores is None:
        raise RuntimeError(f"missing reusable score cache {regime} in {run_dir}")
    return scores


def write_score_cache(run_dir: Path, regime: str, scores: dict[int, dict[str, float]]) -> None:
    shared.write_json(
        run_dir / "score_cache" / f"{regime}.json",
        {
            "schema_version": f"{SCHEMA_VERSION}_score_cache",
            "created_at_utc": shared.utc_now(),
            "regime": regime,
            "scores": {str(index): row for index, row in sorted(scores.items())},
        },
    )


def make_random_top10_sets(protected_sets: dict[str, Any], seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    layers: dict[str, Any] = {}
    for layer_index, layer in sorted(protected_sets["regimes"]["m11b_top10"]["layers"].items(), key=lambda item: int(item[0])):
        channel_count = int(layer["channel_count"])
        count = int(layer["protected_count"])
        channels = sorted(rng.sample(range(channel_count), count))
        layers[layer_index] = {
            "layer_name": layer["layer_name"],
            "channel_count": channel_count,
            "protected_count": count,
            "protected_channels": channels,
            "source": "seeded_random_matched_to_m11b_top10_count",
        }
    return {
        "schema_version": f"{SCHEMA_VERSION}_random_protected_sets",
        "created_at_utc": shared.utc_now(),
        "seed": seed,
        "regimes": {"paroquant_random_top10": {"kind": "random_matched_top10", "layers": layers}},
    }


def protected_for_regime(protected_sets: dict[str, Any], regime: str) -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA_VERSION}_protected_sets_subset",
        "created_at_utc": shared.utc_now(),
        "regimes": {regime: protected_sets["regimes"][regime]},
    }


def quantize_paroquant_with_protection(
    model: Any,
    protected_sets: dict[str, Any],
    regime: str,
    *,
    group_size: int,
    num_rotations: int,
    row_chunk: int,
    scale_clip: tuple[float, float],
) -> dict[str, Any]:
    import torch

    hidden_size = int(getattr(model.config, "hidden_size"))
    layers, layer_origin = shared.discover_transformer_layers(model)
    protected_by_layer = protected_sets["regimes"][regime]["layers"]
    quantized: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    processed_module_ids: set[int] = set()

    def apply_one(name: str, module: Any, protected: set[int], source: str) -> None:
        weight = getattr(module, "weight", None)
        if not torch.is_tensor(weight):
            return
        if weight.ndim not in {2, 3}:
            excluded.append({"name": name, "reason": "weight tensor is not 2D or expert-bank 3D", "shape": list(weight.shape)})
            return
        saved_rows = saved_cols = rows = cols = None
        if weight.ndim == 2:
            row_protected = weight.shape[0] == hidden_size
            col_protected = weight.shape[1] == hidden_size
            rows = torch.tensor(sorted(protected), device=weight.device, dtype=torch.long) if row_protected and protected else None
            cols = torch.tensor(sorted(protected), device=weight.device, dtype=torch.long) if col_protected and protected else None
            saved_rows = weight.index_select(0, rows).clone() if rows is not None else None
            saved_cols = weight.index_select(1, cols).clone() if cols is not None else None
        else:
            row_protected = weight.shape[1] == hidden_size
            col_protected = weight.shape[2] == hidden_size
            rows = torch.tensor(sorted(protected), device=weight.device, dtype=torch.long) if row_protected and protected else None
            cols = torch.tensor(sorted(protected), device=weight.device, dtype=torch.long) if col_protected and protected else None
            saved_rows = weight.index_select(1, rows).clone() if rows is not None else None
            saved_cols = weight.index_select(2, cols).clone() if cols is not None else None
        item = paro_runner.quantize_weight_paroquant_inplace(
            weight,
            group_size=group_size,
            num_rotations=num_rotations,
            row_chunk=row_chunk,
            scale_clip=scale_clip,
        )
        if saved_rows is not None:
            if weight.ndim == 2:
                weight.index_copy_(0, rows, saved_rows)
            else:
                weight.index_copy_(1, rows, saved_rows)
        if saved_cols is not None:
            if weight.ndim == 2:
                weight.index_copy_(1, cols, saved_cols)
            else:
                weight.index_copy_(2, cols, saved_cols)
        item.update(
            {
                "name": name,
                "row_protected": bool(row_protected),
                "col_protected": bool(col_protected),
                "protected_channel_count": len(protected) if (row_protected or col_protected) else 0,
                "protection_source": source,
            }
        )
        if "reason" in item:
            excluded.append(item)
        else:
            quantized.append(item)

    with torch.no_grad():
        for layer_index, (layer_name, layer) in enumerate(layers):
            protected = set(int(ch) for ch in protected_by_layer[str(layer_index)]["protected_channels"])
            for module_name, module in layer.named_modules():
                full_name = f"{layer_name}.{module_name}" if module_name else layer_name
                apply_one(full_name, module, protected, f"layer_{layer_index}")
                processed_module_ids.add(id(module))
        first = set(int(ch) for ch in protected_by_layer["0"]["protected_channels"])
        last_key = str(max(int(key) for key in protected_by_layer))
        last = set(int(ch) for ch in protected_by_layer[last_key]["protected_channels"])
        for name, module in model.named_modules():
            if id(module) in processed_module_ids:
                continue
            weight = getattr(module, "weight", None)
            if not torch.is_tensor(weight):
                continue
            if phase4_runner.is_tied_lm_head(model, name, module):
                excluded.append({"name": name, "reason": "tied input/output embedding excluded", "shape": list(weight.shape)})
                continue
            protected: set[int] = set()
            source = "outside_transformer_stack_no_hidden_axis"
            if weight.ndim == 2 and weight.shape[1] == hidden_size:
                protected.update(last)
                source = "outside_transformer_stack_hidden_input_uses_last_layer"
            if weight.ndim == 2 and weight.shape[0] == hidden_size:
                protected.update(first)
                source = "outside_transformer_stack_hidden_output_uses_first_layer"
            apply_one(name, module, protected, source)
    return {
        "regime": regime,
        "layer_origin": layer_origin,
        "quantized_tensor_count": len(quantized),
        "quantized_tensors": quantized,
        "excluded_tensors": excluded,
    }


def score_composed_regime(
    *,
    model_provenance: dict[str, Any],
    protected_sets: dict[str, Any],
    regime: str,
    prompts: list[dict[str, Any]],
    target_tokens: dict[int, list[int]],
    dtype_name: str,
    device_name: str,
    batch_size: int,
    run_events_path: Path,
    group_size: int,
    num_rotations: int,
    row_chunk: int,
    scale_clip: tuple[float, float],
) -> tuple[dict[int, dict[str, float]], dict[str, Any]]:
    model, tokenizer, device = shared.load_model_and_tokenizer(model_provenance, dtype_name=dtype_name, device_name=device_name)
    excluded = quantize_paroquant_with_protection(
        model,
        protected_sets,
        regime,
        group_size=group_size,
        num_rotations=num_rotations,
        row_chunk=row_chunk,
        scale_clip=scale_clip,
    )
    scores = phase4_runner.score_targets(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompts=prompts,
        target_tokens=target_tokens,
        max_new_tokens=10000,
        batch_size=batch_size,
        use_float16_autocast=True,
        run_events_path=run_events_path,
        regime_name=regime,
    )
    del model, tokenizer, device
    release_model_memory()
    return scores, excluded


def bootstrap_median_30seed(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"ci95_low": None, "ci95_high": None, "bootstrap_seed_count": 30}
    boot: list[float] = []
    seeds = list(range(20260610, 20260640))
    for seed in seeds:
        rng = random.Random(seed)
        for _ in range(1000):
            sample = [values[rng.randrange(len(values))] for _ in values]
            boot.append(float(median(sample)))
    boot.sort()
    return {
        "ci95_low": boot[int(0.025 * (len(boot) - 1))],
        "ci95_high": boot[int(0.975 * (len(boot) - 1))],
        "bootstrap_seed_count": len(seeds),
        "bootstrap_samples_per_seed": 1000,
        "bootstrap_seeds": seeds,
    }


def summarize(values: list[float]) -> dict[str, Any]:
    return {
        "median_recovery": float(median(values)) if values else None,
        "mean_recovery": float(mean(values)) if values else None,
        "bootstrap_ci95": bootstrap_median_30seed(values),
        "included_trace_count": len(values),
        "per_trace_recovery_included": values,
    }


def build_per_trace_rows(prompts: list[dict[str, Any]], all_scores: dict[str, dict[int, dict[str, float]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prompt in prompts:
        index = int(prompt["index"])
        perplexities = {regime: float(all_scores[regime][index]["perplexity"]) for regime in checker.REGIMES}
        mean_nll = {regime: float(all_scores[regime][index]["mean_nll"]) for regime in checker.REGIMES}
        static_gap = perplexities["static_1pct"] - perplexities["bf16"]
        no_gap = static_gap <= 0.0
        recoveries = {
            regime: None if no_gap else 1.0 - (perplexities[regime] - perplexities["bf16"]) / static_gap
            for regime in checker.RECOVERY_REGIMES
        }
        rows.append(
            {
                "prompt_index": index,
                "prompt_id": prompt["prompt_id"],
                "perplexities": perplexities,
                "mean_nll": mean_nll,
                "static_gap": float(static_gap),
                "no_recoverable_static_gap": bool(no_gap),
                "recoveries": recoveries,
                "scored_tokens": int(all_scores["bf16"][index]["scored_tokens"]),
                "score_start": int(all_scores["bf16"][index]["score_start"]),
                "score_end": int(all_scores["bf16"][index]["score_end"]),
            }
        )
    return rows


def build_metrics(run_dir: Path, model_provenance: dict[str, Any], prompt_manifest: dict[str, Any], rows: list[dict[str, Any]], nemotron_deferred: bool) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    included = [row for row in rows if not row["no_recoverable_static_gap"]]
    results = {
        regime: summarize([float(row["recoveries"][regime]) for row in included])
        for regime in checker.RECOVERY_REGIMES
    }
    for summary in results.values():
        summary["total_trace_count"] = len(rows)
        summary["no_recoverable_static_gap_count"] = len(rows) - len(included)
        summary["no_recoverable_static_gap_fraction"] = (len(rows) - len(included)) / len(rows) if rows else 0.0
    comp = results["paroquant_m11b_top10"]["median_recovery"]
    paro = results["paroquant_w4a16"]["median_recovery"]
    m11b = results["m11b_top10"]["median_recovery"]
    random_control = results["paroquant_random_top10"]["median_recovery"]
    headline = {
        "composition_minus_best_individual": None if comp is None else float(comp) - max(float(paro), float(m11b)),
        "composition_minus_paroquant": None if comp is None else float(comp) - float(paro),
        "composition_minus_m11b_top10": None if comp is None else float(comp) - float(m11b),
        "composition_minus_random_control": None if comp is None else float(comp) - float(random_control),
        "nemotron_deferred_for_throughput": nemotron_deferred,
    }
    metrics = {
        "schema_version": f"{SCHEMA_VERSION}_metrics",
        "created_at_utc": shared.utc_now(),
        "preregistration": str(checker.PREREG_PATH.relative_to(ROOT)),
        "preregistration_sha256": shared.file_sha256(checker.PREREG_PATH),
        "model_id": model_provenance.get("model_id"),
        "model_snapshot_commit": model_provenance.get("hf_snapshot_commit"),
        "prompt_sha256": prompt_manifest["prompt_sha256"],
        "trace_count": len(rows),
        "included_trace_count": len(included),
        "scoring_position": 10000,
        "scoring_window_tokens": 512,
        "metric_name": "positive-static-1pct-gap per-trace recovery",
        "metric_formula": "1 - (perplexity_regime - perplexity_BF16) / (perplexity_static_1pct - perplexity_BF16)",
        "results_by_regime": results,
        "headline": headline,
        "artifacts": {"run_dir": str(run_dir)},
    }
    bootstrap = {
        "schema_version": f"{SCHEMA_VERSION}_bootstrap_ci",
        "metric_name": metrics["metric_name"],
        "bootstrap": "30 fixed seeds x 1000 resamples per seed",
        "results_by_regime": results,
    }
    controls = {
        "schema_version": f"{SCHEMA_VERSION}_control_metrics",
        "created_at_utc": shared.utc_now(),
        "controls": {
            "static_1pct": {"median_recovery": 0.0},
            "paroquant_random_top10": results["paroquant_random_top10"],
        },
        "headline": headline,
    }
    return metrics, bootstrap, controls


def main(argv: list[str] | None = None) -> int:
    shared.SCHEMA_VERSION = SCHEMA_VERSION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"om_stage1_e3_granite_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--num-rotations", type=int, default=8)
    parser.add_argument("--row-chunk", type=int, default=256)
    parser.add_argument("--scale-clip-min", type=float, default=0.25)
    parser.add_argument("--scale-clip-max", type=float, default=4.0)
    parser.add_argument("--cap-hours", type=float, default=15.0)
    args = parser.parse_args(argv)

    started = time.monotonic()
    run_dir = args.results_dir / args.run_id
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}")
    (run_dir / "logs").mkdir(parents=True)
    stdout_log = (run_dir / "logs/stdout.log").open("w", encoding="utf-8", buffering=1)
    stderr_log = (run_dir / "logs/stderr.log").open("w", encoding="utf-8", buffering=1)
    sys.stdout = shared.Tee(sys.__stdout__, stdout_log)
    sys.stderr = shared.Tee(sys.__stderr__, stderr_log)
    run_events_path = run_dir / "run_events.jsonl"
    run_events_path.write_text(json.dumps({"created_at_utc": shared.utc_now(), "event": "run_started"}, sort_keys=True) + "\n", encoding="utf-8")
    random.seed(args.seed)

    try:
        model_provenance = json.loads((GRANITE_M11B_DIR / "model_provenance.json").read_text(encoding="utf-8"))
        prompt_manifest = json.loads((GRANITE_M11B_DIR / "prompt_manifest.json").read_text(encoding="utf-8"))
        prompts = prompt_manifest["prompts"]
        prompt_indices = {int(row["index"]) for row in prompts}
        protected_sets = json.loads((GRANITE_M11B_DIR / "protected_sets.json").read_text(encoding="utf-8"))
        random_sets = make_random_top10_sets(protected_sets, args.seed)
        composition_sets = protected_for_regime(protected_sets, "m11b_top10")
        composition_sets["regimes"]["paroquant_m11b_top10"] = composition_sets["regimes"].pop("m11b_top10")
        combined_sets = {
            "schema_version": f"{SCHEMA_VERSION}_protected_sets",
            "created_at_utc": shared.utc_now(),
            "source_m11b": str(GRANITE_M11B_DIR / "protected_sets.json"),
            "regimes": {
                "paroquant_m11b_top10": composition_sets["regimes"]["paroquant_m11b_top10"],
                "paroquant_random_top10": random_sets["regimes"]["paroquant_random_top10"],
            },
        }
        trace_path = run_dir / "bf16_traces.jsonl.gz"
        m11_runner.m2_runner.copy_filtered_jsonl_gz(GRANITE_M11B_DIR / "bf16_traces.jsonl.gz", trace_path, prompt_indices=prompt_indices)
        trace_manifest = json.loads((GRANITE_M11B_DIR / "bf16_trace_manifest.json").read_text(encoding="utf-8"))
        trace_manifest["created_at_utc"] = shared.utc_now()
        trace_manifest["source_run_dir"] = str(GRANITE_M11B_DIR)
        shared.write_json(run_dir / "bf16_trace_manifest.json", trace_manifest)
        target_tokens = phase4_runner.load_trace_tokens(trace_path)

        environment = shared.build_environment(schema_version=SCHEMA_VERSION)
        shared.write_json(run_dir / "environment.json", environment)
        phase4_runner.write_environment_text(run_dir / "environment.txt", environment)
        shared.write_json(run_dir / "model_provenance.json", model_provenance)
        shared.write_json(run_dir / "prompt_manifest.json", prompt_manifest)
        shared.write_json(run_dir / "protected_sets.json", combined_sets)
        scale_clip = (float(args.scale_clip_min), float(args.scale_clip_max))
        shared.write_json(
            run_dir / "command_metadata.json",
            {
                "schema_version": f"{SCHEMA_VERSION}_command",
                "created_at_utc": shared.utc_now(),
                "argv": sys.argv if argv is None else ["run_om_stage1_e3_paroquant_m11b_composition.py", *argv],
                "cwd": str(ROOT),
                "run_dir": str(run_dir),
                "cap_hours": args.cap_hours,
                "scope": "granite_first_nemotron_deferred_if_cap_infeasible",
            },
        )
        shared.write_json(run_dir / "random_seed.json", {"schema_version": f"{SCHEMA_VERSION}_random_seed", "seed": args.seed})
        shared.write_json(
            run_dir / "decoding_config.json",
            {
                "schema_version": f"{SCHEMA_VERSION}_decoding_config",
                "max_new_tokens_for_scoring": 10000,
                "scoring_position": 10000,
                "scoring_window_tokens": 512,
                "target_trace": "BF16 deterministic greedy trace reused from M11b Granite packet",
            },
        )
        shared.write_json(
            run_dir / "quantization_config.json",
            {
                "schema_version": f"{SCHEMA_VERSION}_quantization_config",
                "scheme": "ParoQuant-style folded rotation plus groupwise affine INT4 with protected-channel BF16 restoration",
                "group_size": args.group_size,
                "num_rotations": args.num_rotations,
                "scale_clip": list(scale_clip),
            },
        )
        shared.write_json(
            run_dir / "source_artifacts.json",
            {
                "schema_version": f"{SCHEMA_VERSION}_source_artifacts",
                "created_at_utc": shared.utc_now(),
                "m11b_run_dir": str(GRANITE_M11B_DIR),
                "paroquant_run_dir": str(GRANITE_PARO_DIR),
                "nemotron_m11b_run_dir": str(NEMOTRON_M11B_DIR),
                "nemotron_deferred_reason": "Prior E1 throughput probe projected Nemotron manual decode to exceed the 15 GPU-hour E3 cap.",
            },
        )

        all_scores: dict[str, dict[int, dict[str, float]]] = {
            "bf16": read_score_cache(GRANITE_M11B_DIR, "bf16", prompt_indices),
            "static_1pct": read_score_cache(GRANITE_M11B_DIR, "static_1pct", prompt_indices),
            "m11b_top10": read_score_cache(GRANITE_M11B_DIR, "m11b_top10", prompt_indices),
            "paroquant_w4a16": read_score_cache(GRANITE_PARO_DIR, "paroquant_w4a16", prompt_indices),
        }
        for regime, scores in all_scores.items():
            write_score_cache(run_dir, regime, scores)

        excluded_by_regime: dict[str, Any] = {}
        for regime in ["paroquant_m11b_top10", "paroquant_random_top10"]:
            elapsed_hours = (time.monotonic() - started) / 3600.0
            if elapsed_hours >= args.cap_hours:
                raise RuntimeError(f"E3 cap reached before scoring {regime}: {elapsed_hours:.3f} hours")
            print(json.dumps({"event": "starting_e3_regime", "regime": regime, "elapsed_hours": elapsed_hours}, sort_keys=True))
            scores, excluded = score_composed_regime(
                model_provenance=model_provenance,
                protected_sets=combined_sets,
                regime=regime,
                prompts=prompts,
                target_tokens=target_tokens,
                dtype_name=args.dtype,
                device_name=args.device,
                batch_size=args.batch_size,
                run_events_path=run_events_path,
                group_size=args.group_size,
                num_rotations=args.num_rotations,
                row_chunk=args.row_chunk,
                scale_clip=scale_clip,
            )
            all_scores[regime] = scores
            excluded_by_regime[regime] = excluded
            write_score_cache(run_dir, regime, scores)

        rows = build_per_trace_rows(prompts, all_scores)
        shared.write_json(run_dir / "per_trace_metrics.json", {"schema_version": f"{SCHEMA_VERSION}_per_trace_metrics", "created_at_utc": shared.utc_now(), "traces": rows})
        metrics, bootstrap, controls = build_metrics(run_dir, model_provenance, prompt_manifest, rows, nemotron_deferred=True)
        shared.write_json(run_dir / "metrics.json", metrics)
        shared.write_json(run_dir / "bootstrap_ci.json", bootstrap)
        shared.write_json(run_dir / "control_metrics.json", controls)
        shared.write_json(run_dir / "excluded_tensors.json", {"schema_version": f"{SCHEMA_VERSION}_excluded_tensors", "created_at_utc": shared.utc_now(), "by_regime": excluded_by_regime})
        run_events_path.open("a", encoding="utf-8").write(json.dumps({"created_at_utc": shared.utc_now(), "event": "run_completed"}, sort_keys=True) + "\n")
        shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
        result = checker.evaluate(run_dir)
        print(json.dumps({"checker_decision": result["decision"], "headline": result.get("headline")}, indent=2, sort_keys=True))
        shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
        checker.evaluate(run_dir)
        return 0
    except BaseException as exc:
        write_failure_packet(run_dir, run_events_path, exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
