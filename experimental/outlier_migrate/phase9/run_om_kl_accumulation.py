#!/usr/bin/env python3
"""Run Phase 9 KL accumulation measurement."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experimental.outlier_migrate.phase4 import run_om_phase4_intervention as phase4_runner
from experimental.outlier_migrate.phase9 import check_om_kl_accumulation as checker
from experimental.shared import run_phase0_branch as shared


SCHEMA_VERSION = checker.SCHEMA_VERSION
DEFAULT_RESULTS_DIR = ROOT / "experimental/outlier_migrate/phase9/results"
DEFAULT_M11_RUN_DIR = DEFAULT_RESULTS_DIR / "om_phase9_m11_granite_small_vac12_20260516T010728Z"
DEFAULT_DECDEC_RUN_DIR = DEFAULT_RESULTS_DIR / "om_phase9_decdec_granite_small_vac12_20260517T141500Z"
DEFAULT_M26_RUN_DIR = DEFAULT_RESULTS_DIR / "om_phase9_m26_granite_small_vac12_20260518T203000Z"


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


def load_prompts(prompt_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    prompts = prompt_manifest.get("prompts", [])
    selected = [row for row in prompts if int(row["index"]) < checker.TRACE_COUNT]
    if [int(row["index"]) for row in selected] != list(range(checker.TRACE_COUNT)):
        raise RuntimeError("prompt manifest is not deterministic indices 0-11")
    return selected


def merge_protected_sets(*, m11_dir: Path, decdec_dir: Path, m26_dir: Path) -> dict[str, Any]:
    m11 = json.loads((m11_dir / "protected_sets.json").read_text(encoding="utf-8"))
    decdec = json.loads((decdec_dir / "protected_sets.json").read_text(encoding="utf-8"))
    m26 = json.loads((m26_dir / "protected_sets.json").read_text(encoding="utf-8"))
    return {
        "schema_version": f"{SCHEMA_VERSION}_protected_sets",
        "created_at_utc": shared.utc_now(),
        "source": "merged from completed static/M11/DecDEC packets",
        "regimes": {
            "static_1pct": m26["regimes"]["static_1pct"],
            "decdec_reactive_top1_proxy": decdec["regimes"]["decdec_reactive_top1_proxy"],
            "m11_alpha_0_5": m11["regimes"]["m11_alpha_0_5"],
        },
    }


def source_artifact_manifest(*, m11_dir: Path, decdec_dir: Path, m26_dir: Path) -> dict[str, Any]:
    sources = {
        "m11": m11_dir,
        "decdec": decdec_dir,
        "m26": m26_dir,
    }
    files = [
        "model_provenance.json",
        "prompt_manifest.json",
        "protected_sets.json",
        "bf16_traces.jsonl.gz",
        "bf16_trace_manifest.json",
        "artifact_check.json",
    ]
    payload: dict[str, Any] = {
        "schema_version": f"{SCHEMA_VERSION}_source_artifacts",
        "created_at_utc": shared.utc_now(),
        "sources": {},
    }
    for label, directory in sources.items():
        rows = []
        for rel in files:
            path = directory / rel
            if path.is_file():
                rows.append({"path": str(path), "sha256": shared.file_sha256(path), "bytes": path.stat().st_size})
        payload["sources"][label] = rows
    return payload


def ensure_bf16_traces(
    *,
    run_dir: Path,
    run_events_path: Path,
    model_provenance: dict[str, Any],
    prompts: list[dict[str, Any]],
    source_trace_dir: Path,
    max_new_tokens: int,
    batch_size: int,
    dtype_name: str,
    device_name: str,
) -> tuple[Path, dict[str, Any]]:
    source_manifest_path = source_trace_dir / "bf16_trace_manifest.json"
    source_trace_path = source_trace_dir / "bf16_traces.jsonl.gz"
    output_path = run_dir / "bf16_traces.jsonl.gz"
    if source_manifest_path.is_file() and source_trace_path.is_file():
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if int(source_manifest.get("max_new_tokens", 0)) >= max_new_tokens:
            # Keep the copied packet small and self-contained.
            with gzip.open(source_trace_path, "rt", encoding="utf-8") as src, gzip.open(output_path, "wt", encoding="utf-8") as dst:
                for line in src:
                    row = json.loads(line)
                    if int(row.get("prompt_index", -1)) < checker.TRACE_COUNT:
                        dst.write(json.dumps(row, sort_keys=True) + "\n")
            manifest = {
                **source_manifest,
                "artifact": "bf16_traces.jsonl.gz",
                "artifact_sha256": shared.file_sha256(output_path),
                "source_artifact": str(source_trace_path),
                "source_artifact_sha256": shared.file_sha256(source_trace_path),
                "trace_count": checker.TRACE_COUNT,
                "reuse_note": "copied from source packet with sufficient 20K target length",
            }
            shared.write_json(run_dir / "bf16_trace_manifest.json", manifest)
            return output_path, manifest

    model, tokenizer, device = shared.load_model_and_tokenizer(model_provenance, dtype_name=dtype_name, device_name=device_name)
    manifest = phase4_runner.generate_bf16_traces(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        output_path=output_path,
        run_events_path=run_events_path,
    )
    del model, tokenizer, device
    release_model_memory()
    shared.write_json(run_dir / "bf16_trace_manifest.json", manifest)
    return output_path, manifest


def kl_positions(mode: str) -> list[int]:
    if mode == "full":
        return list(range(1, checker.MAX_NEW_TOKENS + 1))
    if mode == "dense_grid":
        return checker.dense_grid_positions()
    raise ValueError(f"unknown position mode: {mode}")


def collect_log_probs_for_trace(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    prompt: dict[str, Any],
    target_tokens: list[int],
    positions: list[int],
    max_new_tokens: int,
    use_float16_autocast: bool,
    regime_name: str,
) -> dict[int, Any]:
    import torch

    autocast_enabled = bool(use_float16_autocast and torch.cuda.is_available())
    cache_dtype = torch.float16 if autocast_enabled else None
    normalize_cache_inputs, output_cache = phase4_runner.make_cache_helpers(model, cache_dtype=cache_dtype)
    previous_fast_path = phase4_runner.set_granite_fast_path_enabled(False) if autocast_enabled else None
    wanted = set(positions)
    out: dict[int, Any] = {}
    try:
        text = shared.make_prompt_text(str(prompt["prompt"]))
        encoded = tokenizer([text], padding=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        target_tensor = torch.tensor(target_tokens[:max_new_tokens], dtype=torch.long, device=device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=autocast_enabled):
            cache_position = torch.arange(input_ids.shape[1], device=device)
            model_inputs = model.prepare_inputs_for_generation(
                input_ids=input_ids,
                attention_mask=attention_mask,
                cache_position=cache_position,
                use_cache=True,
            )
            outputs = model(**normalize_cache_inputs(model_inputs))
            past_key_values = output_cache(outputs)
            if past_key_values is None:
                raise RuntimeError(f"model did not return cache for KL regime {regime_name}")
            logits = outputs.logits[:, -1, :]
            for decode_position in range(1, max_new_tokens + 1):
                current_target = target_tensor[decode_position - 1]
                if decode_position in wanted:
                    out[decode_position] = torch.log_softmax(logits.float().squeeze(0), dim=-1).cpu()
                if decode_position == max_new_tokens:
                    break
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype),
                    ],
                    dim=1,
                )
                cache_position = torch.tensor([attention_mask.shape[1] - 1], device=device, dtype=torch.long)
                model_inputs = model.prepare_inputs_for_generation(
                    input_ids=current_target.reshape(1, 1),
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    use_cache=True,
                )
                outputs = model(**normalize_cache_inputs(model_inputs))
                past_key_values = output_cache(outputs)
                if past_key_values is None:
                    raise RuntimeError(f"model dropped cache for {regime_name} at decode position {decode_position}")
                logits = outputs.logits[:, -1, :]
        return out
    finally:
        if previous_fast_path is not None:
            phase4_runner.set_granite_fast_path_enabled(bool(previous_fast_path))


def fit_line(xs: list[float], ys: list[float], transform: str) -> dict[str, float]:
    if transform == "linear":
        tx = xs
    elif transform == "sqrt":
        tx = [math.sqrt(x) for x in xs]
    else:
        p = float(transform)
        tx = [x**p for x in xs]
    x_bar = mean(tx)
    y_bar = mean(ys)
    denom = sum((x - x_bar) ** 2 for x in tx)
    slope = 0.0 if denom == 0.0 else sum((x - x_bar) * (y - y_bar) for x, y in zip(tx, ys)) / denom
    intercept = y_bar - slope * x_bar
    residual = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(tx, ys))
    return {"intercept": float(intercept), "slope": float(slope), "rss": float(residual)}


def fit_growth_models(summary_by_regime: dict[str, list[dict[str, float]]]) -> dict[str, Any]:
    regime_fits: dict[str, Any] = {}
    for regime, rows in summary_by_regime.items():
        if regime == "bf16_reference":
            continue
        xs = [float(row["decode_position"]) for row in rows]
        ys = [float(row["mean_kl"]) for row in rows]
        candidates: dict[str, dict[str, float]] = {
            "linear": fit_line(xs, ys, "linear"),
            "sublinear_sqrt": fit_line(xs, ys, "sqrt"),
        }
        for p in [1.1, 1.25, 1.5, 2.0]:
            candidates[f"superlinear_power_{p}"] = fit_line(xs, ys, str(p))
        best_name = min(candidates, key=lambda name: candidates[name]["rss"])
        centered = [y - mean(ys) for y in ys]
        denom = sum(value * value for value in centered[:-1])
        ar1 = None if denom == 0.0 else sum(a * b for a, b in zip(centered[:-1], centered[1:])) / denom
        if max(ys) < 1e-8:
            best_class = "flat"
        elif best_name == "sublinear_sqrt":
            best_class = "sublinear"
        elif best_name == "linear":
            best_class = "linear"
        else:
            best_class = "superlinear"
        regime_fits[regime] = {
            "best_fit": best_name,
            "best_fit_class": best_class,
            "fits": candidates,
            "ar1_decay_estimate": None if ar1 is None else float(ar1),
            "mean_kl_first_position": ys[0],
            "mean_kl_last_position": ys[-1],
            "mean_kl_median": float(median(ys)),
        }
    return {
        "schema_version": f"{SCHEMA_VERSION}_growth_model_fits",
        "created_at_utc": shared.utc_now(),
        "fit_note": "Least-squares fits over trace-mean KL curve; superlinear class selected from fixed p grid.",
        "regime_fits": regime_fits,
    }


def write_kl_summary(run_dir: Path, positions: list[int], rows_by_regime_position: dict[str, dict[int, list[float]]]) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_by_regime: dict[str, list[dict[str, float]]] = {}
    regime_summary: dict[str, Any] = {}
    for regime in checker.REGIMES:
        curve: list[dict[str, float]] = []
        all_values: list[float] = []
        for position in positions:
            values = rows_by_regime_position[regime][position]
            curve.append(
                {
                    "decode_position": int(position),
                    "mean_kl": float(mean(values)),
                    "median_kl": float(median(values)),
                    "max_kl": float(max(values)),
                }
            )
            all_values.extend(values)
        summary_by_regime[regime] = curve
        regime_summary[regime] = {
            "mean_kl_all_positions_traces": float(mean(all_values)) if all_values else None,
            "median_kl_all_positions_traces": float(median(all_values)) if all_values else None,
            "max_kl_all_positions_traces": float(max(all_values)) if all_values else None,
            "position_count": len(positions),
            "trace_count": checker.TRACE_COUNT,
        }
    summary = {
        "schema_version": f"{SCHEMA_VERSION}_kl_summary",
        "created_at_utc": shared.utc_now(),
        "regime_summary": regime_summary,
        "trace_mean_curves": summary_by_regime,
    }
    fits = fit_growth_models(summary_by_regime)
    shared.write_json(run_dir / "kl_summary.json", summary)
    shared.write_json(run_dir / "growth_model_fits.json", fits)
    return summary, fits


def main(argv: list[str] | None = None) -> int:
    shared.SCHEMA_VERSION = SCHEMA_VERSION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"om_phase9_kl_granite_small_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--m11-run-dir", type=Path, default=DEFAULT_M11_RUN_DIR)
    parser.add_argument("--decdec-run-dir", type=Path, default=DEFAULT_DECDEC_RUN_DIR)
    parser.add_argument("--m26-run-dir", type=Path, default=DEFAULT_M26_RUN_DIR)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--position-mode", choices=["full", "dense_grid"], default="dense_grid")
    args = parser.parse_args(argv)

    run_dir = args.results_dir / args.run_id
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}")
    (run_dir / "logs").mkdir(parents=True)
    stdout_log = (run_dir / "logs/stdout.log").open("w", encoding="utf-8", buffering=1)
    stderr_log = (run_dir / "logs/stderr.log").open("w", encoding="utf-8", buffering=1)
    previous_excepthook = sys.excepthook
    sys.stdout = shared.Tee(sys.__stdout__, stdout_log)
    sys.stderr = shared.Tee(sys.__stderr__, stderr_log)
    run_events_path = run_dir / "run_events.jsonl"
    run_events_path.write_text(json.dumps({"created_at_utc": shared.utc_now(), "event": "run_started"}, sort_keys=True) + "\n", encoding="utf-8")

    try:
        model_provenance = json.loads((args.m26_run_dir / "model_provenance.json").read_text(encoding="utf-8"))
        if model_provenance.get("model_id") != checker.MODEL_ID or model_provenance.get("hf_snapshot_commit") != checker.MODEL_SNAPSHOT:
            raise RuntimeError("KL accumulation requires the Granite-Small Phase-1 snapshot")
        prompt_manifest = json.loads((args.m26_run_dir / "prompt_manifest.json").read_text(encoding="utf-8"))
        prompts = load_prompts(prompt_manifest)
        positions = kl_positions(args.position_mode)
        dense_grid_fallback = args.position_mode == "dense_grid"

        shared.write_json(run_dir / "model_provenance.json", model_provenance)
        shared.write_json(run_dir / "prompt_manifest.json", {**prompt_manifest, "prompt_count": checker.TRACE_COUNT, "prompts": prompts})
        shared.write_json(
            run_dir / "command_metadata.json",
            {
                "schema_version": f"{SCHEMA_VERSION}_command",
                "created_at_utc": shared.utc_now(),
                "argv": sys.argv[1:],
                "cwd": str(ROOT),
                "m11_run_dir": str(args.m11_run_dir),
                "decdec_run_dir": str(args.decdec_run_dir),
                "m26_run_dir": str(args.m26_run_dir),
                "position_mode": args.position_mode,
            },
        )
        shared.write_json(run_dir / "random_seed.json", {"schema_version": f"{SCHEMA_VERSION}_random_seed", "seed": 20260602})
        shared.write_json(
            run_dir / "decoding_config.json",
            {
                "schema_version": f"{SCHEMA_VERSION}_decoding_config",
                "decode_policy": "shared-prefix teacher-forced KL along deterministic BF16 target trace",
                "max_new_tokens": checker.MAX_NEW_TOKENS,
                "position_mode": args.position_mode,
            },
        )
        shared.write_json(run_dir / "source_artifacts.json", source_artifact_manifest(m11_dir=args.m11_run_dir, decdec_dir=args.decdec_run_dir, m26_dir=args.m26_run_dir))
        protected_sets = merge_protected_sets(m11_dir=args.m11_run_dir, decdec_dir=args.decdec_run_dir, m26_dir=args.m26_run_dir)
        shared.write_json(run_dir / "protected_sets.json", protected_sets)
        shared.write_json(
            run_dir / "quantization_config.json",
            {
                "schema_version": f"{SCHEMA_VERSION}_quantization_config",
                "weight_quantization": "simple symmetric per-channel INT4 represented as dequantized tensors",
                "activation_dtype": "float16 autocast for quantized regimes",
                "protected_channel_dtype": "bfloat16",
                "regimes": checker.REGIMES,
            },
        )
        shared.write_json(
            run_dir / "kl_positions.json",
            {
                "schema_version": f"{SCHEMA_VERSION}_kl_positions",
                "created_at_utc": shared.utc_now(),
                "dense_grid_fallback": dense_grid_fallback,
                "fallback_reason": "Full every-position full-vocabulary KL is implemented but not used in this run because it is estimated to exceed the sprint runtime budget; this preregistered dense grid is fixed before KL data inspection."
                if dense_grid_fallback
                else None,
                "positions": positions,
                "position_count": len(positions),
            },
        )
        environment = shared.build_environment(schema_version=SCHEMA_VERSION)
        shared.write_json(run_dir / "environment.json", environment)
        phase4_runner.write_environment_text(run_dir / "environment.txt", environment)

        trace_path, trace_manifest = ensure_bf16_traces(
            run_dir=run_dir,
            run_events_path=run_events_path,
            model_provenance=model_provenance,
            prompts=prompts,
            source_trace_dir=args.m26_run_dir,
            max_new_tokens=checker.MAX_NEW_TOKENS,
            batch_size=args.batch_size,
            dtype_name=args.dtype,
            device_name=args.device,
        )
        target_tokens = phase4_runner.load_trace_tokens(trace_path)
        for index in range(checker.TRACE_COUNT):
            if len(target_tokens[index]) < checker.MAX_NEW_TOKENS:
                raise RuntimeError(f"trace {index} has only {len(target_tokens[index])} target tokens")

        rows_by_regime_position: dict[str, dict[int, list[float]]] = {
            regime: defaultdict(list) for regime in checker.REGIMES
        }
        kl_path = run_dir / "kl_rows.jsonl.gz"
        tmp_dir = ROOT / ".debug" / "kl_accumulation" / args.run_id
        tmp_dir.mkdir(parents=True, exist_ok=True)

        print(json.dumps({"event": "collecting_bf16_reference_log_probs", "positions": len(positions)}, sort_keys=True))
        model, tokenizer, device = shared.load_model_and_tokenizer(model_provenance, dtype_name=args.dtype, device_name=args.device)
        for prompt in prompts:
            index = int(prompt["index"])
            log_probs = collect_log_probs_for_trace(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
                target_tokens=target_tokens[index],
                positions=positions,
                max_new_tokens=checker.MAX_NEW_TOKENS,
                use_float16_autocast=False,
                regime_name="bf16_reference",
            )
            import torch

            torch.save({position: tensor.to(torch.float32) for position, tensor in log_probs.items()}, tmp_dir / f"bf16_trace_{index}.pt")
            with run_events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"created_at_utc": shared.utc_now(), "event": "completed_bf16_log_probs", "prompt_index": index}, sort_keys=True) + "\n")
        del model, tokenizer, device
        release_model_memory()

        with gzip.open(kl_path, "wt", encoding="utf-8") as kl_handle:
            for prompt in prompts:
                index = int(prompt["index"])
                for position in positions:
                    row = {
                        "schema_version": f"{SCHEMA_VERSION}_kl_row",
                        "prompt_index": index,
                        "prompt_id": prompt["prompt_id"],
                        "regime": "bf16_reference",
                        "decode_position": position,
                        "kl_bf16_q": 0.0,
                    }
                    kl_handle.write(json.dumps(row, sort_keys=True) + "\n")
                    rows_by_regime_position["bf16_reference"][position].append(0.0)

            for regime in checker.QUANTIZED_REGIMES:
                print(json.dumps({"event": "loading_quantized_kl_regime", "regime": regime, "time": shared.utc_now()}, sort_keys=True))
                model, tokenizer, device = shared.load_model_and_tokenizer(model_provenance, dtype_name=args.dtype, device_name=args.device)
                excluded = phase4_runner.apply_quantization(model, protected_sets, regime)
                shared.write_json(run_dir / f"excluded_tensors_{regime}.json", excluded)
                for prompt in prompts:
                    index = int(prompt["index"])
                    import torch

                    bf16_log_probs = torch.load(tmp_dir / f"bf16_trace_{index}.pt", map_location="cpu")
                    q_log_probs = collect_log_probs_for_trace(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        prompt=prompt,
                        target_tokens=target_tokens[index],
                        positions=positions,
                        max_new_tokens=checker.MAX_NEW_TOKENS,
                        use_float16_autocast=True,
                        regime_name=regime,
                    )
                    for position in positions:
                        ref_logp = bf16_log_probs[position]
                        q_logp = q_log_probs[position].to(ref_logp.dtype)
                        probs = ref_logp.exp()
                        kl_value = float(torch.sum(probs * (ref_logp - q_logp)).item())
                        if kl_value < 0.0 and kl_value > -1e-6:
                            kl_value = 0.0
                        row = {
                            "schema_version": f"{SCHEMA_VERSION}_kl_row",
                            "prompt_index": index,
                            "prompt_id": prompt["prompt_id"],
                            "regime": regime,
                            "decode_position": position,
                            "kl_bf16_q": kl_value,
                        }
                        kl_handle.write(json.dumps(row, sort_keys=True) + "\n")
                        rows_by_regime_position[regime][position].append(kl_value)
                    with run_events_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                {
                                    "created_at_utc": shared.utc_now(),
                                    "event": "completed_kl_trace",
                                    "prompt_index": index,
                                    "regime": regime,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                del model, tokenizer, device
                release_model_memory()

        write_kl_summary(run_dir, positions, rows_by_regime_position)
        shared.write_json(
            run_dir / "bf16_logprob_manifest.json",
            {
                "schema_version": f"{SCHEMA_VERSION}_bf16_logprob_manifest",
                "created_at_utc": shared.utc_now(),
                "storage": "temporary .debug tensors, not result artifacts",
                "position_count": len(positions),
                "trace_count": checker.TRACE_COUNT,
            },
        )
        run_events_path.open("a", encoding="utf-8").write(json.dumps({"created_at_utc": shared.utc_now(), "event": "run_completed"}, sort_keys=True) + "\n")
        sys.stdout.flush()
        sys.stderr.flush()
        shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
        result = checker.evaluate(run_dir)
        print(json.dumps({"checker_decision": result["decision"], "artifact_complete": result.get("artifact_complete", False)}, indent=2, sort_keys=True))
        sys.stdout.flush()
        sys.stderr.flush()
        shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
        checker.evaluate(run_dir)
        sys.excepthook = previous_excepthook
        return 0
    except BaseException as exc:
        write_failure_packet(run_dir, run_events_path, exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
