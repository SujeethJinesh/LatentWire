#!/usr/bin/env python3
"""Run Phase 9 M-FISH Fisher-weighted dynamic channel protection."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import shutil
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experimental.outlier_migrate.phase4 import run_om_phase4_intervention as phase4_runner
from experimental.outlier_migrate.phase9 import check_om_phase9_mfish as checker
from experimental.outlier_migrate.phase9 import run_om_phase9_m11_ema_drift as m11_runner
from experimental.outlier_migrate.phase9 import run_om_phase9_m11b_budget_scaling as m11b_runner
from experimental.shared import run_phase0_branch as shared

DEFAULT_SOURCE_DIR = (
    ROOT / "experimental/outlier_migrate/phase9/results/om_v2_m11b_deepseek_20260527T1210Z"
)
DEFAULT_RESULTS_DIR = ROOT / "experimental/outlier_migrate/phase9/results"
BASELINE_REGIMES = ["bf16", "static_1pct", "m11b_top5", "m11b_top10", "static_top10"]
NEW_REGIMES = ["mfish_top5", "mfish_top10", "random_fisher_top10"]
FISHER_POSITIONS = (9500, 9750, 9999)
UPDATE_POSITIONS = tuple(range(checker.UPDATE_CADENCE, checker.SCORING_POSITION + 1, checker.UPDATE_CADENCE))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_failure_packet(run_dir: Path, run_events_path: Path, exc: BaseException | str) -> None:
    payload: dict[str, Any] = {
        "schema_version": f"{checker.SCHEMA_VERSION}_infra_error",
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
        shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=checker.SCHEMA_VERSION))
        checker.evaluate(run_dir)
    except Exception:
        pass


def ensure_source(source_dir: Path) -> None:
    required = [
        "activation_magnitude_manifest.json",
        "activation_magnitudes.jsonl.gz",
        "bf16_trace_manifest.json",
        "bf16_traces.jsonl.gz",
        "decoding_config.json",
        "model_provenance.json",
        "prompt_manifest.json",
        "protected_sets.json",
        "quantization_config.json",
    ]
    missing = [rel for rel in required if not (source_dir / rel).is_file()]
    missing += [
        f"score_cache/{regime}.json"
        for regime in BASELINE_REGIMES
        if not (source_dir / "score_cache" / f"{regime}.json").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"source packet missing artifacts: {missing}")


def copy_source_artifacts(source_dir: Path, run_dir: Path) -> None:
    for rel in [
        "activation_magnitude_manifest.json",
        "activation_magnitudes.jsonl.gz",
        "bf16_trace_manifest.json",
        "bf16_traces.jsonl.gz",
        "decoding_config.json",
        "model_provenance.json",
        "prompt_manifest.json",
        "quantization_config.json",
    ]:
        dest = run_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_dir / rel, dest)


def load_trace_tokens(path: Path) -> dict[int, list[int]]:
    traces: dict[int, list[int]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            traces[int(row["prompt_index"])] = [int(value) for value in row["token_ids"]]
    return traces


def top_channels(values: list[float], count: int) -> list[int]:
    return sorted(range(len(values)), key=lambda channel: (-float(values[channel]), channel))[:count]


def read_score_cache(source_dir: Path, regime: str, prompt_indices: set[int]) -> dict[int, dict[str, float]]:
    cached = m11_runner.read_score_cache_any(source_dir, regime, expected_prompt_indices=prompt_indices)
    if cached is None:
        raise FileNotFoundError(f"missing score cache for {regime} in {source_dir}")
    return cached


def write_score_cache(run_dir: Path, regime: str, scores: dict[int, dict[str, float]]) -> None:
    shared.write_json(
        run_dir / "score_cache" / f"{regime}.json",
        {
            "schema_version": f"{checker.SCHEMA_VERSION}_score_cache",
            "created_at_utc": shared.utc_now(),
            "regime": regime,
            "scores": {str(index): row for index, row in sorted(scores.items())},
        },
    )


def fisher_weights(
    *,
    model_provenance: dict[str, Any],
    prompts: list[dict[str, Any]],
    trace_tokens: dict[int, list[int]],
    dtype_name: str,
    device_name: str,
    run_events_path: Path,
    sample_positions: tuple[int, ...],
) -> dict[str, Any]:
    model, tokenizer, device = shared.load_model_and_tokenizer(
        model_provenance, dtype_name=dtype_name, device_name=device_name
    )
    previous_fast_paths = phase4_runner.set_autocast_sensitive_fast_paths(False)
    normalize_cache_inputs, output_cache = phase4_runner.make_cache_helpers(model)
    layers, layer_origin = shared.discover_transformer_layers(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    captured: dict[int, torch.Tensor] = {}
    handles = []
    embedding = model.get_input_embeddings()

    def embedding_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        return output.detach().requires_grad_(True)

    def make_layer_hook(layer_index: int):
        def layer_hook(_module: Any, _inputs: Any, output: Any) -> None:
            tensor = shared.tensor_from_hook_output(output)
            if torch.is_tensor(tensor) and tensor.requires_grad:
                tensor.retain_grad()
                captured[layer_index] = tensor

        return layer_hook

    handles.append(embedding.register_forward_hook(embedding_hook))
    for layer_index, (_layer_name, layer) in enumerate(layers):
        handles.append(layer.register_forward_hook(make_layer_hook(layer_index)))

    sums: dict[int, torch.Tensor] = {}
    counts: dict[int, int] = defaultdict(int)
    sample_events: list[dict[str, Any]] = []

    try:
        for prompt in prompts:
            prompt_index = int(prompt["index"])
            tokens = trace_tokens[prompt_index]
            max_position = max(sample_positions)
            if len(tokens) <= max_position:
                raise RuntimeError(f"trace {prompt_index} too short for Fisher positions")
            text = shared.make_prompt_text(str(prompt["prompt"]))
            encoded = tokenizer([text], padding=True, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            with torch.no_grad():
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
                    raise RuntimeError("model did not return cache for Fisher prompt")

            for position in range(1, max_position + 1):
                token = torch.tensor([[tokens[position - 1]]], dtype=torch.long, device=device)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)], dim=1
                )
                cache_position = torch.tensor([attention_mask.shape[1] - 1], dtype=torch.long, device=device)
                if position in sample_positions:
                    label = torch.tensor([tokens[position]], dtype=torch.long, device=device)
                    captured.clear()
                    model_inputs = model.prepare_inputs_for_generation(
                        input_ids=token,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        use_cache=False,
                    )
                    outputs = model(**normalize_cache_inputs(model_inputs))
                    loss = torch.nn.functional.cross_entropy(outputs.logits[:, -1, :].float(), label)
                    loss.backward()
                    missing = sorted(set(range(len(layers))).difference(captured))
                    if missing:
                        raise RuntimeError(f"missing Fisher hook captures: {missing[:8]}")
                    for layer_index, tensor in captured.items():
                        grad = tensor.grad
                        if grad is None:
                            raise RuntimeError(f"missing Fisher gradient for layer {layer_index}")
                        value = (grad[:, -1, :].detach().float().squeeze(0) ** 2).cpu()
                        sums[layer_index] = value if layer_index not in sums else sums[layer_index] + value
                        counts[layer_index] += 1
                    sample_events.append(
                        {"prompt_index": prompt_index, "position": position, "loss": float(loss.item())}
                    )
                    del outputs, loss
                    captured.clear()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                with torch.no_grad():
                    model_inputs = model.prepare_inputs_for_generation(
                        input_ids=token,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        use_cache=True,
                    )
                    outputs = model(**normalize_cache_inputs(model_inputs))
                    past_key_values = output_cache(outputs)
                    if past_key_values is None:
                        raise RuntimeError(f"model dropped cache at Fisher position {position}")
            run_events_path.open("a", encoding="utf-8").write(
                json.dumps(
                    {
                        "created_at_utc": shared.utc_now(),
                        "event": "fisher_prompt_completed",
                        "prompt_index": prompt_index,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            phase4_runner.release_model_memory()
    finally:
        for handle in handles:
            handle.remove()
        phase4_runner.restore_autocast_sensitive_fast_paths(previous_fast_paths)
        del model, tokenizer
        phase4_runner.release_model_memory()

    layer_payload: dict[str, Any] = {}
    for layer_index, (layer_name, _layer) in enumerate(layers):
        if layer_index not in sums:
            raise RuntimeError(f"no Fisher samples accumulated for layer {layer_index}")
        weights = (sums[layer_index] / max(1, counts[layer_index])).tolist()
        layer_payload[str(layer_index)] = {
            "layer_name": layer_name,
            "channel_count": len(weights),
            "sample_count": counts[layer_index],
            "weights": [float(value) for value in weights],
        }
    return {
        "schema_version": f"{checker.SCHEMA_VERSION}_fisher_weights",
        "created_at_utc": shared.utc_now(),
        "model_id": model_provenance.get("model_id"),
        "layer_origin": layer_origin,
        "sample_positions": list(sample_positions),
        "sampled_position_count": len(sample_events),
        "estimator": "detached_prefix_cache_one_step_decode_gradient",
        "events": sample_events,
        "layers": layer_payload,
    }


def build_protected_sets(
    source_sets: dict[str, Any],
    activation_rows: Any,
    fisher: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    means_by_layer, layer_names = m11b_runner.activation_means_by_layer_position(list(activation_rows))
    rng = random.Random(seed)
    layers_by_regime: dict[str, dict[str, Any]] = {}
    trajectories_by_regime: dict[str, dict[str, Any]] = {}
    for regime in BASELINE_REGIMES:
        layers_by_regime[regime] = source_sets["regimes"][regime]["layers"]

    fisher_layers = fisher["layers"]
    for layer_index in sorted(means_by_layer):
        channel_count = len(means_by_layer[layer_index][100])
        fisher_values = [float(value) for value in fisher_layers[str(layer_index)]["weights"]]
        if len(fisher_values) != channel_count:
            raise RuntimeError(f"Fisher/channel mismatch for layer {layer_index}")
        shuffled_fisher = list(fisher_values)
        rng.shuffle(shuffled_fisher)
        for regime, fraction, use_random in [
            ("mfish_top5", 0.05, False),
            ("mfish_top10", 0.10, False),
            ("random_fisher_top10", 0.10, True),
        ]:
            budget_count = max(1, math.ceil(channel_count * fraction))
            active_fisher = shuffled_fisher if use_random else fisher_values
            weighted_100 = [
                float(means_by_layer[layer_index][100][channel]) * active_fisher[channel]
                for channel in range(channel_count)
            ]
            scores = [0.0] * channel_count
            for channel in top_channels(weighted_100, budget_count):
                scores[channel] = 1.0
            selected = sorted(top_channels(scores, budget_count))
            steps: list[dict[str, Any]] = []
            for position in UPDATE_POSITIONS:
                weighted = [
                    float(means_by_layer[layer_index][position][channel]) * active_fisher[channel]
                    for channel in range(channel_count)
                ]
                indicator = set(top_channels(weighted, budget_count))
                scores = [
                    checker.ALPHA * (1.0 if channel in indicator else 0.0)
                    + (1.0 - checker.ALPHA) * float(scores[channel])
                    for channel in range(channel_count)
                ]
                selected = sorted(top_channels(scores, budget_count))
                if position % 1000 == 0 or position in {100, checker.SCORING_POSITION}:
                    steps.append(
                        {
                            "position": position,
                            "protected_count": len(selected),
                            "protected_channels": selected,
                            "mean_score": float(mean(scores)),
                        }
                    )
            layers_by_regime.setdefault(regime, {})[str(layer_index)] = {
                "layer_name": layer_names[layer_index],
                "channel_count": channel_count,
                "budget_fraction": fraction,
                "protected_count": len(selected),
                "protected_channels": selected,
                "source": "fisher_weighted_dynamic_ema" if not use_random else "random_fisher_weighted_dynamic_ema",
            }
            trajectories_by_regime.setdefault(regime, {})[str(layer_index)] = {
                "layer_name": layer_names[layer_index],
                "channel_count": channel_count,
                "budget_fraction": fraction,
                "steps_recorded": steps,
                "final_protected_channels": selected,
            }

    protected_sets = {
        "schema_version": f"{checker.SCHEMA_VERSION}_protected_sets",
        "created_at_utc": shared.utc_now(),
        "selection_basis": "Fisher-weighted activation magnitude tracked by EMA over decode positions",
        "alpha": checker.ALPHA,
        "regimes": {regime: {"layers": layers_by_regime[regime]} for regime in checker.REGIMES if regime != "bf16"},
    }
    trajectories = {
        "schema_version": f"{checker.SCHEMA_VERSION}_protected_trajectories",
        "created_at_utc": shared.utc_now(),
        "alpha": checker.ALPHA,
        "update_cadence": checker.UPDATE_CADENCE,
        "update_positions": list(UPDATE_POSITIONS),
        "by_regime": {regime: {"layers": trajectories_by_regime.get(regime, {})} for regime in checker.RECOVERY_REGIMES},
        "protected_set_count_stats": m11b_runner.summarize_protected_counts(protected_sets),
    }
    return protected_sets, trajectories


def build_per_trace_rows(prompts: list[dict[str, Any]], all_scores: dict[str, dict[int, dict[str, float]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    recovery_regimes = [regime for regime in all_scores if regime not in {"bf16", "static_1pct"}]
    for prompt in prompts:
        index = int(prompt["index"])
        perplexities = {regime: float(all_scores[regime][index]["perplexity"]) for regime in all_scores}
        mean_nll = {regime: float(all_scores[regime][index]["mean_nll"]) for regime in all_scores}
        static_gap = perplexities["static_1pct"] - perplexities["bf16"]
        no_gap = static_gap <= 0.0
        recoveries = {
            regime: None if no_gap else 1.0 - (perplexities[regime] - perplexities["bf16"]) / static_gap
            for regime in recovery_regimes
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


def summarize(values: list[float]) -> dict[str, Any]:
    return {
        "median_recovery": float(median(values)) if values else None,
        "mean_recovery": float(mean(values)) if values else None,
        "bootstrap_ci95": checker.bootstrap_median(values),
        "included_trace_count": len(values),
    }


def build_metrics(
    run_dir: Path,
    prompt_manifest: dict[str, Any],
    model_provenance: dict[str, Any],
    rows: list[dict[str, Any]],
    protected_trajectories: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    included = [row for row in rows if not bool(row["no_recoverable_static_gap"])]
    recovery_regimes = [regime for regime in rows[0]["recoveries"]] if rows else []
    summaries = {
        regime: summarize([float(row["recoveries"][regime]) for row in included])
        for regime in recovery_regimes
    }
    no_gap_count = len(rows) - len(included)
    for summary in summaries.values():
        summary["total_trace_count"] = len(rows)
        summary["no_recoverable_static_gap_count"] = no_gap_count
        summary["no_recoverable_static_gap_fraction"] = no_gap_count / len(rows) if rows else 0.0
    metrics = {
        "schema_version": f"{checker.SCHEMA_VERSION}_metrics",
        "created_at_utc": shared.utc_now(),
        "preregistration": str(checker.PREREG_PATH.relative_to(ROOT)),
        "preregistration_sha256": shared.file_sha256(checker.PREREG_PATH),
        "model_key": checker.MODEL_KEY,
        "model_id": model_provenance.get("model_id"),
        "model_snapshot_commit": model_provenance.get("hf_snapshot_commit"),
        "prompt_sha256": prompt_manifest.get("prompt_sha256"),
        "trace_count": checker.TRACE_COUNT,
        "included_trace_count": len(included),
        "no_recoverable_static_gap_count": no_gap_count,
        "no_recoverable_static_gap_fraction": no_gap_count / len(rows) if rows else 0.0,
        "scoring_position": checker.SCORING_POSITION,
        "scoring_window_tokens": checker.SCORING_WINDOW_TOKENS,
        "alpha": checker.ALPHA,
        "metric_name": "positive-static-1pct-gap per-trace recovery",
        "metric_formula": "1 - (perplexity_regime - perplexity_BF16) / (perplexity_static_1pct - perplexity_BF16)",
        "completed_regimes": list(rows[0]["perplexities"]) if rows else [],
        "results_by_regime": summaries,
        "thresholds": checker.THRESHOLDS,
        "protected_set_count_stats": protected_trajectories.get("protected_set_count_stats", {}),
        "artifacts": {"run_dir": str(run_dir)},
    }
    bootstrap = {
        "schema_version": f"{checker.SCHEMA_VERSION}_bootstrap_ci",
        "metric_name": "median_per_trace_recovery",
        "bootstrap_samples": checker.BOOTSTRAP_SAMPLES,
        "bootstrap_seed": checker.BOOTSTRAP_SEED,
        "results_by_regime": summaries,
    }
    controls = {
        "schema_version": f"{checker.SCHEMA_VERSION}_control_metrics",
        "created_at_utc": shared.utc_now(),
        "mfish_minus_m11b_top10": (
            None
            if summaries["mfish_top10"]["median_recovery"] is None or summaries["m11b_top10"]["median_recovery"] is None
            else float(summaries["mfish_top10"]["median_recovery"]) - float(summaries["m11b_top10"]["median_recovery"])
        ),
        "mfish_minus_random_fisher_top10": (
            None
            if summaries["mfish_top10"]["median_recovery"] is None
            or summaries["random_fisher_top10"]["median_recovery"] is None
            else float(summaries["mfish_top10"]["median_recovery"]) - float(summaries["random_fisher_top10"]["median_recovery"])
        ),
    }
    return metrics, bootstrap, controls


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"om_phase9_mfish_deepseek_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--source-run-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=checker.BOOTSTRAP_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--fisher-positions", type=int, nargs="+", default=list(FISHER_POSITIONS))
    args = parser.parse_args(argv)

    if args.seed != checker.BOOTSTRAP_SEED:
        raise SystemExit(f"M-FISH preregisters seed {checker.BOOTSTRAP_SEED}")
    source_dir = args.source_run_dir.resolve()
    ensure_source(source_dir)
    run_dir = args.results_dir.resolve() / args.run_id
    if run_dir.exists():
        raise SystemExit(f"run dir already exists: {run_dir}")
    (run_dir / "logs").mkdir(parents=True)
    stdout_log = (run_dir / "logs/stdout.log").open("w", encoding="utf-8", buffering=1)
    stderr_log = (run_dir / "logs/stderr.log").open("w", encoding="utf-8", buffering=1)
    sys.stdout = shared.Tee(sys.__stdout__, stdout_log)
    sys.stderr = shared.Tee(sys.__stderr__, stderr_log)
    run_events_path = run_dir / "run_events.jsonl"
    previous_excepthook = sys.excepthook

    def mfish_excepthook(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        write_failure_packet(run_dir, run_events_path, exc)
        previous_excepthook(exc_type, exc, tb)

    sys.excepthook = mfish_excepthook
    random.seed(args.seed)
    run_events_path.write_text(
        json.dumps({"created_at_utc": shared.utc_now(), "event": "mfish_run_started"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    try:
        copy_source_artifacts(source_dir, run_dir)
        environment = shared.build_environment(schema_version=checker.SCHEMA_VERSION)
        phase4_runner.write_environment_text(run_dir / "environment.txt", environment)
        shared.write_json(run_dir / "environment.json", environment)
        prompt_manifest = load_json(run_dir / "prompt_manifest.json")
        model_provenance = load_json(run_dir / "model_provenance.json")
        if model_provenance.get("model_id") != checker.MODEL_ID:
            raise RuntimeError("source model_id mismatch")
        if model_provenance.get("hf_snapshot_commit") != checker.MODEL_SNAPSHOT:
            raise RuntimeError("source model snapshot mismatch")
        shared.write_json(
            run_dir / "command_metadata.json",
            {
                "schema_version": f"{checker.SCHEMA_VERSION}_command",
                "created_at_utc": shared.utc_now(),
                "argv": sys.argv if argv is None else ["run_om_phase9_mfish.py", *argv],
                "cwd": str(Path.cwd()),
                "run_dir": str(run_dir),
                "source_run_dir": str(source_dir),
                "batch_size": args.batch_size,
                "fisher_positions": list(args.fisher_positions),
            },
        )
        shared.write_json(
            run_dir / "random_seed.json",
            {"schema_version": f"{checker.SCHEMA_VERSION}_random_seed", "seed": args.seed, "determinism": {"do_sample": False, "num_beams": 1}},
        )
        prompts = prompt_manifest["prompts"]
        prompt_indices = {int(row["index"]) for row in prompts}
        target_tokens = phase4_runner.load_trace_tokens(run_dir / "bf16_traces.jsonl.gz")
        trace_tokens = load_trace_tokens(run_dir / "bf16_traces.jsonl.gz")

        fisher = fisher_weights(
            model_provenance=model_provenance,
            prompts=prompts,
            trace_tokens=trace_tokens,
            dtype_name=args.dtype,
            device_name=args.device,
            run_events_path=run_events_path,
            sample_positions=tuple(args.fisher_positions),
        )
        shared.write_json(run_dir / "fisher_weights.json", fisher)

        source_sets = load_json(source_dir / "protected_sets.json")
        protected_sets, protected_trajectories = build_protected_sets(
            source_sets,
            shared.iter_activation_rows(run_dir / "activation_magnitudes.jsonl.gz"),
            fisher,
            args.seed,
        )
        shared.write_json(run_dir / "protected_sets.json", protected_sets)
        shared.write_json(run_dir / "protected_trajectories.json", protected_trajectories)

        all_scores: dict[str, dict[int, dict[str, float]]] = {}
        excluded_by_regime: dict[str, Any] = {}
        for regime in BASELINE_REGIMES:
            all_scores[regime] = read_score_cache(source_dir, regime, prompt_indices)
            write_score_cache(run_dir, regime, all_scores[regime])
            if regime != "bf16":
                excluded_by_regime[regime] = {"regime": regime, "reused_score_cache": str(source_dir)}

        for regime in NEW_REGIMES:
            run_events_path.open("a", encoding="utf-8").write(
                json.dumps({"created_at_utc": shared.utc_now(), "event": "score_regime_started", "regime": regime}, sort_keys=True) + "\n"
            )
            all_scores[regime], excluded_by_regime[regime] = m11b_runner.score_regime(
                model_provenance=model_provenance,
                protected_sets=protected_sets,
                regime=regime,
                prompts=prompts,
                target_tokens=target_tokens,
                batch_size=args.batch_size,
                dtype_name=args.dtype,
                device_name=args.device,
                run_events_path=run_events_path,
            )
            write_score_cache(run_dir, regime, all_scores[regime])

        shared.write_json(
            run_dir / "excluded_tensors.json",
            {"schema_version": f"{checker.SCHEMA_VERSION}_excluded_tensors", "created_at_utc": shared.utc_now(), "by_regime": excluded_by_regime},
        )
        per_trace_rows = build_per_trace_rows(prompts, all_scores)
        shared.write_json(
            run_dir / "per_trace_metrics.json",
            {"schema_version": f"{checker.SCHEMA_VERSION}_per_trace_metrics", "created_at_utc": shared.utc_now(), "traces": per_trace_rows},
        )
        metrics, bootstrap, controls = build_metrics(run_dir, prompt_manifest, model_provenance, per_trace_rows, protected_trajectories)
        shared.write_json(run_dir / "metrics.json", metrics)
        shared.write_json(run_dir / "bootstrap_ci.json", bootstrap)
        shared.write_json(run_dir / "control_metrics.json", controls)
        shared.write_json(
            run_dir / "artifact_hashes.json",
            shared.build_artifact_hashes(run_dir, schema_version=checker.SCHEMA_VERSION),
        )
        result = checker.evaluate(run_dir)
        print(json.dumps({"checker_decision": result["decision"], "run_dir": str(run_dir)}, indent=2, sort_keys=True))
        shared.write_json(
            run_dir / "artifact_hashes.json",
            shared.build_artifact_hashes(run_dir, schema_version=checker.SCHEMA_VERSION),
        )
        checker.evaluate(run_dir)
        run_events_path.open("a", encoding="utf-8").write(
            json.dumps({"created_at_utc": shared.utc_now(), "event": "mfish_run_completed"}, sort_keys=True) + "\n"
        )
        sys.excepthook = previous_excepthook
        return 0
    except BaseException as exc:
        write_failure_packet(run_dir, run_events_path, exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
