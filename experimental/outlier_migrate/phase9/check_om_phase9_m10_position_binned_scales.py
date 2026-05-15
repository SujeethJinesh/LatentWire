#!/usr/bin/env python3
"""Check Phase 9 M10 position-binned scale packets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "experimental/outlier_migrate/phase9/results"
PREREG_PATH = ROOT / "experimental/outlier_migrate/phase9/preregister_om_phase9_m10_position_binned_scales.md"
DEFAULT_PROMPT_FILE = ROOT / "experimental/shared/prompts/aime_2025_indices_0_23.jsonl"

SCHEMA_VERSION = "om_phase9_m10_v1"
TRACE_COUNT = 24
MIN_VACATION_TRACE_COUNT = 12
SCORING_POSITION = 10000
SCORING_WINDOW_TOKENS = 512
CALIBRATION_POSITIONS = (100, 200, 500, 1000, 2000, 5000, 7000, 10000, 15000)
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 20260514
EXPECTED_PROMPT_FILE_SHA256 = "sha256:ead004dae0848ad43ad102551f48fa22a0b8ed4a57efecdcf9d7ae387bb6d17a"
EXPECTED_PROMPT_PAYLOAD_SHA256 = "sha256:aa038b29332b6d137d558205ee441163e7ea4cb3cc323eb705a2f5928fd2fe4e"
EXPECTED_PROMPT_SOURCE_DATASET = "opencompass/AIME2025"
EXPECTED_PROMPT_SOURCE_COMMIT = "a6ad95f611d72cf628a80b58bd0432ef6638f958"

MODEL_SNAPSHOTS = {
    "ibm-granite/granite-4.0-h-small": "b8c0982bab7fde4eb48110f5a069527c008fab39",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "cbd3fa9f933d55ef16a84236559f4ee2a0526848",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
    "tiiuae/Falcon-H1-0.5B-Instruct": "8f2587ca06bff78d8fa1adfccbe8c24d5f86b368",
}

PASS_DECISION = "PASS_M10_POSITION_BINNED"
KILL_NO_IMPROVEMENT = "KILL_M10_NO_IMPROVEMENT"
KILL_RANDOM_CONTROL = "KILL_M10_RANDOM_CONTROL_BEATS"
KILL_AMBIGUOUS = "KILL_M10_AMBIGUOUS"
FAIL_INFRA = "FAIL_INFRA_M10"

THRESHOLDS = {
    "pass_median_recovery_min": 0.40,
    "pass_ci95_low_gt": 0.20,
    "pass_beats_static_smoothquant_by_ge": 0.20,
    "pass_beats_midpoint_by_ge": 0.10,
    "pass_beats_random_bin_by_ge": 0.15,
    "kill_within_static_smoothquant_abs_le": 0.05,
    "random_control_beats_by_gt": 0.10,
    "bootstrap_seed": BOOTSTRAP_SEED,
}

REGIMES = [
    "bf16",
    "static_smoothquant",
    "m10_position_binned",
    "midpoint_matched_cost",
    "random_bin_assignment",
]

REQUIRED_FILES = [
    "environment.json",
    "environment.txt",
    "model_provenance.json",
    "prompt_manifest.json",
    "command_metadata.json",
    "random_seed.json",
    "decoding_config.json",
    "activation_magnitude_manifest.json",
    "activation_magnitudes.jsonl.gz",
    "bf16_traces.jsonl.gz",
    "bf16_trace_manifest.json",
    "scale_tables.json",
    "scale_table_manifest.json",
    "quantization_config.json",
    "excluded_tensors.json",
    "per_trace_metrics.json",
    "metrics.json",
    "bootstrap_ci.json",
    "control_metrics.json",
    "artifact_hashes.json",
    "logs/stdout.log",
    "logs/stderr.log",
    "run_events.jsonl",
]
HASHED_FILES = [rel for rel in REQUIRED_FILES if rel != "artifact_hashes.json"]
OPTIONAL_HASHED_FILES = ["vacation_adaptation.json", "diagnostic.md"]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def is_close(left: float, right: float, *, tol: float = 1e-8) -> bool:
    return abs(float(left) - float(right)) <= tol * max(1.0, abs(float(left)), abs(float(right)))


def latest_run_dir() -> Path:
    candidates = [path for path in RESULTS_DIR.iterdir() if path.is_dir()] if RESULTS_DIR.is_dir() else []
    candidates = [path for path in candidates if path.name.startswith("om_phase9_m10_")]
    if not candidates:
        raise FileNotFoundError(f"no M10 result dirs found under {RESULTS_DIR}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def expected_source_file(index: int) -> str:
    return "aime2025-I.jsonl" if index < 15 else "aime2025-II.jsonl"


def expected_prompt_id(index: int) -> str:
    if index < 15:
        return f"opencompass_AIME2025_I_{index}"
    return f"opencompass_AIME2025_II_{index - 15}"


def prompt_payload_sha256(prompts: list[dict[str, Any]]) -> str:
    ordered = sorted(prompts, key=lambda row: int(row["index"]))
    payload = "".join(str(row["prompt"]) for row in ordered).encode("utf-8")
    return bytes_sha256(payload)


def bootstrap_median(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"ci95_low": None, "ci95_high": None}
    rng = random.Random(BOOTSTRAP_SEED)
    boot: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [values[rng.randrange(len(values))] for _ in values]
        boot.append(float(median(sample)))
    boot.sort()
    return {
        "ci95_low": boot[int(0.025 * (len(boot) - 1))],
        "ci95_high": boot[int(0.975 * (len(boot) - 1))],
    }


def summarize_recovery(rows: list[dict[str, Any]], regime: str) -> dict[str, Any]:
    included = [row for row in rows if not bool(row.get("no_recoverable_static_gap"))]
    values = [float(row["recoveries"][regime]) for row in included]
    return {
        "median_recovery": float(median(values)) if values else None,
        "mean_recovery": float(mean(values)) if values else None,
        "bootstrap_ci95": bootstrap_median(values),
        "included_trace_count": len(values),
        "total_trace_count": len(rows),
        "per_trace_recovery_included": values,
        "no_recoverable_static_gap_count": len(rows) - len(values),
        "no_recoverable_static_gap_fraction": (len(rows) - len(values)) / len(rows) if rows else 0.0,
    }


def decision_from_summaries(
    m10: dict[str, Any], midpoint: dict[str, Any], random_bin: dict[str, Any]
) -> tuple[str, list[str]]:
    if m10["median_recovery"] is None:
        return FAIL_INFRA, ["no traces had a positive recoverable static SmoothQuant gap"]
    m10_med = float(m10["median_recovery"])
    m10_low = float(m10["bootstrap_ci95"]["ci95_low"])
    static_med = 0.0
    midpoint_med = float(midpoint["median_recovery"]) if midpoint["median_recovery"] is not None else float("nan")
    random_med = float(random_bin["median_recovery"]) if random_bin["median_recovery"] is not None else float("nan")
    if random_med - m10_med > THRESHOLDS["random_control_beats_by_gt"]:
        return KILL_RANDOM_CONTROL, [
            f"random-bin scale control median {random_med:.8f} beats M10 median {m10_med:.8f} by >0.10"
        ]
    if abs(m10_med - static_med) <= THRESHOLDS["kill_within_static_smoothquant_abs_le"]:
        return KILL_NO_IMPROVEMENT, [
            f"M10 median recovery {m10_med:.8f} is within 0.05 of static SmoothQuant median recovery 0.0"
        ]
    pass_conditions = [
        m10_med >= THRESHOLDS["pass_median_recovery_min"],
        m10_low > THRESHOLDS["pass_ci95_low_gt"],
        m10_med - static_med >= THRESHOLDS["pass_beats_static_smoothquant_by_ge"],
        m10_med - midpoint_med >= THRESHOLDS["pass_beats_midpoint_by_ge"],
        m10_med - random_med >= THRESHOLDS["pass_beats_random_bin_by_ge"],
    ]
    if all(pass_conditions):
        return PASS_DECISION, [
            "M10 median recovery, CI lower bound, static separation, midpoint separation, and random-bin separation all pass"
        ]
    return KILL_AMBIGUOUS, [
        "M10 is neither a preregistered pass nor a no-improvement/random-control kill"
    ]


def effective_trace_count(adaptation: dict[str, Any], infra: list[str]) -> int:
    if not adaptation:
        return TRACE_COUNT
    if adaptation.get("schema_version") != f"{SCHEMA_VERSION}_vacation_adaptation":
        infra.append("vacation_adaptation schema mismatch")
    if adaptation.get("authority") != "Vacation mode V2/V4":
        infra.append("vacation_adaptation authority mismatch")
    if adaptation.get("adaptation") != "deterministic_trace_count_reduction":
        infra.append("vacation_adaptation.adaptation mismatch")
    count = int(adaptation.get("effective_trace_count", -1))
    if not (MIN_VACATION_TRACE_COUNT <= count < TRACE_COUNT):
        infra.append("vacation_adaptation effective_trace_count must be in [12, 23]")
        return TRACE_COUNT
    if adaptation.get("prompt_indices") != list(range(count)):
        infra.append("vacation_adaptation prompt_indices mismatch")
    return count


def validate_prompt_manifest(
    prompt_manifest: dict[str, Any], metrics: dict[str, Any], infra: list[str], *, trace_count: int
) -> None:
    if file_sha256(DEFAULT_PROMPT_FILE) != EXPECTED_PROMPT_FILE_SHA256:
        infra.append("canonical prompt file hash drifted")
    expected_selection = "deterministic_indices_0_23"
    if trace_count != TRACE_COUNT:
        expected_selection = f"deterministic_indices_0_{trace_count - 1}_vacation_revision"
    if prompt_manifest.get("selection") != expected_selection:
        infra.append(f"prompt_manifest.selection must be {expected_selection}")
    if prompt_manifest.get("prompt_file_sha256") != EXPECTED_PROMPT_FILE_SHA256:
        infra.append("prompt file SHA mismatch")
    prompts = prompt_manifest.get("prompts", [])
    if not isinstance(prompts, list):
        infra.append("prompt_manifest.prompts must be a list")
        prompts = []
    if len(prompts) != trace_count or prompt_manifest.get("prompt_count") != trace_count:
        infra.append(f"prompt manifest must contain exactly {trace_count} prompts")
    indices = []
    for row in prompts:
        if not isinstance(row, dict):
            infra.append("prompt row is not an object")
            continue
        index = int(row.get("index", -1))
        indices.append(index)
        if row.get("prompt_id") != expected_prompt_id(index):
            infra.append(f"prompt {index}: prompt_id mismatch")
        if row.get("source_dataset") != EXPECTED_PROMPT_SOURCE_DATASET:
            infra.append(f"prompt {index}: source_dataset mismatch")
        if row.get("source_commit") != EXPECTED_PROMPT_SOURCE_COMMIT:
            infra.append(f"prompt {index}: source_commit mismatch")
        if row.get("source_file") != expected_source_file(index):
            infra.append(f"prompt {index}: source_file mismatch")
    if indices != list(range(trace_count)):
        infra.append(f"prompt indices must be exactly 0-{trace_count - 1} in order")
    if prompts:
        expected = prompt_payload_sha256(prompts)
        if prompt_manifest.get("prompt_sha256") != expected:
            infra.append("prompt payload SHA does not match prompt rows")
        if trace_count == TRACE_COUNT and prompt_manifest.get("prompt_sha256") != EXPECTED_PROMPT_PAYLOAD_SHA256:
            infra.append("prompt payload SHA does not match frozen Phase 1 set")
    if metrics.get("prompt_sha256") != prompt_manifest.get("prompt_sha256"):
        infra.append("metrics.prompt_sha256 must match prompt_manifest")


def validate_artifact_hashes(run_dir: Path, artifact_hashes: dict[str, Any], infra: list[str]) -> None:
    entries = artifact_hashes.get("artifacts", [])
    if not isinstance(entries, list):
        infra.append("artifact_hashes.artifacts must be a list")
        return
    by_path = {str(row.get("path")): row for row in entries if isinstance(row, dict)}
    rels = list(HASHED_FILES)
    for optional_rel in OPTIONAL_HASHED_FILES:
        if (run_dir / optional_rel).is_file():
            rels.append(optional_rel)
    for rel in rels:
        item = by_path.get(rel)
        path = run_dir / rel
        if item is None:
            infra.append(f"artifact_hashes missing {rel}")
            continue
        if not path.is_file():
            infra.append(f"hashed artifact missing on disk: {rel}")
            continue
        if item.get("bytes") != path.stat().st_size:
            infra.append(f"artifact_hashes byte mismatch for {rel}")
        if item.get("sha256") != file_sha256(path):
            infra.append(f"artifact_hashes sha256 mismatch for {rel}")


def validate_trace_rows(rows: list[dict[str, Any]], infra: list[str]) -> None:
    expected_score_start = SCORING_POSITION - SCORING_WINDOW_TOKENS + 1
    for row in rows:
        prompt_index = int(row.get("prompt_index", -1))
        if int(row.get("scored_tokens", -1)) != SCORING_WINDOW_TOKENS:
            infra.append(f"trace {prompt_index}: scored_tokens mismatch")
        if int(row.get("score_start", -1)) != expected_score_start:
            infra.append(f"trace {prompt_index}: score_start mismatch")
        if int(row.get("score_end", -1)) != SCORING_POSITION:
            infra.append(f"trace {prompt_index}: score_end mismatch")
        perplexities = row.get("perplexities", {})
        recoveries = row.get("recoveries", {})
        if set(perplexities) != set(REGIMES):
            infra.append(f"trace {prompt_index}: perplexity regimes mismatch")
            continue
        if set(recoveries) != {"m10_position_binned", "midpoint_matched_cost", "random_bin_assignment"}:
            infra.append(f"trace {prompt_index}: recovery regimes mismatch")
            continue
        static_gap = float(perplexities["static_smoothquant"]) - float(perplexities["bf16"])
        if not is_close(float(row.get("static_gap", float("nan"))), static_gap):
            infra.append(f"trace {prompt_index}: static_gap mismatch")
        no_gap = static_gap <= 0.0
        if bool(row.get("no_recoverable_static_gap")) != no_gap:
            infra.append(f"trace {prompt_index}: no_recoverable_static_gap mismatch")
        for regime in ["m10_position_binned", "midpoint_matched_cost", "random_bin_assignment"]:
            if no_gap:
                if recoveries[regime] is not None:
                    infra.append(f"trace {prompt_index}: no-gap recovery for {regime} must be null")
                continue
            expected = 1.0 - (float(perplexities[regime]) - float(perplexities["bf16"])) / static_gap
            if not is_close(float(recoveries[regime]), expected, tol=1e-7):
                infra.append(f"trace {prompt_index}: recovery formula mismatch for {regime}")


def validate_scale_tables(scale_tables: dict[str, Any], scale_manifest: dict[str, Any], infra: list[str]) -> None:
    if scale_tables.get("schema_version") != f"{SCHEMA_VERSION}_scale_tables":
        infra.append("scale_tables schema mismatch")
    expected_tables = {"static_position_100", "m10_bins", "midpoint_bins", "random_bin_assignment"}
    if set(scale_tables.get("tables", {})) != expected_tables:
        infra.append("scale_tables.tables mismatch")
    bins = scale_tables.get("position_bins", [])
    if len(bins) != 5:
        infra.append("scale_tables must record five position bins")
    if scale_manifest.get("scale_tables_sha256") and scale_manifest.get("scale_tables_sha256") != file_sha256(
        scale_manifest_path(scale_manifest)
    ):
        infra.append("scale_table_manifest hash points at a different file")
    route = scale_manifest.get("implementation_route")
    if route not in {
        "fold_scale_into_dequantized_int4_weights_with_activation_pre_hooks",
        "reused_score_cache",
    }:
        infra.append("scale_table_manifest implementation route mismatch")


def scale_manifest_path(scale_manifest: dict[str, Any]) -> Path:
    path = scale_manifest.get("scale_tables_path")
    return Path(path) if path else ROOT / "__missing_scale_table_path__"


def validate_packet(run_dir: Path) -> tuple[list[str], dict[str, Any], list[dict[str, Any]]]:
    infra: list[str] = []
    for rel in REQUIRED_FILES:
        if not (run_dir / rel).is_file():
            infra.append(f"missing required file: {rel}")
    loaded: dict[str, Any] = {}
    for rel in [
        "metrics.json",
        "bootstrap_ci.json",
        "control_metrics.json",
        "per_trace_metrics.json",
        "prompt_manifest.json",
        "model_provenance.json",
        "bf16_trace_manifest.json",
        "command_metadata.json",
        "random_seed.json",
        "decoding_config.json",
        "scale_tables.json",
        "scale_table_manifest.json",
        "quantization_config.json",
        "excluded_tensors.json",
        "activation_magnitude_manifest.json",
        "artifact_hashes.json",
        "vacation_adaptation.json",
    ]:
        path = run_dir / rel
        if path.is_file():
            try:
                loaded[rel] = load_json(path)
            except Exception as exc:
                infra.append(f"bad JSON {rel}: {exc!r}")
    rows = loaded.get("per_trace_metrics.json", {}).get("traces", [])
    if not isinstance(rows, list):
        infra.append("per_trace_metrics.traces must be a list")
        rows = []
    trace_count = effective_trace_count(loaded.get("vacation_adaptation.json", {}), infra)
    if len(rows) != trace_count:
        infra.append(f"per_trace_metrics must contain exactly {trace_count} traces")
    validate_trace_rows(rows, infra)
    metrics = loaded.get("metrics.json", {})
    if metrics.get("schema_version") != f"{SCHEMA_VERSION}_metrics":
        infra.append("metrics schema_version mismatch")
    model_id = metrics.get("model_id")
    if model_id not in MODEL_SNAPSHOTS:
        infra.append("metrics.model_id not in M10 preregistered model set")
    if metrics.get("thresholds") != THRESHOLDS:
        infra.append("metrics.thresholds mismatch")
    if metrics.get("scoring_position") != SCORING_POSITION:
        infra.append("metrics.scoring_position mismatch")
    if metrics.get("scoring_window_tokens") != SCORING_WINDOW_TOKENS:
        infra.append("metrics.scoring_window_tokens mismatch")
    if metrics.get("calibration_positions") != list(CALIBRATION_POSITIONS):
        infra.append("metrics.calibration_positions mismatch")
    validate_prompt_manifest(loaded.get("prompt_manifest.json", {}), metrics, infra, trace_count=trace_count)
    model = loaded.get("model_provenance.json", {})
    if model.get("model_id") != model_id:
        infra.append("model_provenance.model_id must match metrics.model_id")
    if model_id in MODEL_SNAPSHOTS and model.get("hf_snapshot_commit") != MODEL_SNAPSHOTS[model_id]:
        infra.append("model snapshot commit mismatch")
    if loaded.get("random_seed.json", {}).get("seed") != BOOTSTRAP_SEED:
        infra.append("random_seed.seed mismatch")
    decoding = loaded.get("decoding_config.json", {})
    if decoding.get("do_sample") is not False or decoding.get("num_beams") != 1:
        infra.append("decoding_config must be deterministic greedy")
    if decoding.get("scoring_position") != SCORING_POSITION:
        infra.append("decoding_config.scoring_position mismatch")
    if decoding.get("scoring_window_tokens") != SCORING_WINDOW_TOKENS:
        infra.append("decoding_config.scoring_window_tokens mismatch")
    qcfg = loaded.get("quantization_config.json", {})
    if qcfg.get("weight_bits") != 4 or qcfg.get("scheme") != "symmetric_per_output_channel_int4_after_scale_folding":
        infra.append("quantization_config must specify scaled symmetric per-output-channel INT4")
    if qcfg.get("activation_dtype") != "float16":
        infra.append("quantization_config.activation_dtype must be float16")
    validate_scale_tables(
        loaded.get("scale_tables.json", {}),
        loaded.get("scale_table_manifest.json", {}),
        infra,
    )
    excluded = loaded.get("excluded_tensors.json", {}).get("by_regime", {})
    if set(excluded) != {"static_smoothquant", "m10_position_binned", "midpoint_matched_cost", "random_bin_assignment"}:
        infra.append("excluded_tensors.by_regime mismatch")
    if PREREG_PATH.is_file() and metrics.get("preregistration_sha256") != file_sha256(PREREG_PATH):
        infra.append("metrics.preregistration_sha256 mismatch")
    if "artifact_hashes.json" in loaded:
        validate_artifact_hashes(run_dir, loaded["artifact_hashes.json"], infra)
    return infra, loaded, rows


def evaluate(run_dir: Path) -> dict[str, Any]:
    infra, _loaded, rows = validate_packet(run_dir)
    if infra:
        result = {
            "decision": FAIL_INFRA,
            "run_dir": str(run_dir),
            "artifact_complete": False,
            "reasons": infra,
        }
    else:
        m10 = summarize_recovery(rows, "m10_position_binned")
        midpoint = summarize_recovery(rows, "midpoint_matched_cost")
        random_bin = summarize_recovery(rows, "random_bin_assignment")
        decision, reasons = decision_from_summaries(m10, midpoint, random_bin)
        result = {
            "decision": decision,
            "run_dir": str(run_dir),
            "artifact_complete": decision != FAIL_INFRA,
            "reasons": reasons,
            "primary_result": m10,
            "control_results": {
                "static_smoothquant": {"median_recovery": 0.0},
                "midpoint_matched_cost": midpoint,
                "random_bin_assignment": random_bin,
                "m10_minus_midpoint_median": (
                    None
                    if m10["median_recovery"] is None or midpoint["median_recovery"] is None
                    else float(m10["median_recovery"]) - float(midpoint["median_recovery"])
                ),
                "m10_minus_random_bin_median": (
                    None
                    if m10["median_recovery"] is None or random_bin["median_recovery"] is None
                    else float(m10["median_recovery"]) - float(random_bin["median_recovery"])
                ),
            },
            "thresholds": THRESHOLDS,
        }
    artifact_check = {
        "schema_version": f"{SCHEMA_VERSION}_artifact_check",
        "decision": result["decision"],
        "run_dir": str(run_dir),
        "artifact_complete": result.get("artifact_complete", False),
        "reasons": result["reasons"],
    }
    write_json(run_dir / "checker_result.json", result)
    write_json(run_dir / "artifact_check.json", artifact_check)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve() if args.run_dir else latest_run_dir().resolve()
    result = evaluate(run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result["decision"] == FAIL_INFRA else 0


if __name__ == "__main__":
    raise SystemExit(main())
