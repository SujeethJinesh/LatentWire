import gzip
import json
from pathlib import Path

from experimental.outlier_migrate.phase3 import check_phase3_intervention as checker


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prompts() -> list[dict[str, object]]:
    prompts = []
    for line in checker.DEFAULT_PROMPT_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            prompts.append(
                {
                    "index": int(item["index"]),
                    "prompt_id": item["prompt_id"],
                    "prompt": item["prompt"],
                    "answer": item["answer"],
                    "source_dataset": item["source_dataset"],
                    "source_file": item["source_file"],
                    "source_commit": item["source_commit"],
                }
            )
    return prompts


def _make_trace(prompt: dict[str, object], *, union_recovery: float, static2_recovery: float, avg_recovery: float) -> dict[str, object]:
    bf16 = 10.0
    static = 20.0
    gap = static - bf16
    return {
        "prompt_index": int(prompt["index"]),
        "prompt_id": prompt["prompt_id"],
        "perplexities": {
            "bf16": bf16,
            "static_1pct": static,
            "union_primary": bf16 + (1.0 - union_recovery) * gap,
            "static_2pct": bf16 + (1.0 - static2_recovery) * gap,
            "magnitude_average": bf16 + (1.0 - avg_recovery) * gap,
            "grid_sparse": bf16 + (1.0 - max(0.0, union_recovery - 0.05)) * gap,
            "grid_dense": bf16 + (1.0 - min(1.0, union_recovery + 0.05)) * gap,
        },
        "mean_nll": {
            "bf16": 2.30,
            "static_1pct": 3.00,
            "union_primary": 2.70,
            "static_2pct": 2.80,
            "magnitude_average": 2.85,
            "grid_sparse": 2.75,
            "grid_dense": 2.65,
        },
        "static_gap": gap,
        "no_recoverable_static_gap": False,
        "recoveries": {
            "union_primary": union_recovery,
            "static_2pct": static2_recovery,
            "magnitude_average": avg_recovery,
            "grid_sparse": max(0.0, union_recovery - 0.05),
            "grid_dense": min(1.0, union_recovery + 0.05),
        },
        "scored_tokens": checker.SCORING_WINDOW_TOKENS,
        "score_start": checker.SCORING_POSITION - checker.SCORING_WINDOW_TOKENS + 1,
        "score_end": checker.SCORING_POSITION,
    }


def _summary(values: list[float]) -> dict[str, object]:
    return {
        "median_recovery": values[0],
        "bootstrap_ci95": {"ci95_low": values[0], "ci95_high": values[0]},
        "trace_count": len(values),
        "per_trace_recovery": values,
    }


def _write_packet(
    tmp_path: Path,
    *,
    union_recovery: float,
    static2_recovery: float = 0.40,
    avg_recovery: float = 0.35,
) -> Path:
    run_dir = tmp_path / "packet"
    run_dir.mkdir()
    prompts = _prompts()
    prompt_manifest = {
        "schema_version": f"{checker.SCHEMA_VERSION}_prompt_manifest",
        "source": "AIME-2025",
        "selection": "deterministic_indices_0_23",
        "prompt_file_sha256": checker.EXPECTED_PROMPT_FILE_SHA256,
        "prompt_count": 24,
        "prompt_sha256": checker.prompt_payload_sha256(prompts),
        "prompt_sha256_semantics": "sha256 of concatenated prompt text in deterministic index order",
        "prompts": prompts,
    }
    rows = [
        _make_trace(prompt, union_recovery=union_recovery, static2_recovery=static2_recovery, avg_recovery=avg_recovery)
        for prompt in prompts
    ]
    union_values = [float(row["recoveries"]["union_primary"]) for row in rows]
    static2_values = [float(row["recoveries"]["static_2pct"]) for row in rows]
    avg_values = [float(row["recoveries"]["magnitude_average"]) for row in rows]
    sparse_values = [float(row["recoveries"]["grid_sparse"]) for row in rows]
    dense_values = [float(row["recoveries"]["grid_dense"]) for row in rows]
    _write_json(run_dir / "environment.json", {"schema_version": f"{checker.SCHEMA_VERSION}_environment"})
    (run_dir / "environment.txt").write_text("synthetic\n", encoding="utf-8")
    _write_json(run_dir / "model_provenance.json", {"model_id": checker.MODEL_ID, "hf_snapshot_commit": "synthetic"})
    _write_json(run_dir / "prompt_manifest.json", prompt_manifest)
    _write_json(run_dir / "command_metadata.json", {"branch": "outlier_migrate_phase3_intervention"})
    _write_json(run_dir / "random_seed.json", {"seed": checker.BOOTSTRAP_SEED})
    _write_json(
        run_dir / "decoding_config.json",
        {"scoring_position": checker.SCORING_POSITION, "scoring_window_tokens": checker.SCORING_WINDOW_TOKENS},
    )
    regimes = {}
    for name, positions in {
        "static_1pct": [100],
        "union_primary": list(checker.PRIMARY_GRID),
        "static_2pct": [100],
        "magnitude_average": list(checker.PRIMARY_GRID),
        "grid_sparse": list(checker.SPARSE_GRID),
        "grid_dense": list(checker.DENSE_GRID),
    }.items():
        regimes[name] = {
            "kind": "union" if name.startswith("grid") or name == "union_primary" else "single_position",
            "positions": positions,
            "layers": {"0": {"protected_channels": [0], "protected_count": 1}},
        }
    _write_json(run_dir / "protected_sets.json", {"schema_version": f"{checker.SCHEMA_VERSION}_protected_sets", "regimes": regimes})
    _write_json(
        run_dir / "quantization_config.json",
        {
            "weight_bits": 4,
            "scheme": "symmetric_per_output_channel_int4",
            "activation_dtype": "float16",
            "forbidden_methods": ["AWQ-style activation-aware scaling"],
        },
    )
    _write_json(run_dir / "excluded_tensors.json", {"by_regime": {}})
    _write_json(run_dir / "per_trace_metrics.json", {"traces": rows})
    _write_json(
        run_dir / "metrics.json",
        {
            "schema_version": f"{checker.SCHEMA_VERSION}_metrics",
            "model_id": checker.MODEL_ID,
            "prompt_sha256": prompt_manifest["prompt_sha256"],
            "primary_grid": list(checker.PRIMARY_GRID),
            "scoring_position": checker.SCORING_POSITION,
            "scoring_window_tokens": checker.SCORING_WINDOW_TOKENS,
            "thresholds": checker.THRESHOLDS,
            "primary_result": _summary(union_values),
            "preregistration_sha256": checker.file_sha256(checker.PREREG_PATH),
        },
    )
    _write_json(run_dir / "bootstrap_ci.json", {"primary_result": _summary(union_values)})
    _write_json(
        run_dir / "control_metrics.json",
        {
            "controls": {"static_2pct": _summary(static2_values), "magnitude_average": _summary(avg_values)},
            "union_primary": _summary(union_values),
        },
    )
    _write_json(
        run_dir / "grid_sensitivity_metrics.json",
        {
            "grids": {
                "sparse": {"positions": list(checker.SPARSE_GRID), **_summary(sparse_values)},
                "primary": {"positions": list(checker.PRIMARY_GRID), **_summary(union_values)},
                "dense": {"positions": list(checker.DENSE_GRID), **_summary(dense_values)},
            }
        },
    )
    _write_json(run_dir / "activation_magnitude_manifest.json", {"positions": list(checker.DENSE_GRID)})
    with gzip.open(run_dir / "activation_magnitudes.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write("{}\n")
    with gzip.open(run_dir / "bf16_traces.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write("{}\n")
    (run_dir / "logs").mkdir()
    (run_dir / "logs/stdout.log").write_text("synthetic\n", encoding="utf-8")
    (run_dir / "logs/stderr.log").write_text("synthetic\n", encoding="utf-8")
    (run_dir / "run_events.jsonl").write_text("{}\n", encoding="utf-8")
    artifacts = []
    for rel in checker.HASHED_FILES:
        path = run_dir / rel
        artifacts.append({"path": rel, "bytes": path.stat().st_size, "sha256": checker.file_sha256(path)})
    _write_json(run_dir / "artifact_hashes.json", {"artifacts": artifacts})
    return run_dir


def test_phase3_checker_passes_promotable_packet(tmp_path: Path) -> None:
    result = checker.evaluate(_write_packet(tmp_path, union_recovery=0.60))

    assert result["decision"] == checker.PASS_DECISION
    assert result["artifact_complete"] is True
    assert result["primary_result"]["median_recovery"] == 0.60


def test_phase3_checker_kills_low_recovery_packet(tmp_path: Path) -> None:
    result = checker.evaluate(_write_packet(tmp_path, union_recovery=0.10))

    assert result["decision"] == checker.KILL_DECISION
    assert result["artifact_complete"] is True


def test_phase3_checker_flags_control_stop_condition(tmp_path: Path) -> None:
    result = checker.evaluate(_write_packet(tmp_path, union_recovery=0.60, static2_recovery=0.75))

    assert result["decision"] == checker.PASS_DECISION
    assert result["control_results"]["control_stop_condition"] is True
    assert result["control_results"]["control_stop_details"][0]["control"] == "static_2pct"
