import gzip
import json
from pathlib import Path
from statistics import median

from experimental.outlier_migrate.phase4 import check_om_phase4_intervention as checker


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


def _make_trace(
    prompt: dict[str, object],
    *,
    union_recovery: float,
    static2_recovery: float,
    avg_recovery: float,
    no_recoverable_static_gap: bool = False,
) -> dict[str, object]:
    bf16 = 10.0
    static = 10.0 if no_recoverable_static_gap else 20.0
    gap = static - bf16
    recoveries = {
        "union_primary": 0.0 if no_recoverable_static_gap else union_recovery,
        "static_2pct": 0.0 if no_recoverable_static_gap else static2_recovery,
        "magnitude_average": 0.0 if no_recoverable_static_gap else avg_recovery,
        "grid_sparse": 0.0 if no_recoverable_static_gap else max(0.0, union_recovery - 0.05),
        "grid_dense": 0.0 if no_recoverable_static_gap else min(1.0, union_recovery + 0.05),
    }
    return {
        "prompt_index": int(prompt["index"]),
        "prompt_id": prompt["prompt_id"],
        "perplexities": {
            "bf16": bf16,
            "static_1pct": static,
            "union_primary": bf16 + (1.0 - recoveries["union_primary"]) * gap,
            "static_2pct": bf16 + (1.0 - recoveries["static_2pct"]) * gap,
            "magnitude_average": bf16 + (1.0 - recoveries["magnitude_average"]) * gap,
            "grid_sparse": bf16 + (1.0 - recoveries["grid_sparse"]) * gap,
            "grid_dense": bf16 + (1.0 - recoveries["grid_dense"]) * gap,
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
        "no_recoverable_static_gap": no_recoverable_static_gap,
        "recoveries": recoveries,
        "scored_tokens": checker.SCORING_WINDOW_TOKENS,
        "score_start": checker.SCORING_POSITION - checker.SCORING_WINDOW_TOKENS + 1,
        "score_end": checker.SCORING_POSITION,
    }


def _summary(values: list[float]) -> dict[str, object]:
    no_gap_count = sum(1 for value in values if value == 0.0)
    above_half_count = sum(1 for value in values if value > 0.50)
    return {
        "median_recovery": float(median(values)),
        "mean_recovery": sum(values) / len(values),
        "bootstrap_ci95": checker.bootstrap_median(values),
        "trace_count": len(values),
        "per_trace_recovery": values,
        "no_recoverable_static_gap_count": no_gap_count,
        "no_recoverable_static_gap_fraction": no_gap_count / len(values),
        "recovery_gt_0_50_count": above_half_count,
        "recovery_gt_0_50_fraction": above_half_count / len(values),
    }


def _write_activation_rows(run_dir: Path, prompts: list[dict[str, object]]) -> None:
    positions = list(checker.DENSE_GRID)
    row_count = 0
    with gzip.open(run_dir / "activation_magnitudes.jsonl.gz", "wt", encoding="utf-8") as handle:
        for prompt in prompts:
            for position in positions:
                values = [0.01 * index for index in range(16)]
                if position == 100:
                    values[0] = 10.0
                elif position == 500:
                    values[4] = 10.0
                elif position == 1000:
                    values[1] = 10.0
                elif position == 2000:
                    values[5] = 10.0
                elif position == 5000:
                    values[2] = 10.0
                elif position == 7500:
                    values[6] = 10.0
                elif position == 10000:
                    values[3] = 10.0
                handle.write(
                    json.dumps(
                        {
                            "schema_version": f"{checker.SCHEMA_VERSION}_activation_row",
                            "prompt_index": int(prompt["index"]),
                            "prompt_id": prompt["prompt_id"],
                            "layer_index": 0,
                            "layer_name": "synthetic.layers.0",
                            "decode_position": position,
                            "channel_count": len(values),
                            "channel_magnitudes": values,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                row_count += 1
    _write_json(
        run_dir / "activation_magnitude_manifest.json",
        {
            "positions": positions,
            "layer_count": 1,
            "row_count": row_count,
            "trace_count": len(prompts),
        },
    )


def _write_packet(
    tmp_path: Path,
    *,
    union_recovery: float,
    static2_recovery: float = 0.40,
    avg_recovery: float = 0.35,
    no_gap_count: int = 0,
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
    rows = []
    for offset, prompt in enumerate(prompts):
        rows.append(
            _make_trace(
                prompt,
                union_recovery=union_recovery,
                static2_recovery=static2_recovery,
                avg_recovery=avg_recovery,
                no_recoverable_static_gap=offset < no_gap_count,
            )
        )
    union_values = [float(row["recoveries"]["union_primary"]) for row in rows]
    static2_values = [float(row["recoveries"]["static_2pct"]) for row in rows]
    avg_values = [float(row["recoveries"]["magnitude_average"]) for row in rows]
    sparse_values = [float(row["recoveries"]["grid_sparse"]) for row in rows]
    dense_values = [float(row["recoveries"]["grid_dense"]) for row in rows]
    _write_json(run_dir / "environment.json", {"schema_version": f"{checker.SCHEMA_VERSION}_environment"})
    (run_dir / "environment.txt").write_text("synthetic\n", encoding="utf-8")
    _write_json(
        run_dir / "model_provenance.json",
        {"model_id": checker.MODEL_ID, "hf_snapshot_commit": checker.EXPECTED_MODEL_SNAPSHOT_COMMIT},
    )
    _write_json(run_dir / "prompt_manifest.json", prompt_manifest)
    _write_json(run_dir / "command_metadata.json", {"branch": "outlier_migrate_phase4_intervention"})
    _write_json(run_dir / "random_seed.json", {"seed": checker.BOOTSTRAP_SEED})
    _write_json(
        run_dir / "decoding_config.json",
        {
            "scoring_position": checker.SCORING_POSITION,
            "scoring_window_tokens": checker.SCORING_WINDOW_TOKENS,
            "do_sample": False,
            "num_beams": 1,
        },
    )
    _write_activation_rows(run_dir, prompts)
    _write_json(
        run_dir / "protected_sets.json",
        {
            "schema_version": f"{checker.SCHEMA_VERSION}_protected_sets",
            "regimes": checker.expected_protected_sets_from_activations(run_dir / "activation_magnitudes.jsonl.gz"),
        },
    )
    _write_json(
        run_dir / "quantization_config.json",
        {
            "weight_bits": 4,
            "scheme": "symmetric_per_output_channel_int4",
            "activation_dtype": "float16",
            "protected_channel_dtype": "bfloat16",
            "forbidden_methods": [
                "AWQ-style activation-aware scaling",
                "SmoothQuant-style activation folding",
            ],
        },
    )
    _write_json(
        run_dir / "excluded_tensors.json",
        {
            "by_regime": {
                regime: {
                    "quantized_tensor_count": 1,
                    "quantized_tensors": [
                        {
                            "name": f"synthetic.{regime}",
                            "protected_channel_count": 1,
                            "row_protected": True,
                            "col_protected": False,
                        }
                    ],
                    "excluded_tensors": [],
                }
                for regime in ["static_1pct", *checker.RECOVERY_REGIMES]
            }
        },
    )
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
            "no_recoverable_static_gap_count": no_gap_count,
            "no_recoverable_static_gap_fraction": no_gap_count / len(rows),
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
    with gzip.open(run_dir / "bf16_traces.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write("{}\n")
    _write_json(
        run_dir / "bf16_trace_manifest.json",
        {
            "schema_version": f"{checker.SCHEMA_VERSION}_bf16_trace_manifest",
            "trace_count": len(prompts),
            "max_new_tokens": checker.SCORING_POSITION,
            "decode_policy": "manual greedy decode; EOS recorded but ignored for fixed-length traces",
            "artifact_sha256": checker.file_sha256(run_dir / "bf16_traces.jsonl.gz"),
        },
    )
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


def test_phase4_checker_passes_promotable_packet(tmp_path: Path) -> None:
    result = checker.evaluate(_write_packet(tmp_path, union_recovery=0.60))

    assert result["decision"] == checker.PASS_DECISION
    assert result["artifact_complete"] is True
    assert result["primary_result"]["median_recovery"] == 0.60


def test_phase4_checker_kills_low_recovery_packet(tmp_path: Path) -> None:
    result = checker.evaluate(_write_packet(tmp_path, union_recovery=0.10))

    assert result["decision"] == checker.KILL_DECISION
    assert result["artifact_complete"] is True


def test_phase4_checker_kills_no_gap_packet_even_if_median_passes(tmp_path: Path) -> None:
    result = checker.evaluate(_write_packet(tmp_path, union_recovery=0.80, no_gap_count=7))

    assert result["decision"] == checker.KILL_DECISION
    assert result["artifact_complete"] is True
    assert "no recoverable static gap" in result["reasons"][0]


def test_phase4_checker_flags_control_stop_condition(tmp_path: Path) -> None:
    run_dir = _write_packet(tmp_path, union_recovery=0.60, static2_recovery=0.75)
    result = checker.evaluate(run_dir)

    assert result["decision"] == checker.PASS_DECISION
    assert result["control_results"]["control_stop_condition"] is True
    assert result["control_results"]["control_stop_details"][0]["control"] == "static_2pct"
    assert checker.main(["--run-dir", str(run_dir)]) == 2
