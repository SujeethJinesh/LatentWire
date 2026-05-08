import json
from pathlib import Path

import numpy as np

from experimental.cross_layer_error import check_cle_bound as checker


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


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


def _write_logits(path: Path, values: list[float]) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.array(values, dtype="<f4")
    path.write_bytes(array.tobytes(order="C"))
    return {
        "path": str(path.relative_to(path.parents[1])),
        "dtype": "float32_le",
        "shape": [int(array.size)],
        "bytes": path.stat().st_size,
        "sha256": checker.file_sha256(path),
    }


def _artifact_hashes(run_dir: Path) -> list[dict[str, object]]:
    excluded = {"artifact_hashes.json", "checker_result.json", "artifact_check.json"}
    return [
        {
            "path": str(path.relative_to(run_dir)),
            "bytes": path.stat().st_size,
            "sha256": checker.file_sha256(path),
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name not in excluded
    ]


def _write_packet(tmp_path: Path, *, predicted_value: float) -> Path:
    run_dir = tmp_path / f"cle_{predicted_value:g}"
    run_dir.mkdir()
    prompts = _prompts()
    prompt_sha = checker.prompt_payload_sha256(prompts)
    prompt_manifest = {
        "schema_version": f"{checker.SCHEMA_VERSION}_prompt_manifest",
        "source": checker.PROMPT_SOURCE,
        "selection": checker.PROMPT_SELECTION,
        "prompt_file_sha256": checker.file_sha256(checker.DEFAULT_PROMPT_FILE),
        "prompt_count": checker.TRACE_COUNT,
        "prompt_sha256": prompt_sha,
        "prompt_sha256_semantics": "sha256 of concatenated prompt text in deterministic index order",
        "prompts": prompts,
    }
    derivation = "# Synthetic CLE derivation\n\nF(N) is fixed before measurement.\n"
    (run_dir / "derivation.md").write_text(derivation, encoding="utf-8")
    bounds_by_depth = [
        {
            "depth": depth,
            "predicted_bound_l2": predicted_value,
            "quantized_layers": [f"model.layers.{idx}" for idx in range(depth)],
            "sigma_block_sum": 1.0,
            "sigma_outlier_sum": 1.0,
            "layer_error_l2_sum": 1.0,
        }
        for depth in checker.DEPTHS
    ]
    predicted_bounds = {
        "schema_version": f"{checker.SCHEMA_VERSION}_predicted_bounds",
        "model_id": checker.MODEL_ID,
        "depths": list(checker.DEPTHS),
        "created_before_measurement": True,
        "measurement_free_inputs_only": True,
        "bound_formula": "synthetic fixed upper bound",
        "derivation": "derivation.md",
        "derivation_sha256": checker.file_sha256(run_dir / "derivation.md"),
        "bounds_by_depth": bounds_by_depth,
    }
    prompt_ids = {int(row["index"]): str(row["prompt_id"]) for row in prompts}
    trace_rows = []
    for prompt_index in range(checker.TRACE_COUNT):
        token_ids = [prompt_index] * checker.DECODE_POSITION
        trace_rows.append(
            {
                "schema_version": f"{checker.SCHEMA_VERSION}_trace_tokens_row",
                "prompt_index": prompt_index,
                "prompt_id": prompt_ids[prompt_index],
                "decode_position": checker.DECODE_POSITION,
                "token_count": checker.DECODE_POSITION,
                "token_ids": token_ids,
                "token_ids_sha256": checker.bytes_sha256(
                    json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
                ),
                "source": "bf16_greedy_prefix",
            }
        )
    entries = []
    raw_rows = []
    for prompt_index in range(checker.TRACE_COUNT):
        bf16_rel = Path("logits") / f"bf16_prompt_{prompt_index:03d}.f32"
        bf16_entry = _write_logits(run_dir / bf16_rel, [0.0, 0.0, 0.0])
        entries.append(
            {
                "role": "bf16",
                "prompt_index": prompt_index,
                "prompt_id": prompt_ids[prompt_index],
                **bf16_entry,
            }
        )
        for depth in checker.DEPTHS:
            fp4_rel = Path("logits") / f"fp4_depth_{depth:02d}_prompt_{prompt_index:03d}.f32"
            fp4_entry = _write_logits(run_dir / fp4_rel, [10.0, 0.0, 0.0])
            entries.append(
                {
                    "role": "fp4",
                    "depth": depth,
                    "prompt_index": prompt_index,
                    "prompt_id": prompt_ids[prompt_index],
                    **fp4_entry,
                }
            )
            raw_rows.append(
                {
                    "schema_version": f"{checker.SCHEMA_VERSION}_raw_drift_row",
                    "prompt_index": prompt_index,
                    "prompt_id": prompt_ids[prompt_index],
                    "depth": depth,
                    "decode_position": checker.DECODE_POSITION,
                    "quantization_format": "nvfp4_e2m1_weight_sim",
                    "bf16_logits_path": str(bf16_rel),
                    "fp4_logits_path": str(fp4_rel),
                    "l2_drift": 10.0,
                    "bf16_argmax_token_id": 0,
                    "fp4_argmax_token_id": 0,
                }
            )
    predicted_by_depth = {int(row["depth"]): row for row in bounds_by_depth}
    metrics = checker.compute_metrics(
        raw_rows,
        predicted_by_depth,
        bootstrap_samples=checker.THRESHOLDS["bootstrap_samples"],
        seed=20260508,
    )
    metrics["prompt_sha256"] = prompt_sha
    bootstrap_ci = {
        "schema_version": f"{checker.SCHEMA_VERSION}_bootstrap_ci",
        "metric_name": metrics["metric_name"],
        "bootstrap_samples": metrics["bootstrap_samples"],
        "bootstrap_seed": metrics["bootstrap_seed"],
        "depth_metrics": [
            {
                "depth": row["depth"],
                "mean_l2_drift": row["mean_l2_drift"],
                "bootstrap_ci95": row["bootstrap_ci95"],
            }
            for row in metrics["depth_metrics"]
        ],
    }
    _write_json(run_dir / "environment.json", {"schema_version": f"{checker.SCHEMA_VERSION}_environment"})
    _write_json(
        run_dir / "model_provenance.json",
        {
            "schema_version": f"{checker.SCHEMA_VERSION}_model_provenance",
            "model_id": checker.MODEL_ID,
            "local_files_only": True,
            "hf_snapshot_commit": "synthetic",
            "snapshot_path": "/workspace/hf_cache/synthetic",
        },
    )
    _write_json(run_dir / "prompt_manifest.json", prompt_manifest)
    _write_json(
        run_dir / "command_metadata.json",
        {
            "schema_version": f"{checker.SCHEMA_VERSION}_command",
            "branch": "cross_layer_error",
            "model_id": checker.MODEL_ID,
            "depths": list(checker.DEPTHS),
            "decode_position": checker.DECODE_POSITION,
        },
    )
    _write_json(run_dir / "random_seed.json", {"seed": 20260508})
    _write_json(
        run_dir / "quantization_config.json",
        {
            "schema_version": f"{checker.SCHEMA_VERSION}_quantization_config",
            "quantization_format": "nvfp4_e2m1_weight_sim",
            "native_kernel_claim": False,
            "block_size": 16,
            "depths": list(checker.DEPTHS),
            "decode_position": checker.DECODE_POSITION,
        },
    )
    _write_json(run_dir / "predicted_bounds.json", predicted_bounds)
    _write_jsonl(run_dir / "trace_tokens.jsonl", trace_rows)
    _write_json(
        run_dir / "logits_manifest.json",
        {
            "schema_version": f"{checker.SCHEMA_VERSION}_logits_manifest",
            "dtype": "float32_le",
            "decode_position": checker.DECODE_POSITION,
            "entries": entries,
        },
    )
    _write_jsonl(run_dir / "raw_drift_rows.jsonl", raw_rows)
    _write_json(run_dir / "drift_metrics.json", metrics)
    _write_json(run_dir / "bootstrap_ci.json", bootstrap_ci)
    (run_dir / "logs").mkdir()
    (run_dir / "logs/stdout.log").write_text("synthetic\n", encoding="utf-8")
    (run_dir / "logs/stderr.log").write_text("synthetic\n", encoding="utf-8")
    _write_jsonl(
        run_dir / "run_events.jsonl",
        [
            {"event": "run_started"},
            {"event": "bound_predictions_written"},
            {"event": "derivation_locked"},
            {"event": "measurement_started"},
            {"event": "trace_tokens_written"},
            {"event": "run_completed"},
        ],
    )
    _write_json(run_dir / "artifact_hashes.json", {"artifacts": _artifact_hashes(run_dir)})
    return run_dir


def test_cle_checker_passes_tight_packet(tmp_path: Path) -> None:
    result = checker.evaluate(_write_packet(tmp_path, predicted_value=15.0))

    assert result["decision"] == checker.PASS
    assert result["artifact_complete"] is True


def test_cle_checker_kills_loose_bound_packet(tmp_path: Path) -> None:
    result = checker.evaluate(_write_packet(tmp_path, predicted_value=60.0))

    assert result["decision"] == checker.KILL
    assert result["artifact_complete"] is True
    assert any("ratio > 5.0" in reason for reason in result["reasons"])


def test_cle_checker_kills_underpredicted_bound_packet(tmp_path: Path) -> None:
    result = checker.evaluate(_write_packet(tmp_path, predicted_value=5.0))

    assert result["decision"] == checker.KILL
    assert result["artifact_complete"] is True
    assert result["depth_metrics"][0]["predicted_to_measured_ratio"] == 0.5


def test_cle_checker_rejects_bad_trace_token_hash(tmp_path: Path) -> None:
    run_dir = _write_packet(tmp_path, predicted_value=15.0)
    trace_path = run_dir / "trace_tokens.jsonl"
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["token_ids_sha256"] = "sha256:bad"
    _write_jsonl(trace_path, rows)
    _write_json(run_dir / "artifact_hashes.json", {"artifacts": _artifact_hashes(run_dir)})

    result = checker.evaluate(run_dir)

    assert result["decision"] == checker.FAIL_INFRA
    assert any("token_ids_sha256 mismatch" in reason for reason in result["reasons"])
