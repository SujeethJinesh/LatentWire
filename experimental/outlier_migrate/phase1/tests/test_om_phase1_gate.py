import gzip
import json
from pathlib import Path

from experimental.outlier_migrate.phase1 import check_om_phase1 as checker


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return checker.file_sha256(path)


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


def _vector(*, migrated: bool) -> tuple[list[float], list[float]]:
    base = [0.0] * 100
    final = [0.0] * 100
    base[0] = 100.0
    if migrated:
        for channel in range(1, 6):
            final[channel] = float(100 - channel)
        final[0] = 1.0
    else:
        final[0] = 100.0
    return base, final


def _write_packet(tmp_path: Path, *, migrated: bool) -> Path:
    run_dir = tmp_path / ("dynamic" if migrated else "failed_at_scale")
    run_dir.mkdir()
    prompts = _prompts()
    prompt_manifest = {
        "source": "AIME-2025",
        "selection": "deterministic_indices_0_23",
        "prompt_file_sha256": checker.EXPECTED_PROMPT_FILE_SHA256,
        "prompt_count": 24,
        "prompt_sha256": checker.prompt_payload_sha256(prompts),
        "prompt_sha256_semantics": "sha256 of concatenated prompt text in deterministic index order",
        "prompts": prompts,
    }
    rows = []
    for prompt in prompts:
        for layer_index in range(2):
            base, final = _vector(migrated=migrated)
            for position in checker.POSITIONS:
                values = base if position == 100 else final
                rows.append(
                    {
                        "prompt_index": prompt["index"],
                        "prompt_id": prompt["prompt_id"],
                        "layer_index": layer_index,
                        "layer_name": f"model.layers.{layer_index}",
                        "decode_position": position,
                        "channel_count": 100,
                        "channel_magnitudes": values,
                    }
                )
    with gzip.open(run_dir / "activation_magnitudes.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    metrics_core = checker.compute_metrics(rows, bootstrap_samples=1000, seed=20260508)
    metrics = {
        **metrics_core,
        "thresholds": checker.THRESHOLDS,
        "model_id": checker.MODEL_ID,
        "phase0_decision": "PASS_OM_PHASE0_DECODE_TIME_MIGRATION",
        "prompt_source": "AIME-2025",
        "prompt_selection": "deterministic_indices_0_23",
        "prompt_sha256": prompt_manifest["prompt_sha256"],
        "positions": list(checker.POSITIONS),
        "layer_count": 2,
        "hidden_size": 100,
        "activation_artifact": "activation_magnitudes.jsonl.gz",
        "activation_artifact_sha256": _sha(run_dir / "activation_magnitudes.jsonl.gz"),
    }
    _write_json(run_dir / "environment.json", {"schema_version": "om_phase1_v1_environment"})
    _write_json(
        run_dir / "model_provenance.json",
        {
            "model_id": checker.MODEL_ID,
            "hf_snapshot_commit": "synthetic-commit",
            "snapshot_path": "/workspace/hf_cache/synthetic",
        },
    )
    _write_json(run_dir / "prompt_manifest.json", prompt_manifest)
    _write_json(run_dir / "command_metadata.json", {"branch": "outlier_migrate_phase1"})
    _write_json(run_dir / "random_seed.json", {"seed": 20260508})
    _write_json(
        run_dir / "activation_magnitude_manifest.json",
        {
            "positions": list(checker.POSITIONS),
            "trace_count": 24,
            "layer_count": 2,
            "layer_names": ["model.layers.0", "model.layers.1"],
            "row_count": len(rows),
        },
    )
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(
        run_dir / "bootstrap_ci.json",
        {
            "bootstrap_samples": 1000,
            "bootstrap_ci95": metrics["bootstrap_ci95"],
            "migration_fraction": metrics["migration_fraction"],
        },
    )
    (run_dir / "logs").mkdir()
    (run_dir / "logs/stdout.log").write_text("synthetic\n", encoding="utf-8")
    (run_dir / "logs/stderr.log").write_text("synthetic\n", encoding="utf-8")
    (run_dir / "run_events.jsonl").write_text("{}\n", encoding="utf-8")
    artifacts = []
    for rel in checker.HASHED_FILES:
        path = run_dir / rel
        artifacts.append({"path": rel, "bytes": path.stat().st_size, "sha256": _sha(path)})
    _write_json(run_dir / "artifact_hashes.json", {"artifacts": artifacts})
    return run_dir


def test_outlier_migrate_phase1_checker_passes_dynamic_packet(tmp_path: Path) -> None:
    result = checker.evaluate(_write_packet(tmp_path, migrated=True))

    assert result["decision"] == checker.PASS_DYNAMIC
    assert result["artifact_complete"] is True


def test_outlier_migrate_phase1_checker_kills_failed_at_scale_packet(tmp_path: Path) -> None:
    result = checker.evaluate(_write_packet(tmp_path, migrated=False))

    assert result["decision"] == checker.KILL_FAILED_AT_SCALE
    assert result["artifact_complete"] is True


def test_phase1_top_channels_are_selected_by_layer_mean_not_per_trace() -> None:
    rows = []
    for prompt_index in range(24):
        base = [0.0] * 100
        final = [0.0] * 100
        if prompt_index == 0:
            base[0] = 100.0
            base[1] = 90.0
            final[0] = 1.0
            final[1] = 100.0
        else:
            base[1] = 100.0
            base[0] = 0.0
            final[1] = 100.0
            final[0] = 0.0
        for position in checker.POSITIONS:
            rows.append(
                {
                    "prompt_index": prompt_index,
                    "layer_index": 0,
                    "decode_position": position,
                    "channel_count": 100,
                    "channel_magnitudes": base if position == 100 else final,
                }
            )

    metrics = checker.compute_metrics(rows, bootstrap_samples=1000, seed=20260508)

    assert metrics["top_channels_by_layer"] == {"0": [1]}
    assert metrics["migration_fraction"] == 0.0


def test_phase1_decision_rule_surfaces_ci_overlap_as_prereg_ambiguous() -> None:
    decision, reasons = checker.decision_from_metrics(0.05, 0.0, 0.1)

    assert decision == checker.PREREG_AMBIGUOUS
    assert "preregistration ambiguous" in reasons[0]
