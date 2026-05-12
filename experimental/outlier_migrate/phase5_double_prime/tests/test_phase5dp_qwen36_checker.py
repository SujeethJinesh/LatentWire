import json
from pathlib import Path

from experimental.outlier_migrate.phase5_double_prime import check_om_phase5dp_qwen36 as checker


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_phase5dp_checker_accepts_artifact_complete_infra_packet(tmp_path: Path) -> None:
    run_dir = tmp_path / "packet"
    (run_dir / "logs").mkdir(parents=True)
    _write_json(run_dir / "environment.json", {"schema_version": f"{checker.SCHEMA_VERSION}_environment"})
    _write_json(
        run_dir / "model_provenance.json",
        {
            "model_id": checker.MODEL_ID,
            "hf_snapshot_commit": checker.MODEL_SNAPSHOT_COMMIT,
            "snapshot_path": "/workspace/hf_cache/synthetic",
        },
    )
    _write_json(
        run_dir / "prompt_manifest.json",
        {"prompt_sha256": "sha256:aa038b29332b6d137d558205ee441163e7ea4cb3cc323eb705a2f5928fd2fe4e"},
    )
    _write_json(run_dir / "command_metadata.json", {"branch": "outlier_migrate_phase5_double_prime_qwen36"})
    _write_json(
        run_dir / "capability_probe.json",
        {
            "activation_capture_accessible": False,
            "failure_classification": "per_layer_activation_capture_unavailable",
        },
    )
    _write_json(run_dir / "infra_error.json", {"reasons": ["no per-layer hook"]})
    (run_dir / "logs/stdout.log").write_text("synthetic\n", encoding="utf-8")
    (run_dir / "logs/stderr.log").write_text("synthetic\n", encoding="utf-8")
    (run_dir / "run_events.jsonl").write_text("{}\n", encoding="utf-8")
    artifacts = []
    for rel in checker.HASHED_FILES:
        path = run_dir / rel
        artifacts.append({"path": rel, "bytes": path.stat().st_size, "sha256": checker.file_sha256(path)})
    _write_json(run_dir / "artifact_hashes.json", {"artifacts": artifacts})

    result = checker.evaluate(run_dir)

    assert result["decision"] == checker.FAIL_INFRA
    assert result["artifact_complete"] is True
