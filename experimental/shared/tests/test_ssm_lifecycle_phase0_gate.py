import json
from pathlib import Path

import numpy as np

from experimental.shared import check_phase0_gate as checker


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


def _state_arrays(*, ages: bool) -> dict[str, np.ndarray]:
    arrays = {}
    for prompt_index in range(checker.TRACE_COUNT):
        base = np.ones((32,), dtype=np.float32)
        final = np.full((32,), 3.0, dtype=np.float32) if ages else base.copy()
        for position in checker.POSITIONS:
            arrays[checker.ssml_npz_key(prompt_index, position)] = (
                base if position == checker.POSITIONS[0] else final
            )
    return arrays


def _write_packet(tmp_path: Path, *, ages: bool) -> Path:
    run_dir = tmp_path / ("ssml_ages" if ages else "ssml_stable")
    (run_dir / "ssm_states").mkdir(parents=True)
    prompts = _prompts()
    prompt_manifest = {
        "schema_version": f"{checker.SSML_SCHEMA_VERSION}_prompt_manifest",
        "source": "AIME-2025",
        "selection": "deterministic_indices_0_11",
        "prompt_file_sha256": checker.file_sha256(checker.DEFAULT_PROMPT_FILE),
        "prompt_count": 12,
        "prompt_sha256": checker.prompt_payload_sha256(prompts),
        "prompt_sha256_semantics": "sha256 of concatenated prompt text in deterministic index order",
        "prompts": prompts,
    }
    layers = [
        {
            "layer_index": layer_index,
            "layer_type": layer_type,
            "is_mamba": layer_type == "mamba",
            "observed_cache_state_shape": [1, *checker.GRANITE_TINY_SSM_STATE_SHAPE_WITHOUT_BATCH]
            if layer_type == "mamba"
            else [0],
        }
        for layer_index, layer_type in enumerate(checker.GRANITE_TINY_LAYER_TYPES)
    ]
    artifacts = []
    for layer in [row for row in layers if row["is_mamba"]]:
        layer_index = int(layer["layer_index"])
        rel = Path("ssm_states") / f"layer_{layer_index:03d}.npz"
        path = run_dir / rel
        np.savez_compressed(path, **_state_arrays(ages=ages))
        layer.update(
            {
                "artifact": str(rel),
                "state_shape_without_batch": list(checker.GRANITE_TINY_SSM_STATE_SHAPE_WITHOUT_BATCH),
                "record_count": checker.TRACE_COUNT * len(checker.POSITIONS),
                "artifact_sha256": checker.file_sha256(path),
                "artifact_bytes": path.stat().st_size,
            }
        )
        artifacts.append(
            {
                "layer_index": layer_index,
                "path": str(rel),
                "bytes": path.stat().st_size,
                "sha256": checker.file_sha256(path),
            }
        )
    manifest = {
        "schema_version": f"{checker.SSML_SCHEMA_VERSION}_ssm_state_manifest",
        "artifact_format": "numpy_npz_compressed_per_mamba_layer",
        "artifact_key_format": "p{prompt_index:03d}_pos{decode_position:05d}",
        "trace_count": checker.TRACE_COUNT,
        "positions": list(checker.POSITIONS),
        "layer_count": len(layers),
        "mamba_layer_count": len(checker.GRANITE_TINY_MAMBA_LAYER_INDICES),
        "layer_type_source": checker.REAL_SSML_LAYER_TYPE_SOURCE,
        "cache_state_source": "past_key_values.ssm_states",
        "layers": layers,
        "capture_record_count": checker.TRACE_COUNT
        * len(checker.GRANITE_TINY_MAMBA_LAYER_INDICES)
        * len(checker.POSITIONS),
        "expected_capture_record_count": checker.TRACE_COUNT
        * len(checker.GRANITE_TINY_MAMBA_LAYER_INDICES)
        * len(checker.POSITIONS),
        "artifacts": artifacts,
    }
    metrics = checker.compute_ssml_metrics(run_dir, manifest)
    metrics.update(
        {
            "branch": "ssm_lifecycle",
            "model_id": checker.MODEL_ID,
            "prompt_source": "AIME-2025",
            "prompt_selection": "deterministic_indices_0_11",
            "prompt_sha256": prompt_manifest["prompt_sha256"],
        }
    )
    _write_json(run_dir / "environment.json", {"schema_version": f"{checker.SSML_SCHEMA_VERSION}_environment"})
    _write_json(
        run_dir / "model_provenance.json",
        {
            "schema_version": f"{checker.SSML_SCHEMA_VERSION}_model_provenance",
            "model_id": checker.MODEL_ID,
            "local_files_only": True,
            "hf_snapshot_commit": "791e0d3d28c86e106c9b6e0b4cecdee0375b6124",
            "snapshot_path": "/workspace/hf_cache/synthetic",
        },
    )
    _write_json(run_dir / "prompt_manifest.json", prompt_manifest)
    _write_json(
        run_dir / "command_metadata.json",
        {
            "schema_version": f"{checker.SSML_SCHEMA_VERSION}_command",
            "branch": "ssm_lifecycle",
            "positions": list(checker.POSITIONS),
            "generation": {"do_sample": False, "num_beams": 1, "local_files_only": True},
        },
    )
    _write_json(run_dir / "random_seed.json", {"seed": 20260508})
    _write_json(run_dir / "ssm_state_manifest.json", manifest)
    _write_json(run_dir / "metrics.json", metrics)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "logs/stdout.log").write_text("synthetic\n", encoding="utf-8")
    (run_dir / "logs/stderr.log").write_text("synthetic\n", encoding="utf-8")
    (run_dir / "run_events.jsonl").write_text("{}\n", encoding="utf-8")
    artifact_entries = []
    state_artifacts = [
        f"ssm_states/layer_{layer_index:03d}.npz"
        for layer_index in sorted(checker.GRANITE_TINY_MAMBA_LAYER_INDICES)
    ]
    for rel in [*checker.SSML_HASHED_FILES, *state_artifacts]:
        path = run_dir / rel
        artifact_entries.append({"path": rel, "bytes": path.stat().st_size, "sha256": checker.file_sha256(path)})
    _write_json(run_dir / "artifact_hashes.json", {"artifacts": artifact_entries})
    return run_dir


def test_ssm_lifecycle_checker_passes_state_ages_packet(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(checker, "GRANITE_TINY_SSM_STATE_SHAPE_WITHOUT_BATCH", (32,))
    result = checker.evaluate(_write_packet(tmp_path, ages=True), branch="ssm_lifecycle")

    assert result["decision"] == checker.PASS_SSML
    assert result["artifact_complete"] is True
    assert result["mamba_layer_pass_fraction"] == 1.0


def test_ssm_lifecycle_checker_kills_stable_state_packet(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(checker, "GRANITE_TINY_SSM_STATE_SHAPE_WITHOUT_BATCH", (32,))
    result = checker.evaluate(_write_packet(tmp_path, ages=False), branch="ssm_lifecycle")

    assert result["decision"] == checker.KILL_SSML
    assert result["artifact_complete"] is True
    assert result["mamba_layer_pass_fraction"] == 0.0


def test_ssm_lifecycle_checker_rejects_bad_state_artifact_hash(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(checker, "GRANITE_TINY_SSM_STATE_SHAPE_WITHOUT_BATCH", (32,))
    run_dir = _write_packet(tmp_path, ages=True)
    with (run_dir / "ssm_states/layer_000.npz").open("ab") as handle:
        handle.write(b"tamper")

    result = checker.evaluate(run_dir, branch="ssm_lifecycle")

    assert result["decision"] == checker.FAIL_INFRA_SSML
    assert any("sha256 mismatch" in reason for reason in result["reasons"])


def test_ssm_lifecycle_checker_requires_full_granite_tiny_layer_layout(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(checker, "GRANITE_TINY_SSM_STATE_SHAPE_WITHOUT_BATCH", (32,))
    run_dir = _write_packet(tmp_path, ages=True)
    manifest_path = run_dir / "ssm_state_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["layers"] = manifest["layers"][:-1]
    manifest["layer_count"] = len(manifest["layers"])
    _write_json(manifest_path, manifest)

    result = checker.evaluate(run_dir, branch="ssm_lifecycle")

    assert result["decision"] == checker.FAIL_INFRA_SSML
    assert any("all 40 Granite-4.0-H-Tiny layers" in reason for reason in result["reasons"])


def test_ssm_lifecycle_checker_rejects_synthetic_cache_provenance(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(checker, "GRANITE_TINY_SSM_STATE_SHAPE_WITHOUT_BATCH", (32,))
    run_dir = _write_packet(tmp_path, ages=True)
    manifest_path = run_dir / "ssm_state_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cache_state_source"] = "synthetic"
    _write_json(manifest_path, manifest)

    result = checker.evaluate(run_dir, branch="ssm_lifecycle")

    assert result["decision"] == checker.FAIL_INFRA_SSML
    assert any("real model cache path" in reason for reason in result["reasons"])
