import json
from pathlib import Path

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


def _wrong_answer(answer: object) -> str:
    normalized = checker.normalize_aime_answer(answer)
    return "999" if normalized != "999" else "998"


def _generation_row(prompt: dict[str, object], *, phase: str, correct: bool) -> dict[str, object]:
    answer = checker.normalize_aime_answer(prompt["answer"])
    emitted = answer if correct else _wrong_answer(answer)
    generated_text = f"Final answer: {emitted}"
    extracted = checker.extract_aime_answer(generated_text)
    return {
        "phase": phase,
        "schema_version": f"{checker.RM_SCHEMA_VERSION}_generation_row",
        "prompt_index": int(prompt["index"]),
        "prompt_id": str(prompt["prompt_id"]),
        "input_token_count": 32,
        "max_new_tokens": 2048,
        "generated_token_count": 8,
        "generated_text": generated_text,
        "canonical_answer": answer,
        "extracted_answer": extracted,
        "correct": extracted == answer,
        "scoring_rule": "synthetic exact-match row",
    }


def _write_packet(tmp_path: Path, *, ablation_wrong_indices: set[int]) -> Path:
    run_dir = tmp_path / ("rm_" + "_".join(map(str, sorted(ablation_wrong_indices) or ["none"])))
    run_dir.mkdir()
    prompts = _prompts()
    prompt_manifest = {
        "schema_version": f"{checker.RM_SCHEMA_VERSION}_prompt_manifest",
        "source": "AIME-2025",
        "selection": "deterministic_indices_0_11",
        "prompt_file_sha256": checker.file_sha256(checker.DEFAULT_PROMPT_FILE),
        "prompt_count": 12,
        "prompt_sha256": checker.prompt_payload_sha256(prompts),
        "prompt_sha256_semantics": "sha256 of concatenated prompt text in deterministic index order",
        "prompts": prompts,
    }
    rows = []
    for prompt in prompts:
        prompt_index = int(prompt["index"])
        rows.append(_generation_row(prompt, phase="baseline", correct=True))
        rows.append(
            _generation_row(prompt, phase="ablation", correct=prompt_index not in ablation_wrong_indices)
        )
    with (run_dir / "generations.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    metrics_core = checker.compute_residual_metrics(rows, bootstrap_samples=1000, seed=20260508)
    metrics = {
        **metrics_core,
        "schema_version": f"{checker.RM_SCHEMA_VERSION}_metrics",
        "metric_name": "aime_accuracy_drop_after_residual_95p_clip",
        "thresholds": checker.RM_THRESHOLDS,
        "branch": "residual_migration",
        "model_id": checker.MODEL_ID,
        "prompt_source": "AIME-2025",
        "prompt_selection": "deterministic_indices_0_11",
        "prompt_sha256": prompt_manifest["prompt_sha256"],
        "trace_count": 12,
        "generation_artifact": "generations.jsonl",
        "generation_artifact_sha256": checker.file_sha256(run_dir / "generations.jsonl"),
    }
    _write_json(run_dir / "environment.json", {"schema_version": f"{checker.RM_SCHEMA_VERSION}_environment"})
    _write_json(
        run_dir / "model_provenance.json",
        {
            "schema_version": f"{checker.RM_SCHEMA_VERSION}_model_provenance",
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
            "schema_version": f"{checker.RM_SCHEMA_VERSION}_command",
            "branch": "residual_migration",
            "frozen_generation_limit": {"max_new_tokens": 2048, "set_before_analysis": True},
            "generation": {"do_sample": False, "num_beams": 1, "local_files_only": True},
            "layer_count": 1,
        },
    )
    _write_json(run_dir / "random_seed.json", {"seed": 20260508})
    _write_json(
        run_dir / "ablation_config.json",
        {
            "schema_version": f"{checker.RM_SCHEMA_VERSION}_ablation_config",
            "clip_quantile": 0.95,
            "clip_rule": "synthetic forward pre-hook residual clip",
            "clip_stats": {
                "hook_type": "forward_pre_hook",
                "layers": {
                    "0": {
                        "layer_index": 0,
                        "layer_name": "model.layers.0",
                        "invocations": 12,
                        "total_values": 1200,
                        "clipped_values": 60,
                        "clip_fraction": 0.05,
                    }
                },
            },
            "layer_count": 1,
        },
    )
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(
        run_dir / "bootstrap_ci.json",
        {
            "schema_version": f"{checker.RM_SCHEMA_VERSION}_bootstrap_ci",
            "metric_name": metrics["metric_name"],
            "bootstrap_samples": 1000,
            "bootstrap_seed": 20260508,
            "bootstrap_ci95": metrics["bootstrap_ci95"],
            "accuracy_drop": metrics["accuracy_drop"],
        },
    )
    (run_dir / "logs").mkdir()
    (run_dir / "logs/stdout.log").write_text("synthetic\n", encoding="utf-8")
    (run_dir / "logs/stderr.log").write_text("synthetic\n", encoding="utf-8")
    (run_dir / "run_events.jsonl").write_text("{}\n", encoding="utf-8")
    artifacts = []
    for rel in checker.RM_HASHED_FILES:
        path = run_dir / rel
        artifacts.append({"path": rel, "bytes": path.stat().st_size, "sha256": checker.file_sha256(path)})
    _write_json(run_dir / "artifact_hashes.json", {"artifacts": artifacts})
    return run_dir


def test_residual_migration_checker_passes_rethinking_replicates(tmp_path: Path) -> None:
    result = checker.evaluate(_write_packet(tmp_path, ablation_wrong_indices=set()), branch="residual_migration")

    assert result["decision"] == checker.PASS_RM_REPLICATES
    assert result["artifact_complete"] is True


def test_residual_migration_checker_passes_hybrids_depend_on_residual(tmp_path: Path) -> None:
    result = checker.evaluate(
        _write_packet(tmp_path, ablation_wrong_indices=set(range(12))),
        branch="residual_migration",
    )

    assert result["decision"] == checker.PASS_RM_REJECTS
    assert result["artifact_complete"] is True


def test_residual_migration_checker_kills_ambiguous_drop(tmp_path: Path) -> None:
    result = checker.evaluate(_write_packet(tmp_path, ablation_wrong_indices={0}), branch="residual_migration")

    assert result["decision"] == checker.KILL_RM_AMBIGUOUS
    assert result["artifact_complete"] is True


def test_residual_migration_checker_rejects_incomplete_layer_hook_coverage(tmp_path: Path) -> None:
    run_dir = _write_packet(tmp_path, ablation_wrong_indices=set())
    command = json.loads((run_dir / "command_metadata.json").read_text(encoding="utf-8"))
    command["layer_count"] = 2
    _write_json(run_dir / "command_metadata.json", command)
    artifacts = []
    for rel in checker.RM_HASHED_FILES:
        path = run_dir / rel
        artifacts.append({"path": rel, "bytes": path.stat().st_size, "sha256": checker.file_sha256(path)})
    _write_json(run_dir / "artifact_hashes.json", {"artifacts": artifacts})

    result = checker.evaluate(run_dir, branch="residual_migration")

    assert result["decision"] == checker.FAIL_INFRA_RM
    assert any("every discovered transformer layer" in reason for reason in result["reasons"])


def test_residual_migration_checker_rejects_nonfrozen_generation_limit(tmp_path: Path) -> None:
    run_dir = _write_packet(tmp_path, ablation_wrong_indices=set())
    command = json.loads((run_dir / "command_metadata.json").read_text(encoding="utf-8"))
    command["frozen_generation_limit"]["max_new_tokens"] = 1024
    _write_json(run_dir / "command_metadata.json", command)
    rows = []
    for line in (run_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        row["max_new_tokens"] = 1024
        rows.append(row)
    with (run_dir / "generations.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics["generation_artifact_sha256"] = checker.file_sha256(run_dir / "generations.jsonl")
    _write_json(run_dir / "metrics.json", metrics)
    artifacts = []
    for rel in checker.RM_HASHED_FILES:
        path = run_dir / rel
        artifacts.append({"path": rel, "bytes": path.stat().st_size, "sha256": checker.file_sha256(path)})
    _write_json(run_dir / "artifact_hashes.json", {"artifacts": artifacts})

    result = checker.evaluate(run_dir, branch="residual_migration")

    assert result["decision"] == checker.FAIL_INFRA_RM
    assert any("frozen value 2048" in reason for reason in result["reasons"])


def test_residual_answer_extraction_requires_explicit_final_answer() -> None:
    assert checker.extract_aime_answer("I tried 123 and then changed direction.") is None
    assert checker.extract_aime_answer("Therefore 123.") is None
    assert checker.extract_aime_answer("Final answer: 123") == "123"
    assert checker.extract_aime_answer("Therefore, the answer is 123.") == "123"
