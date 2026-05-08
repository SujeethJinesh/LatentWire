import json
from pathlib import Path

from experimental.residual_migration.phase1 import check_rm_phase1 as checker


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
        "ablation_set": None if phase == "baseline" else phase,
        "schema_version": f"{checker.SCHEMA_VERSION}_generation_row",
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


def _clip_stats(layer_indices: list[int]) -> dict[str, object]:
    return {
        "ablation_set": "synthetic",
        "clip_quantile": 0.95,
        "threshold_scope": "per layer, per batch element, per forwarded token position, over hidden channels",
        "hook_type": "forward_pre_hook",
        "target_layer_indices": layer_indices,
        "layers": {
            str(index): {
                "layer_index": index,
                "layer_name": f"model.layers.{index}",
                "invocations": 24,
                "total_values": 2400,
                "clipped_values": 120,
                "clip_fraction": 0.05,
                "max_abs_observed": 2.0,
                "max_threshold_observed": 1.0,
            }
            for index in layer_indices
        },
    }


def _write_packet(
    tmp_path: Path,
    *,
    baseline_correct_indices: set[int] | None = None,
    full_ablation_wrong_indices: set[int] | None = None,
    omit_attention_stats: bool = False,
) -> Path:
    baseline_correct_indices = set(range(24)) if baseline_correct_indices is None else baseline_correct_indices
    full_ablation_wrong_indices = set() if full_ablation_wrong_indices is None else full_ablation_wrong_indices
    run_dir = tmp_path / "rm_phase1_synthetic"
    run_dir.mkdir()
    prompts = _prompts()
    prompt_manifest = {
        "schema_version": f"{checker.SCHEMA_VERSION}_prompt_manifest",
        "source": "AIME-2025",
        "selection": "deterministic_indices_0_23",
        "source_dataset": checker.EXPECTED_PROMPT_SOURCE_DATASET,
        "source_dataset_commit": checker.EXPECTED_PROMPT_SOURCE_COMMIT,
        "source_file_order": ["aime2025-I.jsonl", "aime2025-II.jsonl"],
        "prompt_file_sha256": checker.file_sha256(checker.DEFAULT_PROMPT_FILE),
        "prompt_count": 24,
        "prompt_sha256": checker.prompt_payload_sha256(prompts),
        "prompt_sha256_semantics": "sha256 of concatenated prompt text in deterministic index order",
        "prompts": prompts,
    }
    rows_by_phase = {phase: [] for phase in checker.ALL_PHASES}
    for prompt in prompts:
        prompt_index = int(prompt["index"])
        rows_by_phase["baseline"].append(
            _generation_row(prompt, phase="baseline", correct=prompt_index in baseline_correct_indices)
        )
        for phase in checker.ABLATION_SETS:
            correct = prompt_index in baseline_correct_indices
            if phase == "full_ablation" and prompt_index in full_ablation_wrong_indices:
                correct = False
            rows_by_phase[phase].append(_generation_row(prompt, phase=phase, correct=correct))
    with (run_dir / "generations.jsonl").open("w", encoding="utf-8") as handle:
        for phase in checker.ALL_PHASES:
            for row in rows_by_phase[phase]:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    computed_by_set = {
        phase: checker.compute_ablation_metrics(
            rows_by_phase["baseline"],
            rows_by_phase[phase],
            bootstrap_samples=1000,
            seed=20260508,
            ablation_set=phase,
        )
        for phase in checker.ABLATION_SETS
    }
    full_metrics = computed_by_set["full_ablation"]
    metrics = {
        **full_metrics,
        "schema_version": f"{checker.SCHEMA_VERSION}_metrics",
        "metric_name": "aime_accuracy_drop_after_residual_95p_clip",
        "branch": "residual_migration_phase1",
        "phase0_decision": checker.PHASE0_DECISION,
        "model_id": checker.MODEL_ID,
        "prompt_source": "AIME-2025",
        "prompt_selection": "deterministic_indices_0_23",
        "prompt_sha256": prompt_manifest["prompt_sha256"],
        "trace_count": 24,
        "generation_artifact": "generations.jsonl",
        "generation_artifact_sha256": checker.file_sha256(run_dir / "generations.jsonl"),
        "thresholds": checker.THRESHOLDS,
    }
    attention = [5, 15, 25, 35]
    mamba = [index for index in range(40) if index not in attention]
    layer_groups = {
        "full_ablation": list(range(40)),
        "first_half": list(range(20)),
        "second_half": list(range(20, 40)),
        "attention_only": attention,
        "mamba_only": mamba,
    }
    clip_stats = {name: _clip_stats(indices) for name, indices in layer_groups.items()}
    if omit_attention_stats:
        clip_stats.pop("attention_only")
    _write_json(run_dir / "environment.json", {"schema_version": f"{checker.SCHEMA_VERSION}_environment"})
    _write_json(
        run_dir / "model_provenance.json",
        {
            "schema_version": f"{checker.SCHEMA_VERSION}_model_provenance",
            "model_id": checker.MODEL_ID,
            "local_files_only": True,
            "hf_snapshot_commit": "3f201b081dd005ecfa5a050f526f9948cc8cba00",
            "snapshot_path": "/workspace/hf_cache/synthetic",
        },
    )
    _write_json(run_dir / "prompt_manifest.json", prompt_manifest)
    _write_json(
        run_dir / "command_metadata.json",
        {
            "schema_version": f"{checker.SCHEMA_VERSION}_command",
            "branch": "residual_migration_phase1",
            "phase0_decision": checker.PHASE0_DECISION,
            "frozen_generation_limit": {"max_new_tokens": 2048, "set_before_analysis": True},
            "generation": {"do_sample": False, "num_beams": 1, "local_files_only": True},
            "headroom_guard": {
                "baseline_accuracy_recorded": True,
                "oracle_answer_key_diagnostic_recorded": True,
                "decision_thresholds_unchanged": True,
            },
            "layer_count": 40,
        },
    )
    _write_json(run_dir / "random_seed.json", {"seed": 20260508})
    _write_json(
        run_dir / "ablation_config.json",
        {
            "schema_version": f"{checker.SCHEMA_VERSION}_ablation_config",
            "clip_quantile": 0.95,
            "clip_rule": "synthetic forward pre-hook residual clip",
            "layer_count": 40,
            "layer_groups": layer_groups,
            "clip_stats_by_ablation_set": clip_stats,
        },
    )
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(
        run_dir / "bootstrap_ci.json",
        {
            "schema_version": f"{checker.SCHEMA_VERSION}_bootstrap_ci",
            "metric_name": metrics["metric_name"],
            "bootstrap_samples": 1000,
            "bootstrap_seed": 20260508,
            "bootstrap_ci95": metrics["bootstrap_ci95"],
            "accuracy_drop": metrics["accuracy_drop"],
        },
    )
    _write_json(
        run_dir / "stratified_metrics.json",
        {
            "schema_version": f"{checker.SCHEMA_VERSION}_stratified_metrics",
            "ablation_sets": computed_by_set,
            "baseline_accuracy": full_metrics["baseline_accuracy"],
            "layer_groups": layer_groups,
            "attribution_only": True,
            "decision_ablation_set": "full_ablation",
        },
    )
    _write_json(
        run_dir / "headroom_diagnostics.json",
        {
            "schema_version": f"{checker.SCHEMA_VERSION}_headroom_diagnostics",
            "prompt_count": 24,
            "baseline_correct_count": sum(1 for row in full_metrics["per_prompt"] if row["baseline_correct"]),
            "baseline_accuracy": full_metrics["baseline_accuracy"],
            "full_ablation_accuracy": full_metrics["ablation_accuracy"],
            "extractor_failure_count": sum(
                1 for row in full_metrics["per_prompt"] if row["baseline_extracted_answer"] is None
            ),
            "lenient_oracle_answer_mention_count": sum(
                1
                for row in full_metrics["per_prompt"]
                if bool(row["baseline_generated_text_contains_canonical"])
            ),
            "lenient_oracle_accuracy": sum(
                1
                for row in full_metrics["per_prompt"]
                if bool(row["baseline_generated_text_contains_canonical"])
            )
            / 24,
            "headroom_status": "NO_BASELINE_HEADROOM"
            if not any(row["baseline_correct"] for row in full_metrics["per_prompt"])
            else "USABLE_BASELINE_HEADROOM",
            "oracle_answer_key_available": True,
            "oracle_answer_key_correct_count": 24,
            "oracle_answer_key_source": "prompt_manifest canonical AIME-2025 answers",
            "decision_thresholds_unchanged": True,
            "capability_claim_blocked": not any(row["baseline_correct"] for row in full_metrics["per_prompt"]),
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


def test_residual_migration_phase1_checker_passes_replicated_drop(tmp_path: Path) -> None:
    result = checker.evaluate(_write_packet(tmp_path))

    assert result["decision"] == checker.PASS_REPLICATED
    assert result["artifact_complete"] is True
    assert result["headroom_warning"] is False


def test_residual_migration_phase1_checker_kills_failed_at_scale(tmp_path: Path) -> None:
    result = checker.evaluate(_write_packet(tmp_path, full_ablation_wrong_indices={0}))

    assert result["decision"] == checker.KILL_FAILED_AT_SCALE
    assert result["artifact_complete"] is True


def test_residual_migration_phase1_checker_flags_zero_baseline_headroom(tmp_path: Path) -> None:
    result = checker.evaluate(_write_packet(tmp_path, baseline_correct_indices=set()))

    assert result["decision"] == checker.PASS_REPLICATED
    assert result["artifact_complete"] is True
    assert result["headroom_warning"] is True
    assert result["headroom_diagnostics"]["capability_claim_blocked"] is True


def test_residual_migration_phase1_checker_requires_stratified_clip_stats(tmp_path: Path) -> None:
    result = checker.evaluate(_write_packet(tmp_path, omit_attention_stats=True))

    assert result["decision"] == checker.FAIL_INFRA
    assert any("attention_only" in reason for reason in result["reasons"])


def test_residual_migration_phase1_checker_rejects_empty_generations(tmp_path: Path) -> None:
    run_dir = _write_packet(tmp_path)
    (run_dir / "generations.jsonl").write_text("", encoding="utf-8")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics["generation_artifact_sha256"] = checker.file_sha256(run_dir / "generations.jsonl")
    _write_json(run_dir / "metrics.json", metrics)
    artifacts = []
    for rel in checker.HASHED_FILES:
        path = run_dir / rel
        artifacts.append({"path": rel, "bytes": path.stat().st_size, "sha256": checker.file_sha256(path)})
    _write_json(run_dir / "artifact_hashes.json", {"artifacts": artifacts})

    result = checker.evaluate(run_dir)

    assert result["decision"] == checker.FAIL_INFRA
    assert any("generations.jsonl must contain generation rows" in reason for reason in result["reasons"])
