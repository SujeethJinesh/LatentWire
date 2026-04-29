from __future__ import annotations

import json

from scripts.run_source_private_tool_trace_compression_baselines import run_gate


def test_tool_trace_compression_baselines_write_artifacts(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path,
        train_examples=64,
        eval_examples=32,
        train_family_set="all",
        eval_family_set="all",
        candidates=4,
        feature_dim=128,
        budgets=[6],
        train_seed=5,
        eval_seed=6,
        ridge=1e-2,
    )

    summary = json.loads((tmp_path / "summary.json").read_text())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    row = payload["budget_summaries"][0]
    assert payload["exact_id_parity"] is True
    assert summary["budget_summaries"][0]["budget_bytes"] == 6
    assert "predictions_budget6.jsonl" in manifest["artifacts"]
    assert (tmp_path / "predictions_budget6.jsonl").exists()
    assert "best_compression_baseline_accuracy" in row


def test_tool_trace_compression_baselines_include_source_destroying_controls(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path,
        train_examples=128,
        eval_examples=64,
        train_family_set="all",
        eval_family_set="all",
        candidates=4,
        feature_dim=256,
        budgets=[6],
        train_seed=3,
        eval_seed=4,
        ridge=1e-2,
    )

    metrics = payload["budget_summaries"][0]["metrics"]
    assert metrics["zero_source"]["accuracy"] == metrics["target_only"]["accuracy"]
    assert "scalar_quantized_source" in metrics
    assert "scalar_shuffled_source" in metrics
    assert "scalar_constrained_shuffled_source" in metrics
    assert "scalar_answer_masked_source" in metrics
    assert "scalar_label_shuffled_ridge" in metrics
    assert "raw_source_sign_sketch" in metrics


def test_tool_trace_compression_qjl_variant_is_opt_in(tmp_path) -> None:
    default = run_gate(
        output_dir=tmp_path / "default",
        train_examples=64,
        eval_examples=32,
        train_family_set="all",
        eval_family_set="all",
        candidates=4,
        feature_dim=128,
        budgets=[6],
        train_seed=5,
        eval_seed=6,
        ridge=1e-2,
        candidate_view="slot",
        fit_intercept=False,
    )
    opt_in = run_gate(
        output_dir=tmp_path / "qjl",
        train_examples=64,
        eval_examples=32,
        train_family_set="all",
        eval_family_set="all",
        candidates=4,
        feature_dim=128,
        budgets=[6],
        train_seed=5,
        eval_seed=6,
        ridge=1e-2,
        candidate_view="slot",
        fit_intercept=False,
        packet_variants=["qjl_residual"],
    )

    assert default["packet_variants"] == []
    assert "qjl_residual_source" not in default["budget_summaries"][0]["metrics"]
    assert opt_in["packet_variants"] == ["qjl_residual"]
    assert "qjl_residual_source" in opt_in["budget_summaries"][0]["metrics"]
    assert opt_in["pass_gate"] == opt_in["budget_summaries"][0]["scalar_source_packet_pass"]
    qjl_rows = [
        json.loads(line)
        for line in (tmp_path / "qjl" / "predictions_budget6.jsonl").read_text().splitlines()
        if json.loads(line)["condition"] == "qjl_residual_source"
    ]
    assert qjl_rows
    assert {row["payload_bytes"] for row in qjl_rows} == {6}
    assert all(row["metadata"]["scalar_bytes"] + row["metadata"]["sign_bytes"] == 6 for row in qjl_rows)


def test_tool_trace_compression_relative_scores_variant_is_opt_in(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "relative",
        train_examples=64,
        eval_examples=32,
        train_family_set="all",
        eval_family_set="all",
        candidates=4,
        feature_dim=128,
        budgets=[6],
        train_seed=5,
        eval_seed=6,
        ridge=1e-2,
        candidate_view="slot",
        fit_intercept=False,
        packet_variants=["relative_scores"],
    )

    row = payload["budget_summaries"][0]
    assert payload["packet_variants"] == ["relative_scores"]
    assert "relative_score_source" in row["metrics"]
    assert row["relative_score_source_accuracy"] is not None
    assert payload["pass_gate"] == row["scalar_source_packet_pass"]
    relative_rows = [
        json.loads(line)
        for line in (tmp_path / "relative" / "predictions_budget6.jsonl").read_text().splitlines()
        if json.loads(line)["condition"] == "relative_score_source"
    ]
    assert relative_rows
    assert {row["payload_bytes"] for row in relative_rows} == {4}
    assert all(row["metadata"]["score_bytes"] == 4 for row in relative_rows)


def test_tool_trace_compression_canonical_relative_scores_variant_is_opt_in(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "relative_canonical",
        train_examples=64,
        eval_examples=32,
        train_family_set="all",
        eval_family_set="all",
        candidates=4,
        feature_dim=128,
        budgets=[4],
        train_seed=5,
        eval_seed=6,
        ridge=1e-2,
        candidate_view="slot",
        fit_intercept=False,
        remap_slot_seed=101,
        packet_variants=["relative_scores_canonical"],
    )

    row = payload["budget_summaries"][0]
    assert payload["packet_variants"] == ["relative_scores_canonical"]
    assert "relative_canonical_score_source" in row["metrics"]
    assert row["relative_canonical_score_source_accuracy"] is not None
    assert payload["pass_gate"] == row["scalar_source_packet_pass"]
    canonical_rows = [
        json.loads(line)
        for line in (tmp_path / "relative_canonical" / "predictions_budget4.jsonl").read_text().splitlines()
        if json.loads(line)["condition"] == "relative_canonical_score_source"
    ]
    assert canonical_rows
    assert {row["payload_bytes"] for row in canonical_rows} == {4}
    assert all(row["metadata"]["canonical_order"] is True for row in canonical_rows)
    assert any(
        row["metadata"]["display_candidate_labels"] != row["metadata"]["packet_candidate_labels"]
        for row in canonical_rows
    )


def test_tool_trace_consistent_posterior_packet_variant_is_opt_in(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "consistent",
        train_examples=64,
        eval_examples=32,
        train_family_set="all",
        eval_family_set="all",
        candidates=4,
        feature_dim=128,
        budgets=[4],
        train_seed=5,
        eval_seed=6,
        ridge=1e-2,
        candidate_view="slot",
        fit_intercept=False,
        remap_slot_seed=101,
        packet_variants=["consistent_posterior_packet"],
    )

    row = payload["budget_summaries"][0]
    assert payload["packet_variants"] == ["consistent_posterior_packet"]
    assert "consistent_posterior_packet_source" in row["metrics"]
    assert row["consistent_posterior_packet_accuracy"] is not None
    assert row["consistent_posterior_controls_ok"] in {True, False}
    assert payload["pass_gate"] == row["scalar_source_packet_pass"]
    consistent_rows = [
        json.loads(line)
        for line in (tmp_path / "consistent" / "predictions_budget4.jsonl").read_text().splitlines()
        if json.loads(line)["condition"] == "consistent_posterior_packet_source"
    ]
    assert consistent_rows
    assert {row["payload_bytes"] for row in consistent_rows} == {4}
    assert all(row["metadata"]["canonical_order"] is True for row in consistent_rows)
    assert any(
        row["metadata"]["display_candidate_labels"] != row["metadata"]["packet_candidate_labels"]
        for row in consistent_rows
    )


def test_tool_trace_slot_no_intercept_control_gate(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path,
        train_examples=128,
        eval_examples=64,
        train_family_set="all",
        eval_family_set="all",
        candidates=4,
        feature_dim=256,
        budgets=[6],
        train_seed=7,
        eval_seed=8,
        ridge=1e-2,
        candidate_view="slot",
        fit_intercept=False,
    )

    row = payload["budget_summaries"][0]
    metrics = row["metrics"]
    assert payload["pass_gate"] is True
    assert row["scalar_source_packet_pass"] is True
    assert metrics["scalar_quantized_source"]["accuracy"] >= 0.90
    assert metrics["scalar_answer_masked_source"]["accuracy"] <= metrics["target_only"]["accuracy"] + 0.05
    assert metrics["scalar_constrained_shuffled_source"]["accuracy"] <= metrics["target_only"]["accuracy"] + 0.05


def test_tool_trace_slot_remap_changes_codebook_and_passes(tmp_path) -> None:
    baseline = run_gate(
        output_dir=tmp_path / "base",
        train_examples=128,
        eval_examples=64,
        train_family_set="all",
        eval_family_set="all",
        candidates=4,
        feature_dim=256,
        budgets=[6],
        train_seed=7,
        eval_seed=8,
        ridge=1e-2,
        candidate_view="slot",
        fit_intercept=False,
    )
    remapped = run_gate(
        output_dir=tmp_path / "remap",
        train_examples=128,
        eval_examples=64,
        train_family_set="all",
        eval_family_set="all",
        candidates=4,
        feature_dim=256,
        budgets=[6],
        train_seed=7,
        eval_seed=8,
        ridge=1e-2,
        candidate_view="slot",
        fit_intercept=False,
        remap_slot_seed=101,
    )

    assert remapped["pass_gate"] is True
    assert remapped["remap_slot_seed"] == 101
    assert baseline["encoder_sha256"] != remapped["encoder_sha256"]
