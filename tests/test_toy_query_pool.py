from __future__ import annotations

import json

from scripts import run_toy_query_pool


def test_toy_query_pool_experiment_returns_interpretable_rows() -> None:
    config = run_toy_query_pool.ToyConfig(
        seed=3,
        train_examples=24,
        test_examples=12,
        dim=8,
        slots=6,
        classes=3,
        top_k=2,
        pool_slots=2,
        route_atoms=2,
        protected_channels=2,
        epochs=3,
        lr=1e-2,
        rec_weight=0.1,
    )

    rows = run_toy_query_pool.run_experiment(config, ["aligned", "rotated"])

    assert {(row["scenario"], row["method"]) for row in rows} == {
        ("aligned", "topk"),
        ("aligned", "query_pool"),
        ("aligned", "preconditioned_query_pool"),
        ("aligned", "constrained_preconditioned_query_pool"),
        ("aligned", "asymmetric_kv_budget"),
        ("aligned", "codebook_remap"),
        ("aligned", "residual_codebook_remap"),
        ("aligned", "protected_channel_residual_codebook_remap"),
        ("aligned", "gauge_aware_protected_channel_residual_codebook_remap"),
        ("aligned", "signal_aware_protected_channel_residual_codebook_remap"),
        ("aligned", "route_atom"),
        ("rotated", "topk"),
        ("rotated", "query_pool"),
        ("rotated", "preconditioned_query_pool"),
        ("rotated", "constrained_preconditioned_query_pool"),
        ("rotated", "asymmetric_kv_budget"),
        ("rotated", "codebook_remap"),
        ("rotated", "residual_codebook_remap"),
        ("rotated", "protected_channel_residual_codebook_remap"),
        ("rotated", "gauge_aware_protected_channel_residual_codebook_remap"),
        ("rotated", "signal_aware_protected_channel_residual_codebook_remap"),
        ("rotated", "route_atom"),
    }
    for row in rows:
        assert 0.0 <= row["task_acc"] <= 1.0
        assert row["rec_mse"] >= 0.0
        assert row["route_entropy"] >= 0.0
        assert 0.0 <= row["slot_collision_rate"] <= 1.0
        assert 0.0 <= row["dead_slot_rate"] <= 1.0
        assert "atom_entropy" in row
        assert "atom_collision_rate" in row
        assert "dead_atom_rate" in row
        assert "atom_top_margin" in row
        assert "precondition_condition_proxy" in row
        assert "precondition_cosine_drift" in row
        assert "precondition_norm_ratio" in row
        assert "precondition_abs_scale_ratio" in row
        assert "constrained_scale_min" in row
        assert "constrained_scale_max" in row
        assert "constrained_scale_mean" in row
        assert "kv_route_entropy" in row
        assert "kv_value_entropy" in row
        assert "kv_route_value_overlap" in row
        assert "kv_route_value_jaccard" in row
        assert "kv_route_value_kl" in row
        assert "kv_route_value_cosine" in row
        assert "kv_route_value_gap" in row
        assert "kv_gate_mean" in row
        assert "kv_gate_std" in row
        assert "codebook_entropy" in row
        assert "codebook_collision_rate" in row
        assert "dead_code_rate" in row
        assert "codebook_top_margin" in row
        assert "slot_code_entropy" in row
        assert "slot_code_collision_rate" in row
        assert "dead_slot_code_rate" in row
        assert "slot_code_top_margin" in row
        assert "codebook_recon_mse" in row
        assert "codebook_recon_cosine" in row
        assert "slot_remap_recon_mse" in row
        assert "slot_remap_recon_cosine" in row
        assert "codebook_support_mean" in row
        assert "codebook_remap_overlap" in row
        assert "codebook_remap_jaccard" in row
        assert "residual_codebook_entropy" in row
        assert "residual_codebook_collision_rate" in row
        assert "residual_dead_code_rate" in row
        assert "residual_codebook_top_margin" in row
        assert "residual_slot_code_entropy" in row
        assert "residual_slot_code_collision_rate" in row
        assert "residual_dead_slot_code_rate" in row
        assert "residual_slot_code_top_margin" in row
        assert "residual_codebook_recon_mse" in row
        assert "residual_codebook_recon_cosine" in row
        assert "residual_slot_remap_recon_mse" in row
        assert "residual_slot_remap_recon_cosine" in row
        assert "residual_codebook_support_mean" in row
        assert "residual_codebook_remap_overlap" in row
        assert "residual_codebook_remap_jaccard" in row
        assert "residual_query_energy_ratio" in row
        assert "residual_slot_energy_ratio" in row
        assert "residual_gate" in row
        assert "protected_channels" in row
        assert "protected_channel_fraction" in row
        assert "protected_residual_codebook_entropy" in row
        assert "protected_residual_codebook_collision_rate" in row
        assert "protected_residual_codebook_top_margin" in row
        assert "protected_residual_codebook_recon_mse" in row
        assert "protected_residual_codebook_recon_cosine" in row
        assert "protected_query_energy_ratio" in row
        assert "protected_slot_energy_ratio" in row
        assert "protected_gate" in row
        assert "gauge_basis_orthogonality_error" in row
        assert "gauge_protected_energy_fraction" in row
        assert "gauge_eigenvalue_top_margin" in row
        assert "gauge_selected_channels" in row
        if row["method"] != "topk":
            assert row["atom_entropy"] >= 0.0
            assert 0.0 <= row["atom_collision_rate"] <= 1.0
            assert 0.0 <= row["dead_atom_rate"] <= 1.0
        if row["method"] == "preconditioned_query_pool":
            assert row["precondition_condition_proxy"] >= 1.0
            assert 0.0 < row["precondition_cosine_drift"] <= 1.0
            assert row["precondition_norm_ratio"] > 0.0
        if row["method"] == "constrained_preconditioned_query_pool":
            assert 0.75 <= row["constrained_scale_min"] <= row["constrained_scale_max"] <= 1.25
            assert 0.75 <= row["constrained_scale_mean"] <= 1.25
        if row["method"] == "asymmetric_kv_budget":
            assert row["kv_route_budget"] == 2
            assert row["kv_value_budget"] == 4
            assert row["kv_route_entropy"] >= 0.0
            assert row["kv_value_entropy"] >= 0.0
            assert 0.0 <= row["kv_route_value_overlap"] <= 1.0
            assert 0.0 <= row["kv_route_value_jaccard"] <= 1.0
            assert row["kv_route_value_kl"] >= 0.0
            assert -1.0 <= row["kv_route_value_cosine"] <= 1.0
            assert row["kv_route_value_gap"] >= 0.0
            assert 0.0 <= row["kv_gate_mean"] <= 1.0
            assert row["kv_gate_std"] >= 0.0
        if row["method"] == "codebook_remap":
            assert row["codebook_size"] == 4
            assert row["codebook_entropy"] >= 0.0
            assert 0.0 <= row["codebook_collision_rate"] <= 1.0
            assert 0.0 <= row["dead_code_rate"] <= 1.0
            assert row["codebook_recon_mse"] >= 0.0
            assert -1.0 <= row["codebook_recon_cosine"] <= 1.0
            assert row["slot_remap_recon_mse"] >= 0.0
            assert -1.0 <= row["slot_remap_recon_cosine"] <= 1.0
            assert row["codebook_support_mean"] >= 1.0
            assert 0.0 <= row["codebook_remap_overlap"] <= 1.0
            assert 0.0 <= row["codebook_remap_jaccard"] <= 1.0
        if row["method"] == "residual_codebook_remap":
            assert row["codebook_size"] == 4
            assert row["residual_codebook_size"] == 4
            assert row["codebook_entropy"] >= 0.0
            assert row["residual_codebook_entropy"] >= 0.0
            assert 0.0 <= row["residual_codebook_collision_rate"] <= 1.0
            assert 0.0 <= row["residual_dead_code_rate"] <= 1.0
            assert row["residual_codebook_recon_mse"] >= 0.0
            assert -1.0 <= row["residual_codebook_recon_cosine"] <= 1.0
            assert row["residual_slot_remap_recon_mse"] >= 0.0
            assert -1.0 <= row["residual_slot_remap_recon_cosine"] <= 1.0
            assert row["residual_codebook_support_mean"] >= 1.0
            assert 0.0 <= row["residual_codebook_remap_overlap"] <= 1.0
            assert 0.0 <= row["residual_codebook_remap_jaccard"] <= 1.0
            assert row["residual_query_energy_ratio"] >= 0.0
            assert row["residual_slot_energy_ratio"] >= 0.0
            assert 0.0 <= row["residual_gate"] <= 1.0
        if row["method"] == "protected_channel_residual_codebook_remap":
            assert row["codebook_size"] == 4
            assert row["residual_codebook_size"] == 4
            assert row["protected_channels"] == 2
            assert 0.0 < row["protected_channel_fraction"] <= 1.0
            assert row["protected_residual_codebook_entropy"] >= 0.0
            assert 0.0 <= row["protected_residual_codebook_collision_rate"] <= 1.0
            assert row["protected_residual_codebook_recon_mse"] >= 0.0
            assert -1.0 <= row["protected_residual_codebook_recon_cosine"] <= 1.0
            assert row["protected_query_energy_ratio"] >= 0.0
            assert row["protected_slot_energy_ratio"] >= 0.0
            assert 0.0 <= row["protected_gate"] <= 1.0
        if row["method"] == "gauge_aware_protected_channel_residual_codebook_remap":
            assert row["codebook_size"] == 4
            assert row["residual_codebook_size"] == 4
            assert row["protected_channels"] == 2
            assert 0.0 < row["protected_channel_fraction"] <= 1.0
            assert row["protected_residual_codebook_entropy"] >= 0.0
            assert 0.0 <= row["protected_residual_codebook_collision_rate"] <= 1.0
            assert row["protected_residual_codebook_recon_mse"] >= 0.0
            assert -1.0 <= row["protected_residual_codebook_recon_cosine"] <= 1.0
            assert row["protected_query_energy_ratio"] >= 0.0
            assert row["protected_slot_energy_ratio"] >= 0.0
            assert 0.0 <= row["protected_gate"] <= 1.0
            assert row["gauge_basis_orthogonality_error"] >= 0.0
            assert 0.0 <= row["gauge_protected_energy_fraction"] <= 1.0
            assert row["gauge_eigenvalue_top_margin"] >= 0.0
            assert row["gauge_selected_channels"] == 2.0
        if row["method"] == "signal_aware_protected_channel_residual_codebook_remap":
            assert row["codebook_size"] == 4
            assert row["residual_codebook_size"] == 4
            assert row["protected_channels"] == 2
            assert 0.0 < row["protected_channel_fraction"] <= 1.0
            assert row["signal_basis_orthogonality_error"] >= 0.0
            assert 0.0 <= row["signal_task_energy_fraction"] <= 1.0
            assert row["signal_eigenvalue_top_margin"] >= 0.0
            assert row["signal_selected_channels"] == 2.0
            assert 0.0 <= row["signal_variance_alignment"] <= 1.0
            assert row["signal_query_energy_ratio"] >= 0.0
            assert row["signal_slot_energy_ratio"] >= 0.0
            assert 0.0 <= row["signal_gate"] <= 1.0


def test_toy_query_pool_cli_writes_json(tmp_path) -> None:
    output = tmp_path / "toy.json"
    markdown = tmp_path / "toy.md"

    # Exercise the core serialization shape without paying for the default CLI run.
    config = run_toy_query_pool.ToyConfig(
        seed=1,
        train_examples=16,
        test_examples=8,
        dim=8,
        slots=5,
        classes=3,
        top_k=2,
        pool_slots=2,
        route_atoms=2,
        epochs=2,
    )
    payload = {
        "config": config.__dict__,
        "rows": run_toy_query_pool.run_experiment(config, ["outlier"]),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    run_toy_query_pool.write_markdown_summary(payload["rows"], markdown)
    loaded = json.loads(output.read_text())

    assert loaded["config"]["pool_slots"] == 2
    assert {row["method"] for row in loaded["rows"]} == {
        "topk",
        "query_pool",
        "preconditioned_query_pool",
        "constrained_preconditioned_query_pool",
        "asymmetric_kv_budget",
        "codebook_remap",
        "residual_codebook_remap",
        "protected_channel_residual_codebook_remap",
        "gauge_aware_protected_channel_residual_codebook_remap",
        "signal_aware_protected_channel_residual_codebook_remap",
        "route_atom",
    }
    summary = markdown.read_text()
    assert "Atom entropy" in summary
    assert "Precond cond." in summary
    assert "Constrained scale min" in summary
    assert "KV route entropy" in summary
    assert "KV value entropy" in summary
    assert "KV overlap" in summary
    assert "Codebook entropy" in summary
    assert "Remap overlap" in summary
    assert "Residual codebook entropy" in summary
    assert "Residual query energy" in summary
    assert "Protected channels" in summary
    assert "Protected residual entropy" in summary
    assert "Protected query energy" in summary
    assert "Gauge orth." in summary
    assert "Gauge energy frac." in summary
    assert "Gauge top margin" in summary
    assert "Gauge selected" in summary
    assert "Signal orth." in summary
    assert "Signal task energy" in summary
    assert "Signal top margin" in summary
    assert "Signal selected" in summary
    assert "Signal var. align" in summary
    assert "Signal query energy" in summary
    assert "Signal slot energy" in summary
    assert "Signal gate" in summary
    assert "| outlier | query_pool |" in summary
    assert "| outlier | preconditioned_query_pool |" in summary
    assert "| outlier | constrained_preconditioned_query_pool |" in summary
    assert "| outlier | asymmetric_kv_budget |" in summary
    assert "| outlier | codebook_remap |" in summary
    assert "| outlier | residual_codebook_remap |" in summary
    assert "| outlier | gauge_aware_protected_channel_residual_codebook_remap |" in summary
    assert "| outlier | signal_aware_protected_channel_residual_codebook_remap |" in summary
    assert "| outlier | route_atom |" in summary
