from __future__ import annotations

import json
import pathlib

from scripts import build_gsm8k_contract_manifest as manifest


def test_build_manifest_extracts_current_evidence_chain(tmp_path: pathlib.Path) -> None:
    smoke_json = tmp_path / "smoke.json"
    live_json = tmp_path / "live.json"
    control_json = tmp_path / "control.json"
    campaign_json = tmp_path / "campaign.json"
    health_json = tmp_path / "health.json"

    smoke_json.write_text(
        json.dumps(
            {
                "config": {"slice_size": 32},
                "rows": {
                    "target_alone": {"accuracy": 0.0625},
                    "rotalign_kv": {"accuracy": 0.0625, "numeric_extraction_coverage": 28},
                    "c2c_generate": {"accuracy": 0.1250},
                },
                "checks": {
                    "row_counts_match_slice": {"passed": True},
                    "numeric_extraction_coverage": {"passed": False},
                },
            }
        )
    )
    live_json.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "label": "dynalign_module_replace_residrank16",
                        "accuracy": 0.125,
                        "numeric_extraction_coverage": 32,
                        "checkpoint_path": "checkpoints/live.pt",
                        "paired_vs_target": {"win": 2, "loss": 0, "tie": 30},
                    }
                ]
            }
        )
    )
    control_json.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "label": "tokenbasis_replace_residrank16",
                        "accuracy": 0.0625,
                        "numeric_extraction_coverage": 32,
                        "checkpoint_path": "checkpoints/control.pt",
                        "paired_vs_target": {"win": 0, "loss": 0, "tie": 32},
                    }
                ]
            }
        )
    )
    campaign_json.write_text(
        json.dumps(
            {
                "aggregate_rows": {
                    "dynalign_module_replace_residrank16": {
                        "paired_n": 70,
                        "seeds": [0],
                        "accuracy_mean": 0.1142857143,
                        "accuracy_min": 0.1142857143,
                        "accuracy_max": 0.1142857143,
                        "delta_mean": 0.0571428571,
                        "delta_ci_low_mean": -0.0142857143,
                        "delta_ci_high_mean": 0.1428571429,
                        "wins_mean": 6.0,
                        "losses_mean": 2.0,
                    }
                },
                "baseline_summary": {
                    "target_alone": {"accuracy": 0.0571428571},
                    "c2c_generate": {"accuracy": 0.1285714286},
                },
            }
        )
    )
    health_json.write_text(
        json.dumps(
            {
                "seed": 1,
                "checkpoint_path": "checkpoints/bad_seed1.pt",
                "nonfinite_numel": 2381056,
                "first_bad_key": "W_V.8",
                "nonfinite_keys": ["W_V.8", "quant_proj_V.8"],
                "top_abs_tensors": [{"key": "W_V.8", "nonfinite_numel": 131072}],
            }
        )
    )

    built = manifest._build_manifest(
        smoke_payload=json.loads(smoke_json.read_text()),
        live_payload=json.loads(live_json.read_text()),
        control_payload=json.loads(control_json.read_text()),
        campaign_payload=json.loads(campaign_json.read_text()),
        health_payload=json.loads(health_json.read_text()),
        smoke_json=smoke_json,
        live_json=live_json,
        control_json=control_json,
        campaign_json=campaign_json,
        health_json=health_json,
    )

    assert built["smoke_contract"]["rotalign_numeric_coverage"] == 28
    assert built["same_pair_live_row"]["paired_vs_target"]["win"] == 2
    assert built["matched_control"]["label"] == "tokenbasis_replace_residrank16"
    assert built["larger_slice_campaign"]["slice_size"] == 70
    assert built["seed1_health"]["first_bad_key"] == "W_V.8"


def test_write_markdown_renders_key_rows(tmp_path: pathlib.Path) -> None:
    payload = {
        "date": "2026-04-22",
        "current_story": {
            "live_label": "dynalign_module_replace_residrank16",
            "main_blocker": "seed_stability_and_cross_family_falsification",
        },
        "artifacts": {
            "smoke_contract_json": "results/smoke.json",
            "live_contract_json": "results/live.json",
            "matched_control_json": "results/control.json",
            "larger_slice_campaign_json": "results/campaign.json",
            "seed1_health_json": "checkpoints/seed1.pt.health.json",
        },
        "smoke_contract": {
            "target_accuracy": 0.0625,
            "rotalign_accuracy": 0.0625,
            "c2c_accuracy": 0.125,
            "rotalign_numeric_coverage": 28,
        },
        "same_pair_live_row": {
            "label": "dynalign_module_replace_residrank16",
            "accuracy": 0.125,
            "paired_vs_target": {"win": 2, "loss": 0, "tie": 30},
            "numeric_coverage": 32,
        },
        "matched_control": {
            "label": "tokenbasis_replace_residrank16",
            "accuracy": 0.0625,
            "paired_vs_target": {"win": 0, "loss": 0, "tie": 32},
            "numeric_coverage": 32,
        },
        "larger_slice_campaign": {
            "candidate_accuracy_mean": 0.1143,
            "target_accuracy": 0.0571,
            "c2c_accuracy": 0.1286,
            "delta_mean_vs_target": 0.0571,
            "delta_ci_low_mean": -0.0143,
            "delta_ci_high_mean": 0.1429,
        },
        "seed1_health": {
            "seed": 1,
            "nonfinite_numel": 2381056,
            "first_bad_key": "W_V.8",
            "top_abs_tensors": [{"key": "W_V.8"}],
        },
        "next_exact_gates": ["finish seed repeats"],
    }
    path = tmp_path / "manifest.md"
    manifest._write_markdown(path, payload)
    text = path.read_text()
    assert "# GSM8K Contract Artifact Manifest" in text
    assert "live 32-example row" in text
    assert "seed-1 health: nonfinite=`2381056`, first_bad_key=`W_V.8`" in text
