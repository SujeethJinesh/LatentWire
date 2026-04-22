from __future__ import annotations

import json

from scripts import run_gsm8k_contract_campaign as campaign


def test_materialized_eval_path_uses_results_root_and_slice() -> None:
    config = campaign.CampaignConfig(results_root="results/my_campaign", slice_size=128)
    path = campaign._materialized_eval_path(config)
    assert path.endswith("my_campaign_gsm8k_eval_128.jsonl")


def test_aggregate_rows_computes_seed_stats() -> None:
    payloads = {
        0: {
            "rows": [
                {
                    "label": "dynalign_module_replace_residrank16",
                    "accuracy": 0.125,
                    "paired_vs_target": {"win": 2, "loss": 0, "tie": 30},
                    "base_label": "dynalign_module_replace",
                    "residual_rank": 16,
                }
            ]
        },
        1: {
            "rows": [
                {
                    "label": "dynalign_module_replace_residrank16",
                    "accuracy": 0.09375,
                    "paired_vs_target": {"win": 1, "loss": 0, "tie": 31},
                    "base_label": "dynalign_module_replace",
                    "residual_rank": 16,
                }
            ]
        },
    }
    summary = campaign._aggregate_rows(payloads)
    row = summary["dynalign_module_replace_residrank16"]
    assert row["n_seeds"] == 2
    assert row["seeds"] == [0, 1]
    assert row["accuracy_mean"] == (0.125 + 0.09375) / 2.0
    assert row["accuracy_min"] == 0.09375
    assert row["accuracy_max"] == 0.125
    assert row["wins_mean"] == 1.5
    assert row["losses_mean"] == 0.0


def test_parse_args_accepts_repeated_seed_base_rank_and_candidate(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_gsm8k_contract_campaign.py",
            "--results-root",
            "results/campaign",
            "--seed",
            "0",
            "--seed",
            "3",
            "--base",
            "dynalign_module_replace",
            "--rank",
            "16",
            "--candidate-label",
            "dynalign_module_replace_residrank16",
        ],
    )
    args = campaign._parse_args()
    assert args.results_root == "results/campaign"
    assert args.seeds == [0, 3]
    assert args.bases == ["dynalign_module_replace"]
    assert args.ranks == [16]
    assert args.candidate_labels == ["dynalign_module_replace_residrank16"]


def test_write_markdown_renders_aggregate_rows(tmp_path) -> None:
    payload = {
        "date": "2026-04-22",
        "config": {
            "source_model": "src",
            "target_model": "tgt",
            "slice_size": 128,
            "seeds": [0, 1],
            "bases": ["dynalign_module_replace"],
            "ranks": [16],
        },
        "artifacts": {"baseline_results_dir": "results/base"},
        "aggregate_rows": {
            "dynalign_module_replace_residrank16": {
                "n_seeds": 2,
                "accuracy_mean": 0.109375,
                "accuracy_min": 0.09375,
                "accuracy_max": 0.125,
                "wins_mean": 1.5,
                "losses_mean": 0.0,
            }
        },
        "seed_artifacts": {
            0: {"residual_payload": "results/campaign/seed0/gsm8k_contract_residual_sweep_20260421.json", "diagnostics": []}
        },
    }
    path = tmp_path / "out.md"
    campaign._write_markdown(path, payload)
    text = path.read_text()
    assert "# GSM8K Contract Campaign" in text
    assert "| dynalign_module_replace_residrank16 | 2 | 0.1094 | 0.0938 | 0.1250 | 1.50 | 0.00 |" in text


def test_payload_round_trip_json() -> None:
    payload = {"aggregate_rows": {"foo": {"accuracy_mean": 0.5}}}
    dumped = json.dumps(payload, sort_keys=True)
    assert json.loads(dumped)["aggregate_rows"]["foo"]["accuracy_mean"] == 0.5
