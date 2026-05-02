from __future__ import annotations

import json
import pathlib

from scripts import run_gsm8k_contract_campaign as campaign


def test_materialized_eval_path_uses_results_root_and_slice() -> None:
    config = campaign.CampaignConfig(results_root="results/my_campaign", slice_size=128)
    path = campaign._materialized_eval_path(config)
    assert path.endswith("results/my_campaign/_artifacts/gsm8k_eval_128.jsonl")


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


def test_bootstrap_mean_ci_is_bounded() -> None:
    low, high = campaign._bootstrap_mean_ci([0.0, 1.0, 0.0, 1.0], samples=200, seed=7)
    assert low is not None and high is not None
    assert 0.0 <= low <= high <= 1.0


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
            "--bootstrap-samples",
            "250",
            "--bootstrap-seed",
            "17",
        ],
    )
    args = campaign._parse_args()
    assert args.results_root == "results/campaign"
    assert args.seeds == [0, 3]
    assert args.bases == ["dynalign_module_replace"]
    assert args.ranks == [16]
    assert args.candidate_labels == ["dynalign_module_replace_residrank16"]
    assert args.bootstrap_samples == 250
    assert args.bootstrap_seed == 17


def test_parse_args_accepts_selective_conditioning(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_gsm8k_contract_campaign.py",
            "--results-root",
            "results/campaign",
            "--whitening",
            "--target-whitening",
            "--whitening-streams",
            "v",
            "--target-whitening-streams",
            "v",
            "--conditioning-target-layer",
            "8",
        ],
    )
    args = campaign._parse_args()
    assert args.whitening is True
    assert args.target_whitening is True
    assert args.whitening_streams == "v"
    assert args.target_whitening_streams == "v"
    assert args.conditioning_target_layers == [8]


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
                "delta_mean": 0.046875,
                "delta_ci_low_mean": 0.0,
                "delta_ci_high_mean": 0.09375,
                "wins_mean": 1.5,
                "losses_mean": 0.0,
                "positive_seed_count": 2,
            }
        },
        "diagnostic_rows": {
            "dynalign_module_replace_residrank16": {
                "oracle_accuracy_mean": 0.125,
                "oracle_accuracy_min": 0.125,
                "oracle_accuracy_max": 0.125,
                "oracle_headroom_mean": 0.0,
                "candidate_only_win_n": 2,
                "candidate_only_win_source_correct": 0,
                "candidate_only_win_text_correct": 0,
                "text_only_loss_n": 1,
                "text_only_loss_source_correct": 0,
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
    assert "| dynalign_module_replace_residrank16 | 2 | 0.1094 | 0.0938 | 0.1250 | 0.0469 | [0.0000, 0.0938] | 1.50 | 0.00 | 2 |" in text
    assert "| dynalign_module_replace_residrank16 | 0.1250 | 0.1250 | 0.1250 | 0.0000 | 2 | 0 | 0 | 1 | 0 |" in text


def test_aggregate_diagnostics_summarizes_oracle_and_support() -> None:
    summary = campaign._aggregate_diagnostics(
        {
            "dynalign_module_replace_residrank16": [
                {
                    "summary_metrics": {"oracle_accuracy": 0.125, "candidate_accuracy": 0.125},
                    "candidate_only_win_support": {"n": 2, "source_correct": 0, "text_correct": 0},
                    "text_to_text_loss_support": {"n": 1, "source_correct": 0},
                },
                {
                    "summary_metrics": {"oracle_accuracy": 0.15625, "candidate_accuracy": 0.09375},
                    "candidate_only_win_support": {"n": 1, "source_correct": 1, "text_correct": 0},
                    "text_to_text_loss_support": {"n": 2, "source_correct": 1},
                },
            ]
        }
    )
    row = summary["dynalign_module_replace_residrank16"]
    assert row["n_seeds"] == 2
    assert row["oracle_accuracy_mean"] == (0.125 + 0.15625) / 2.0
    assert row["oracle_headroom_mean"] == (0.0 + (0.15625 - 0.09375)) / 2.0
    assert row["candidate_only_win_n"] == 3
    assert row["candidate_only_win_source_correct"] == 1
    assert row["text_only_loss_n"] == 3
    assert row["text_only_loss_source_correct"] == 1


def test_payload_round_trip_json() -> None:
    payload = {"aggregate_rows": {"foo": {"accuracy_mean": 0.5}}}
    dumped = json.dumps(payload, sort_keys=True)
    assert json.loads(dumped)["aggregate_rows"]["foo"]["accuracy_mean"] == 0.5


def test_candidate_row_finds_matching_label() -> None:
    payload = {
        "rows": [
            {"label": "a", "accuracy": 0.0},
            {"label": "b", "accuracy": 1.0},
        ]
    }
    assert campaign._candidate_row(payload, "b") == {"label": "b", "accuracy": 1.0}
    assert campaign._candidate_row(payload, "c") is None


def test_run_campaign_skips_failed_candidate_rows_without_outputs(tmp_path: pathlib.Path, monkeypatch) -> None:
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    (baseline_dir / campaign.SMOKE_PAYLOAD_NAME).write_text(json.dumps({"rows": {}}))

    seed0_dir = tmp_path / "results" / "seed0"
    seed0_dir.mkdir(parents=True, exist_ok=True)
    (seed0_dir / campaign.RESIDUAL_PAYLOAD_NAME).write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "label": "dynalign_module_replace_residrank16",
                        "accuracy": 0.0,
                        "paired_vs_target": {"win": 0, "loss": 0, "tie": 0},
                        "base_label": "dynalign_module_replace",
                        "residual_rank": 16,
                        "seed": 0,
                        "status": "checkpoint_nonfinite",
                    }
                ]
            }
        )
    )

    monkeypatch.setattr(campaign, "_run_smoke_contract", lambda config, baseline_dir: None)
    monkeypatch.setattr(campaign, "_run_residual_sweep", lambda config, baseline_dir, seed: seed0_dir)
    monkeypatch.setattr(
        campaign.harness,
        "materialize_slice",
        lambda src, dst, limit: (pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True), pathlib.Path(dst).write_text("")),
    )

    def _unexpected(*args, **kwargs):
        raise AssertionError("should not be called for failed candidate rows")

    monkeypatch.setattr(campaign, "_candidate_paired_stats", _unexpected)
    monkeypatch.setattr(campaign, "_run_diagnostics", _unexpected)

    config = campaign.CampaignConfig(
        results_root=str(tmp_path / "results"),
        eval_file="data/gsm8k_eval_70.jsonl",
        slice_size=70,
        seeds=(0,),
        bases=("dynalign_module_replace",),
        ranks=(16,),
        candidate_labels=("dynalign_module_replace_residrank16",),
        baseline_results_dir=str(baseline_dir),
        skip_smoke=True,
    )
    payload = campaign.run_campaign(config)

    row = payload["aggregate_rows"]["dynalign_module_replace_residrank16"]
    assert row["n_seeds"] == 1
    assert row["accuracy_mean"] == 0.0
    assert payload["paired_stats_by_label"] == {}
    assert payload["diagnostic_rows"] == {}
    assert payload["seed_artifacts"][0]["diagnostics"] == []


def test_run_residual_sweep_forwards_selective_conditioning_flags(tmp_path: pathlib.Path, monkeypatch) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(campaign, "_run", lambda cmd: commands.append(cmd))

    config = campaign.CampaignConfig(
        results_root=".debug/test_campaign_selective_conditioning",
        whitening=True,
        target_whitening=True,
        whitening_streams="v",
        target_whitening_streams="v",
        conditioning_target_layers=(8,),
    )
    baseline_dir = campaign.ROOT / ".debug" / "test_campaign_selective_conditioning_baseline"
    campaign._run_residual_sweep(config, baseline_dir, seed=1)

    cmd = commands[0]
    assert "--whitening" in cmd
    assert "--target-whitening" in cmd
    assert "--whitening-streams" in cmd
    assert "--target-whitening-streams" in cmd
    assert "--conditioning-target-layer" in cmd
    assert "v" in cmd
    assert "8" in cmd
