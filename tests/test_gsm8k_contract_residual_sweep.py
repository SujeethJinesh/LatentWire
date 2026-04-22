from __future__ import annotations

import json

from scripts import run_gsm8k_contract_residual_sweep as sweep


def test_checkpoint_path_reuses_existing_rank8() -> None:
    config = sweep.ResidualSweepConfig()
    path = sweep._checkpoint_path("dynalign_module_replace", 8, config)
    assert str(path).endswith(
        "checkpoints/bridge_ridge_qk_dynalign_module_replace_20260420/"
        "qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_module_replace_cal64_chat.pt"
    )


def test_checkpoint_path_builds_new_rank_path() -> None:
    config = sweep.ResidualSweepConfig()
    path = sweep._checkpoint_path("tokenbasis_replace", 16, config)
    assert str(path).endswith(
        "checkpoints/gsm8k_contract_residual_sweep_20260421/tokenbasis_replace/"
        "qwen25_to_qwen3_grouped_subspace_transport_w010_r16_tokenbasis_replace_cal64_chat.pt"
    )


def test_checkpoint_path_builds_new_preserve_rank_path() -> None:
    config = sweep.ResidualSweepConfig()
    path = sweep._checkpoint_path("dynalign_preserve_module_replace", 16, config)
    assert str(path).endswith(
        "checkpoints/gsm8k_contract_residual_sweep_20260421/dynalign_preserve_module_replace/"
        "qwen25_to_qwen3_grouped_subspace_transport_w010_r16_dynalign_preserve_module_replace_cal64_chat.pt"
    )


def test_checkpoint_path_builds_new_eigenspace_rank_path() -> None:
    config = sweep.ResidualSweepConfig()
    path = sweep._checkpoint_path("dynalign_eigenspace_module_replace", 16, config)
    assert str(path).endswith(
        "checkpoints/gsm8k_contract_residual_sweep_20260421/dynalign_eigenspace_module_replace/"
        "qwen25_to_qwen3_grouped_subspace_transport_w010_r16_dynalign_eigenspace_module_replace_cal64_chat.pt"
    )


def test_checkpoint_path_builds_new_saliency_rank_path() -> None:
    config = sweep.ResidualSweepConfig()
    path = sweep._checkpoint_path("dynalign_saliency_module_replace", 16, config)
    assert str(path).endswith(
        "checkpoints/gsm8k_contract_residual_sweep_20260421/dynalign_saliency_module_replace/"
        "qwen25_to_qwen3_grouped_subspace_transport_w010_r16_dynalign_saliency_module_replace_cal64_chat.pt"
    )


def test_checkpoint_path_builds_new_saliency_preserve_rank_path() -> None:
    config = sweep.ResidualSweepConfig()
    path = sweep._checkpoint_path("dynalign_saliency_preserve_module_replace", 16, config)
    assert str(path).endswith(
        "checkpoints/gsm8k_contract_residual_sweep_20260421/dynalign_saliency_preserve_module_replace/"
        "qwen25_to_qwen3_grouped_subspace_transport_w010_r16_dynalign_saliency_preserve_module_replace_cal64_chat.pt"
    )


def test_checkpoint_path_builds_new_routed_rank_path() -> None:
    config = sweep.ResidualSweepConfig()
    path = sweep._checkpoint_path("dynalign_routed_module_replace", 16, config)
    assert str(path).endswith(
        "checkpoints/gsm8k_contract_residual_sweep_20260421/dynalign_routed_module_replace/"
        "qwen25_to_qwen3_grouped_subspace_transport_w010_r16_dynalign_routed_module_replace_cal64_chat.pt"
    )


def test_parse_args_accepts_multiple_ranks_and_bases(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_gsm8k_contract_residual_sweep.py",
            "--rank",
            "4",
            "--rank",
            "16",
            "--base",
            "dynalign_module_replace",
            "--base",
            "tokenbasis_replace",
        ],
    )
    args = sweep._parse_args()
    assert args.ranks == [4, 16]
    assert args.bases == ["dynalign_module_replace", "tokenbasis_replace"]


def test_write_markdown_renders_rows(tmp_path) -> None:
    payload = {
        "date": "2026-04-21",
        "baseline_contract": "/tmp/gsm8k_smoke_contract_20260421.md",
        "config": {
            "source_model": "src",
            "target_model": "tgt",
            "calibration_file": ".debug/calibration_64.txt",
            "slice_size": 32,
            "eval_file": "data/gsm8k_eval_70.jsonl",
        },
        "rows": [
            {
                "label": "dynalign_module_replace_residrank16",
                "base_label": "dynalign_module_replace",
                "residual_rank": 16,
                "accuracy": 0.125,
                "paired_vs_target": {"win": 3, "loss": 1, "tie": 28},
                "numeric_extraction_coverage": 32,
                "empty_predictions": 0,
                "reused_existing_checkpoint": False,
            }
        ],
        "checks": {
            "dynalign_module_replace_residrank16": {
                "row_count_matches_slice": True,
                "example_ids_match_target": True,
                "no_empty_predictions": True,
                "numeric_extraction_coverage": True,
                "beats_target": True,
            }
        },
    }
    path = tmp_path / "out.md"
    sweep._write_markdown(path, payload)
    text = path.read_text()
    assert "| dynalign_module_replace | 16 | 0.1250 | 3 | 1 | 28 | 32 | 0 | no | yes |" in text
    assert "`dynalign_module_replace_residrank16` — row_count_matches_slice=PASS" in text


def test_payload_round_trip_json() -> None:
    payload = {"rows": [{"label": "foo"}], "checks": {"foo": {"beats_target": False}}}
    dumped = json.dumps(payload, sort_keys=True)
    assert json.loads(dumped)["rows"][0]["label"] == "foo"
