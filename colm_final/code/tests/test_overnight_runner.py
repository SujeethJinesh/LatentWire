from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import csv

import scripts.run_overnight_rotalign as overnight


def test_parse_summary_metrics_keeps_system_metrics() -> None:
    summary = """
    noise
    === Summary ===
      target_alone: 0.100
      text_to_text: 0.200
      rotalign_kv: 0.300
      rotalign_kv_bytes: 256.000
      rotalign_kv_ttft_sec: 0.125
      rotalign_kv_tokens_per_sec: 8.500
    """

    metrics = overnight.parse_summary_metrics(summary)

    assert metrics["target_alone"] == 0.1
    assert metrics["text_to_text"] == 0.2
    assert metrics["rotalign_kv"] == 0.3
    assert metrics["rotalign_kv_bytes"] == 256.0
    assert metrics["rotalign_kv_ttft_sec"] == 0.125
    assert metrics["rotalign_kv_tokens_per_sec"] == 8.5


def test_build_ablation_cmd_includes_protocols_and_reasoning_modes(tmp_path) -> None:
    args = Namespace(
        calibration_file="cal.txt",
        eval_file="eval.jsonl",
        bits=4,
        whitening=True,
        device="mps",
        dtype="float32",
        selection_ratio=0.5,
        seed=1,
        ablation_source_reasoning_modes=["plain", "cot"],
        ablation_protocols=["translated_only", "fused", "text_kv_hybrid"],
    )

    cmd = overnight.build_ablation_cmd(
        python_exe="python",
        repo_root=Path("/repo"),
        pair=("src", "tgt"),
        args=args,
        ablation_output=tmp_path / "out.jsonl",
        ablation_checkpoint_dir=tmp_path / "ckpts",
    )

    assert "--source-reasoning-modes" in cmd
    reason_idx = cmd.index("--source-reasoning-modes")
    assert cmd[reason_idx + 1 : reason_idx + 3] == ["plain", "cot"]
    proto_idx = cmd.index("--protocols")
    assert cmd[proto_idx + 1 : proto_idx + 4] == ["translated_only", "fused", "text_kv_hybrid"]


def test_checkpoint_is_usable_rejects_unreadable_files(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "bad.pt"
    checkpoint.write_bytes(b"not a checkpoint")

    def fake_load(path: str, map_location: str = "cpu"):
        raise RuntimeError("bad checkpoint")

    monkeypatch.setattr(overnight.RotAlignKVTranslator, "load", fake_load)
    assert overnight.checkpoint_is_usable(checkpoint) is False


def test_write_pair_summaries_surfaces_system_metrics(tmp_path) -> None:
    out_dir = tmp_path / "results"
    pair_records = [
        {
            "pair_index": 1,
            "pair_tag": "qwen_control",
            "source_model": "src",
            "target_model": "tgt",
            "elapsed_sec": 42.0,
            "checkpoint_path": "/tmp/ckpt.pt",
            "log_file": "/tmp/run.log",
            "metrics": {
                "target_alone": 0.04,
                "target_alone_ttft_sec": 0.10,
                "target_alone_tokens_per_sec": 12.0,
                "text_to_text": 0.10,
                "text_to_text_ttft_sec": 0.20,
                "text_to_text_tokens_per_sec": 8.0,
                "rotalign_kv": 0.05,
                "rotalign_kv_bytes": 128.0,
                "rotalign_kv_ttft_sec": 0.15,
                "rotalign_kv_tokens_per_sec": 10.0,
            },
        }
    ]

    overnight.write_pair_summaries(pair_records, "rotalign_kv", out_dir)

    with (out_dir / "pair_results.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["rotalign_kv_bytes"] == "128.0"
    assert rows[0]["rotalign_kv_ttft_sec"] == "0.15"
    summary = (out_dir / "latest_summary.md").read_text(encoding="utf-8")
    assert "Bytes" in summary
    assert "Tok/s" in summary
    assert "qwen_control" in summary
