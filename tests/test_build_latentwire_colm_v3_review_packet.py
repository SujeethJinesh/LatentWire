from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts import build_latentwire_colm_v3_review_packet as packet_builder


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _input_paths(tmp_path: Path) -> dict[str, Path]:
    v2_packet = {
        "baseline_matrix": [
            {"baseline": "Cache-to-Cache (C2C)", "category": "dense_cache_transfer"},
        ],
        "claim_audit": [
            {"claim": "legacy", "support_level": "supported"},
        ],
        "contribution_table": [
            {
                "evidence": "legacy evidence",
                "gap": "legacy gap",
                "name": "legacy_control_suite",
                "status": "alive",
            }
        ],
        "main_results": [
            {"branch": "ARC packet", "score": 0.344, "baseline": 0.265},
        ],
        "negative_results": [
            {"branch": "Phi cross-family", "status": "failed"},
        ],
        "strict_controls": [
            {"condition": "same_source_choice_wrong_row_packet", "receiver_accuracy": 0.422},
        ],
    }
    systems = {
        "checks": [{"check": "packet_rows_remain_source_private", "pass": True}],
        "headline": {"native_nvidia_complete": False, "max_packet_framed_bytes": 11},
        "rows": [
            {
                "batch64_bytes": 704,
                "cacheline_bytes": 64,
                "claim_allowed": "packet byte accounting only",
                "communicated_object": "cached packet",
                "framed_bytes": 11,
                "measurement_status": "cached_source_communication_object",
                "method": "LatentWire ARC packet",
                "native_measured": False,
                "raw_bytes": 8,
                "row_group": "LatentWire packet",
                "source_kv_exposed": False,
                "source_private": True,
            },
            {
                "claim_allowed": "byte floor only",
                "communicated_object": "KV cache",
                "framed_bytes": 768,
                "measurement_status": "analytical_floor",
                "method": "C2C one-token KV floor",
                "native_measured": False,
                "row_group": "dense cache",
                "source_kv_exposed": True,
                "source_private": False,
            },
        ],
    }
    benchmark_gate = {
        "rows": [
            {
                "budget_bytes": 3,
                "dataset": "OpenBookQA",
                "eval_rows": 500,
                "matched_accuracy_mean": 0.378,
                "matched_minus_same_byte_text_min": 0.028,
                "matched_minus_target_min": 0.102,
                "packet_target_pass": True,
                "packet_text_margin_pass": True,
                "paired_ci95_low_vs_target_min": 0.038,
                "row_id": "openbookqa_test_3b",
                "same_byte_text_accuracy": 0.35,
                "seed_artifact": "results/obqa.json",
                "seed_count": 5,
                "selection_role": "receiver_candidate",
                "split": "test",
                "target_accuracy": 0.276,
            },
        ]
    }
    latest_model_matrix = {
        "models": [
            {
                "architecture": "qwen3_5 conditional generation",
                "expected_device": "cpu",
                "family": "Qwen3.5 small hybrid",
                "local_rung": "CPU n160 passed",
                "model": "Qwen/Qwen3.5-0.8B",
                "params": "0.8B",
                "status": "CPU n160 seed repeat passed",
            }
        ]
    }
    cpu_frontier = {
        "rows": [
            {
                "accuracy": 1.0,
                "contribution": "model-emitted source packet",
                "method": "Qwen3.5-0.8B",
                "note": "n=160; exact_id_parity=True",
                "status": "pass",
                "surface": "n160 seed29",
                "valid_rate": 1.0,
            }
        ]
    }
    return {
        "benchmark_selection_gate": _write_json(tmp_path / "benchmark_gate.json", benchmark_gate),
        "colm_v2_review_packet": _write_json(tmp_path / "v2.json", v2_packet),
        "colm_v3_readiness": _write_text(tmp_path / "readiness.md", "readiness"),
        "colm_v3_tex": _write_text(tmp_path / "paper.tex", "paper"),
        "cpu_systems_frontier": _write_json(tmp_path / "cpu_frontier.json", cpu_frontier),
        "experiment_ledger": _write_text(tmp_path / "ledger.md", "ledger"),
        "experimental_status": _write_text(tmp_path / "experimental.md", "experimental"),
        "latest_model_matrix": _write_json(tmp_path / "latest_model_matrix.json", latest_model_matrix),
        "reviewer_feedback": _write_text(tmp_path / "reviewer.md", "reviewer"),
        "systems_boundary": _write_json(tmp_path / "systems.json", systems),
    }


def test_build_review_packet_sets_colm_v3_claim_boundaries(tmp_path: Path) -> None:
    packet = packet_builder.build_review_packet(_input_paths(tmp_path))

    assert packet["readiness"]["colm_v3"] == "reviewer_hardened_draft_pending_human_review"
    assert "candidate-transfer" in packet["main_claim"]
    assert any(
        row["claim"] == "The current packet beats source-index communication or selected-candidate codes."
        and row["support_level"] == "not_supported"
        for row in packet["claim_audit"]
    )
    assert any(
        row["claim"] == "LatentWire beats C2C or dense KV/cache transfer."
        and row["support_level"] == "not_supported"
        for row in packet["claim_audit"]
    )
    assert any(row["artifact"] == "systems boundary table" for row in packet["table_figure_inventory"])
    assert any(row["experiment"] == "ThoughtFlow-FP8" for row in packet["experiment_scoping"])
    assert any(row["benchmark"] == "OpenBookQA" for row in packet["benchmark_breadth"])
    assert any(row["model"] == "Qwen/Qwen3.5-0.8B" for row in packet["latest_model_breadth"])

    systems_classes = {row["measured_vs_estimated"] for row in packet["systems_measured_vs_estimated"]}
    assert "measured_packet_object_bytes" in systems_classes
    assert "analytical_or_literature_byte_floor" in systems_classes


def test_write_outputs_emits_colm_v3_reviewer_packet(tmp_path: Path) -> None:
    packet = packet_builder.build_review_packet(_input_paths(tmp_path))
    output_dir = tmp_path / "packet"
    paper_path = tmp_path / "paper.md"

    packet_builder.write_outputs(packet, output_dir, paper_path)

    assert (output_dir / "review_packet.json").exists()
    assert (output_dir / "benchmark_breadth.csv").exists()
    assert (output_dir / "latest_model_breadth.csv").exists()
    assert (output_dir / "nvidia_native_runbook.md").exists()
    assert (output_dir / "manifest.json").exists()
    assert paper_path.exists()

    rows = list(csv.DictReader((output_dir / "claim_audit.csv").open(encoding="utf-8")))
    assert any(row["support_level"] == "not_supported" for row in rows)

    systems_rows = list(
        csv.DictReader((output_dir / "systems_measured_vs_estimated.csv").open(encoding="utf-8"))
    )
    assert any(row["measured_vs_estimated"] == "measured_packet_object_bytes" for row in systems_rows)

    markdown = paper_path.read_text(encoding="utf-8")
    assert "## Reviewer Claim Audit" in markdown
    assert "## Benchmark Breadth Audit" in markdown
    assert "## Latest Model Breadth Audit" in markdown
    assert "## Experimental Side-Branch Scope" in markdown
