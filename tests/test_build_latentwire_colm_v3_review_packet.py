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
    return {
        "colm_v2_review_packet": _write_json(tmp_path / "v2.json", v2_packet),
        "colm_v3_readiness": _write_text(tmp_path / "readiness.md", "readiness"),
        "colm_v3_tex": _write_text(tmp_path / "paper.tex", "paper"),
        "experiment_ledger": _write_text(tmp_path / "ledger.md", "ledger"),
        "experimental_status": _write_text(tmp_path / "experimental.md", "experimental"),
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

    systems_classes = {row["measured_vs_estimated"] for row in packet["systems_measured_vs_estimated"]}
    assert "measured_packet_object_bytes" in systems_classes
    assert "analytical_or_literature_byte_floor" in systems_classes


def test_write_outputs_emits_colm_v3_reviewer_packet(tmp_path: Path) -> None:
    packet = packet_builder.build_review_packet(_input_paths(tmp_path))
    output_dir = tmp_path / "packet"
    paper_path = tmp_path / "paper.md"

    packet_builder.write_outputs(packet, output_dir, paper_path)

    assert (output_dir / "review_packet.json").exists()
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
    assert "## Experimental Side-Branch Scope" in markdown
