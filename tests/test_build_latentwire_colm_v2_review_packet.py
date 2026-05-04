from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts import build_latentwire_colm_v2_review_packet as packet_builder


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def _row(branch: str, status: str, *, score: float = 0.6, baseline: float = 0.5) -> dict:
    return {
        "artifact": branch.lower().replace(" ", "_"),
        "baseline": baseline,
        "branch": branch,
        "ci95_low": score - baseline,
        "decision": f"decision for {branch}",
        "delta": score - baseline,
        "evidence": f"evidence for {branch}",
        "record_bytes": 7,
        "score": score,
        "status": status,
    }


def _evidence() -> dict:
    return {
        "claim_boundaries": ["Do not claim better than C2C."],
        "colm_v2_core_rows": [
            _row("Conditional PQ", "promote_for_colm_v2_only", score=0.7),
        ],
        "colm_v2_supporting_rows": [
            _row("OpenBookQA Hardened", "weakened_openbookqa_source_choice_control", score=0.424, baseline=0.422),
        ],
        "current_contributions": [
            {
                "gap": "needs broader receiver evidence",
                "name": "strict_source_private_controls",
                "status": "alive_for_colm_v2",
            },
            {
                "gap": "needs native systems rows",
                "name": "byte_systems_accounting",
                "status": "alive_for_colm_v2",
            },
        ],
        "iclr_blocker_rows": [
            _row("ARC Behavior Atom", "ruled_out_behavior_atom", score=0.4, baseline=0.5),
        ],
        "paper_decision": {
            "colm_v2_claim": "Scoped packet framework.",
            "iclr_claim_not_yet_supported": "Broad Sparse Resonance Packets.",
            "single_highest_priority": "Package COLM_v2 evidence.",
        },
        "readiness": {
            "colm_v2": "scoped_positive_ready_for_writeup_if_claims_are_narrow",
            "iclr": "blocked_by_lack_of_broad_or_learned_positive_receiver",
        },
        "story": "Scoped source-private packets plus strict controls.",
    }


def _triage() -> dict:
    return {
        "next_exact_gate": {
            "fallback_path": "target behavior-transcoder fallback",
            "name": "arc_n32_tokenwise_source_evidence_preflight",
            "pass_bar": "positive paired CI over controls",
            "primary_path": "materialize tokenwise source-evidence cache",
            "required_controls": ["target_only", "wrong_row"],
        },
        "submission_gap": "ICLR needs a source-causal receiver.",
    }


def _openbookqa() -> dict:
    return {
        "headline": {"default_seed": 47},
        "per_seed": [
            {
                "condition_metrics": {
                    "matched_source_private_packet": {
                        "base_accuracy": 0.378,
                        "harm_count": 35,
                        "help_count": 58,
                        "override_rate": 0.28,
                        "receiver_accuracy": 0.424,
                        "receiver_minus_base": 0.046,
                        "target_public_accuracy": 0.372,
                    },
                    "same_source_choice_wrong_row_packet": {
                        "base_accuracy": 0.378,
                        "harm_count": 54,
                        "help_count": 76,
                        "override_rate": 0.414,
                        "receiver_accuracy": 0.422,
                        "receiver_minus_base": 0.044,
                        "target_public_accuracy": 0.372,
                    },
                },
                "seed": 47,
            }
        ],
    }


def _systems() -> dict:
    return {
        "headline": {
            "claim_scope": "Byte/exposure accounting only.",
            "native_nvidia_complete": False,
        },
        "rows": [
            {
                "batch64_bytes": 704,
                "cacheline_bytes": 64,
                "claim_allowed": "packet byte accounting only",
                "communicated_object": "source-private packet",
                "framed_bytes": 11,
                "method": "LatentWire packet",
                "native_measured": False,
                "raw_bytes": 8,
                "row_group": "LatentWire packet",
                "source_kv_exposed": False,
                "source_private": True,
            }
        ],
    }


def _input_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "conditional_pq_status": _write_json(tmp_path / "conditional.json", {"ok": True}),
        "evidence_table": _write_json(tmp_path / "evidence.json", _evidence()),
        "hellaswag_fixed_hybrid": _write_json(tmp_path / "hellaswag.json", {"ok": True}),
        "live_branch_triage": _write_json(tmp_path / "triage.json", _triage()),
        "openbookqa_receiver_headroom": _write_json(tmp_path / "openbookqa.json", _openbookqa()),
        "systems_boundary": _write_json(tmp_path / "systems.json", _systems()),
    }


def test_build_review_packet_collects_tables_and_baselines(tmp_path: Path) -> None:
    packet = packet_builder.build_review_packet(_input_paths(tmp_path))

    assert packet["reviewer_packet_status"]["colm_v2_review_ready_after_human_paper_pass"]
    assert not packet["reviewer_packet_status"]["iclr_positive_method_ready"]
    assert [row["branch"] for row in packet["main_results"]] == [
        "Conditional PQ",
        "OpenBookQA Hardened",
    ]
    assert any(
        row["condition"] == "same_source_choice_wrong_row_packet"
        for row in packet["strict_controls"]
    )

    baselines = {row["baseline"] for row in packet["baseline_matrix"]}
    assert "Cache-to-Cache (C2C)" in baselines
    assert "CIPHER / Let Models Speak Ciphers" in baselines
    assert "Latent Space Communication via K-V Cache Alignment" in baselines
    assert "LLMLingua / LongLLMLingua" in baselines
    assert "vLLM / PagedAttention" in baselines


def test_write_outputs_emits_reviewer_packet_artifacts(tmp_path: Path) -> None:
    packet = packet_builder.build_review_packet(_input_paths(tmp_path))
    output_dir = tmp_path / "packet"
    paper_path = tmp_path / "paper.md"

    packet_builder.write_outputs(packet, output_dir, paper_path)

    assert (output_dir / "review_packet.json").exists()
    assert (output_dir / "manifest.json").exists()
    assert paper_path.exists()

    rows = list(csv.DictReader((output_dir / "baseline_matrix.csv").open(encoding="utf-8")))
    assert any(row["baseline"] == "CIPHER / Let Models Speak Ciphers" for row in rows)

    claim_rows = list(csv.DictReader((output_dir / "claim_audit.csv").open(encoding="utf-8")))
    assert any(row["support_level"] == "not_supported" for row in claim_rows)

    markdown = paper_path.read_text(encoding="utf-8")
    assert "## Reviewer Claim Audit" in markdown
    assert "Do not claim better than C2C." in markdown
