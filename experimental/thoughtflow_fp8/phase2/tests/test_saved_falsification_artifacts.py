from __future__ import annotations

import hashlib
import json
from pathlib import Path


PHASE2 = Path(__file__).resolve().parents[1]
DIAGNOSTIC_PACKET = PHASE2 / "diagnostic_packets/thoughtflow_diagnostic_packet_20260506"


def _load(name: str) -> dict[str, object]:
    return json.loads((PHASE2 / name).read_text(encoding="utf-8"))


def test_independent_rdu_artifact_locks_cross_family_failure() -> None:
    report = _load("rdu_independent_trace_reproduction_check.json")

    assert report["method_branch"] == "rdu_topk"
    assert report["reproduction_pass"] is False
    assert str(report["status"]).startswith("NOT REPRODUCED")
    measured = report["measured_decision"]
    assert measured["best_compressed_policy"] == "rkv_like"
    assert measured["rdu_is_best_compressed"] is False
    strict = report["strict_family_pass"]
    assert strict["same_family_positive"] is True
    assert strict["cross_family_positive"] is False
    assert strict["cross_family_min_margin"] < 0.0


def test_alternate_rdu_artifact_locks_same_family_failure() -> None:
    report = _load("rdu_alt_surface_reproduction_check.json")

    assert report["method_branch"] == "rdu_topk"
    assert report["reproduction_pass"] is False
    assert str(report["status"]).startswith("NOT REPRODUCED")
    measured = report["measured_decision"]
    assert measured["best_compressed_policy"] == "tf_sparse_r0.55_p0.05_m0.12_a2"
    assert measured["rdu_is_best_compressed"] is False
    same_family = report["measured_family_separation"]["same_family_margin_nll_vs_rdu"]
    assert same_family["tf_sparse_r0.55_p0.05_m0.12_a2"] < 0.0


def test_psi_artifact_locks_fresh_surface_kill() -> None:
    report = _load("psi_fresh_sparse_cache_check.json")

    assert report["policy_name"] == "psi_topk"
    assert report["n_scored_traces"] == 70
    assert str(report["status"]).startswith("KILLED")
    assert report["decision"]["promotion_pass"] is False
    summary = report["summary"]
    assert summary["psi_topk"]["nll"] > summary["rkv_like"]["nll"]
    assert summary["psi_topk"]["nll"] > summary["thin_kv_like"]["nll"]
    assert report["decision"]["paired_delta_vs_rkv_like"]["ci95_low"] > 0.0
    assert report["decision"]["paired_delta_vs_thin_kv_like"]["ci95_low"] > 0.0


def test_vwac_artifact_locks_fresh_surface_kill() -> None:
    report = _load("vwac_fresh_sparse_cache_check.json")

    assert report["policy_name"] == "vwac_topk"
    assert report["n_scored_traces"] == 64
    assert str(report["status"]).startswith("KILLED")
    assert report["decision"]["promotion_pass"] is False
    summary = report["summary"]
    assert summary["vwac_topk"]["nll"] > summary["rkv_like"]["nll"]
    assert summary["vwac_topk"]["nll"] > summary["thin_kv_like"]["nll"]
    assert report["decision"]["paired_delta_vs_rkv_like"]["ci95_low"] > 0.0
    assert report["decision"]["paired_delta_vs_thin_kv_like"]["ci95_low"] > 0.0


def test_diagnostic_packet_hashes_saved_falsification_artifacts() -> None:
    manifest = json.loads((DIAGNOSTIC_PACKET / "manifest.json").read_text(encoding="utf-8"))

    assert "not a positive method claim" in manifest["claim_boundary"]
    assert manifest["script"]["sha256"].startswith("sha256:")
    artifacts = manifest["artifacts"]
    assert {artifact["id"] for artifact in artifacts} == {
        "frozen_sparse_cache_probe",
        "rdu_robustness_diagnostic",
        "rdu_same_surface_rerun",
        "rdu_alternate_surface",
        "rdu_independent_surface",
        "psi_fresh_surface",
        "vwac_fresh_surface",
    }
    for artifact in artifacts:
        path = PHASE2 / artifact["path"]
        expected = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        assert artifact["sha256"] == expected
        assert artifact["provenance"]["command"].startswith("./venv_arm64/bin/python")
        assert isinstance(artifact["provenance"]["source_metadata"], dict)
        assert isinstance(artifact["provenance"]["input_hashes"], dict)
    table = (DIAGNOSTIC_PACKET / "falsification_table.md").read_text(encoding="utf-8")
    assert "stale_positive_first_surface" in table
    assert "historical_positive_same_surface" in table
    assert "same_family_falsification" in table
    assert "cross_family_falsification" in table
