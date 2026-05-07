from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from experimental.thoughtflow_fp8.phase2 import build_diagnostic_packet


PHASE2 = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
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
    assert manifest["git"]["thoughtflow_path_dirty_at_generation"] is False
    assert manifest["git"]["thoughtflow_path_status_at_generation"] == ""
    script_path = REPO_ROOT / str(manifest["script"]["path"])
    expected_script_hash = "sha256:" + hashlib.sha256(script_path.read_bytes()).hexdigest()
    assert manifest["script"]["sha256"] == expected_script_hash
    assert "current builder-file integrity hash" in manifest["script"]["sha256_role"]
    artifacts = manifest["artifacts"]
    preregistrations = manifest["preregistrations"]
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
        assert artifact["provenance"]["input_paths"]
        assert artifact["provenance"]["input_hashes"]
        if artifact["id"] in {
            "frozen_sparse_cache_probe",
            "psi_fresh_surface",
            "vwac_fresh_surface",
        }:
            command = artifact["provenance"]["command"]
            assert "--model-revision 2290a62682d06624634c1f46a6ad5be0f47f38aa" in command
            assert "--json-output .debug/thoughtflow_replay/" in command
            assert "--md-output .debug/thoughtflow_replay/" in command
            source_metadata = artifact["provenance"]["source_metadata"]
            assert source_metadata["model_revision"] == "2290a62682d06624634c1f46a6ad5be0f47f38aa"
            assert (
                source_metadata["tokenizer_revision"]
                == "2290a62682d06624634c1f46a6ad5be0f47f38aa"
            )
        if artifact["id"] in {"psi_fresh_surface", "vwac_fresh_surface"}:
            command = artifact["provenance"]["command"]
            assert "--input-jsonl results/c2c_" in command
        for input_path in artifact["provenance"]["input_hashes"]:
            assert (REPO_ROOT / input_path).is_file()
        if artifact["id"] in {
            "frozen_sparse_cache_probe",
            "rdu_same_surface_rerun",
            "rdu_alternate_surface",
        }:
            assert artifact["provenance"]["input_path_inference"]["source"] == (
                "run_real_trace_retention.DEFAULT_TRACES"
            )
        if artifact["id"] in {
            "rdu_same_surface_rerun",
            "rdu_alternate_surface",
            "rdu_independent_surface",
        }:
            measured = artifact["provenance"]["source_metadata"]["measured_reproduction"]
            for field in (
                "model_name",
                "keep_fraction",
                "max_length",
                "continuation_tokens",
                "n_scored_traces",
            ):
                assert field in measured

        if artifact["role"].startswith(("stale_positive", "historical_positive")):
            assert artifact["historical_status"] in {"ALIVE", "PROMOTED", "REPRODUCED"}
            assert str(artifact["summary"]["status"]).startswith("HISTORICAL/SUPERSEDED: ")
            assert artifact["current_status"] == "superseded_diagnostic_only"
            assert artifact["current_claim_allowed"] is False
            assert artifact["positive_method_claim_allowed"] is False
        else:
            assert not str(artifact["summary"]["status"]).startswith(
                ("ALIVE", "PROMOTED", "REPRODUCED")
            )
            assert artifact["current_status"] == "current_falsification_evidence"
            assert artifact["positive_method_claim_allowed"] is False
    assert {item["id"] for item in preregistrations} == {
        "rdu_preregistration",
        "psi_preregistration",
        "vwac_preregistration",
    }
    for preregistration in preregistrations:
        path = PHASE2 / preregistration["path"]
        expected = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        assert preregistration["sha256"] == expected
        assert preregistration["role"].endswith("preregistration")
    table = (DIAGNOSTIC_PACKET / "falsification_table.md").read_text(encoding="utf-8")
    assert "stale_positive_first_surface" in table
    assert "historical_positive_same_surface" in table
    assert "SUPERSEDED historical readout; later gates demote or kill this row" in table
    for stale_word in ["ALIVE", "PROMOTED", "REPRODUCED"]:
        assert f"historical readout: {stale_word}" not in table
    assert "same_family_falsification" in table
    assert "cross_family_falsification" in table
    assert "## Preregistrations" in table


def test_diagnostic_packet_input_hashes_are_tracked_for_clean_checkout() -> None:
    manifest = json.loads((DIAGNOSTIC_PACKET / "manifest.json").read_text(encoding="utf-8"))
    tracked_paths = set(
        subprocess.check_output(["git", "ls-files"], cwd=REPO_ROOT, text=True).splitlines()
    )

    input_paths = {
        input_path
        for artifact in manifest["artifacts"]
        for input_path in artifact["provenance"].get("input_hashes", {})
    }

    assert input_paths
    assert input_paths.issubset(tracked_paths)


def test_diagnostic_packet_has_no_stale_results_duplicate() -> None:
    stale_results_packet = PHASE2 / "results/thoughtflow_diagnostic_packet_20260506"

    assert not stale_results_packet.exists()


def test_diagnostic_packet_builder_refuses_dirty_thoughtflow_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        build_diagnostic_packet,
        "_thoughtflow_path_status",
        lambda: " M experimental/thoughtflow_fp8/phase2/build_diagnostic_packet.py",
    )

    output_dir = tmp_path / "packet"
    with pytest.raises(RuntimeError, match="refusing to build diagnostic packet"):
        build_diagnostic_packet.build_packet(output_dir)

    assert not output_dir.exists()


def test_diagnostic_packet_rejects_missing_declared_input_path() -> None:
    with pytest.raises(ValueError, match="unresolved diagnostic packet input path"):
        build_diagnostic_packet._hash_existing_input_paths(["missing/thoughtflow/input.jsonl"])
