from __future__ import annotations

import json

from scripts import run_gsm8k_contract_gauge_wrapper_sweep as sweep


def test_parse_candidate_accepts_alignment_rank_and_canonical_rank() -> None:
    label, spec = sweep._parse_candidate("fitted=grouped_fitted_rotation_transport:16:8")
    assert label == "fitted"
    assert spec["alignment"] == "grouped_fitted_rotation_transport"
    assert spec["quantization_correction"] == "bridge_ridge_qk_dynalign_module_replace"
    assert spec["quantization_correction_rank"] == 16
    assert spec["canonical_subspace_rank"] == 8


def test_markdown_writer_marks_promotion_gate(tmp_path) -> None:
    payload = {
        "date": "2026-04-21",
        "baseline_contract": "results/gsm8k_smoke_contract_20260421/gsm8k_smoke_contract_20260421.md",
        "config": {
            "source_model": "src",
            "target_model": "tgt",
            "slice_size": 32,
            "eval_file": "data.jsonl",
            "calibration_file": ".debug/calibration_64.txt",
        },
        "rows": [
            {
                "label": "fitted_rotation",
                "alignment": "grouped_fitted_rotation_transport",
                "quantization_correction_rank": 16,
                "accuracy": 0.125,
                "paired_vs_target": {"win": 2, "loss": 0, "tie": 30},
                "numeric_extraction_coverage": 32,
                "empty_predictions": 0,
            },
            {
                "label": "shared_basis",
                "alignment": "grouped_shared_basis_transport",
                "quantization_correction_rank": 16,
                "accuracy": 0.0625,
                "paired_vs_target": {"win": 0, "loss": 0, "tie": 32},
                "numeric_extraction_coverage": 32,
                "empty_predictions": 0,
            },
        ],
        "checks": {
            "fitted_rotation": {
                "row_count_matches_slice": True,
                "example_ids_match_target": True,
                "no_empty_predictions": True,
                "numeric_extraction_coverage": True,
                "beats_target": True,
            },
            "shared_basis": {
                "row_count_matches_slice": True,
                "example_ids_match_target": True,
                "no_empty_predictions": True,
                "numeric_extraction_coverage": True,
                "beats_target": False,
            },
        },
    }
    path = tmp_path / "sweep.md"
    sweep._write_markdown(path, payload)
    markdown = path.read_text()
    assert "| fitted_rotation | grouped_fitted_rotation_transport | 16 | 0.1250 | 2 | 0 | 30 | 32 | 0 | yes |" in markdown
    assert "| shared_basis | grouped_shared_basis_transport | 16 | 0.0625 | 0 | 0 | 32 | 32 | 0 | no |" in markdown


def test_default_candidates_are_json_serializable() -> None:
    json.dumps(sweep.DEFAULT_ALIGNMENT_CANDIDATES, sort_keys=True)
