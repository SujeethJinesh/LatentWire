from __future__ import annotations

import json

from scripts import run_gsm8k_contract_checkpoint_sweep as sweep


def test_parse_candidate_requires_label_and_path() -> None:
    label, path = sweep._parse_candidate("dynalign=checkpoints/foo.pt")
    assert label == "dynalign"
    assert path == "checkpoints/foo.pt"


def test_markdown_writer_marks_promotion_gate(tmp_path) -> None:
    payload = {
        "date": "2026-04-21",
        "baseline_contract": "results/gsm8k_smoke_contract_20260421/gsm8k_smoke_contract_20260421.md",
        "config": {"source_model": "src", "target_model": "tgt", "slice_size": 32, "eval_file": "data.jsonl"},
        "rows": [
            {
                "label": "dynalign",
                "accuracy": 0.125,
                "paired_vs_target": {"win": 3, "loss": 1, "tie": 28},
                "numeric_extraction_coverage": 32,
                "empty_predictions": 0,
            },
            {
                "label": "sae_adapter",
                "accuracy": 0.03125,
                "paired_vs_target": {"win": 1, "loss": 2, "tie": 29},
                "numeric_extraction_coverage": 28,
                "empty_predictions": 0,
            },
        ],
        "checks": {
            "dynalign": {
                "row_count_matches_slice": True,
                "example_ids_match_target": True,
                "no_empty_predictions": True,
                "numeric_extraction_coverage": True,
                "beats_target": True,
            },
            "sae_adapter": {
                "row_count_matches_slice": True,
                "example_ids_match_target": True,
                "no_empty_predictions": True,
                "numeric_extraction_coverage": False,
                "beats_target": False,
            },
        },
    }
    path = tmp_path / "sweep.md"
    sweep._write_markdown(path, payload)
    markdown = path.read_text()
    assert "| dynalign | 0.1250 | 3 | 1 | 28 | 32 | 0 | yes |" in markdown
    assert "| sae_adapter | 0.0312 | 1 | 2 | 29 | 28 | 0 | no |" in markdown


def test_default_candidates_are_json_serializable() -> None:
    json.dumps(sweep.DEFAULT_CANDIDATES, sort_keys=True)
