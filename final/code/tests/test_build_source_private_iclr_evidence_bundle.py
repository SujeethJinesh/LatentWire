from __future__ import annotations

import csv
import json

from scripts import build_source_private_iclr_evidence_bundle as bundle


def test_build_bundle_passes_and_writes_expected_artifacts(tmp_path) -> None:
    payload = bundle.build_bundle(output_dir=tmp_path)

    assert payload["pass_gate"] is True
    assert len(payload["contribution_rows"]) >= 5
    assert len(payload["novelty_matrix"]) >= 8
    assert all(check["pass"] for check in payload["pass_checks"])
    assert (tmp_path / "iclr_evidence_bundle.json").exists()
    assert (tmp_path / "iclr_evidence_bundle.md").exists()
    assert (tmp_path / "novelty_matrix.csv").exists()
    assert (tmp_path / "contribution_matrix.csv").exists()
    assert (tmp_path / "reproduce_iclr_evidence_bundle.sh").exists()
    assert (tmp_path / "manifest.json").exists()


def test_bundle_highlights_source_private_and_systems_axes(tmp_path) -> None:
    payload = bundle.build_bundle(output_dir=tmp_path)
    novelty_rows = {row["comparison"]: row for row in payload["novelty_matrix"]}
    contribution_rows = {row["contribution"]: row for row in payload["contribution_rows"]}

    assert novelty_rows["LatentWire source-private packet"]["source_private"] is True
    assert novelty_rows["C2C cache-to-cache communication"]["requires_model_internals"] is True
    assert "QJL" in novelty_rows["QJL 1-bit sign sketch"]["comparison"]
    assert "1000x" in next(
        check["check"] for check in payload["pass_checks"] if check["check"].endswith("above_1000x")
    )
    assert contribution_rows["Systems byte/KV-cache accounting frontier"]["status"]


def test_written_bundle_is_valid_json_and_csv(tmp_path) -> None:
    bundle.build_bundle(output_dir=tmp_path)
    summary = json.loads((tmp_path / "iclr_evidence_bundle.json").read_text(encoding="utf-8"))
    with (tmp_path / "novelty_matrix.csv").open(encoding="utf-8", newline="") as handle:
        novelty_rows = list(csv.DictReader(handle))

    assert summary["pass_gate"] is True
    assert novelty_rows
    assert "comparison" in novelty_rows[0]
