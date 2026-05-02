from __future__ import annotations

import csv
import json

from scripts import build_source_private_cross_family_negative_boundary as boundary


def test_cross_family_boundary_has_no_claim_ready_methods(tmp_path) -> None:
    payload = boundary.build_boundary(output_dir=tmp_path)

    assert payload["pass_gate"] is True
    assert payload["headline"]["claim_ready_cross_family_methods"] == 0
    assert payload["headline"]["negative_boundary_rows"] >= 10
    assert payload["headline"]["oracle_headroom_rows"] >= 2
    assert (tmp_path / "cross_family_negative_boundary.json").exists()
    assert (tmp_path / "cross_family_negative_boundary.md").exists()
    assert (tmp_path / "cross_family_negative_boundary.csv").exists()


def test_boundary_includes_key_failed_and_asymmetric_families(tmp_path) -> None:
    payload = boundary.build_boundary(output_dir=tmp_path)
    families = payload["headline"]["by_family"]

    assert "learned Wyner-Ziv / scalar syndrome" in families
    assert "masked innovation receiver" in families
    assert "learned target-preserving receiver" in families
    assert families["masked innovation receiver"]["negative_boundary_rows"] >= 2
    assert families["learned Wyner-Ziv / scalar syndrome"]["negative_boundary_rows"] >= 5


def test_boundary_outputs_are_parseable(tmp_path) -> None:
    boundary.build_boundary(output_dir=tmp_path)
    summary = json.loads((tmp_path / "cross_family_negative_boundary.json").read_text(encoding="utf-8"))
    with (tmp_path / "cross_family_negative_boundary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert summary["pass_gate"] is True
    assert rows
    assert "claim_status" in rows[0]
