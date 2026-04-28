from __future__ import annotations

import json

from scripts import run_source_private_codebook_remap_gate as gate


def test_codebook_remap_changes_codes_but_keeps_public_surface() -> None:
    payload = gate.run_gate(examples=64, candidates=4, family_set="all", seeds=[29, 31], budgets=[2])

    assert payload["pass_gate"] is True
    assert payload["exact_id_parity_across_seeds"] is True
    assert payload["public_surface_parity_across_seeds"] is True
    assert payload["codebook_remapped"] is True
    assert payload["unique_codebook_count"] == 2
    assert payload["seed_rows"][0]["diagnostic_preview"] != payload["seed_rows"][1]["diagnostic_preview"]


def test_markdown_and_cli_payload_are_serializable(tmp_path) -> None:
    payload = gate.run_gate(examples=16, candidates=4, family_set="core", seeds=[29], budgets=[2, 4])
    output = tmp_path / "summary.md"

    gate._write_markdown(output, payload)
    encoded = json.dumps(payload, sort_keys=True)

    assert "Source-Private Codebook-Remap Gate" in output.read_text(encoding="utf-8")
    assert '"pass_gate": true' in encoded
    assert payload["seed_rows"][0]["budget_summaries"][0]["matched"] == 1.0
