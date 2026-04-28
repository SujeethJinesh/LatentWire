from __future__ import annotations

from scripts import run_source_private_evidence_packet_gate as gate


def test_matched_syndrome_passes_source_private_gate() -> None:
    rows, summary = gate.run_gate(examples=32, candidates=4, seed=3, syndrome_bytes=2)

    assert len(rows) == 32
    assert summary["pass_gate"] is True
    assert summary["metrics"]["matched_syndrome"]["accuracy"] == 1.0
    assert summary["matched_minus_best_no_source"] >= 0.15
    assert summary["metrics"]["structured_text_matched"]["accuracy"] == summary["metrics"]["target_only"]["accuracy"]


def test_answer_only_and_random_controls_do_not_recover_gain() -> None:
    _, summary = gate.run_gate(examples=40, candidates=4, seed=11, syndrome_bytes=2)

    target_accuracy = summary["metrics"]["target_only"]["accuracy"]
    for condition in [
        "zero_source",
        "shuffled_source",
        "random_same_byte",
        "answer_only",
        "answer_masked",
        "target_only_sidecar",
    ]:
        assert summary["metrics"][condition]["accuracy"] <= target_accuracy + 0.02


def test_full_structured_text_is_oracle_but_uses_more_bytes() -> None:
    _, summary = gate.run_gate(examples=16, candidates=4, seed=19, syndrome_bytes=2)

    assert summary["metrics"]["structured_text_full"]["accuracy"] == 1.0
    assert summary["metrics"]["structured_text_full"]["mean_payload_bytes"] > summary["syndrome_bytes"]
