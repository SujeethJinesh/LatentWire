from __future__ import annotations

import json

from scripts import summarize_source_private_masked_consistency_public_gate as summary


def _write_run(path, payload: dict) -> None:
    path.mkdir(parents=True)
    (path / "run_summary.json").write_text(json.dumps(payload), encoding="utf-8")


def _packet_payload(*, learned: float, public_hash: str = "hash") -> dict:
    return {
        "candidate_view": "diag_only",
        "diagnostic_table_mode": "plausible_decoys",
        "train_eval_id_intersection_count": 0,
        "summary": {
            "n": 10,
            "exact_id_sha256": public_hash,
            "exact_id_parity": True,
            "target_only_accuracy": 0.25,
            "learned_matched_accuracy": learned,
            "hamming_matched_accuracy": learned,
            "best_control_condition": "zero_source",
            "best_control_accuracy": 0.25,
            "learned_controls_ok": True,
            "pass_gate": learned >= 0.40,
            "paired_bootstrap": {
                "learned_matched_vs_target": {"ci95_low": 0.20},
                "learned_matched_vs_best_control": {"ci95_low": 0.20},
            },
        },
    }


def _public_payload(*, public_acc: float, public_hash: str = "hash") -> dict:
    return {
        "candidate_view": "diag_only",
        "diagnostic_table_mode": "plausible_decoys",
        "summary": {
            "exact_id_sha256": public_hash,
            "target_only_accuracy": 0.25,
            "public_only_accuracy": public_acc,
            "max_allowed_lift": 0.05,
            "train_eval_id_intersection_count": 0,
            "paired_bootstrap_public_vs_target": {"ci95_high": public_acc - 0.25},
        },
    }


def test_public_separation_gate_passes_when_packet_lift_is_private(tmp_path) -> None:
    packet_dir = tmp_path / "packet"
    public_dir = tmp_path / "public"
    _write_run(packet_dir, _packet_payload(learned=0.60))
    _write_run(public_dir, _public_payload(public_acc=0.27))

    payload = summary.summarize_runs(
        packet_dirs=[packet_dir],
        public_dirs=[public_dir],
        output_dir=tmp_path / "summary",
        public_fraction_limit=0.25,
    )

    assert payload["headline"]["pass_gate"] is True
    assert payload["rows"][0]["public_explained_fraction"] < 0.25
    assert (tmp_path / "summary" / "summary.md").exists()


def test_public_separation_gate_fails_when_packet_lift_is_too_small(tmp_path) -> None:
    packet_dir = tmp_path / "packet"
    public_dir = tmp_path / "public"
    _write_run(packet_dir, _packet_payload(learned=0.33))
    _write_run(public_dir, _public_payload(public_acc=0.25))

    payload = summary.summarize_runs(
        packet_dirs=[packet_dir],
        public_dirs=[public_dir],
        output_dir=tmp_path / "summary",
        public_fraction_limit=0.25,
    )

    assert payload["headline"]["pass_gate"] is False
    assert payload["rows"][0]["learned_minus_target"] == 0.08000000000000002
