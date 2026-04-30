from __future__ import annotations

import json
import pathlib

from scripts import summarize_source_private_masked_consistency_label_blind_stress as summary_script


def _write_run(
    run_dir: pathlib.Path,
    *,
    candidate_view: str,
    remap_slot_seed: int | None,
    pass_gate: bool,
    matched: float,
    hamming: float,
    target: float = 0.25,
    control: float = 0.25,
) -> None:
    run_dir.mkdir(parents=True)
    conditions = ["target_only", "matched_consistency_packet", "zero_source", "shuffled_source"]
    metrics = {
        "target_only": {"accuracy": target},
        "matched_consistency_packet": {"accuracy": matched},
        "zero_source": {"accuracy": target},
        "shuffled_source": {"accuracy": control},
    }
    payload = {
        "candidate_view": candidate_view,
        "remap_slot_seed": remap_slot_seed,
        "train_seed": 1,
        "eval_seed": 2,
        "summary": {
            "n": 256,
            "conditions": conditions,
            "pass_gate": pass_gate,
            "source_packet_pass": pass_gate,
            "exact_id_parity": True,
            "target_only_accuracy": target,
            "learned_matched_accuracy": matched,
            "hamming_matched_accuracy": hamming,
            "best_control_condition": "shuffled_source",
            "best_control_accuracy": control,
            "learned_minus_target": matched - target,
            "learned_minus_best_control": matched - control,
            "paired_bootstrap": {
                "learned_matched_vs_target": {"ci95_low": matched - target, "ci95_high": matched - target},
                "learned_matched_vs_best_control": {"ci95_low": matched - control, "ci95_high": matched - control},
            },
            "learned_metrics": metrics,
            "hamming_metrics": {
                "target_only": {"accuracy": target},
                "matched_consistency_packet": {"accuracy": hamming},
                "zero_source": {"accuracy": target},
                "shuffled_source": {"accuracy": control},
            },
        },
    }
    (run_dir / "run_summary.json").write_text(json.dumps(payload), encoding="utf-8")


def test_label_blind_summary_passes_when_slot_remap_collapses(tmp_path) -> None:
    full = tmp_path / "full"
    slot = tmp_path / "slot"
    no_diag = tmp_path / "no_diag"
    _write_run(full, candidate_view="full", remap_slot_seed=None, pass_gate=True, matched=0.95, hamming=0.95)
    _write_run(slot, candidate_view="slot", remap_slot_seed=901, pass_gate=False, matched=0.25, hamming=0.25)
    _write_run(no_diag, candidate_view="no_diag", remap_slot_seed=None, pass_gate=True, matched=0.90, hamming=0.90)

    payload = summary_script.summarize_runs(run_dirs=[full, slot, no_diag], output_dir=tmp_path / "out")

    assert payload["headline"]["pass_gate"] is True
    assert payload["headline"]["opaque_slot_collapse"] is True
    assert (tmp_path / "out" / "summary.md").exists()


def test_label_blind_summary_fails_when_slot_remap_has_lift(tmp_path) -> None:
    full = tmp_path / "full"
    slot = tmp_path / "slot"
    _write_run(full, candidate_view="full", remap_slot_seed=None, pass_gate=True, matched=0.95, hamming=0.95)
    _write_run(slot, candidate_view="slot", remap_slot_seed=901, pass_gate=True, matched=0.50, hamming=0.50)

    payload = summary_script.summarize_runs(run_dirs=[full, slot], output_dir=tmp_path / "out")

    assert payload["headline"]["pass_gate"] is False
    assert payload["headline"]["opaque_slot_collapse"] is False
