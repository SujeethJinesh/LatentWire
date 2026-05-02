from __future__ import annotations

import json

from scripts import build_source_private_hellaswag_tinyllama_full_mac_systems_card as card


def _write(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _full_payload(tmp_path) -> dict[str, object]:
    score_cache = tmp_path / "source_eval_score_cache.json"
    hidden_cache = tmp_path / "source_eval_hidden_cache.npz"
    hidden_meta = tmp_path / "source_eval_hidden_cache.json"
    score_cache.write_text("{}\n", encoding="utf-8")
    hidden_cache.write_bytes(b"hidden")
    hidden_meta.write_text("{}\n", encoding="utf-8")
    return {
        "headline": {"eval_rows": 10042},
        "timing": {"total_seconds": 100.0},
        "eval_cache_metadata": {
            "score_cache": str(score_cache),
            "hidden_cache": str(hidden_cache),
            "hidden_cache_meta": str(hidden_meta),
            "score_model": {
                "latency_s": 20.0,
                "model_path": "tinyllama",
                "device": "mps",
                "dtype": "float16",
            },
            "hidden_model": {
                "latency_s": 30.0,
                "model_path": "tinyllama",
                "device": "mps",
                "dtype": "float16",
            },
        },
        "bagged_gate": {
            "pass_gate": True,
            "headline": {
                "selected_eval_accuracy": 0.51,
                "best_label_copy_eval_accuracy": 0.46,
                "source_label_copy_eval_accuracy": 0.45,
                "score_only_bagged_control_accuracy": 0.45,
                "selected_minus_best_label_copy": 0.05,
                "selected_minus_score_only_bagged_control": 0.06,
                "paired_ci95_low_vs_best_label_copy": 0.03,
            },
            "packet_contract": {
                "raw_payload_bytes": 2,
                "framed_record_bytes": 5,
                "source_text_exposed": False,
                "source_kv_exposed": False,
                "raw_hidden_vector_transmitted": False,
                "raw_scores_transmitted": False,
            },
        },
    }


def test_mac_systems_card_records_byte_and_timing_boundary(tmp_path) -> None:
    full = tmp_path / "full.json"
    _write(full, _full_payload(tmp_path))

    payload = card.build_card(output_dir=tmp_path / "out", full_eval=full)

    assert payload["pass_gate"] is True
    assert payload["headline"]["packet_raw_bytes"] == 2
    assert payload["headline"]["packet_framed_bytes"] == 5
    assert payload["headline"]["single_request_cacheline_bytes"] == 64.0
    assert payload["headline"]["batch64_cacheline_bytes_per_request"] == 5.0
    assert payload["headline"]["source_scoring_wall_time_s"] == 20.0
    assert payload["headline"]["source_hidden_extraction_wall_time_s"] == 30.0
    assert payload["headline"]["packet_build_and_gate_wall_time_s"] == 50.0
    assert payload["headline"]["native_gpu_claims_allowed"] is False
    assert (tmp_path / "out" / "tinyllama_full_mac_systems_card.json").exists()
    assert (tmp_path / "out" / "manifest.json").exists()


def test_mac_systems_card_fails_if_source_exposure_changes(tmp_path) -> None:
    full = tmp_path / "full.json"
    payload = _full_payload(tmp_path)
    payload["bagged_gate"]["packet_contract"]["source_text_exposed"] = True
    _write(full, payload)

    result = card.build_card(output_dir=tmp_path / "out", full_eval=full)

    assert result["pass_gate"] is False
