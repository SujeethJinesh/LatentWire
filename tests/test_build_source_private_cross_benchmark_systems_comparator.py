from __future__ import annotations

import csv
import json

from scripts import build_source_private_cross_benchmark_systems_comparator as comparator


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _seed_payload(*, budget: int, eval_rows: int, text_gap: float = 0.03) -> dict:
    target = 0.25
    matched = 0.38
    return {
        "pass_gate": True,
        "budget_bytes": budget,
        "eval_rows": eval_rows,
        "aggregate": {
            "all_seeds_pass": True,
            "candidate_derangement_accuracy_max": 0.22,
            "matched_accuracy_max": matched + 0.002,
            "matched_accuracy_mean": matched,
            "matched_accuracy_min": matched - 0.002,
            "matched_minus_best_destructive_min": 0.11,
            "matched_minus_same_byte_text_min": text_gap,
            "matched_minus_target_mean": matched - target,
            "matched_minus_target_min": matched - target - 0.002,
            "paired_ci95_low_vs_target_min": 0.04,
            "pass_count": 5,
            "same_byte_structured_text_accuracy": matched - text_gap,
            "seed_count": 5,
            "target_accuracy": target,
        },
    }


def _phase_payload(*, eval_rows: int) -> dict:
    return {
        "eval_rows": eval_rows,
        "condition_metrics": {
            "matched_source_private_packet": {
                "p50_latency_ms": 0.01,
                "p95_latency_ms": 0.02,
            }
        },
        "systems_trace": {
            "batch64_cacheline_bytes_per_request": 15.0,
            "batch64_dma_bytes_per_request": 16.0,
            "phase_timings_s": {"source_scoring": 2.0},
            "peak_rss_mib": 123.0,
            "source_kv_exposed": False,
            "source_text_exposed": False,
        },
    }


def _fixture(tmp_path):
    config = tmp_path / "config.json"
    _write_json(
        config,
        {
            "model_type": "qwen2",
            "num_hidden_layers": 24,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "hidden_size": 896,
        },
    )
    arc = tmp_path / "arc_seed.json"
    arc_phase = tmp_path / "arc_phase.json"
    openbook = tmp_path / "openbook_seed.json"
    hellaswag = tmp_path / "hellaswag_seed.json"
    hellaswag_phase = tmp_path / "hellaswag_phase.json"
    hellaswag_label = tmp_path / "hellaswag_label.json"
    hellaswag_compact = tmp_path / "hellaswag_compact.json"
    _write_json(arc, _seed_payload(budget=12, eval_rows=1172))
    _write_json(arc_phase, _phase_payload(eval_rows=1172))
    _write_json(openbook, _seed_payload(budget=3, eval_rows=500))
    _write_json(hellaswag, _seed_payload(budget=2, eval_rows=1024, text_gap=0.07))
    _write_json(hellaswag_phase, _phase_payload(eval_rows=1024))
    _write_json(
        hellaswag_label,
        {
            "headline": {
                "label_copy_threat_present": True,
                "source_label_text_copy_accuracy": 0.462,
                "matched_minus_source_label_text_copy": -0.002,
            }
        },
    )
    _write_json(
        hellaswag_compact,
        {
            "pass_gate": True,
            "headline": {
                "all_prediction_equivalent": True,
                "compact_framed_record_bytes": 4,
                "compact_raw_payload_bytes": 1,
                "source_exposure_clear": True,
                "source_row_count": 2,
                "total_rows_covered": 200,
            },
            "source_rows": [
                {
                    "baseline_accuracy": 0.45,
                    "compact_accuracy": 0.50,
                    "compact_ci95_low_vs_baseline": 0.02,
                    "compact_delta_vs_baseline": 0.05,
                    "packet_accounting": {
                        "batch64_packed_compact_framed_bytes": 256,
                        "compact_framed_record_bytes_per_request": 4,
                        "compact_raw_payload_bytes_per_request": 1,
                    },
                    "positive_source_gate_passed": True,
                    "prediction_equivalence": True,
                    "row_count": 100,
                    "source_kv_exposed": False,
                    "source_name": "qwen",
                    "source_text_exposed": False,
                },
                {
                    "baseline_accuracy": 0.48,
                    "compact_accuracy": 0.56,
                    "compact_ci95_low_vs_baseline": 0.03,
                    "compact_delta_vs_baseline": 0.08,
                    "packet_accounting": {
                        "batch64_packed_compact_framed_bytes": 256,
                        "compact_framed_record_bytes_per_request": 4,
                        "compact_raw_payload_bytes_per_request": 1,
                    },
                    "positive_source_gate_passed": True,
                    "prediction_equivalence": True,
                    "row_count": 100,
                    "source_kv_exposed": False,
                    "source_name": "tiny",
                    "source_text_exposed": False,
                },
            ],
        },
    )
    benchmarks = (
        {
            "row_id": "arc",
            "dataset": "ARC-Challenge",
            "split": "test",
            "paper_role": "headline",
            "seed_artifact": arc,
            "phase_artifact": arc_phase,
            "label_copy_artifact": None,
        },
        {
            "row_id": "openbookqa",
            "dataset": "OpenBookQA",
            "split": "test",
            "paper_role": "headline",
            "seed_artifact": openbook,
            "phase_artifact": None,
            "label_copy_artifact": None,
        },
        {
            "row_id": "hellaswag",
            "dataset": "HellaSwag",
            "split": "validation",
            "paper_role": "diagnostic",
            "seed_artifact": hellaswag,
            "phase_artifact": hellaswag_phase,
            "label_copy_artifact": hellaswag_label,
        },
        {
            "row_id": "hellaswag_compact_1b",
            "dataset": "HellaSwag",
            "split": "validation_full_compaction",
            "paper_role": "systems_rate_candidate_not_native",
            "compaction_artifact": hellaswag_compact,
            "phase_artifact": None,
            "label_copy_artifact": None,
            "headline_eligible_override": False,
        },
    )
    return config, benchmarks


def test_cross_benchmark_comparator_passes_and_writes_outputs(tmp_path) -> None:
    config, benchmarks = _fixture(tmp_path)
    payload = comparator.build_comparator(
        output_dir=tmp_path / "out",
        source_config=config,
        benchmarks=benchmarks,
    )

    assert payload["pass_gate"] is True
    assert payload["headline"]["headline_eligible_benchmarks"] == 2
    assert payload["headline"]["diagnostic_benchmarks"] == 2
    assert payload["headline"]["native_systems_complete"] is False
    assert payload["headline"]["min_qjl_1bit_ratio_vs_framed"] >= 50.0
    assert (tmp_path / "out" / "cross_benchmark_systems_comparator.json").exists()
    assert (tmp_path / "out" / "cross_benchmark_systems_comparator.csv").exists()
    assert (tmp_path / "out" / "cross_benchmark_systems_comparator.md").exists()
    assert (tmp_path / "out" / "manifest.json").exists()


def test_hellaswag_is_marked_diagnostic_and_kv_formula_is_conservative(tmp_path) -> None:
    config, benchmarks = _fixture(tmp_path)
    payload = comparator.build_comparator(
        output_dir=tmp_path / "out",
        source_config=config,
        benchmarks=benchmarks,
    )
    rows = {row["row_id"]: row for row in payload["rows"]}

    assert rows["hellaswag"]["label_copy_threat"] is True
    assert rows["hellaswag"]["headline_eligible"] is False
    assert rows["arc"]["one_token_qjl_1bit_bytes"] == 768.0
    assert abs(rows["arc"]["one_token_kvcomm30_fp16_bytes"] - 3686.4) < 1e-9
    assert rows["arc"]["source_text_exposed"] is False
    assert rows["arc"]["source_kv_exposed"] is False
    assert "No native" in rows["arc"]["claim_forbidden"]
    compact = next(row for row in payload["rows"] if row["row_id"] == "hellaswag_compact_1b")
    assert compact["payload_bytes"] == 1.0
    assert compact["framed_record_bytes"] == 4.0
    assert compact["headline_eligible"] is False
    assert compact["pass_gate"] is True


def test_written_csv_and_manifest_are_parseable(tmp_path) -> None:
    config, benchmarks = _fixture(tmp_path)
    comparator.build_comparator(
        output_dir=tmp_path / "out",
        source_config=config,
        benchmarks=benchmarks,
    )

    with (tmp_path / "out" / "cross_benchmark_systems_comparator.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))

    assert len(rows) == 4
    assert rows[0]["row_id"]
    assert manifest["headline"]["pass_gate"] is True
