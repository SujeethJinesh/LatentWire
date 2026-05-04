from __future__ import annotations

import csv
import json

from scripts.build_source_private_systems_boundary_figure_table import (
    _kv_bytes,
    _round_up,
    build_systems_boundary_figure_table,
)


def _row(
    row_id: str,
    *,
    dataset: str,
    split: str,
    payload: float,
    framed: float,
    accuracy: float = 0.5,
    headline: bool = True,
) -> dict[str, object]:
    return {
        "artifact_path": f"results/{row_id}.json",
        "dataset": dataset,
        "eval_rows": 100,
        "framed_record_bytes": framed,
        "headline_eligible": headline,
        "matched_accuracy_mean": accuracy,
        "payload_bytes": payload,
        "receiver_decode_p50_us": 10.0,
        "receiver_decode_p95_us": 20.0,
        "row_id": row_id,
        "same_byte_text_accuracy": accuracy - 0.02,
        "source_scoring_ms_per_question": 2.5,
        "source_kv_exposed": False,
        "source_text_exposed": False,
        "split": split,
    }


def _write_comparator(path) -> None:
    payload = {
        "rows": [
            _row(
                "arc_challenge_test_12b",
                dataset="ARC-Challenge",
                split="test",
                payload=12,
                framed=15,
                accuracy=0.344,
            ),
            _row(
                "openbookqa_test_3b",
                dataset="OpenBookQA",
                split="test",
                payload=3,
                framed=6,
                accuracy=0.378,
            ),
            _row(
                "hellaswag_validation1024_2b",
                dataset="HellaSwag",
                split="validation1024",
                payload=2,
                framed=5,
                accuracy=0.461,
                headline=False,
            ),
            _row(
                "hellaswag_full_compact_1b",
                dataset="HellaSwag",
                split="validation_full_compaction",
                payload=1,
                framed=4,
                accuracy=0.619,
                headline=False,
            ),
        ],
        "source_model_config": {
            "hidden_size": 896,
            "kv_elements_per_source_token": 6144,
            "model_type": "qwen2",
            "num_attention_heads": 14,
            "num_hidden_layers": 24,
            "num_key_value_heads": 2,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_byte_helpers_match_qwen_config() -> None:
    config = {"kv_elements_per_source_token": 6144}
    assert _round_up(15, 64) == 64.0
    assert _round_up(0, 64) == 0.0
    assert _kv_bytes(config, bits_per_element=16) == 12288.0
    assert _kv_bytes(config, bits_per_element=1) == 768.0
    assert abs(_kv_bytes(config, bits_per_element=16, layer_fraction=0.30) - 3686.4) < 1e-9


def test_systems_boundary_writes_paper_artifacts_and_guards(tmp_path) -> None:
    comparator = tmp_path / "cross_benchmark_systems_comparator.json"
    _write_comparator(comparator)

    payload = build_systems_boundary_figure_table(
        comparator_path=comparator,
        output_dir=tmp_path / "out",
    )

    assert payload["pass_gate"] is True
    assert payload["headline"]["packet_row_count"] == 8
    assert payload["headline"]["cached_source_packet_row_count"] == 4
    assert payload["headline"]["end_to_end_source_scoring_packet_row_count"] == 4
    assert payload["headline"]["min_packet_framed_bytes"] == 4.0
    assert payload["headline"]["max_packet_framed_bytes"] == 15.0
    assert payload["headline"]["min_source_state_floor_ratio_vs_max_packet"] >= 50.0
    rows = {row["row_id"]: row for row in payload["rows"]}
    cached = rows["latentwire_arc_challenge_test_12b_cached_source"]
    e2e = rows["latentwire_arc_challenge_test_12b_end_to_end_source_scoring"]
    assert cached["systems_row_id"] == "latentwire_packet_cached_source"
    assert cached["source_private"] is True
    assert cached["source_packet_cached"] is True
    assert cached["source_scoring_included"] is False
    assert cached["native_measured"] is False
    assert cached["native_claim_allowed"] is False
    assert e2e["systems_row_id"] == "latentwire_packet_end_to_end_source_scoring"
    assert e2e["source_private"] is True
    assert e2e["source_scoring_included"] is True
    assert e2e["source_scoring_ms_per_question"] == 2.5
    assert e2e["source_scoring_total_s"] == 0.25
    assert e2e["receiver_decode_p50_us"] == 10.0
    assert rows["same_byte_text_control_arc"]["source_text_exposed"] is True
    assert rows["source_score_vector_fp16_floor"]["source_private"] is False
    assert rows["source_logit_vector_fp16_floor"]["source_private"] is False
    assert rows["c2c_fp16_kv_floor"]["source_kv_exposed"] is True
    assert rows["c2c_fp16_kv_floor"]["nvidia_vllm_required"] is True
    assert "native C2C" in rows["c2c_fp16_kv_floor"]["overclaim_guard"]
    assert rows["qjl_1bit_kv_floor"]["framed_bytes"] == 768.0
    assert rows["vllm_pagedattention_fp16_kv_floor"]["measurement_status"] == "native_nvidia_pending"

    out = tmp_path / "out"
    assert (out / "systems_boundary_figure_data.json").exists()
    assert (out / "systems_boundary_figure_data.csv").exists()
    assert (out / "systems_boundary_table.md").exists()
    assert (out / "systems_boundary_table.tex").exists()
    assert (out / "systems_boundary_waterfall.svg").exists()
    assert (out / "manifest.json").exists()


def test_written_csv_tex_svg_are_parseable(tmp_path) -> None:
    comparator = tmp_path / "cross_benchmark_systems_comparator.json"
    _write_comparator(comparator)
    build_systems_boundary_figure_table(
        comparator_path=comparator,
        output_dir=tmp_path / "out",
    )

    with (tmp_path / "out" / "systems_boundary_figure_data.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    tex = (tmp_path / "out" / "systems_boundary_table.tex").read_text(encoding="utf-8")
    svg = (tmp_path / "out" / "systems_boundary_waterfall.svg").read_text(encoding="utf-8")
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))

    assert len(rows) == 22
    assert "overclaim_guard" in rows[0]
    assert "\\label{tab:systems-boundary}" in tex
    assert "Log-scale framed or state bytes" in svg
    assert manifest["pass_gate"] is True
