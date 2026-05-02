from __future__ import annotations

import csv
import json

from scripts.build_source_private_byte_amplification_ablation import (
    _kv_bytes,
    _packed_bytes_per_request,
    _round_up,
    build_byte_amplification_ablation,
)


def _row(
    row_id: str,
    *,
    dataset: str,
    split: str,
    payload: float,
    framed: float,
    accuracy: float,
) -> dict[str, object]:
    return {
        "artifact_path": f"results/{row_id}/artifact.json",
        "best_destructive_accuracy": accuracy - 0.15,
        "dataset": dataset,
        "framed_record_bytes": framed,
        "matched_accuracy_mean": accuracy,
        "payload_bytes": payload,
        "receiver_decode_p50_us": 10.0,
        "receiver_decode_p95_us": 20.0,
        "row_id": row_id,
        "split": split,
        "target_accuracy": accuracy - 0.08,
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
                "hellaswag_full_compact_1b",
                dataset="HellaSwag",
                split="validation_full_compaction",
                payload=1,
                framed=4,
                accuracy=0.619,
            ),
        ],
        "source_model_config": {
            "hidden_size": 896,
            "kv_elements_per_source_token": 6144,
            "model_type": "qwen2",
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_byte_helpers_match_expected_packet_and_kv_floors() -> None:
    config = {"kv_elements_per_source_token": 6144}
    assert _round_up(15, 64) == 64.0
    assert _round_up(64, 64) == 64.0
    assert _packed_bytes_per_request(5, quantum=64, batch_size=64) == 5.0
    assert _packed_bytes_per_request(5, quantum=128, batch_size=64) == 6.0
    assert _kv_bytes(config, bits_per_element=1.0) == 768.0
    assert _kv_bytes(config, bits_per_element=3.5) == 2688.0


def test_byte_amplification_ablation_writes_guarded_rows(tmp_path) -> None:
    comparator = tmp_path / "cross_benchmark_systems_comparator.json"
    _write_comparator(comparator)

    payload = build_byte_amplification_ablation(
        comparator_path=comparator,
        output_dir=tmp_path / "out",
        choice_count=4,
    )

    assert payload["pass_gate"] is True
    assert payload["headline"]["benchmark_row_count"] == 2
    assert payload["headline"]["interface_row_count"] == 20
    assert payload["headline"]["min_packet_framed_bytes"] == 4.0
    assert payload["headline"]["max_packet_framed_bytes"] == 15.0
    assert payload["headline"]["max_single_request_cacheline_amplification"] == 16.0
    assert payload["headline"]["min_kv_floor_ratio_vs_cacheline_packet"] == 12.0
    assert payload["headline"]["score_floor_ratio_vs_min_packet"] == 2.0

    rows = {
        (row["benchmark_row_id"], row["interface_id"]): row for row in payload["rows"]
    }
    packet = rows[("arc_challenge_test_12b", "latentwire_framed_packet")]
    padded = rows[("hellaswag_full_compact_1b", "latentwire_cacheline_padded_packet")]
    score = rows[("hellaswag_full_compact_1b", "source_score_vector_fp16_floor")]
    qjl = rows[("arc_challenge_test_12b", "qjl_1bit_kv_floor")]
    c2c = rows[("arc_challenge_test_12b", "c2c_fp16_kv_floor")]

    assert packet["source_private"] is True
    assert packet["native_measured"] is True
    assert padded["source_private"] is True
    assert padded["framed_or_state_bytes"] == 64.0
    assert padded["exact_prediction_equivalence_to_packet"] == 1.0
    assert score["source_private"] is False
    assert score["source_score_vector_exposed"] is True
    assert score["framed_or_state_bytes"] == 8.0
    assert qjl["source_kv_exposed"] is True
    assert qjl["framed_or_state_bytes"] == 768.0
    assert c2c["framed_or_state_bytes"] == 12288.0
    assert "native" in c2c["overclaim_guard"].lower()

    out = tmp_path / "out"
    assert (out / "byte_amplification_ablation.json").exists()
    assert (out / "byte_amplification_ablation.csv").exists()
    assert (out / "byte_amplification_ablation.md").exists()
    assert (out / "manifest.json").exists()


def test_byte_amplification_csv_and_markdown_are_parseable(tmp_path) -> None:
    comparator = tmp_path / "cross_benchmark_systems_comparator.json"
    _write_comparator(comparator)
    build_byte_amplification_ablation(
        comparator_path=comparator,
        output_dir=tmp_path / "out",
    )

    with (tmp_path / "out" / "byte_amplification_ablation.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    md = (tmp_path / "out" / "byte_amplification_ablation.md").read_text(
        encoding="utf-8"
    )
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))

    assert len(rows) == 20
    assert "overclaim_guard" in rows[0]
    assert "Source-Private Byte-Amplification Ablation" in md
    assert "QJL" in md or "qjl" in md
    assert manifest["pass_gate"] is True
