from __future__ import annotations

import json

from scripts.build_source_private_candidate_local_systems_boundary_trace import (
    build_candidate_local_systems_boundary_trace,
)


def _systems_row(condition: str, *, accuracy: float, pass_gate: bool | None, text: bool = False) -> dict[str, object]:
    return {
        "accuracy": accuracy,
        "batch_dma_bytes_per_request": 12.0,
        "batch_line_bytes_per_request": 11.0,
        "best_control_accuracy": 0.25,
        "condition": condition,
        "payload_bytes": 8.0,
        "record_bytes": 11.0,
        "resident_sparse_decode_p50_us": 5.0 if condition == "learned_synonym_dictionary_packet" else None,
        "single_request_cacheline_bytes": 64.0,
        "single_request_dma_bytes": 128.0,
        "source_kv_exposed": False,
        "source_text_exposed": text,
        "target_accuracy": 0.25,
        "pass_gate": pass_gate,
    }


def _memory_row(method: str, *, raw: float, text: bool = False, kv: bool = False) -> dict[str, object]:
    return {
        "accuracy": None,
        "batch64_packet_dma_bytes_per_request": None,
        "batch64_packet_line_bytes_per_request": None,
        "method": method,
        "raw_payload_bytes": raw,
        "single_request_cacheline_bytes": 64.0 if raw < 64 else raw,
        "single_request_dma_bytes": 128.0 if raw < 128 else raw,
        "source_kv_exposed": kv,
        "source_private": not text and not kv,
        "source_text_exposed": text,
        "target_accuracy": None,
    }


def _kv_proxy_row(condition: str, *, delta: float) -> dict[str, object]:
    fp16_bytes = delta * 1024.0
    return {
        "condition": condition,
        "prompt_token_delta_vs_packet": delta,
        "kv_payload_bytes_fp16_bf16": fp16_bytes,
        "kv_payload_bytes_turboquant_3p5bit_proxy": fp16_bytes * 3.5 / 16.0,
        "kv_payload_bytes_turboquant_2p5bit_proxy": fp16_bytes * 2.5 / 16.0,
    }


def _write(path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_candidate_local_systems_boundary_trace_labels_native_rows(tmp_path) -> None:
    systems_path = tmp_path / "systems.json"
    memory_path = tmp_path / "memory.json"
    rate_path = tmp_path / "rate.json"
    competitor_path = tmp_path / "competitor.json"
    kv_path = tmp_path / "kv.json"
    _write(
        systems_path,
        {
            "pass_gate": True,
            "rows": [
                _systems_row("learned_synonym_dictionary_packet", accuracy=0.625, pass_gate=True),
                _systems_row("structured_text_matched", accuracy=0.25, pass_gate=None, text=True),
                _systems_row("random_same_byte", accuracy=0.25, pass_gate=None),
            ],
        },
    )
    _write(
        memory_path,
        {
            "rows": [
                _memory_row("query-aware diagnostic text", raw=14.0, text=True),
                _memory_row("full hidden-log relay", raw=370.0, text=True),
                _memory_row("QJL-style 1-bit source KV byte floor", raw=21504.0, kv=True),
                _memory_row("KIVI/KVQuant-style 2-bit source KV byte floor", raw=43008.0, kv=True),
            ],
        },
    )
    _write(rate_path, {"related_work": [{"method": "C2C", "source": "https://arxiv.org/abs/2510.03215"}]})
    _write(competitor_path, {"headline": {"measured_table_rows": 14, "pending_required_rows": 4}})
    _write(
        kv_path,
        {
            "rows": [
                _kv_proxy_row("matched_packet", delta=0.0),
                _kv_proxy_row("query_aware_diag_span", delta=3.0),
                _kv_proxy_row("full_hidden_log", delta=80.0),
            ]
        },
    )

    payload = build_candidate_local_systems_boundary_trace(
        systems_waterfall=systems_path,
        memory_ledger=memory_path,
        systems_rate_frontier=rate_path,
        competitor_table=competitor_path,
        kv_cache_table=kv_path,
        output_dir=tmp_path / "out",
    )

    assert payload["pass_gate"] is True
    assert payload["headline"]["live_pass_rows"] == 1
    assert payload["headline"]["pending_native_systems_rows"] >= 5
    assert payload["headline"]["kv_native_proxy_byte_floor_rows"] == 6
    assert payload["headline"]["min_kv_native_proxy_payload_bytes"] == 480.0
    assert payload["headline"]["min_kv_native_proxy_record_ratio_vs_live"] > 40.0
    rows = {row["method"]: row for row in payload["rows"]}
    live = rows["candidate-local residual chart with row/payload normalization"]
    assert live["source_private"] is True
    assert live["source_text_exposed"] is False
    assert live["source_kv_exposed"] is False
    assert live["single_request_page_bytes"] == 4096.0
    c2c = rows["C2C cache-to-cache communication"]
    assert c2c["measurement_status"] == "pending_native_systems_row"
    assert c2c["source_kv_exposed"] is True
    assert c2c["nvidia_vllm_required"] is True
    vllm = rows["vLLM / PagedAttention serving substrate"]
    assert vllm["source_kv_exposed"] is False
    c2c_proxy = rows["C2C fp16 source-KV lower-bound proxy"]
    assert c2c_proxy["measurement_status"] == "mac_proxy_byte_floor_only"
    assert c2c_proxy["payload_bytes"] == 3072.0
    assert c2c_proxy["source_kv_exposed"] is True
    turbo_proxy = rows["TurboQuant 3.5-bit source-KV lower-bound proxy"]
    assert turbo_proxy["kv_bits_k"] == 3.5
    assert turbo_proxy["payload_bytes"] == 672.0
    assert (tmp_path / "out" / "candidate_local_systems_boundary_trace.json").exists()
    assert (tmp_path / "out" / "manifest.json").exists()
