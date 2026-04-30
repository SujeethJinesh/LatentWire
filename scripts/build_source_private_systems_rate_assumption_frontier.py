from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_SYSTEMS_CAVEAT = pathlib.Path("results/source_private_systems_caveat_frontier_20260429/systems_caveat_frontier.json")
DEFAULT_RATE_FRONTIER = pathlib.Path("results/source_private_rate_frontier_20260429/rate_frontier.json")
DEFAULT_KV_TABLE = pathlib.Path("results/source_private_kv_cache_baseline_table_20260429/kv_cache_baseline_table.json")
DEFAULT_SEMANTIC_MEDIUM = pathlib.Path(
    "results/source_private_semantic_anchor_heldout_medium_confirmation_20260430/summary.json"
)

CSV_COLUMNS = (
    "row_class",
    "method",
    "native_problem",
    "communicated_object",
    "artifact",
    "surface",
    "source_private",
    "private_payload_bytes",
    "rate_unit",
    "public_side_info",
    "receiver_side_info",
    "source_destroying_controls",
    "same_model_required",
    "cross_model_supported",
    "training_or_calibration",
    "source_text_exposed",
    "source_kv_exposed",
    "accuracy",
    "target_accuracy",
    "delta_vs_target",
    "best_control_accuracy",
    "delta_vs_best_control",
    "paired_ci95_low",
    "valid_rate",
    "prompt_token_delta",
    "kv_byte_floor",
    "latency_metric_scope",
    "ttft_ms",
    "p50_ttft_delta_vs_packet_ms",
    "tpot_ms",
    "throughput_qps",
    "peak_memory_gb",
    "byte_ratio_vs_packet",
    "native_claim",
    "claim_allowed",
    "paper_use",
    "overclaim_guard",
    "caveat",
)

RELATED_WORK = (
    {
        "method": "C2C / cache-to-cache communication",
        "source": "https://arxiv.org/abs/2510.03215",
        "role": "closest high-rate internal-state communication baseline",
        "positioning": "different access model: dense source/target KV or cache state, not public endpoint packet",
    },
    {
        "method": "KVComm / KV sharing",
        "source": "https://openreview.net/forum?id=F7rUng23nw",
        "role": "KV communication systems neighbor",
        "positioning": "selects KV tensors; useful as assumption contrast, not same source-private packet task",
    },
    {
        "method": "TurboQuant",
        "source": "https://arxiv.org/abs/2504.19874",
        "role": "low-bit online vector/KV quantization neighbor",
        "positioning": "byte-floor comparator only unless run on native KV task",
    },
    {
        "method": "QJL",
        "source": "https://arxiv.org/abs/2406.03482",
        "role": "sign-sketch / low-bit KV quantization neighbor",
        "positioning": "compares estimated KV payload bytes, not endpoint accuracy",
    },
    {
        "method": "KIVI / KVQuant",
        "source": "https://arxiv.org/abs/2402.02750; https://arxiv.org/abs/2401.18079",
        "role": "2-bit / ultra-long-context KV compression neighbor",
        "positioning": "same-model cache compression, not source-private communication",
    },
    {
        "method": "LLMLingua / Gist tokens",
        "source": "https://aclanthology.org/2023.emnlp-main.825/; https://openreview.net/forum?id=2DtxPCL3T5",
        "role": "prompt compression baseline family",
        "positioning": "text/context compression; include rate ladder rather than overclaiming packet superiority",
    },
)


def _resolve(path: pathlib.Path) -> pathlib.Path:
    return path if path.is_absolute() else ROOT / path


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: pathlib.Path) -> str:
    resolved = _resolve(path)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _row(
    *,
    row_class: str,
    method: str,
    artifact: pathlib.Path,
    surface: str,
    private_payload_bytes: float | None,
    public_side_info: str,
    source_text_exposed: bool,
    source_kv_exposed: bool,
    accuracy: float | None,
    target_accuracy: float | None,
    best_control_accuracy: float | None,
    p50_ttft_delta_vs_packet_ms: float | None,
    byte_ratio_vs_packet: float | None,
    native_claim: str,
    paper_use: str,
    caveat: str,
    native_problem: str | None = None,
    communicated_object: str | None = None,
    source_private: bool | None = None,
    receiver_side_info: str | None = None,
    source_destroying_controls: str | None = None,
    same_model_required: bool = False,
    cross_model_supported: bool = True,
    training_or_calibration: str = "none",
    rate_unit: str = "payload_bytes",
    prompt_token_delta: float | None = None,
    kv_byte_floor: float | None = None,
    paired_ci95_low: float | None = None,
    valid_rate: float | None = None,
    latency_metric_scope: str = "not_measured",
    ttft_ms: float | None = None,
    tpot_ms: float | None = None,
    throughput_qps: float | None = None,
    peak_memory_gb: float | None = None,
    claim_allowed: str | None = None,
    overclaim_guard: str | None = None,
) -> dict[str, Any]:
    delta = None if accuracy is None or target_accuracy is None else accuracy - target_accuracy
    delta_vs_best = None if accuracy is None or best_control_accuracy is None else accuracy - best_control_accuracy
    inferred_source_private = (
        (not source_text_exposed and not source_kv_exposed) if source_private is None else source_private
    )
    inferred_native_problem = native_problem or {
        "endpoint_packet": "source-private endpoint communication",
        "semantic_anchor_medium": "source-private held-out paraphrase communication",
        "endpoint_text_relay": "visible private text relay",
        "rate_frontier_text": "visible private text relay",
        "kv_byte_floor": "same-model KV/cache compression accounting",
        "contract_failure": "receiver-contract stress",
        "external_reference": "external native task reference",
    }.get(row_class, "source-private communication")
    inferred_object = communicated_object or {
        "endpoint_packet": "diagnostic atom packet",
        "semantic_anchor_medium": "semantic-anchor atom packet",
        "endpoint_text_relay": "private diagnostic text",
        "rate_frontier_text": "private diagnostic text",
        "kv_byte_floor": "quantized source KV/cache tensor",
        "contract_failure": "diagnostic atom packet",
        "external_reference": "external method native object",
    }.get(row_class, "payload")
    inferred_controls = source_destroying_controls or (
        "passed" if row_class in {"endpoint_packet", "semantic_anchor_medium"} else "not_applicable"
    )
    inferred_claim = claim_allowed or {
        "endpoint_packet": "headline_endpoint_proxy",
        "semantic_anchor_medium": "headline_method_medium",
        "endpoint_text_relay": "text_rate_comparator",
        "rate_frontier_text": "text_rate_comparator",
        "kv_byte_floor": "accounting_only",
        "contract_failure": "failure_boundary",
        "external_reference": "reference_only",
    }.get(row_class, "supporting")
    inferred_guard = overclaim_guard or {
        "endpoint_packet": "Do not claim production GPU serving throughput from this row.",
        "semantic_anchor_medium": "Do not claim prompt-contract-free or activation-level latent transfer from this row.",
        "endpoint_text_relay": "Do not call higher-byte text failure when it catches up.",
        "rate_frontier_text": "Do not call higher-byte text failure when it catches up.",
        "kv_byte_floor": "Do not claim superiority over native KV/cache compression.",
        "contract_failure": "Do not hide that the public receiver contract is required.",
        "external_reference": "Do not compare as exact-ID evidence unless rerun locally.",
    }.get(row_class, "Scope the claim to the measured artifact.")
    return {
        "row_class": row_class,
        "method": method,
        "native_problem": inferred_native_problem,
        "communicated_object": inferred_object,
        "artifact": _rel(artifact),
        "surface": surface,
        "source_private": inferred_source_private,
        "private_payload_bytes": private_payload_bytes,
        "rate_unit": rate_unit,
        "public_side_info": public_side_info,
        "receiver_side_info": receiver_side_info or public_side_info,
        "source_destroying_controls": inferred_controls,
        "same_model_required": same_model_required,
        "cross_model_supported": cross_model_supported,
        "training_or_calibration": training_or_calibration,
        "source_text_exposed": source_text_exposed,
        "source_kv_exposed": source_kv_exposed,
        "accuracy": accuracy,
        "target_accuracy": target_accuracy,
        "delta_vs_target": delta,
        "best_control_accuracy": best_control_accuracy,
        "delta_vs_best_control": delta_vs_best,
        "paired_ci95_low": paired_ci95_low,
        "valid_rate": valid_rate,
        "prompt_token_delta": prompt_token_delta,
        "kv_byte_floor": kv_byte_floor,
        "latency_metric_scope": latency_metric_scope,
        "ttft_ms": ttft_ms,
        "p50_ttft_delta_vs_packet_ms": p50_ttft_delta_vs_packet_ms,
        "tpot_ms": tpot_ms,
        "throughput_qps": throughput_qps,
        "peak_memory_gb": peak_memory_gb,
        "byte_ratio_vs_packet": byte_ratio_vs_packet,
        "native_claim": native_claim,
        "claim_allowed": inferred_claim,
        "paper_use": paper_use,
        "overclaim_guard": inferred_guard,
        "caveat": caveat,
    }


def _endpoint_rows(systems_path: pathlib.Path, systems: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in systems["rows"]:
        if not item["scope"].startswith("Mac CPU endpoint-proxy, not production"):
            continue
        rows.append(
            _row(
                row_class="endpoint_packet",
                method="LatentWire endpoint packet",
                artifact=systems_path,
                surface=item["surface"],
                private_payload_bytes=item["packet_payload_bytes"],
                public_side_info="candidate set + receiver prompt contract",
                source_text_exposed=False,
                source_kv_exposed=False,
                accuracy=item["packet_accuracy"],
                target_accuracy=item["target_accuracy"],
                best_control_accuracy=item["best_source_destroying_control_accuracy"],
                p50_ttft_delta_vs_packet_ms=0.0,
                byte_ratio_vs_packet=1.0,
                native_claim="Mac-local endpoint proxy source-private task evidence",
                paper_use="endpoint systems positive",
                caveat="Local CPU proxy; not production serving throughput.",
                training_or_calibration="receiver prompt contract",
                paired_ci95_low=item["min_packet_vs_target_ci95_low"],
                valid_rate=item["packet_valid_rate"],
                latency_metric_scope="mac_cpu_endpoint_proxy",
                ttft_ms=item["packet_p50_ttft_ms"],
            )
        )
        rows.append(
            _row(
                row_class="endpoint_text_relay",
                method="query-aware diagnostic text",
                artifact=systems_path,
                surface=item["surface"],
                private_payload_bytes=item["query_aware_payload_bytes"],
                public_side_info="candidate set + receiver prompt contract",
                source_text_exposed=True,
                source_kv_exposed=False,
                accuracy=None,
                target_accuracy=item["target_accuracy"],
                best_control_accuracy=None,
                p50_ttft_delta_vs_packet_ms=None,
                byte_ratio_vs_packet=item["packet_vs_query_payload_compression"],
                native_claim="higher-rate text relay",
                paper_use="prompt/text compression frontier",
                caveat="Text exposes private diagnostic content; rate is higher than packet.",
            )
        )
        rows.append(
            _row(
                row_class="endpoint_text_relay",
                method="full hidden-log relay",
                artifact=systems_path,
                surface=item["surface"],
                private_payload_bytes=item["full_log_payload_bytes"],
                public_side_info="candidate set + receiver prompt contract",
                source_text_exposed=True,
                source_kv_exposed=False,
                accuracy=None,
                target_accuracy=item["target_accuracy"],
                best_control_accuracy=None,
                p50_ttft_delta_vs_packet_ms=item["full_log_ttft_delta_vs_packet_ms"],
                byte_ratio_vs_packet=item["packet_vs_full_log_payload_compression"],
                native_claim="visible text oracle relay",
                paper_use="upper text relay / TTFT contrast",
                caveat="Not source-private; reports local CPU proxy TTFT delta.",
                latency_metric_scope="mac_cpu_endpoint_proxy",
                ttft_ms=item["full_log_p50_ttft_ms"],
            )
        )
    terse = next(row for row in systems["rows"] if row["prompt_contract"] == "terse")
    rows.append(
        _row(
            row_class="contract_failure",
            method="LatentWire packet with under-specified receiver",
            artifact=systems_path,
            surface=terse["surface"],
            private_payload_bytes=terse["packet_payload_bytes"],
            public_side_info="candidate set + weak receiver prompt",
            source_text_exposed=False,
            source_kv_exposed=False,
            accuracy=terse["packet_accuracy"],
            target_accuracy=terse["target_accuracy"],
            best_control_accuracy=terse["best_source_destroying_control_accuracy"],
            p50_ttft_delta_vs_packet_ms=0.0,
            byte_ratio_vs_packet=1.0,
            native_claim="negative prompt-contract ablation",
            paper_use="required public side-info boundary",
            caveat="Shows receiver contract is part of the method.",
            training_or_calibration="weak receiver prompt contract",
            valid_rate=terse["packet_valid_rate"],
            latency_metric_scope="mac_cpu_endpoint_proxy",
            ttft_ms=terse["packet_p50_ttft_ms"],
        )
    )
    return rows


def _semantic_medium_rows(semantic_path: pathlib.Path, semantic: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for budget in (4, 8):
        budget_summary = semantic["headline"]["by_budget"][str(budget)]
        rows.append(
            _row(
                row_class="semantic_anchor_medium",
                method="semantic-anchor source-private packet",
                artifact=semantic_path,
                surface=f"heldout paraphrase n{semantic['eval_examples']} x {len(semantic['seeds'])} seeds",
                private_payload_bytes=float(budget),
                public_side_info="candidate set + public semantic-anchor receiver",
                source_text_exposed=False,
                source_kv_exposed=False,
                accuracy=None,
                target_accuracy=0.25,
                best_control_accuracy=budget_summary["max_best_control_accuracy"],
                p50_ttft_delta_vs_packet_ms=None,
                byte_ratio_vs_packet=1.0,
                native_claim="medium seed-stable held-out paraphrase source-private communication",
                paper_use="method-depth systems row",
                caveat=(
                    f"All {budget_summary['pass_rows']}/{budget_summary['total_rows']} rows pass; "
                    "accuracy varies by direction, so aggregate row reports margins rather than one accuracy."
                ),
                training_or_calibration="public semantic-anchor lexicon + ridge dictionary",
                paired_ci95_low=semantic["headline"]["min_passing_ci95_low_vs_target"],
            )
        )
    return rows


def _rate_rows(rate_path: pathlib.Path, rate: dict[str, Any]) -> list[dict[str, Any]]:
    h = rate["headline"]
    return [
        _row(
            row_class="rate_frontier_text",
            method="same-byte structured text",
            artifact=rate_path,
            surface="core+holdout rate frontier",
            private_payload_bytes=h["packet_oracle_bytes_max"],
            public_side_info="candidate set + text decoder",
            source_text_exposed=True,
            source_kv_exposed=False,
            accuracy=h["matched_byte_text_at_packet_accuracy_max"],
            target_accuracy=0.25,
            best_control_accuracy=None,
            p50_ttft_delta_vs_packet_ms=None,
            byte_ratio_vs_packet=1.0,
            native_claim="same-byte visible text control",
            paper_use="text baseline",
            caveat="Fails at packet byte rate on frozen rate-frontier surfaces.",
        ),
        _row(
            row_class="rate_frontier_text",
            method="query-aware structured text oracle",
            artifact=rate_path,
            surface="core+holdout rate frontier",
            private_payload_bytes=h["query_aware_oracle_bytes_min"],
            public_side_info="candidate set + text decoder",
            source_text_exposed=True,
            source_kv_exposed=False,
            accuracy=1.0,
            target_accuracy=0.25,
            best_control_accuracy=None,
            p50_ttft_delta_vs_packet_ms=None,
            byte_ratio_vs_packet=h["packet_vs_query_aware_oracle_compression_min"],
            native_claim="higher-rate visible text oracle",
            paper_use="honest text catches-up row",
            caveat="Shows structured text can catch up when given more bytes and private text exposure.",
        ),
        _row(
            row_class="rate_frontier_text",
            method="JSON/free-text oracle relay",
            artifact=rate_path,
            surface="core+holdout rate frontier",
            private_payload_bytes=h["json_oracle_bytes_min"],
            public_side_info="candidate set + text decoder",
            source_text_exposed=True,
            source_kv_exposed=False,
            accuracy=1.0,
            target_accuracy=0.25,
            best_control_accuracy=None,
            p50_ttft_delta_vs_packet_ms=None,
            byte_ratio_vs_packet=h["packet_vs_json_oracle_compression_min"],
            native_claim="higher-rate visible text oracle",
            paper_use="structured text rate ladder",
            caveat="Uses the smaller JSON oracle byte count; free text needs at least 17 bytes.",
        ),
    ]


def _kv_rows(kv_path: pathlib.Path, kv: dict[str, Any], packet_bytes: float) -> list[dict[str, Any]]:
    h = kv["headline"]
    return [
        _row(
            row_class="kv_byte_floor",
            method="QJL-style 1-bit source KV byte floor",
            artifact=kv_path,
            surface="endpoint source context byte accounting",
            private_payload_bytes=packet_bytes * h["min_non_packet_qjl_1bit_bytes_vs_packet"],
            public_side_info="model internals + KV transport",
            source_text_exposed=False,
            source_kv_exposed=True,
            accuracy=None,
            target_accuracy=None,
            best_control_accuracy=None,
            p50_ttft_delta_vs_packet_ms=None,
            byte_ratio_vs_packet=h["min_non_packet_qjl_1bit_bytes_vs_packet"],
            native_claim="KV/cache compression byte-floor neighbor",
            paper_use="assumption contrast, not accuracy baseline",
            caveat="No claim of beating QJL on native KV compression; this is byte-floor accounting only.",
        ),
        _row(
            row_class="kv_byte_floor",
            method="KIVI/KVQuant-style 2-bit source KV byte floor",
            artifact=kv_path,
            surface="endpoint source context byte accounting",
            private_payload_bytes=packet_bytes * h["min_non_packet_kivi_2bit_bytes_vs_packet"],
            public_side_info="model internals + KV transport",
            source_text_exposed=False,
            source_kv_exposed=True,
            accuracy=None,
            target_accuracy=None,
            best_control_accuracy=None,
            p50_ttft_delta_vs_packet_ms=None,
            byte_ratio_vs_packet=h["min_non_packet_kivi_2bit_bytes_vs_packet"],
            native_claim="KV/cache compression byte-floor neighbor",
            paper_use="assumption contrast, not accuracy baseline",
            caveat="No claim of beating KIVI/KVQuant on native KV compression; this is byte-floor accounting only.",
        ),
    ]


def _external_reference_rows(artifact: pathlib.Path) -> list[dict[str, Any]]:
    specs = (
        (
            "C2C cache-to-cache communication",
            "cross-model internal-state communication",
            "projected/fused KV cache",
            "learned cache translator/fuser",
            False,
            True,
            True,
            "reference_only",
            "Do not compare bytes or accuracy unless rerun on exact IDs and cache payloads are measured.",
        ),
        (
            "KVComm / KVCOMM selective KV communication",
            "cross-context or multi-agent KV-cache communication",
            "selected KV tensors / offsets",
            "KV selector/router",
            True,
            True,
            True,
            "reference_only",
            "Different native problem: KV reuse/sharing, not endpoint packet communication.",
        ),
        (
            "TurboQuant",
            "same-model online vector/KV quantization",
            "low-bit KV/vector state",
            "quantization calibration",
            True,
            False,
            True,
            "accounting_only",
            "Do not claim superiority on native quantization kernels without GPU/server runs.",
        ),
        (
            "QJL",
            "same-model KV/sign-sketch compression",
            "quantized Johnson-Lindenstrauss sketch",
            "random projection/sketch",
            True,
            False,
            True,
            "accounting_only",
            "Current direct QJL comparator is on a local feature surface; native KV result is external/accounting only.",
        ),
        (
            "LLMLingua / LLMLingua-2",
            "prompt compression",
            "compressed visible text prompt",
            "learned prompt compressor",
            False,
            True,
            False,
            "reference_only",
            "Fair comparison requires running on the exact private diagnostic text ladder.",
        ),
        (
            "Gist tokens",
            "learned prompt/context compression",
            "learned soft/gist prompt tokens",
            "task/model training",
            False,
            True,
            False,
            "reference_only",
            "Soft-token compression is not a measured source-private packet baseline here.",
        ),
    )
    rows = []
    for method, native_problem, communicated_object, training, same_model, cross_model, kv_exposed, claim, guard in specs:
        rows.append(
            _row(
                row_class="external_reference",
                method=method,
                artifact=artifact,
                surface="external primary-source reference",
                private_payload_bytes=None,
                public_side_info="native method assumptions",
                source_text_exposed=False,
                source_kv_exposed=kv_exposed,
                accuracy=None,
                target_accuracy=None,
                best_control_accuracy=None,
                p50_ttft_delta_vs_packet_ms=None,
                byte_ratio_vs_packet=None,
                native_claim=native_problem,
                paper_use=claim,
                caveat=guard,
                native_problem=native_problem,
                communicated_object=communicated_object,
                source_private=False,
                receiver_side_info="native method assumptions",
                source_destroying_controls="not_applicable",
                same_model_required=same_model,
                cross_model_supported=cross_model,
                training_or_calibration=training,
                claim_allowed=claim,
                overclaim_guard=guard,
            )
        )
    return rows


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(row.get(key)) for key in CSV_COLUMNS})


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Source-Private Systems Rate And Assumption Frontier",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- claim scope: {payload['claim_scope']}",
        "",
        "## Headline",
        "",
    ]
    for key, value in payload["headline"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Frontier Rows",
            "",
            "| Group | Method | Surface | Bytes | Text exposed | KV exposed | Accuracy | Target | Best control | Byte ratio | Native claim |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["rows"]:
        accuracy = "" if row["accuracy"] is None else f"{row['accuracy']:.3f}"
        target = "" if row["target_accuracy"] is None else f"{row['target_accuracy']:.3f}"
        control = "" if row["best_control_accuracy"] is None else f"{row['best_control_accuracy']:.3f}"
        ratio = "" if row["byte_ratio_vs_packet"] is None else f"{row['byte_ratio_vs_packet']:.1f}"
        bytes_value = "" if row["private_payload_bytes"] is None else f"{row['private_payload_bytes']:.1f}"
        lines.append(
            f"| {row['row_class']} | {row['method']} | {row['surface']} | {bytes_value} | "
            f"`{row['source_text_exposed']}` | `{row['source_kv_exposed']}` | {accuracy} | {target} | "
            f"{control} | {ratio} | {row['native_claim']} |"
        )
    lines.extend(
        [
            "",
            "## Related Work Positioning",
            "",
            "| Method | Source | Role | Positioning |",
            "|---|---|---|---|",
        ]
    )
    for row in payload["related_work"]:
        lines.append(f"| {row['method']} | {row['source']} | {row['role']} | {row['positioning']} |")
    lines.extend(["", "## Non-Claims", ""])
    for item in payload["non_claims"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Interpretation", "", payload["interpretation"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_systems_rate_assumption_frontier(
    *,
    systems_caveat: pathlib.Path,
    rate_frontier: pathlib.Path,
    kv_table: pathlib.Path,
    semantic_medium: pathlib.Path,
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    systems = _read_json(systems_caveat)
    rate = _read_json(rate_frontier)
    kv = _read_json(kv_table)
    semantic = _read_json(semantic_medium)
    rows = (
        _endpoint_rows(systems_caveat, systems)
        + _semantic_medium_rows(semantic_medium, semantic)
        + _rate_rows(rate_frontier, rate)
        + _kv_rows(kv_table, kv, systems["headline"]["packet_payload_bytes"])
        + _external_reference_rows(kv_table)
    )
    endpoint_packets = [row for row in rows if row["row_class"] == "endpoint_packet"]
    semantic_packets = [row for row in rows if row["row_class"] == "semantic_anchor_medium"]
    text_rows = [row for row in rows if row["row_class"] in {"endpoint_text_relay", "rate_frontier_text"}]
    kv_rows = [row for row in rows if row["row_class"] == "kv_byte_floor"]
    headline = {
        "endpoint_packet_rows": len(endpoint_packets),
        "endpoint_packet_rows_passing": systems["headline"]["passing_endpoint_rows"],
        "semantic_medium_pass_rows": semantic["headline"]["pass_rows"],
        "semantic_medium_total_rows": semantic["headline"]["total_rows"],
        "packet_payload_bytes_min": min(row["private_payload_bytes"] for row in endpoint_packets),
        "semantic_packet_budget_bytes": semantic["budgets"],
        "min_endpoint_packet_delta_vs_target": min(row["delta_vs_target"] for row in endpoint_packets),
        "min_semantic_packet_delta_vs_target": semantic["headline"]["min_passing_learned_minus_target"],
        "same_byte_text_accuracy_max": rate["headline"]["matched_byte_text_at_packet_accuracy_max"],
        "query_aware_text_oracle_bytes_min": rate["headline"]["query_aware_oracle_bytes_min"],
        "query_aware_text_bytes_vs_packet": rate["headline"]["packet_vs_query_aware_oracle_compression_min"],
        "full_log_bytes_vs_packet_min": rate["headline"]["packet_vs_full_log_compression_min"],
        "min_full_log_ttft_delta_vs_packet_ms": systems["headline"]["min_full_log_ttft_delta_vs_packet_ms"],
        "min_kv_byte_floor_vs_packet": min(row["byte_ratio_vs_packet"] for row in kv_rows),
        "contract_failure_packet_accuracy": next(row for row in rows if row["row_class"] == "contract_failure")[
            "accuracy"
        ],
        "external_reference_rows": sum(1 for row in rows if row["row_class"] == "external_reference"),
    }
    pass_gate = (
        systems["pass_gate"]
        and rate["pass_gate"]
        and semantic["headline"]["all_seed_pass"]
        and headline["semantic_medium_pass_rows"] == headline["semantic_medium_total_rows"]
        and headline["min_semantic_packet_delta_vs_target"] >= 0.50
        and headline["same_byte_text_accuracy_max"] <= 0.25
        and headline["query_aware_text_bytes_vs_packet"] >= 7.0
        and headline["min_kv_byte_floor_vs_packet"] >= 1000.0
        and headline["contract_failure_packet_accuracy"] == 0.25
    )
    payload = {
        "gate": "source_private_systems_rate_assumption_frontier",
        "pass_gate": pass_gate,
        "claim_scope": (
            "Source-private extreme-rate task communication with explicit public side information; "
            "KV/cache rows are byte-floor assumption contrasts, not native accuracy baselines."
        ),
        "sources": {
            "systems_caveat": _rel(systems_caveat),
            "rate_frontier": _rel(rate_frontier),
            "kv_table": _rel(kv_table),
            "semantic_medium": _rel(semantic_medium),
        },
        "headline": headline,
        "rows": rows,
        "related_work": RELATED_WORK,
        "non_claims": [
            "No claim of beating TurboQuant, QJL, KIVI, KVQuant, C2C, or KVComm on native KV/cache tasks.",
            "No production GPU serving throughput claim from Mac-local CPU proxy rows.",
            "No prompt-contract-free claim; the terse receiver row is a recorded failure.",
            "No broad latent-transfer claim; the semantic-anchor receiver uses public candidate side information.",
        ],
        "interpretation": (
            "The systems win is an assumption-aware rate frontier: LatentWire packets occupy a 2-8 byte "
            "source-private regime where same-byte text fails, higher-byte text relays catch up by exposing "
            "private text, and KV/cache methods require internal-state transport and much larger byte floors."
        ),
    }

    json_path = output_dir / "systems_rate_assumption_frontier.json"
    csv_path = output_dir / "systems_rate_assumption_frontier.csv"
    md_path = output_dir / "systems_rate_assumption_frontier.md"
    manifest_path = output_dir / "manifest.json"
    manifest_md_path = output_dir / "manifest.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(csv_path, rows)
    _write_markdown(md_path, payload)
    manifest = {
        "gate": payload["gate"],
        "pass_gate": pass_gate,
        "artifacts": [json_path.name, csv_path.name, md_path.name, manifest_path.name, manifest_md_path.name],
        "artifact_sha256": {
            json_path.name: _sha256_file(json_path),
            csv_path.name: _sha256_file(csv_path),
            md_path.name: _sha256_file(md_path),
        },
        "headline": headline,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest_md_path.write_text(
        "\n".join(
            [
                "# Systems Rate And Assumption Frontier Manifest",
                "",
                f"- pass gate: `{pass_gate}`",
                f"- semantic pass rows: `{headline['semantic_medium_pass_rows']}/{headline['semantic_medium_total_rows']}`",
                f"- min KV byte floor / packet: `{headline['min_kv_byte_floor_vs_packet']:.1f}x`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems-caveat", type=pathlib.Path, default=DEFAULT_SYSTEMS_CAVEAT)
    parser.add_argument("--rate-frontier", type=pathlib.Path, default=DEFAULT_RATE_FRONTIER)
    parser.add_argument("--kv-table", type=pathlib.Path, default=DEFAULT_KV_TABLE)
    parser.add_argument("--semantic-medium", type=pathlib.Path, default=DEFAULT_SEMANTIC_MEDIUM)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/source_private_systems_rate_assumption_frontier_20260430"),
    )
    args = parser.parse_args()
    output_dir = _resolve(args.output_dir)
    payload = build_systems_rate_assumption_frontier(
        systems_caveat=args.systems_caveat,
        rate_frontier=args.rate_frontier,
        kv_table=args.kv_table,
        semantic_medium=args.semantic_medium,
        output_dir=output_dir,
    )
    print(json.dumps({"output_dir": str(output_dir), "pass_gate": payload["pass_gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
