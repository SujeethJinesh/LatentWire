from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_CONDITIONAL_SUMMARY = (
    ROOT
    / "results/source_private_conditional_pq_innovation_gate_20260430/summary/conditional_pq_innovation_summary.json"
)
DEFAULT_SCHEMA_GRID = (
    ROOT
    / "results/source_private_conditional_pq_basis_schema_grid_20260430/summary/conditional_pq_basis_schema_grid_summary.json"
)
DEFAULT_SYSTEMS_WATERFALL = (
    ROOT
    / "results/source_private_conditional_pq_packet_isa_waterfall_20260430/conditional_pq_packet_isa_waterfall.json"
)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _method_rows(waterfall: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in waterfall["rows"] if row.get("row_type") == "method"]


def _systems_byte_summary(waterfall: dict[str, Any]) -> dict[str, Any]:
    method_rows = _method_rows(waterfall)
    min_record_bytes = min(row["record_bytes"] for row in method_rows)
    max_record_bytes = max(row["record_bytes"] for row in method_rows)
    kv_rows = [row for row in waterfall["rows"] if row.get("source_kv_exposed")]
    min_kv_record_bytes = None if not kv_rows else min(row["record_bytes"] for row in kv_rows)
    return {
        "method_record_bytes_range": [min_record_bytes, max_record_bytes],
        "method_payload_bytes": sorted({row["payload_bytes"] for row in method_rows}),
        "min_kv_floor_record_bytes": min_kv_record_bytes,
        "private_state_exposure_separated": bool(waterfall["checks"].get("private_state_exposure_separated")),
        "receiver_exact": bool(waterfall["checks"].get("receiver_exact")),
        "native_gpu_claim_allowed": False,
    }


def build_status(
    *,
    conditional_summary_path: pathlib.Path,
    schema_grid_path: pathlib.Path,
    systems_waterfall_path: pathlib.Path,
) -> dict[str, Any]:
    conditional = _read_json(conditional_summary_path)
    grid = _read_json(schema_grid_path)
    waterfall = _read_json(systems_waterfall_path)

    conditional_summary = conditional["summary"]
    grid_summary = grid["summary"]
    systems_summary = _systems_byte_summary(waterfall)
    same_family_pass = (
        conditional_summary["decisive_disjoint_n500_rows"] > 0
        and conditional_summary["decisive_disjoint_n500_pass_rows"]
        == conditional_summary["decisive_disjoint_n500_rows"]
    )
    budget2_pass = (
        conditional_summary["budget2_decisive_rows"] > 0
        and conditional_summary["budget2_decisive_pass_rows"] == conditional_summary["budget2_decisive_rows"]
    )
    cross_family_blocked = grid_summary["pass_rows"] == 0 and conditional_summary["cross_family_pass_rows"] == 0
    systems_ready = bool(waterfall["pass_gate"]) and systems_summary["private_state_exposure_separated"]

    colm_v2_status = "scoped_positive_method_ready_for_writeup" if same_family_pass and budget2_pass and systems_ready else "blocked"
    iclr_status = "blocked_by_cross_family_or_broader_benchmark_positive_gate"
    if not same_family_pass:
        iclr_status = "blocked_by_positive_method_gate"

    return {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "gate": "source_private_conditional_pq_iclr_colm_v2_status",
        "input_artifacts": {
            "conditional_summary": str(conditional_summary_path),
            "schema_grid": str(schema_grid_path),
            "systems_waterfall": str(systems_waterfall_path),
        },
        "input_sha256": {
            "conditional_summary": _sha256_file(conditional_summary_path),
            "schema_grid": _sha256_file(schema_grid_path),
            "systems_waterfall": _sha256_file(systems_waterfall_path),
        },
        "readiness": {
            "colm_v2": colm_v2_status,
            "iclr": iclr_status,
            "same_family_disjoint_n500_positive": same_family_pass,
            "budget2_positive": budget2_pass,
            "cross_family_blocked": cross_family_blocked,
            "systems_accounting_ready": systems_ready,
        },
        "story": (
            "Conditional PQ innovation is the current strongest LatentWire_v2 branch: a source-private "
            "conditional innovation packet is product-quantized into 2-4 payload bytes and decoded against "
            "target/public candidate side information. The defensible claim is shared-schema source-private "
            "communication with utility-per-byte accounting, not broad unseen-family latent transfer."
        ),
        "evidence": {
            "decisive_disjoint_n500_pass_rows": conditional_summary["decisive_disjoint_n500_pass_rows"],
            "decisive_disjoint_n500_rows": conditional_summary["decisive_disjoint_n500_rows"],
            "less_diagnostic_pass_rows": conditional_summary["less_diagnostic_decisive_pass_rows"],
            "less_diagnostic_rows": conditional_summary["less_diagnostic_decisive_rows"],
            "budget2_pass_rows": conditional_summary["budget2_decisive_pass_rows"],
            "budget2_rows": conditional_summary["budget2_decisive_rows"],
            "min_decisive_source_accuracy": conditional_summary["min_decisive_source_accuracy"],
            "max_decisive_best_control_accuracy": conditional_summary["max_decisive_best_control_accuracy"],
            "min_decisive_ci95_low_vs_best_control": conditional_summary["min_decisive_ci95_low_vs_best_control"],
            "cross_family_original_pass_rows": conditional_summary["cross_family_pass_rows"],
            "cross_family_original_rows": conditional_summary["cross_family_rows"],
            "cross_family_schema_grid_pass_rows": grid_summary["pass_rows"],
            "cross_family_schema_grid_rows": grid_summary["rows"],
            "cross_family_max_source_minus_control": grid_summary["max_source_minus_best_control"],
            "cross_family_max_ci95_low_vs_control": grid_summary["max_ci95_low_vs_best_control"],
            "systems": systems_summary,
        },
        "current_contributions": [
            {
                "name": "source_private_conditional_innovation_packet",
                "status": "alive_positive_shared_schema",
                "needs_work": "show broader benchmark or held-out-family transfer",
            },
            {
                "name": "strict_destructive_controls",
                "status": "strong_for_current_synthetic_surface",
                "needs_work": "add paper-facing source-index/rank/score and same-byte-text comparators where meaningful",
            },
            {
                "name": "utility_per_byte_systems_accounting",
                "status": "ready_for_mac_local_packet_boundary_claim",
                "needs_work": "native GPU/C2C/KV measurements before throughput, HBM, PCIe, NVLink, energy, or serving claims",
            },
        ],
        "submission_gap": (
            "COLM_v2 needs the scoped conditional-PQ table, systems waterfall, and limitations integrated. "
            "ICLR needs one more positive gate: either held-out-family/public-conditioned residual codebooks, "
            "a less synthetic benchmark, or a learned receiver that keeps the same byte and exposure advantages."
        ),
        "next_exact_gate": {
            "name": "public_conditioned_conditional_pq_resurrection_gate",
            "decision_surface": "n256 bidirectional held-out family first, then n500/remap repeat if positive",
            "method": (
                "replace static public bases with target-public conditioned residual/codebook decoding while "
                "keeping the same source-private conditional innovation packet interface"
            ),
            "pass_bar": "source minus best destructive/shortcut control >= +0.10 with positive paired CI95 low",
            "required_controls": [
                "target_only",
                "answer_masked_source",
                "constrained_wrong_row_source",
                "same_source_choice_wrong_row",
                "candidate_roll_or_deranged_public_basis",
                "permuted_codes",
                "random_same_byte",
                "opaque_slot_or_deranged_basis",
                "source_index_rank_score_comparators_when_not_answer_oracles",
                "same_byte_visible_text",
            ],
        },
        "claim_boundaries": [
            "Do not claim unseen-family transfer: both original cross-family rows and the 28-row basis/schema grid fail.",
            "Do not claim product quantization, rotations, or side-information coding as standalone novelty.",
            "Do not claim GPU throughput, HBM savings, latency, energy, PCIe, or NVLink wins from Mac-local packet accounting.",
            "Do frame C2C/KV methods as dense or cache-sharing baselines with different exposure and byte regimes.",
        ],
        "references": [
            "references/543_conditional_pq_innovation_refs_20260430.md",
            "references/544_conditional_pq_semantic_schema_systems_refs_20260430.md",
            "references/727_srp_competitor_basis_quant_benchmark_lateral_refresh_20260504.md",
            "references/728_event_triggered_defer_syndrome_packet_refs_20260504.md",
            "references/739_conditional_pq_iclr_colm_v2_status_refs_20260504.md",
        ],
    }


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    readiness = payload["readiness"]
    evidence = payload["evidence"]
    systems = evidence["systems"]
    lines = [
        "# Conditional PQ ICLR / COLM_v2 Status",
        "",
        f"- COLM_v2 readiness: `{readiness['colm_v2']}`",
        f"- ICLR readiness: `{readiness['iclr']}`",
        f"- same-family n500 pass rows: `{evidence['decisive_disjoint_n500_pass_rows']}/{evidence['decisive_disjoint_n500_rows']}`",
        f"- budget-2 pass rows: `{evidence['budget2_pass_rows']}/{evidence['budget2_rows']}`",
        f"- less-diagnostic pass rows: `{evidence['less_diagnostic_pass_rows']}/{evidence['less_diagnostic_rows']}`",
        f"- original cross-family pass rows: `{evidence['cross_family_original_pass_rows']}/{evidence['cross_family_original_rows']}`",
        f"- schema-grid cross-family pass rows: `{evidence['cross_family_schema_grid_pass_rows']}/{evidence['cross_family_schema_grid_rows']}`",
        "",
        "## Story",
        "",
        payload["story"],
        "",
        "## Evidence",
        "",
        f"- min decisive source accuracy: `{evidence['min_decisive_source_accuracy']}`",
        f"- max decisive best-control accuracy: `{evidence['max_decisive_best_control_accuracy']}`",
        f"- min decisive CI95 low vs best control: `{evidence['min_decisive_ci95_low_vs_best_control']}`",
        f"- cross-family max source-minus-control: `{evidence['cross_family_max_source_minus_control']}`",
        f"- cross-family max CI95 low vs control: `{evidence['cross_family_max_ci95_low_vs_control']}`",
        f"- method record bytes range: `{systems['method_record_bytes_range']}`",
        f"- method payload bytes: `{systems['method_payload_bytes']}`",
        f"- min KV floor record bytes: `{systems['min_kv_floor_record_bytes']}`",
        f"- native GPU claim allowed: `{systems['native_gpu_claim_allowed']}`",
        "",
        "## Contributions",
        "",
    ]
    for contribution in payload["current_contributions"]:
        lines.append(
            f"- `{contribution['name']}`: {contribution['status']}; needs {contribution['needs_work']}."
        )
    lines.extend(
        [
            "",
            "## Submission Gap",
            "",
            payload["submission_gap"],
            "",
            "## Next Exact Gate",
            "",
            f"- name: `{payload['next_exact_gate']['name']}`",
            f"- decision surface: {payload['next_exact_gate']['decision_surface']}",
            f"- method: {payload['next_exact_gate']['method']}",
            f"- pass bar: {payload['next_exact_gate']['pass_bar']}",
            "- required controls: "
            + ", ".join(f"`{control}`" for control in payload["next_exact_gate"]["required_controls"]),
            "",
            "## Claim Boundaries",
            "",
        ]
    )
    for boundary in payload["claim_boundaries"]:
        lines.append(f"- {boundary}")
    lines.extend(["", "## References", ""])
    for reference in payload["references"]:
        lines.append(f"- `{reference}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditional-summary", type=pathlib.Path, default=DEFAULT_CONDITIONAL_SUMMARY)
    parser.add_argument("--schema-grid", type=pathlib.Path, default=DEFAULT_SCHEMA_GRID)
    parser.add_argument("--systems-waterfall", type=pathlib.Path, default=DEFAULT_SYSTEMS_WATERFALL)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=ROOT / "results/source_private_conditional_pq_iclr_colm_v2_status_20260504",
    )
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_status(
        conditional_summary_path=args.conditional_summary if args.conditional_summary.is_absolute() else ROOT / args.conditional_summary,
        schema_grid_path=args.schema_grid if args.schema_grid.is_absolute() else ROOT / args.schema_grid,
        systems_waterfall_path=args.systems_waterfall if args.systems_waterfall.is_absolute() else ROOT / args.systems_waterfall,
    )
    json_path = output_dir / "conditional_pq_iclr_colm_v2_status.json"
    md_path = output_dir / "conditional_pq_iclr_colm_v2_status.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(md_path, payload)
    manifest = {
        "artifacts": [
            "conditional_pq_iclr_colm_v2_status.json",
            "conditional_pq_iclr_colm_v2_status.md",
            "manifest.json",
            "manifest.md",
        ],
        "artifact_sha256": {
            json_path.name: _sha256_file(json_path),
            md_path.name: _sha256_file(md_path),
        },
        "colm_v2_readiness": payload["readiness"]["colm_v2"],
        "iclr_readiness": payload["readiness"]["iclr"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Conditional PQ ICLR / COLM_v2 Status Manifest",
                "",
                f"- COLM_v2 readiness: `{payload['readiness']['colm_v2']}`",
                f"- ICLR readiness: `{payload['readiness']['iclr']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "colm_v2_readiness": payload["readiness"]["colm_v2"],
                "iclr_readiness": payload["readiness"]["iclr"],
                "output_dir": str(output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
