from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_INPUT_PATHS = {
    "colm_v2_review_packet": ROOT
    / "results/latentwire_colm_v2_review_packet_20260504/review_packet.json",
    "systems_boundary": ROOT
    / "results/source_private_systems_boundary_figure_table_split_20260504/systems_boundary_figure_data.json",
    "colm_v3_readiness": ROOT / "paper/latentwire_colm_v3_readiness_20260505.md",
    "experimental_status": ROOT / "experimental/status_20260505.md",
    "reviewer_feedback": ROOT / "paper/reviewer_feedback.md",
    "experiment_ledger": ROOT / "paper/experiment_ledger_20260421.md",
    "colm_v3_tex": ROOT / "colm_final/paper/latentwire_colm2026.tex",
    "colm_v3_reviewer_panel": ROOT
    / "colm_final/audits/colm_v3_10_reviewer_panel_20260505.md",
}

DEFAULT_OUTPUT_DIR = ROOT / "results/latentwire_colm_v3_review_packet_20260505"
DEFAULT_PAPER_PATH = ROOT / "paper/latentwire_colm_v3_review_packet_20260505.md"

MAIN_CLAIM = (
    "LatentWire provides a practical protocol and evaluation framework for "
    "source-private candidate-transfer packets, with controlled evidence of "
    "narrow fixed-byte packet utility, explicit utility-per-byte accounting, "
    "and destructive controls that expose shortcut claims."
)


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(path: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _input_manifest(input_paths: dict[str, pathlib.Path]) -> list[dict[str, str]]:
    return [
        {
            "key": key,
            "path": _repo_path(path),
            "sha256": _sha256_file(path),
        }
        for key, path in sorted(input_paths.items())
    ]


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _safe_md(value: Any) -> str:
    return _fmt(value).replace("|", "\\|").replace("\n", " ")


def _systems_classification(row: dict[str, Any]) -> str:
    if row.get("native_measured"):
        if row.get("native_claim_allowed") or row.get("nvidia_vllm_required"):
            return "native_measured"
        return "measured_local_control_or_accounting_row"

    measurement_status = str(row.get("measurement_status", ""))
    row_group = str(row.get("row_group", "")).lower()
    method = str(row.get("method", "")).lower()
    row_id = str(row.get("row_id", "")).lower()

    if row_group == "latentwire packet":
        if measurement_status == "cached_source_communication_object":
            return "measured_packet_object_bytes"
        if "source_scoring" in measurement_status or "end_to_end" in row_id:
            return "local_partial_measurement_or_missing_phase_trace"
        return "packet_accounting_without_native_gpu_claim"

    if (
        "floor" in measurement_status
        or "floor" in row_id
        or "floor" in str(row.get("claim_allowed", "")).lower()
        or "kv" in method
        or "cache" in method
    ):
        return "analytical_or_literature_byte_floor"

    if row.get("nvidia_vllm_required"):
        return "future_native_nvidia_run_needed"

    return "accounting_boundary_or_related_work"


def _systems_measured_vs_estimated(systems: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in systems.get("rows", []):
        rows.append(
            {
                "method": row.get("method"),
                "communicated_object": row.get("communicated_object"),
                "raw_bytes": row.get("raw_bytes"),
                "framed_bytes": row.get("framed_bytes"),
                "cacheline_bytes": row.get("cacheline_bytes"),
                "batch64_bytes": row.get("batch64_bytes"),
                "source_private": row.get("source_private"),
                "source_kv_exposed": row.get("source_kv_exposed"),
                "native_measured": row.get("native_measured"),
                "measurement_status": row.get("measurement_status"),
                "measured_vs_estimated": _systems_classification(row),
                "claim_allowed": row.get("claim_allowed"),
                "overclaim_guard": row.get("overclaim_guard"),
            }
        )
    return rows


def _claim_audit() -> list[dict[str, str]]:
    return [
        {
            "claim": "LatentWire defines a source-private candidate-transfer packet protocol and strict evaluation framework.",
            "support_level": "supported",
            "evidence_artifact": "COLM_v2 review packet plus COLM_v3 review packet",
            "controls_passed": "source-private interface, wrong-row/source-choice controls where available",
            "required_wording": "safe as a protocol/evaluation contribution",
        },
        {
            "claim": "Low-byte packets show narrow same-family utility on ARC-style rows.",
            "support_level": "supported_but_narrow",
            "evidence_artifact": "main_results.csv; strict_controls.csv; systems_measured_vs_estimated.csv",
            "controls_passed": "target-only and same-byte/text controls on the reported rows; source-index remains a hard boundary",
            "required_wording": "narrow source-private candidate-transfer utility, not broad latent communication",
        },
        {
            "claim": "The current packet beats source-index communication or selected-candidate codes.",
            "support_level": "not_supported",
            "evidence_artifact": "main_results.csv; source-index audit",
            "controls_passed": "packet-source lower bounds remain negative or zero",
            "required_wording": "do not claim; source-index is the main boundary",
        },
        {
            "claim": "Many apparent wins collapse into source-choice, source-rank, or target-cache artifacts.",
            "support_level": "supported",
            "evidence_artifact": "negative_results.csv; source-choice controls; reviewer feedback",
            "controls_passed": "same-source-choice wrong-row, source-index/rank/score, and destructive controls where available",
            "required_wording": "use as a reviewer-strengthening result, not as the headline alone",
        },
        {
            "claim": "LatentWire beats C2C or dense KV/cache transfer.",
            "support_level": "not_supported",
            "evidence_artifact": "systems boundary table only",
            "controls_passed": "none; no native matched C2C row",
            "required_wording": "do not claim; compare as byte/exposure boundary only",
        },
        {
            "claim": "LatentWire has native GPU latency, HBM, energy, or throughput wins.",
            "support_level": "not_supported",
            "evidence_artifact": "NVIDIA native benchmark runbook",
            "controls_passed": "not run",
            "required_wording": "future work until native measurements exist",
        },
        {
            "claim": "LatentWire solves broad latent model-to-model communication or cross-family transfer.",
            "support_level": "not_supported",
            "evidence_artifact": "negative_results.csv; cross-family failure rows",
            "controls_passed": "cross-family falsification weakened the broad claim",
            "required_wording": "do not claim; present as an open ICLR goal",
        },
    ]


def _table_figure_inventory() -> list[dict[str, str]]:
    return [
        {
            "artifact": "unified abstract and introduction",
            "status": "draft_integrated",
            "source": "colm_final/paper/latentwire_colm2026.tex",
            "next_action": "human copyedit and page-budget review",
        },
        {
            "artifact": "method/protocol definition",
            "status": "draft_integrated",
            "source": "COLM_v1 method intuition plus COLM_v2 packet protocol",
            "next_action": "verify notation consistency after copyedit",
        },
        {
            "artifact": "source-private threat model",
            "status": "draft_integrated",
            "source": "COLM_v2 controls and systems boundary notes",
            "next_action": "check against reviewer claim audit",
        },
        {
            "artifact": "strict-control table",
            "status": "draft_integrated",
            "source": "strict_controls.csv",
            "next_action": "validate table placement in PDF",
        },
        {
            "artifact": "main positive result table",
            "status": "draft_integrated_source_index_bounded",
            "source": "main_results.csv",
            "next_action": "keep ARC as narrow same-family positive evidence",
        },
        {
            "artifact": "uncertainty summary table",
            "status": "draft_integrated",
            "source": "source-index audit lower bounds",
            "next_action": "verify table placement in final PDF",
        },
        {
            "artifact": "utility-per-byte / packet-byte table",
            "status": "data_ready",
            "source": "systems_measured_vs_estimated.csv",
            "next_action": "separate raw, framed, cacheline, and batch64 bytes",
        },
        {
            "artifact": "systems boundary table",
            "status": "draft_integrated",
            "source": "systems_measured_vs_estimated.csv",
            "next_action": "validate measured-vs-estimated labels in PDF",
        },
        {
            "artifact": "baseline/related-work matrix",
            "status": "draft_integrated",
            "source": "baseline_matrix.csv",
            "next_action": "check for overflow and page-budget pressure",
        },
        {
            "artifact": "negative-results / failure-boundary table",
            "status": "data_ready",
            "source": "negative_results.csv",
            "next_action": "use to define claim boundaries",
        },
        {
            "artifact": "claim audit table",
            "status": "draft_integrated",
            "source": "claim_audit.csv",
            "next_action": "keep appendix or move to internal audit depending on page limit",
        },
        {
            "artifact": "reproducibility checklist",
            "status": "partial",
            "source": "artifact_manifest.csv and input_manifest",
            "next_action": "convert to workshop checklist before submission",
        },
        {
            "artifact": "NVIDIA native benchmark runbook",
            "status": "generated_future_work",
            "source": "nvidia_native_runbook.md",
            "next_action": "run only on native NVIDIA hardware later",
        },
        {
            "artifact": "ten-reviewer COLM stress panel",
            "status": "recorded",
            "source": "colm_final/audits/colm_v3_10_reviewer_panel_20260505.md",
            "next_action": "use for human copyedit and final reviewer-risk pass",
        },
    ]


def _submission_checklist() -> list[dict[str, str]]:
    return [
        {
            "item": "Main claim agrees across abstract, intro, results, limitations.",
            "status": "reviewer_hardened_pending_human_review",
            "blocker": "requires human copyedit and page-budget review",
        },
        {
            "item": "Every table and figure maps to a claim in the claim audit.",
            "status": "draft_integrated",
            "blocker": "verify final PDF table placement",
        },
        {
            "item": "Systems claims separate measured packet bytes from analytical KV/cache floors.",
            "status": "ready",
            "blocker": "native GPU claims remain forbidden",
        },
        {
            "item": "Related work distinguishes dense KV/cache transfer, compression, and packet controls.",
            "status": "draft_integrated_compressed",
            "blocker": "page-budget review may require moving matrix to appendix",
        },
        {
            "item": "Limitations explicitly cover source-choice artifacts and cross-family failures.",
            "status": "draft_integrated",
            "blocker": "human copyedit",
        },
        {
            "item": "Ten-reviewer stress panel is recorded and actioned.",
            "status": "ready",
            "blocker": "remaining panel risks are claim-boundary risks, not missing paper sections",
        },
        {
            "item": "Experimental side projects are scoped away from COLM_v3 claims.",
            "status": "ready",
            "blocker": "only future-work wording should remain",
        },
    ]


def _experiment_scoping() -> list[dict[str, str]]:
    return [
        {
            "experiment": "HybridKernel",
            "folder": "experimental/hybridkernel",
            "colm_v3_scope": (
                "separate systems spinout; exclude from COLM_v3 claims unless Phase 1 confirms novelty, "
                "Phase 2 shows at least 3% theoretical benefit, and native GPU profiling confirms overhead"
            ),
            "highest_value_gate": "vLLM hybrid SSM/disaggregated serving source audit",
            "novelty_risk": "boundary fusion may already be covered by vLLM/vendor hybrid serving optimizations",
            "status": "alive_but_deferred",
        },
        {
            "experiment": "SinkAware",
            "folder": "experimental/sinkaware",
            "colm_v3_scope": (
                "separate systems spinout only; mention in COLM_v3 only as future work unless Phase 1-4 "
                "produce source-backed novelty plus a reference artifact"
            ),
            "highest_value_gate": "FlashInfer prefill/decode attention path audit",
            "novelty_risk": "static sink priors may already be expressible through generic mask or sparse-block APIs",
            "status": "quick_kill_candidate",
        },
        {
            "experiment": "ThoughtFlow-FP8",
            "folder": "experimental/thoughtflow_fp8",
            "colm_v3_scope": "separate systems spinout candidate after Phase 1, not current COLM_v3 evidence",
            "highest_value_gate": "LongFlow OpenReview/arXiv forensics and failure-mode audit",
            "novelty_risk": "could collapse into LongFlow plus FP8/anchor tweaks unless a concrete failure mode is documented",
            "status": "high_upside_high_crowding",
        },
    ]


def _artifact_manifest(input_paths: dict[str, pathlib.Path]) -> list[dict[str, str]]:
    artifacts = [
        ("review_packet.json", "machine-readable COLM_v3 review packet"),
        ("review_packet.md", "human-readable COLM_v3 review packet"),
        ("claim_audit.csv", "claim to evidence and wording boundary table"),
        ("contribution_table.csv", "current contribution status table"),
        ("systems_measured_vs_estimated.csv", "systems rows with measured-vs-estimated labels"),
        ("table_figure_inventory.csv", "paper table and figure readiness tracker"),
        ("submission_checklist.csv", "remaining workshop submission blockers"),
        ("experiment_scoping.csv", "three systems side experiments scoped for COLM_v3"),
        ("nvidia_native_runbook.md", "future native GPU measurement runbook"),
        ("manifest.json", "input and output manifest"),
    ]
    rows = [
        {
            "artifact": name,
            "role": role,
            "status": "generated",
        }
        for name, role in artifacts
    ]
    for key, path in sorted(input_paths.items()):
        rows.append(
            {
                "artifact": _repo_path(path),
                "role": f"input:{key}",
                "status": "read",
            }
        )
    return rows


def _contribution_table(v2_packet: dict[str, Any]) -> list[dict[str, str]]:
    rows = [
        {
            "contribution": "source-private packet protocol",
            "status": "supported_for_colm_v3",
            "evidence": "packet rows and source-private interface definition",
            "still_needs_work": "paper prose must avoid broad latent-language claims",
        },
        {
            "contribution": "strict destructive controls",
            "status": "supported_for_colm_v3",
            "evidence": "wrong-row, same-source-choice, source-index/rank/score, same-byte/text controls",
            "still_needs_work": "compress into one main table plus appendix",
        },
        {
            "contribution": "narrow low-byte packet utility",
            "status": "supported_but_narrow",
            "evidence": "main_results.csv",
            "still_needs_work": "state that source-index remains a strong boundary",
        },
        {
            "contribution": "systems byte and exposure accounting",
            "status": "supported_as_accounting",
            "evidence": "systems_measured_vs_estimated.csv",
            "still_needs_work": "no native GPU/HBM/energy claim until NVIDIA runbook is executed",
        },
        {
            "contribution": "broad positive latent communication method",
            "status": "not_supported_for_colm_v3",
            "evidence": "negative_results.csv and ICLR triage",
            "still_needs_work": "keep as ICLR future method target",
        },
    ]

    for row in v2_packet.get("contribution_table", []):
        name = row.get("name") or row.get("contribution")
        if name and not any(existing["contribution"] == name for existing in rows):
            rows.append(
                {
                    "contribution": str(name),
                    "status": str(row.get("status", "imported_from_colm_v2")),
                    "evidence": str(row.get("evidence", "")),
                    "still_needs_work": str(row.get("gap", "")),
                }
            )
    return rows


def build_review_packet(input_paths: dict[str, pathlib.Path] | None = None) -> dict[str, Any]:
    paths = dict(DEFAULT_INPUT_PATHS if input_paths is None else input_paths)
    v2_packet = _read_json(paths["colm_v2_review_packet"])
    systems = _read_json(paths["systems_boundary"])

    readiness_text = _read_text(paths["colm_v3_readiness"])
    experimental_status_text = _read_text(paths["experimental_status"])

    systems_rows = _systems_measured_vs_estimated(systems)

    return {
        "packet": "latentwire_colm_v3_review_packet",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "main_claim": MAIN_CLAIM,
        "readiness": {
            "colm_v3": "reviewer_hardened_draft_pending_human_review",
            "workshop_blocker": (
                "human copyedit, page-budget review, and final PDF/table placement; no new speculative experiment "
                "is required unless review exposes a missing claim-supporting row"
            ),
            "iclr": "still blocked by lack of broad source-causal positive method",
        },
        "story": (
            "LatentWire studies whether compact source-private candidate packets can transmit useful model evidence "
            "without dense cache transfer, and uses byte accounting plus destructive controls to separate "
            "real packet utility from answer-choice and target-cache shortcuts."
        ),
        "source_text_sanity": {
            "colm_v3_readiness_chars": len(readiness_text),
            "experimental_status_chars": len(experimental_status_text),
        },
        "systems_headline": systems.get("headline", {}),
        "systems_checks": systems.get("checks", []),
        "claim_audit": _claim_audit(),
        "contribution_table": _contribution_table(v2_packet),
        "table_figure_inventory": _table_figure_inventory(),
        "submission_checklist": _submission_checklist(),
        "systems_measured_vs_estimated": systems_rows,
        "experiment_scoping": _experiment_scoping(),
        "baseline_matrix": v2_packet.get("baseline_matrix", []),
        "main_results": v2_packet.get("main_results", []),
        "strict_controls": v2_packet.get("strict_controls", []),
        "negative_results": v2_packet.get("negative_results", []),
        "input_manifest": _input_manifest(paths),
        "artifact_manifest": _artifact_manifest(paths),
        "next_exact_gate": (
            "human copyedit, page-budget review, final PDF/table placement, and consistency check "
            "between paper, review packet, and artifact manifest"
        ),
    }


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(row.get(key)) for key in fieldnames})


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    if not rows:
        return ["_No rows._", ""]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_safe_md(row.get(column)) for column in columns) + " |")
    lines.append("")
    return lines


def _render_nvidia_runbook() -> str:
    lines = [
        "# COLM v3 Native NVIDIA Systems Runbook",
        "",
        "This runbook is for future native GPU evidence. It does not authorize any COLM_v3",
        "latency, HBM, energy, throughput, or C2C superiority claim until the measurements",
        "are actually run on NVIDIA hardware.",
        "",
        "## Setup",
        "",
        "1. Work on a native NVIDIA host. Do not use SSH from this agent session.",
        "2. Create a fresh repo-local virtual environment on that host.",
        "3. Install the pinned CUDA/PyTorch/vLLM/SGLang stack recorded by the run.",
        "4. Record GPU model, driver, CUDA version, PyTorch version, and clock/power settings.",
        "",
        "## Measurements",
        "",
        "- LatentWire packet encode/decode microbenchmarks with cached-source and end-to-end source-scoring rows separated.",
        "- Dense C2C or KV/cache transfer byte movement for matched source/target/task rows.",
        "- vLLM and SGLang serving baselines: TTFT, TPOT, goodput, peak memory, and cache movement where instrumentable.",
        "- Nsight Systems or PyTorch profiler traces for packet decode, source scoring, and any KV/cache baseline.",
        "- Cacheline/DMA-rounded bytes and batch-level framed bytes for every communicated object.",
        "",
        "## Required Outputs",
        "",
        "- `results/native_nvidia_colm_v3_<date>/metadata.json`",
        "- `results/native_nvidia_colm_v3_<date>/packet_microbench.csv`",
        "- `results/native_nvidia_colm_v3_<date>/dense_cache_baselines.csv`",
        "- `results/native_nvidia_colm_v3_<date>/serving_baselines.csv`",
        "- `results/native_nvidia_colm_v3_<date>/profiler_manifest.md`",
        "",
        "## Claim Bar",
        "",
        "A systems win can be claimed only if the native rows use matched tasks/models, include",
        "packet-source scoring separately, include a dense cache baseline, and pass the same",
        "source-private claim audit used by COLM_v3.",
        "",
    ]
    return "\n".join(lines)


def render_markdown(packet: dict[str, Any]) -> str:
    lines = [
        "# LatentWire COLM v3 Review Packet",
        "",
        f"- created_utc: `{packet['created_utc']}`",
        f"- main_claim: {packet['main_claim']}",
        f"- next_exact_gate: {packet['next_exact_gate']}",
        "",
        "## Readiness",
        "",
        _markdown_table([packet["readiness"]], ["colm_v3", "workshop_blocker", "iclr"])[0],
    ]
    lines.extend(_markdown_table([packet["readiness"]], ["colm_v3", "workshop_blocker", "iclr"])[1:])
    lines.extend(
        [
            "## Contribution Status",
            "",
            *_markdown_table(
                packet["contribution_table"],
                ["contribution", "status", "evidence", "still_needs_work"],
            ),
            "## Reviewer Claim Audit",
            "",
            *_markdown_table(
                packet["claim_audit"],
                ["claim", "support_level", "evidence_artifact", "controls_passed", "required_wording"],
            ),
            "## Table And Figure Inventory",
            "",
            *_markdown_table(
                packet["table_figure_inventory"],
                ["artifact", "status", "source", "next_action"],
            ),
            "## Systems Measured Vs Estimated",
            "",
            *_markdown_table(
                packet["systems_measured_vs_estimated"],
                [
                    "method",
                    "raw_bytes",
                    "framed_bytes",
                    "cacheline_bytes",
                    "batch64_bytes",
                    "measured_vs_estimated",
                    "claim_allowed",
                ],
            ),
            "## Experimental Side-Branch Scope",
            "",
            *_markdown_table(
                packet["experiment_scoping"],
                ["experiment", "colm_v3_scope", "highest_value_gate", "novelty_risk", "status"],
            ),
            "## Submission Checklist",
            "",
            *_markdown_table(packet["submission_checklist"], ["item", "status", "blocker"]),
            "## Input Manifest",
            "",
            *_markdown_table(packet["input_manifest"], ["key", "path", "sha256"]),
        ]
    )
    return "\n".join(lines)


def write_outputs(packet: dict[str, Any], output_dir: pathlib.Path, paper_path: pathlib.Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "review_packet.json").write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = render_markdown(packet)
    (output_dir / "review_packet.md").write_text(markdown, encoding="utf-8")
    paper_path.parent.mkdir(parents=True, exist_ok=True)
    paper_path.write_text(markdown, encoding="utf-8")

    _write_csv(output_dir / "claim_audit.csv", packet["claim_audit"])
    _write_csv(output_dir / "contribution_table.csv", packet["contribution_table"])
    _write_csv(output_dir / "systems_measured_vs_estimated.csv", packet["systems_measured_vs_estimated"])
    _write_csv(output_dir / "table_figure_inventory.csv", packet["table_figure_inventory"])
    _write_csv(output_dir / "submission_checklist.csv", packet["submission_checklist"])
    _write_csv(output_dir / "experiment_scoping.csv", packet["experiment_scoping"])
    _write_csv(output_dir / "artifact_manifest.csv", packet["artifact_manifest"])
    _write_csv(output_dir / "baseline_matrix.csv", packet["baseline_matrix"])
    _write_csv(output_dir / "main_results.csv", packet["main_results"])
    _write_csv(output_dir / "strict_controls.csv", packet["strict_controls"])
    _write_csv(output_dir / "negative_results.csv", packet["negative_results"])
    (output_dir / "nvidia_native_runbook.md").write_text(_render_nvidia_runbook(), encoding="utf-8")

    manifest = {
        "packet": packet["packet"],
        "created_utc": packet["created_utc"],
        "input_manifest": packet["input_manifest"],
        "outputs": [
            _repo_path(path)
            for path in sorted(output_dir.iterdir())
            if path.is_file()
        ]
        + [_repo_path(paper_path)],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--paper-path", type=pathlib.Path, default=DEFAULT_PAPER_PATH)
    args = parser.parse_args()

    packet = build_review_packet()
    write_outputs(packet, args.output_dir, args.paper_path)
    print(f"Wrote COLM_v3 review packet to {args.output_dir}")
    print(f"Wrote paper memo to {args.paper_path}")


if __name__ == "__main__":
    main()
