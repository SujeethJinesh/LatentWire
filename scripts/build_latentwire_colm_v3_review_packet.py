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
    "benchmark_selection_gate": ROOT
    / "results/source_private_benchmark_selection_gate_20260502/benchmark_selection_gate.json",
    "hellaswag_seed_stability": ROOT
    / "results/source_private_hellaswag_seed_stability_20260501_qwen05_hashed_validation1024_2b_5seed/arc_challenge_seed_stability.json",
    "sciq_bridge_contract": ROOT
    / "results/source_private_sciq_bridge_contract_20260501/sciq_bridge_contract.json",
    "latest_model_matrix": ROOT
    / "results/source_private_latest_model_matrix_20260428/latest_model_matrix.json",
    "cpu_systems_frontier": ROOT
    / "results/source_private_cpu_systems_frontier_20260429/cpu_systems_frontier.json",
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
            "artifact": "benchmark breadth audit",
            "status": "review_packet_integrated",
            "source": "benchmark_breadth.csv",
            "next_action": "use as reviewer-pack support; do not turn diagnostics into headline claims",
        },
        {
            "artifact": "latest model breadth audit",
            "status": "review_packet_integrated",
            "source": "latest_model_breadth.csv",
            "next_action": "treat as source-packet emitter smoke, not ARC/OBQA benchmark evidence",
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
            "status": "phase1_quick_audit_complete_phase2_architecture_map_pending",
        },
        {
            "experiment": "SinkAware",
            "folder": "experimental/sinkaware",
            "colm_v3_scope": (
                "separate systems spinout only; mention in COLM_v3 only as future work unless Phase 1-4 "
                "produce source-backed novelty plus a reference artifact"
            ),
            "highest_value_gate": "CPU-only fixed sink-token decomposition",
            "novelty_risk": "learned/per-head sink denominator handling is already occupied by FlashInfer/FlashMLA/GPT-OSS-style paths",
            "status": "phase1_quick_audit_complete_narrow_phase2_or_kill",
        },
        {
            "experiment": "ThoughtFlow-FP8",
            "folder": "experimental/thoughtflow_fp8",
            "colm_v3_scope": "separate systems spinout candidate after Phase 1, not current COLM_v3 evidence",
            "highest_value_gate": "Mac-only trace simulation for protected reasoning-token classes",
            "novelty_risk": "field is crowded by LongFlow, ThinKV, R-KV/R-KVHash, RaaS, LazyEviction, ForesightKV, and PM-KVQ",
            "status": "phase1_forensics_complete_high_upside_phase2_pending",
        },
    ]


def _triton_kernel_scaffolds() -> list[dict[str, str]]:
    rows = [
        {
            "experiment": "HybridKernel",
            "cpu_reference": "experimental/hybridkernel/phase3/reference/boundary.py",
            "cpu_test": "experimental/hybridkernel/phase3/tests/test_boundary_reference.py",
            "triton_kernel": "experimental/hybridkernel/phase4/kernel/boundary_triton.py",
            "triton_test": "experimental/hybridkernel/phase4/tests/test_boundary_triton_interpret.py",
            "local_status": "cpu_reference_passes_triton_skips_missing_dependency",
            "claim_boundary": "future correctness scaffold only; not COLM_v3 systems evidence",
        },
        {
            "experiment": "SinkAware",
            "cpu_reference": "experimental/sinkaware/phase2/reference/sink_decomposition.py",
            "cpu_test": "experimental/sinkaware/phase2/tests/test_sink_decomposition_reference.py",
            "triton_kernel": "experimental/sinkaware/phase4/kernel/sink_decomposition_triton.py",
            "triton_test": "experimental/sinkaware/phase4/tests/test_sink_decomposition_triton_interpret.py",
            "local_status": "cpu_reference_passes_triton_skips_missing_dependency",
            "claim_boundary": "future exact scalar decomposition scaffold only; not COLM_v3 systems evidence",
        },
        {
            "experiment": "ThoughtFlow-FP8",
            "cpu_reference": "experimental/thoughtflow_fp8/phase2/reference/anchor_phase_quant.py",
            "cpu_test": "experimental/thoughtflow_fp8/phase2/tests/test_anchor_phase_quant_reference.py",
            "triton_kernel": "experimental/thoughtflow_fp8/phase4/kernel/anchor_phase_quant_triton.py",
            "triton_test": "experimental/thoughtflow_fp8/phase4/tests/test_anchor_phase_quant_triton_interpret.py",
            "local_status": "cpu_reference_passes_triton_skips_missing_dependency",
            "claim_boundary": "future anchor/phase retention scaffold only; not COLM_v3 systems evidence",
        },
    ]
    for row in rows:
        row["files_exist"] = (
            "true"
            if all((ROOT / row[key]).exists() for key in ("cpu_reference", "cpu_test", "triton_kernel", "triton_test"))
            else "false"
        )
    return rows


def _benchmark_breadth(paths: dict[str, pathlib.Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if "benchmark_selection_gate" in paths and paths["benchmark_selection_gate"].exists():
        gate = _read_json(paths["benchmark_selection_gate"])
        for row in gate.get("rows", []):
            row_id = str(row.get("row_id", ""))
            if row_id in {"openbookqa_test_3b", "arc_challenge_test_12b"}:
                paper_use = "main_or_secondary_test_row"
            elif row.get("packet_text_margin_pass") and row.get("packet_target_pass"):
                paper_use = "diagnostic_breadth"
            else:
                paper_use = "diagnostic_not_headline"
            rows.append(
                {
                    "benchmark": row.get("dataset"),
                    "split": row.get("split"),
                    "examples": row.get("eval_rows"),
                    "packet_bytes": row.get("budget_bytes"),
                    "seeds": row.get("seed_count"),
                    "packet_accuracy": row.get("matched_accuracy_mean"),
                    "target_accuracy": row.get("target_accuracy"),
                    "same_byte_text_accuracy": row.get("same_byte_text_accuracy"),
                    "packet_minus_target_min": row.get("matched_minus_target_min"),
                    "packet_minus_text_min": row.get("matched_minus_same_byte_text_min"),
                    "ci95_low_vs_target_min": row.get("paired_ci95_low_vs_target_min"),
                    "role": row.get("selection_role"),
                    "paper_use": paper_use,
                    "caveat": (
                        "headline-safe only when paired with source-index and source-choice boundary wording"
                        if paper_use == "main_or_secondary_test_row"
                        else "reviewer-pack diagnostic"
                    ),
                    "artifact": row.get("seed_artifact"),
                }
            )

    if "hellaswag_seed_stability" in paths and paths["hellaswag_seed_stability"].exists():
        hellaswag = _read_json(paths["hellaswag_seed_stability"])
        aggregate = hellaswag.get("aggregate", {})
        rows.append(
            {
                "benchmark": "HellaSwag",
                "split": hellaswag.get("split_name", "validation1024"),
                "examples": hellaswag.get("eval_rows"),
                "packet_bytes": hellaswag.get("budget_bytes"),
                "seeds": aggregate.get("seed_count") or len(hellaswag.get("seeds", [])),
                "packet_accuracy": aggregate.get("matched_accuracy_mean"),
                "target_accuracy": aggregate.get("target_accuracy"),
                "same_byte_text_accuracy": aggregate.get("same_byte_structured_text_accuracy"),
                "packet_minus_target_min": aggregate.get("matched_minus_target_min"),
                "packet_minus_text_min": aggregate.get("matched_minus_same_byte_text_min"),
                "ci95_low_vs_target_min": aggregate.get("paired_ci95_low_vs_target_min"),
                "role": "bounded_validation_breadth",
                "paper_use": "reviewer_pack_breadth_not_full_validation_headline",
                "caveat": "passes validation1024 seed gate; full-validation terminal tail blocks benchmark-complete claim",
                "artifact": _repo_path(paths["hellaswag_seed_stability"]),
            }
        )

    if "sciq_bridge_contract" in paths and paths["sciq_bridge_contract"].exists():
        sciq = _read_json(paths["sciq_bridge_contract"])
        validation = sciq.get("official_summaries", {}).get("validation", {})
        test = sciq.get("official_summaries", {}).get("test", {})
        rows.append(
            {
                "benchmark": "SciQ",
                "split": "validation/test materialized",
                "examples": f"{validation.get('n', '')}/{test.get('n', '')}",
                "packet_bytes": sciq.get("method_contract", {}).get("fixed_packet_budget_bytes"),
                "seeds": "",
                "packet_accuracy": "",
                "target_accuracy": "",
                "same_byte_text_accuracy": "",
                "packet_minus_target_min": "",
                "packet_minus_text_min": "",
                "ci95_low_vs_target_min": "",
                "role": "bridge_contract_only",
                "paper_use": "future_gate_not_evidence",
                "caveat": "source-private split/control contract is frozen, but no positive packet row exists yet",
                "artifact": _repo_path(paths["sciq_bridge_contract"]),
            }
        )

    return rows


def _summarize_cpu_frontier_models(paths: dict[str, pathlib.Path]) -> dict[str, dict[str, Any]]:
    if "cpu_systems_frontier" not in paths or not paths["cpu_systems_frontier"].exists():
        return {}
    frontier = _read_json(paths["cpu_systems_frontier"])
    summaries: dict[str, dict[str, Any]] = {}
    for row in frontier.get("rows", []):
        if row.get("contribution") != "model-emitted source packet":
            continue
        method = str(row.get("method", ""))
        current = summaries.setdefault(
            method,
            {
                "passes": 0,
                "fails": 0,
                "max_n": 0,
                "min_valid_rate": None,
                "max_accuracy": None,
                "surfaces": [],
            },
        )
        if row.get("status") == "pass":
            current["passes"] += 1
        else:
            current["fails"] += 1
        n = 0
        note = str(row.get("note", ""))
        if "n=" in note:
            try:
                n = int(note.split("n=", 1)[1].split(";", 1)[0])
            except ValueError:
                n = 0
        current["max_n"] = max(int(current["max_n"]), n)
        valid_rate = row.get("valid_rate")
        if valid_rate is not None:
            current["min_valid_rate"] = (
                valid_rate
                if current["min_valid_rate"] is None
                else min(float(current["min_valid_rate"]), float(valid_rate))
            )
        accuracy = row.get("accuracy")
        if accuracy is not None:
            current["max_accuracy"] = (
                accuracy
                if current["max_accuracy"] is None
                else max(float(current["max_accuracy"]), float(accuracy))
            )
        current["surfaces"].append(str(row.get("surface", "")))
    return summaries


def _summarize_latest_model_artifacts(paths: dict[str, pathlib.Path]) -> dict[str, dict[str, Any]]:
    if "latest_model_matrix" not in paths or not paths["latest_model_matrix"].exists():
        return {}
    root = paths["latest_model_matrix"].parent
    if not root.exists():
        return {}
    prefix_to_model = {
        "qwen35_0_8b": "Qwen/Qwen3.5-0.8B",
        "qwen35_2b": "Qwen/Qwen3.5-2B",
        "qwen35_4b": "Qwen/Qwen3.5-4B",
        "gemma4_e2b": "google/gemma-4-E2B-it",
        "granite33_2b": "ibm-granite/granite-3.3-2b-instruct",
    }
    summaries: dict[str, dict[str, Any]] = {}
    for summary_path in root.glob("*/summary.json"):
        dirname = summary_path.parent.name
        model_id = None
        for prefix, candidate in prefix_to_model.items():
            if dirname.startswith(prefix):
                model_id = candidate
                break
        if model_id is None:
            continue
        summary = _read_json(summary_path)
        current = summaries.setdefault(
            model_id,
            {
                "passes": 0,
                "fails": 0,
                "max_n": 0,
                "min_valid_rate": None,
                "max_accuracy": None,
                "artifacts": [],
            },
        )
        if summary.get("pass_gate"):
            current["passes"] += 1
        else:
            current["fails"] += 1
        current["max_n"] = max(int(current["max_n"]), int(summary.get("n", 0) or 0))
        valid_rate = summary.get("packet_valid_rate")
        if valid_rate is not None and summary.get("pass_gate"):
            current["min_valid_rate"] = (
                valid_rate
                if current["min_valid_rate"] is None
                else min(float(current["min_valid_rate"]), float(valid_rate))
            )
        matched = summary.get("metrics", {}).get("matched_model_packet", {})
        accuracy = matched.get("accuracy")
        if accuracy is not None and summary.get("pass_gate"):
            current["max_accuracy"] = (
                accuracy
                if current["max_accuracy"] is None
                else max(float(current["max_accuracy"]), float(accuracy))
            )
        current["artifacts"].append(_repo_path(summary_path))
    return summaries


def _latest_model_breadth(paths: dict[str, pathlib.Path]) -> list[dict[str, Any]]:
    if "latest_model_matrix" not in paths or not paths["latest_model_matrix"].exists():
        return []
    matrix = _read_json(paths["latest_model_matrix"])
    cpu_summaries = _summarize_cpu_frontier_models(paths)
    artifact_summaries = _summarize_latest_model_artifacts(paths)
    paper_models = {
        "Qwen/Qwen3.5-0.8B",
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-4B",
        "google/gemma-4-E2B-it",
        "ibm-granite/granite-3.3-2b-instruct",
        "allenai/OLMo-2-0425-1B-Instruct",
    }
    label_aliases = {
        "Qwen/Qwen3.5-0.8B": "Qwen3.5-0.8B",
        "Qwen/Qwen3.5-2B": "Qwen3.5-2B",
        "Qwen/Qwen3.5-4B": "Qwen3.5-4B",
        "google/gemma-4-E2B-it": "Gemma 4 E2B",
        "ibm-granite/granite-3.3-2b-instruct": "Granite 3.3 2B",
        "allenai/OLMo-2-0425-1B-Instruct": "OLMo-2 1B",
    }
    rows = []
    for model in matrix.get("models", []):
        model_id = model.get("model")
        if model_id not in paper_models:
            continue
        summary = artifact_summaries.get(
            str(model_id), cpu_summaries.get(label_aliases.get(model_id, str(model_id)), {})
        )
        status = str(model.get("status", ""))
        if summary.get("passes", 0):
            paper_use = "reviewer_pack_emitter_breadth"
        elif "candidate" in status.lower() or "optional" in status.lower():
            paper_use = "planned_not_evidence"
        else:
            paper_use = "negative_or_unverified"
        rows.append(
            {
                "model": model_id,
                "family": model.get("family"),
                "params": model.get("params"),
                "architecture": model.get("architecture"),
                "device": model.get("expected_device"),
                "local_rung": model.get("local_rung"),
                "status": status,
                "passes": summary.get("passes", ""),
                "fails": summary.get("fails", ""),
                "max_n": summary.get("max_n", ""),
                "min_valid_rate": summary.get("min_valid_rate", ""),
                "paper_use": paper_use,
                "caveat": "source-packet emitter smoke on synthetic hidden-repair benchmark, not ARC/OBQA headline evidence",
            }
        )
    return rows


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
        ("triton_kernel_scaffolds.csv", "Macbook Triton interpreter correctness scaffold tracker"),
        ("benchmark_breadth.csv", "additional benchmark breadth and claim boundary audit"),
        ("latest_model_breadth.csv", "newer model source-packet emitter breadth audit"),
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
        "triton_kernel_scaffolds": _triton_kernel_scaffolds(),
        "benchmark_breadth": _benchmark_breadth(paths),
        "latest_model_breadth": _latest_model_breadth(paths),
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
            "## Benchmark Breadth Audit",
            "",
            *_markdown_table(
                packet["benchmark_breadth"],
                [
                    "benchmark",
                    "split",
                    "examples",
                    "packet_bytes",
                    "packet_accuracy",
                    "target_accuracy",
                    "same_byte_text_accuracy",
                    "paper_use",
                    "caveat",
                ],
            ),
            "## Latest Model Breadth Audit",
            "",
            *_markdown_table(
                packet["latest_model_breadth"],
                [
                    "model",
                    "params",
                    "architecture",
                    "device",
                    "local_rung",
                    "passes",
                    "max_n",
                    "paper_use",
                    "caveat",
                ],
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
            "## Triton Kernel Correctness Scaffolds",
            "",
            "These are Macbook-side correctness hooks only. They do not support COLM_v3 GPU systems claims.",
            "",
            *_markdown_table(
                packet["triton_kernel_scaffolds"],
                [
                    "experiment",
                    "cpu_reference",
                    "triton_kernel",
                    "local_status",
                    "files_exist",
                    "claim_boundary",
                ],
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
    _write_csv(output_dir / "triton_kernel_scaffolds.csv", packet["triton_kernel_scaffolds"])
    _write_csv(output_dir / "benchmark_breadth.csv", packet["benchmark_breadth"])
    _write_csv(output_dir / "latest_model_breadth.csv", packet["latest_model_breadth"])
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
