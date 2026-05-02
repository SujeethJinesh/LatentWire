from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_COMMON_BASIS = pathlib.Path(
    "results/source_private_candidate_local_common_basis_falsification_20260430/"
    "candidate_local_common_basis_falsification.json"
)
DEFAULT_SYSTEMS = pathlib.Path(
    "results/source_private_candidate_local_residual_systems_waterfall_20260430/"
    "candidate_local_residual_systems_waterfall.json"
)
DEFAULT_OUTPUT = pathlib.Path("results/source_private_candidate_local_competitor_basis_table_20260430")

CSV_COLUMNS = (
    "method_id",
    "method",
    "category",
    "status",
    "same_slice_measured",
    "rows",
    "pass_rows",
    "payload_bytes",
    "record_bytes",
    "source_text_exposed",
    "source_kv_exposed",
    "matched_accuracy_min",
    "matched_accuracy_max",
    "target_accuracy_min",
    "target_accuracy_max",
    "best_control_accuracy_max",
    "control_leak_over_target_max",
    "delta_vs_best_control_min",
    "claim_use",
    "next_action",
    "sources",
)


def _resolve(path: pathlib.Path) -> pathlib.Path:
    return path if path.is_absolute() else ROOT / path


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _stats(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return min(values), max(values)


def _row(
    *,
    method_id: str,
    method: str,
    category: str,
    status: str,
    same_slice_measured: bool,
    rows: int | None = None,
    pass_rows: int | None = None,
    payload_bytes: int | None = None,
    record_bytes: int | None = None,
    source_text_exposed: bool | None = None,
    source_kv_exposed: bool | None = None,
    matched_values: list[float] | None = None,
    target_values: list[float] | None = None,
    best_control_values: list[float] | None = None,
    control_leak_values: list[float] | None = None,
    delta_vs_best_control_values: list[float] | None = None,
    claim_use: str,
    next_action: str,
    sources: str,
) -> dict[str, Any]:
    matched_min, matched_max = _stats(matched_values or [])
    target_min, target_max = _stats(target_values or [])
    best_control_max = max(best_control_values) if best_control_values else None
    control_leak_max = max(control_leak_values) if control_leak_values else None
    delta_min = min(delta_vs_best_control_values) if delta_vs_best_control_values else None
    return {
        "method_id": method_id,
        "method": method,
        "category": category,
        "status": status,
        "same_slice_measured": same_slice_measured,
        "rows": rows,
        "pass_rows": pass_rows,
        "payload_bytes": payload_bytes,
        "record_bytes": record_bytes,
        "source_text_exposed": source_text_exposed,
        "source_kv_exposed": source_kv_exposed,
        "matched_accuracy_min": matched_min,
        "matched_accuracy_max": matched_max,
        "target_accuracy_min": target_min,
        "target_accuracy_max": target_max,
        "best_control_accuracy_max": best_control_max,
        "control_leak_over_target_max": control_leak_max,
        "delta_vs_best_control_min": delta_min,
        "claim_use": claim_use,
        "next_action": next_action,
        "sources": sources,
    }


def _rows_for_group(rows: list[dict[str, Any]], row_group: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("row_group") == row_group]


def _aggregate_measured_group(
    *,
    method_id: str,
    method: str,
    rows: list[dict[str, Any]],
    status: str,
    claim_use: str,
    next_action: str,
    sources: str,
    record_bytes: int | None,
    payload_bytes: int = 8,
) -> dict[str, Any]:
    return _row(
        method_id=method_id,
        method=method,
        category="measured_method",
        status=status,
        same_slice_measured=True,
        rows=len(rows),
        pass_rows=sum(bool(row.get("pass_gate")) for row in rows),
        payload_bytes=payload_bytes,
        record_bytes=record_bytes,
        source_text_exposed=False,
        source_kv_exposed=False,
        matched_values=[float(row["matched_accuracy"]) for row in rows],
        target_values=[float(row["target_accuracy"]) for row in rows],
        best_control_values=[float(row["best_control_accuracy"]) for row in rows],
        control_leak_values=[float(row["control_leak_over_target"]) for row in rows],
        delta_vs_best_control_values=[float(row["delta_vs_best_control"]) for row in rows],
        claim_use=claim_use,
        next_action=next_action,
        sources=sources,
    )


def _aggregate_condition(
    *,
    method_id: str,
    method: str,
    rows: list[dict[str, Any]],
    metric_key: str,
    category: str,
    status: str,
    source_text_exposed: bool,
    source_kv_exposed: bool,
    payload_bytes: int | None,
    record_bytes: int | None,
    claim_use: str,
    next_action: str,
    sources: str,
) -> dict[str, Any]:
    values = [float(row[metric_key]) for row in rows if row.get(metric_key) not in {None, ""}]
    targets = [float(row["target_accuracy"]) for row in rows]
    deltas = [value - target for value, target in zip(values, targets, strict=True)]
    return _row(
        method_id=method_id,
        method=method,
        category=category,
        status=status,
        same_slice_measured=True,
        rows=len(values),
        pass_rows=None,
        payload_bytes=payload_bytes,
        record_bytes=record_bytes,
        source_text_exposed=source_text_exposed,
        source_kv_exposed=source_kv_exposed,
        matched_values=values,
        target_values=targets,
        best_control_values=None,
        control_leak_values=None,
        delta_vs_best_control_values=deltas,
        claim_use=claim_use,
        next_action=next_action,
        sources=sources,
    )


def _pending_row(
    *,
    method_id: str,
    method: str,
    category: str,
    claim_use: str,
    next_action: str,
    sources: str,
    source_text_exposed: bool | None = None,
    source_kv_exposed: bool | None = None,
) -> dict[str, Any]:
    return _row(
        method_id=method_id,
        method=method,
        category=category,
        status="pending_required_for_iclr",
        same_slice_measured=False,
        source_text_exposed=source_text_exposed,
        source_kv_exposed=source_kv_exposed,
        claim_use=claim_use,
        next_action=next_action,
        sources=sources,
    )


def build_competitor_basis_table(
    *,
    common_basis_path: pathlib.Path,
    systems_path: pathlib.Path | None,
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    common = _read_json(common_basis_path)
    systems = _read_json(systems_path) if systems_path is not None and _resolve(systems_path).exists() else {}
    system_headline = systems.get("headline", {})
    record_bytes = int(system_headline["packet_record_bytes"]) if "packet_record_bytes" in system_headline else None
    rows = common["rows"]
    live_rows = _rows_for_group(rows, "live")
    global_rows = _rows_for_group(rows, "global_common_basis")
    procrustes_rows = _rows_for_group(rows, "procrustes_common_basis")
    ridge_cca_rows = _rows_for_group(rows, "ridge_cca_common_basis")
    ridge_cca_stack_rows = _rows_for_group(rows, "ridge_cca_local_stack")
    lstirp_rows = _rows_for_group(rows, "lstirp_relative_translation")
    lstirp_stack_rows = _rows_for_group(rows, "lstirp_relative_local_stack")
    sinkhorn_ot_rows = _rows_for_group(rows, "sinkhorn_ot_transport")
    sinkhorn_ot_stack_rows = _rows_for_group(rows, "sinkhorn_ot_local_stack")
    gw_rows = _rows_for_group(rows, "gw_transport")
    gw_stack_rows = _rows_for_group(rows, "gw_local_stack")
    relative_rows = _rows_for_group(rows, "relative_anchor_common_basis")
    relative_stack_rows = _rows_for_group(rows, "relative_anchor_local_stack")
    relative_innovation_rows = _rows_for_group(rows, "relative_anchor_innovation_stack")
    relative_rank_innovation_rows = _rows_for_group(rows, "relative_anchor_rank_innovation_stack")
    diagnostic_rows = _rows_for_group(rows, "diagnostic_ablation")
    table_rows = [
        _aggregate_condition(
            method_id="target_prior",
            method="target prior / no source",
            rows=live_rows,
            metric_key="target_accuracy",
            category="measured_floor",
            status="floor",
            source_text_exposed=False,
            source_kv_exposed=False,
            payload_bytes=0,
            record_bytes=0,
            claim_use="Target-side prior floor for all packet rows.",
            next_action="Keep as paired floor in every wider benchmark.",
            sources="current n512 live summaries",
        ),
        _aggregate_condition(
            method_id="structured_text_8b",
            method="matched-byte structured text/log prefix",
            rows=live_rows,
            metric_key="structured_text_matched_accuracy",
            category="measured_text_control",
            status="floor_at_8b",
            source_text_exposed=True,
            source_kv_exposed=False,
            payload_bytes=8,
            record_bytes=None,
            claim_use="Shows the 8B packet win is not matched by a same-byte text/log prefix.",
            next_action="Add a longer structured-text rate curve before claiming text is broadly worse.",
            sources="current n512 live summaries",
        ),
        _aggregate_condition(
            method_id="random_same_byte",
            method="random same-byte packet",
            rows=live_rows,
            metric_key="random_same_byte_accuracy",
            category="measured_source_destroying_control",
            status="floor",
            source_text_exposed=False,
            source_kv_exposed=False,
            payload_bytes=8,
            record_bytes=record_bytes,
            claim_use="Checks that byte budget alone does not explain the gain.",
            next_action="Keep as destructive control in all packet variants.",
            sources="current n512 live summaries",
        ),
        _aggregate_measured_group(
            method_id="candidate_local_residual_norm",
            method="candidate-local residual chart with row/payload normalization",
            rows=live_rows,
            status="passes_strict_controls",
            claim_use="Main positive method row.",
            next_action="Compare against full Relative Representation / CCA / OT rows on the same slice.",
            sources="candidate-local residual n512 summaries",
            record_bytes=record_bytes,
        ),
        _aggregate_measured_group(
            method_id="global_public_anchor_dot",
            method="global public-anchor dot product",
            rows=global_rows,
            status="fails_controls",
            claim_use="Nearest implemented common-basis falsification.",
            next_action="Use as a measured warning: high matched accuracy is invalid if controls rise.",
            sources="common-basis falsification artifact",
            record_bytes=record_bytes,
        ),
    ]
    if procrustes_rows:
        table_rows.append(
            _aggregate_measured_group(
                method_id="orthogonal_procrustes_dot",
                method="public-calibration orthogonal Procrustes dot product",
                rows=procrustes_rows,
                status="fails_controls"
                if any(row["best_control_accuracy"] > row["target_accuracy"] + 0.03 for row in procrustes_rows)
                else "fails_gate",
                claim_use="Measured orthogonal shared-space packet baseline; invalid if permuted/public controls rise.",
                next_action="Keep as an unsafe common-basis row; run ridge CCA/SVCCA next instead of calling this CCA.",
                sources="common-basis falsification artifact; https://web.stanford.edu/class/cs273/refs/procrustes.pdf",
                record_bytes=record_bytes,
            )
        )
    if ridge_cca_rows:
        table_rows.append(
            _aggregate_measured_group(
                method_id="ridge_cca_dot",
                method="ridge CCA/SVCCA-style canonical-coordinate dot product",
                rows=ridge_cca_rows,
                status="fails_controls"
                if any(row["best_control_accuracy"] > row["target_accuracy"] + 0.03 for row in ridge_cca_rows)
                else "passes_strict_controls"
                if all(row["pass_gate"] for row in ridge_cca_rows)
                else "fails_gate",
                claim_use="Measured non-orthogonal linear shared-subspace packet baseline.",
                next_action="If partial or clean, compare directly against RR/LSTIRP; if control-leaky, keep as unsafe common-basis row.",
                sources=(
                    "common-basis falsification artifact; https://arxiv.org/abs/1706.05806; "
                    "https://arxiv.org/abs/1806.05759; https://arxiv.org/abs/1905.00414"
                ),
                record_bytes=record_bytes,
            )
        )
    if relative_rows:
        if all(row["pass_gate"] for row in relative_rows):
            relative_status = "passes_strict_controls"
        elif any(row["pass_gate"] for row in relative_rows) and all(
            row["best_control_accuracy"] <= row["target_accuracy"] + 0.03 for row in relative_rows
        ):
            relative_status = "partial_clean_competitor"
        else:
            relative_status = "fails_controls"
        table_rows.append(
            _aggregate_measured_group(
                method_id="relative_representations_anchor_dot",
                method="Relative Representations anchor-coordinate dot product",
                rows=relative_rows,
                status=relative_status,
                claim_use="Measured anchor-coordinate common-basis competitor; strong in core-to-holdout and same-family, weak in holdout-to-core.",
                next_action=(
                    "If failed, cite as the first real RR-style same-slice falsification; if passed, promote as competitor."
                ),
                sources="common-basis falsification artifact; https://arxiv.org/abs/2209.15430",
                record_bytes=record_bytes,
            )
        )
    else:
        table_rows.append(
            _pending_row(
                method_id="relative_representations",
                method="Relative Representations / anchor-coordinate baseline",
                category="pending_common_basis_baseline",
                claim_use="Closest anchor-relative latent-communication prior.",
                next_action="Implement anchor-similarity coordinates with source-destroying controls on n512.",
                sources="https://arxiv.org/abs/2209.15430",
            )
        )
    table_rows.extend(
        [
        *(
            [
                _aggregate_measured_group(
                    method_id="lstirp_inverse_relative_dot",
                    method="LSTIRP-lite inverse-relative dot product",
                    rows=lstirp_rows,
                    status="fails_controls"
                    if any(row["best_control_accuracy"] > row["target_accuracy"] + 0.03 for row in lstirp_rows)
                    else "passes_strict_controls"
                    if all(row["pass_gate"] for row in lstirp_rows)
                    else "fails_gate",
                    claim_use="Measured inverse-relative public-calibration translation baseline.",
                    next_action="If clean and all-direction, promote as closest RR/LSTIRP competitor; if leaky, keep as unsafe relative-translation row.",
                    sources="common-basis falsification artifact; https://arxiv.org/abs/2406.15057",
                    record_bytes=record_bytes,
                )
            ]
            if lstirp_rows
            else []
        ),
        *(
            [
                _aggregate_measured_group(
                    method_id="lstirp_inverse_relative_residual_norm_stack",
                    method="LSTIRP-lite inverse-relative residual chart with row/payload normalization",
                    rows=lstirp_stack_rows,
                    status="fails_controls"
                    if any(row["best_control_accuracy"] > row["target_accuracy"] + 0.03 for row in lstirp_stack_rows)
                    else "passes_strict_controls"
                    if all(row["pass_gate"] for row in lstirp_stack_rows)
                    else "fails_gate",
                    claim_use="Stack diagnostic: inverse-relative reconstruction plus local residual normalization.",
                    next_action="If clean and all-direction, this becomes the strongest relative-space threat to the live method.",
                    sources="common-basis falsification artifact; https://arxiv.org/abs/2406.15057",
                    record_bytes=record_bytes,
                )
            ]
            if lstirp_stack_rows
            else []
        ),
        *(
            [
                _aggregate_measured_group(
                    method_id="ridge_cca_residual_norm_stack",
                    method="ridge CCA/SVCCA-style residual chart with row/payload normalization",
                    rows=ridge_cca_stack_rows,
                    status="fails_controls"
                    if any(row["best_control_accuracy"] > row["target_accuracy"] + 0.03 for row in ridge_cca_stack_rows)
                    else "passes_strict_controls"
                    if all(row["pass_gate"] for row in ridge_cca_stack_rows)
                    else "fails_gate",
                    claim_use="Stack diagnostic: CCA canonical coordinates plus local residual normalization.",
                    next_action="If clean and all-direction, this becomes the strongest common-basis threat to the live method.",
                    sources="common-basis falsification artifact; https://arxiv.org/abs/1706.05806",
                    record_bytes=record_bytes,
                )
            ]
            if ridge_cca_stack_rows
            else []
        ),
        *(
            [
                _aggregate_measured_group(
                    method_id="sinkhorn_ot_dot",
                    method="Sinkhorn OT public-calibration transport dot product",
                    rows=sinkhorn_ot_rows,
                    status="fails_controls"
                    if any(row["best_control_accuracy"] > row["target_accuracy"] + 0.03 for row in sinkhorn_ot_rows)
                    else "passes_strict_controls"
                    if all(row["pass_gate"] for row in sinkhorn_ot_rows)
                    else "fails_gate",
                    claim_use="Measured feature-cost transport baseline over public atom-axis calibration supports.",
                    next_action="If clean and all-direction, compare directly against GW and candidate-local residual; otherwise keep as the feature-transport falsification row.",
                    sources=(
                        "common-basis falsification artifact; "
                        "https://papers.neurips.cc/paper/4927-sinkhorn-distances-lightspeed-computation-of-optimal-transport"
                    ),
                    record_bytes=record_bytes,
                )
            ]
            if sinkhorn_ot_rows
            else []
        ),
        *(
            [
                _aggregate_measured_group(
                    method_id="sinkhorn_ot_residual_norm_stack",
                    method="Sinkhorn OT transport with residual chart normalization",
                    rows=sinkhorn_ot_stack_rows,
                    status="fails_controls"
                    if any(
                        row["best_control_accuracy"] > row["target_accuracy"] + 0.03
                        for row in sinkhorn_ot_stack_rows
                    )
                    else "passes_strict_controls"
                    if all(row["pass_gate"] for row in sinkhorn_ot_stack_rows)
                    else "fails_gate",
                    claim_use="Stack diagnostic: feature-cost OT transport plus candidate-pool residual normalization.",
                    next_action="If clean and all-direction, promote as a stronger systems-light transport method; if leaky, prune this stack.",
                    sources=(
                        "common-basis falsification artifact; "
                        "https://papers.neurips.cc/paper/4927-sinkhorn-distances-lightspeed-computation-of-optimal-transport"
                    ),
                    record_bytes=record_bytes,
                )
            ]
            if sinkhorn_ot_stack_rows
            else []
        ),
        *(
            [
                _aggregate_measured_group(
                    method_id="gromov_wasserstein_dot",
                    method="Gromov-Wasserstein public-calibration transport dot product",
                    rows=gw_rows,
                    status="fails_controls"
                    if any(row["best_control_accuracy"] > row["target_accuracy"] + 0.03 for row in gw_rows)
                    else "passes_strict_controls"
                    if all(row["pass_gate"] for row in gw_rows)
                    else "fails_gate",
                    claim_use="Measured relational-geometry transport competitor for common-basis and gauge objections.",
                    next_action="If clean and all-direction, promote as the strongest basis-transport threat; otherwise cite as a controlled negative.",
                    sources=(
                        "common-basis falsification artifact; https://aclanthology.org/D18-1214/; "
                        "https://arxiv.org/abs/1805.11222"
                    ),
                    record_bytes=record_bytes,
                )
            ]
            if gw_rows
            else []
        ),
        *(
            [
                _aggregate_measured_group(
                    method_id="gromov_wasserstein_residual_norm_stack",
                    method="Gromov-Wasserstein transport with residual chart normalization",
                    rows=gw_stack_rows,
                    status="fails_controls"
                    if any(row["best_control_accuracy"] > row["target_accuracy"] + 0.03 for row in gw_stack_rows)
                    else "passes_strict_controls"
                    if all(row["pass_gate"] for row in gw_stack_rows)
                    else "fails_gate",
                    claim_use="Stack diagnostic: relational transport plus candidate-local residual normalization.",
                    next_action="If clean and all-direction, promote as a fourth positive method branch; if leaky, keep as the OT/GW falsification row.",
                    sources=(
                        "common-basis falsification artifact; https://aclanthology.org/D18-1214/; "
                        "https://arxiv.org/abs/1805.11222"
                    ),
                    record_bytes=record_bytes,
                )
            ]
            if gw_stack_rows
            else []
        ),
        *(
            [
                _aggregate_measured_group(
                    method_id="relative_anchor_residual_norm_stack",
                    method="relative-anchor residual chart with row/payload normalization",
                    rows=relative_stack_rows,
                    status="fails_controls"
                    if any(row["best_control_accuracy"] > row["target_accuracy"] + 0.03 for row in relative_stack_rows)
                    else "fails_gate",
                    claim_use="Stack diagnostic: RR coordinates plus local residual normalization.",
                    next_action="Prune this exact stack unless a better guard removes control leakage.",
                    sources="common-basis falsification artifact",
                    record_bytes=record_bytes,
                )
            ]
            if relative_stack_rows
            else []
        ),
        *(
            [
                _aggregate_measured_group(
                    method_id="relative_anchor_innovation_residual_norm_stack",
                    method="relative-anchor innovation residual chart",
                    rows=relative_innovation_rows,
                    status="fails_controls"
                    if any(
                        row["best_control_accuracy"] > row["target_accuracy"] + 0.03
                        for row in relative_innovation_rows
                    )
                    else "passes_strict_controls"
                    if all(row["pass_gate"] for row in relative_innovation_rows)
                    else "fails_gate",
                    claim_use="RR repair probe: subtract receiver-local anchor prior before scoring.",
                    next_action="Pruned for ICLR main method unless a future guard fixes holdout-to-core and control leakage.",
                    sources="common-basis falsification artifact; https://arxiv.org/abs/2209.15430",
                    record_bytes=record_bytes,
                )
            ]
            if relative_innovation_rows
            else []
        ),
        *(
            [
                _aggregate_measured_group(
                    method_id="relative_anchor_rank_innovation_residual_norm_stack",
                    method="ranked relative-anchor innovation residual chart",
                    rows=relative_rank_innovation_rows,
                    status="fails_controls"
                    if any(
                        row["best_control_accuracy"] > row["target_accuracy"] + 0.03
                        for row in relative_rank_innovation_rows
                    )
                    else "passes_strict_controls"
                    if all(row["pass_gate"] for row in relative_rank_innovation_rows)
                    else "fails_gate",
                    claim_use="RR repair probe: use only anchor-similarity ranks before local residual scoring.",
                    next_action="Pruned: rank invariance removes too much signal and does not rescue holdout-to-core.",
                    sources="common-basis falsification artifact; https://arxiv.org/abs/2209.15430",
                    record_bytes=record_bytes,
                )
            ]
            if relative_rank_innovation_rows
            else []
        ),
        _aggregate_measured_group(
            method_id="candidate_local_residual_no_norm",
            method="candidate-local residual chart without row/payload normalization",
            rows=diagnostic_rows,
            status="fails_controls",
            claim_use="Mechanism ablation: local centering alone is insufficient.",
            next_action="Keep normalization as part of the promoted method, not cosmetic cleanup.",
            sources="common-basis falsification artifact",
            record_bytes=record_bytes,
        ),
        _aggregate_condition(
            method_id="oracle_candidate_atoms",
            method="oracle learned candidate-atom packet",
            rows=live_rows,
            metric_key="oracle_accuracy",
            category="measured_upper_bound",
            status="upper_bound",
            source_text_exposed=False,
            source_kv_exposed=False,
            payload_bytes=8,
            record_bytes=record_bytes,
            claim_use="Headroom bound for better packet selection/encoding.",
            next_action="Use only as oracle headroom, never as a method row.",
            sources="current n512 live summaries",
        ),
        *(
            []
            if lstirp_rows
            else [
                _pending_row(
                    method_id="lstirp_inverse_relative",
                    method="LSTIRP / inverse-relative public transport baseline",
                    category="pending_transport_baseline",
                    claim_use="Closest relative-space translation competitor for the measured RR partial row.",
                    next_action="Run inverse-relative projection with the same source-private packet and controls.",
                    sources="https://arxiv.org/abs/2406.15057; https://openreview.net/forum?id=SrC-nwieGJ",
                )
            ]
        ),
        *(
            []
            if sinkhorn_ot_rows or gw_rows
            else [
                _pending_row(
                    method_id="ot_lstirp_gw",
                    method="OT / Gromov-Wasserstein public transport baseline",
                    category="pending_transport_baseline",
                    claim_use="Nonlinear/local transport competitor for gauge and basis objections.",
                    next_action="Run small candidate-pool OT/GW transport with deranged/permuted controls.",
                    sources="https://arxiv.org/abs/2406.15057; https://aclanthology.org/D18-1214/",
                )
            ]
        ),
        _pending_row(
            method_id="c2c_cache_fuser",
            method="C2C cache-to-cache fuser or scoped cache-access proxy",
            category="pending_cache_communication_baseline",
            claim_use="Closest direct KV-cache communication competitor.",
            next_action="Run native/proxy row only if source KV exposure and byte accounting are explicit.",
            sources="https://arxiv.org/abs/2510.03215",
            source_text_exposed=False,
            source_kv_exposed=True,
        ),
        _pending_row(
            method_id="kvcomm_qkvcomm",
            method="KVComm / Q-KVComm selective or compressed KV sharing",
            category="pending_cache_communication_baseline",
            claim_use="Closest multi-agent KV-sharing systems competitor.",
            next_action="Add supported-task smoke or same-slice proxy with source KV exposure flagged.",
            sources="https://arxiv.org/abs/2510.03346; https://arxiv.org/abs/2512.17914",
            source_text_exposed=False,
            source_kv_exposed=True,
        ),
        _pending_row(
            method_id="turboquant_kv_floor",
            method="TurboQuant / KIVI / KVQuant / CacheGen byte-floor systems rows",
            category="pending_kv_compression_floor",
            claim_use="Systems byte/latency caveat, not privacy-equivalent communication.",
            next_action="Report KV-state byte floors separately from source-private packet bytes.",
            sources=(
                "https://arxiv.org/abs/2504.19874; https://arxiv.org/abs/2402.02750; "
                "https://arxiv.org/abs/2401.18079; https://arxiv.org/abs/2310.07240"
            ),
            source_text_exposed=False,
            source_kv_exposed=True,
        ),
        ]
    )
    pending_rows = [row for row in table_rows if row["status"] == "pending_required_for_iclr"]
    measured_rows = [row for row in table_rows if row["same_slice_measured"]]
    headline = {
        "common_basis_pass_gate": bool(common["headline"]["pass_gate"]),
        "live_pass_rows": common["headline"]["live_pass_rows"],
        "live_rows": common["headline"]["live_rows"],
        "global_common_basis_pass_rows": common["headline"]["global_common_basis_pass_rows"],
        "global_common_basis_rows": common["headline"]["global_common_basis_rows"],
        "procrustes_common_basis_pass_rows": common["headline"].get("procrustes_common_basis_pass_rows", 0),
        "procrustes_common_basis_rows": common["headline"].get("procrustes_common_basis_rows", 0),
        "ridge_cca_common_basis_pass_rows": common["headline"].get("ridge_cca_common_basis_pass_rows", 0),
        "ridge_cca_common_basis_rows": common["headline"].get("ridge_cca_common_basis_rows", 0),
        "ridge_cca_stack_pass_rows": common["headline"].get("ridge_cca_stack_pass_rows", 0),
        "ridge_cca_stack_rows": common["headline"].get("ridge_cca_stack_rows", 0),
        "lstirp_pass_rows": common["headline"].get("lstirp_pass_rows", 0),
        "lstirp_rows": common["headline"].get("lstirp_rows", 0),
        "lstirp_stack_pass_rows": common["headline"].get("lstirp_stack_pass_rows", 0),
        "lstirp_stack_rows": common["headline"].get("lstirp_stack_rows", 0),
        "sinkhorn_ot_pass_rows": common["headline"].get("sinkhorn_ot_pass_rows", 0),
        "sinkhorn_ot_rows": common["headline"].get("sinkhorn_ot_rows", 0),
        "sinkhorn_ot_stack_pass_rows": common["headline"].get("sinkhorn_ot_stack_pass_rows", 0),
        "sinkhorn_ot_stack_rows": common["headline"].get("sinkhorn_ot_stack_rows", 0),
        "gw_pass_rows": common["headline"].get("gw_pass_rows", 0),
        "gw_rows": common["headline"].get("gw_rows", 0),
        "gw_stack_pass_rows": common["headline"].get("gw_stack_pass_rows", 0),
        "gw_stack_rows": common["headline"].get("gw_stack_rows", 0),
        "relative_anchor_pass_rows": common["headline"].get("relative_anchor_pass_rows", 0),
        "relative_anchor_rows": common["headline"].get("relative_anchor_rows", 0),
        "relative_anchor_stack_pass_rows": common["headline"].get("relative_anchor_stack_pass_rows", 0),
        "relative_anchor_stack_rows": common["headline"].get("relative_anchor_stack_rows", 0),
        "relative_anchor_innovation_pass_rows": common["headline"].get("relative_anchor_innovation_pass_rows", 0),
        "relative_anchor_innovation_rows": common["headline"].get("relative_anchor_innovation_rows", 0),
        "relative_anchor_rank_innovation_pass_rows": common["headline"].get(
            "relative_anchor_rank_innovation_pass_rows", 0
        ),
        "relative_anchor_rank_innovation_rows": common["headline"].get("relative_anchor_rank_innovation_rows", 0),
        "diagnostic_pass_rows": common["headline"]["diagnostic_pass_rows"],
        "diagnostic_rows": common["headline"]["diagnostic_rows"],
        "measured_table_rows": len(measured_rows),
        "pending_required_rows": len(pending_rows),
        "iclr_competitor_complete": False,
        "colm_competitor_table_ready": bool(live_rows and measured_rows),
        "packet_record_bytes": record_bytes,
        "source_text_exposed": system_headline.get("source_text_exposed"),
        "source_kv_exposed": system_headline.get("source_kv_exposed"),
        "resident_sparse_decode_p50_us": system_headline.get("max_resident_sparse_decode_p50_us"),
    }
    payload = {
        "gate": "source_private_candidate_local_competitor_basis_table",
        "headline": headline,
        "rows": table_rows,
        "interpretation": (
            "The same-slice measured table now separates a passing candidate-local residual receiver from "
            "unsafe high-accuracy common-basis baselines, byte-matched controls, a measured Procrustes row, "
            "measured ridge CCA/SVCCA-style rows, measured LSTIRP-lite rows, and a measured Relative "
            "Representations-style anchor-coordinate competitor plus two guarded RR repair probes. Sinkhorn OT "
            "and GW rows are included when available as public-calibration transport baselines. It is COLM-useful, "
            "but not ICLR-complete because C2C/KVComm and KV byte-floor rows remain pending."
        ),
    }
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "candidate_local_competitor_basis_table.json"
    csv_path = output_dir / "candidate_local_competitor_basis_table.csv"
    md_path = output_dir / "candidate_local_competitor_basis_table.md"
    manifest_path = output_dir / "manifest.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in table_rows:
            writer.writerow({key: _fmt(row.get(key)) for key in CSV_COLUMNS})
    _write_markdown(md_path, payload)
    manifest = {
        "artifacts": [json_path.name, csv_path.name, md_path.name, manifest_path.name],
        "artifact_sha256": {
            json_path.name: _sha256_file(json_path),
            csv_path.name: _sha256_file(csv_path),
            md_path.name: _sha256_file(md_path),
        },
        "headline": headline,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Candidate-Local Competitor/Basis Table Manifest",
                "",
                f"- common-basis pass gate: `{headline['common_basis_pass_gate']}`",
                f"- measured rows: `{headline['measured_table_rows']}`",
                f"- pending ICLR-required rows: `{headline['pending_required_rows']}`",
                f"- ICLR competitor complete: `{headline['iclr_competitor_complete']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    headline = payload["headline"]
    lines = [
        "# Candidate-Local Competitor/Basis Table",
        "",
        "This table converts the n512 candidate-local common-basis falsification into",
        "a reviewer-facing competitor ledger. Measured same-slice rows are separated",
        "from baselines that remain required before an ICLR-complete claim.",
        "",
        "## Headline",
        "",
        f"- common-basis pass gate: `{headline['common_basis_pass_gate']}`",
        (
            "- live candidate-local rows: "
            f"`{headline['live_pass_rows']}/{headline['live_rows']}` pass"
        ),
        (
            "- global common-basis rows: "
            f"`{headline['global_common_basis_pass_rows']}/{headline['global_common_basis_rows']}` pass"
        ),
        (
            "- Procrustes common-basis rows: "
            f"`{headline['procrustes_common_basis_pass_rows']}/{headline['procrustes_common_basis_rows']}` pass"
        ),
        (
            "- ridge CCA common-basis rows: "
            f"`{headline['ridge_cca_common_basis_pass_rows']}/{headline['ridge_cca_common_basis_rows']}` pass"
        ),
        (
            "- ridge CCA local-stack rows: "
            f"`{headline['ridge_cca_stack_pass_rows']}/{headline['ridge_cca_stack_rows']}` pass"
        ),
        (
            "- LSTIRP-lite rows: "
            f"`{headline['lstirp_pass_rows']}/{headline['lstirp_rows']}` pass"
        ),
        (
            "- LSTIRP-lite local-stack rows: "
            f"`{headline['lstirp_stack_pass_rows']}/{headline['lstirp_stack_rows']}` pass"
        ),
        (
            "- Sinkhorn OT rows: "
            f"`{headline['sinkhorn_ot_pass_rows']}/{headline['sinkhorn_ot_rows']}` pass"
        ),
        (
            "- Sinkhorn OT local-stack rows: "
            f"`{headline['sinkhorn_ot_stack_pass_rows']}/{headline['sinkhorn_ot_stack_rows']}` pass"
        ),
        (
            "- GW transport rows: "
            f"`{headline['gw_pass_rows']}/{headline['gw_rows']}` pass"
        ),
        (
            "- GW local-stack rows: "
            f"`{headline['gw_stack_pass_rows']}/{headline['gw_stack_rows']}` pass"
        ),
        (
            "- relative-anchor rows: "
            f"`{headline['relative_anchor_pass_rows']}/{headline['relative_anchor_rows']}` pass"
        ),
        (
            "- relative-anchor local-stack rows: "
            f"`{headline['relative_anchor_stack_pass_rows']}/{headline['relative_anchor_stack_rows']}` pass"
        ),
        (
            "- relative-anchor innovation-stack rows: "
            f"`{headline['relative_anchor_innovation_pass_rows']}/{headline['relative_anchor_innovation_rows']}` pass"
        ),
        (
            "- relative-anchor rank-innovation rows: "
            f"`{headline['relative_anchor_rank_innovation_pass_rows']}/{headline['relative_anchor_rank_innovation_rows']}` pass"
        ),
        (
            "- no-normalization local rows: "
            f"`{headline['diagnostic_pass_rows']}/{headline['diagnostic_rows']}` pass"
        ),
        f"- measured table rows: `{headline['measured_table_rows']}`",
        f"- pending ICLR-required rows: `{headline['pending_required_rows']}`",
        f"- ICLR competitor complete: `{headline['iclr_competitor_complete']}`",
        "",
        "## Rows",
        "",
        "| Method | Status | Same-slice | Rows | Pass | Acc range | Best ctrl max | Exposure | Claim use |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in payload["rows"]:
        acc = ""
        if row["matched_accuracy_min"] is not None:
            acc = f"{row['matched_accuracy_min']:.3f}-{row['matched_accuracy_max']:.3f}"
        best = "" if row["best_control_accuracy_max"] is None else f"{row['best_control_accuracy_max']:.3f}"
        exposure = (
            f"text={_fmt(row['source_text_exposed'])}, kv={_fmt(row['source_kv_exposed'])}"
        )
        lines.append(
            "| {method} | `{status}` | `{same}` | {rows} | {passes} | {acc} | {best} | {exposure} | {claim} |".format(
                method=row["method"],
                status=row["status"],
                same=row["same_slice_measured"],
                rows="" if row["rows"] is None else row["rows"],
                passes="" if row["pass_rows"] is None else row["pass_rows"],
                acc=acc,
                best=best,
                exposure=exposure,
                claim=row["claim_use"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
            "",
            "Layman explanation: the current method has now beaten the simplest fake-clue",
            "and shared-dictionary explanations on the same examples, but the paper still",
            "owes stronger outside baselines before it can claim ICLR-level completeness.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--common-basis", type=pathlib.Path, default=DEFAULT_COMMON_BASIS)
    parser.add_argument("--systems", type=pathlib.Path, default=DEFAULT_SYSTEMS)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_competitor_basis_table(
        common_basis_path=args.common_basis,
        systems_path=args.systems,
        output_dir=args.output_dir,
    )
    print(json.dumps({"output_dir": str(_resolve(args.output_dir)), "headline": payload["headline"]}, indent=2))


if __name__ == "__main__":
    main()
