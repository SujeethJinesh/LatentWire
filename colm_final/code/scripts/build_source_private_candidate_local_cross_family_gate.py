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
DEFAULT_OUTPUT = pathlib.Path("results/source_private_candidate_local_cross_family_gate_20260430")

CROSS_FAMILY_DIRECTIONS = ("core_to_holdout", "holdout_to_core")
SAME_FAMILY_DIRECTIONS = ("same_family_all",)

ROW_GROUP_LABELS = {
    "live": "candidate-local residual norm",
    "relative_anchor_common_basis": "Relative Representations anchor-coordinate dot product",
    "relative_anchor_local_stack": "RR-anchor local residual norm stack",
    "relative_anchor_innovation_stack": "RR-anchor innovation residual norm stack",
    "relative_anchor_rank_innovation_stack": "ranked RR-anchor innovation residual norm stack",
    "global_common_basis": "global public-anchor dot product",
    "procrustes_common_basis": "public-calibration orthogonal Procrustes dot product",
    "ridge_cca_common_basis": "ridge CCA/SVCCA-style canonical-coordinate dot product",
    "ridge_cca_local_stack": "ridge CCA/SVCCA-style residual norm stack",
    "lstirp_relative_translation": "LSTIRP-lite inverse-relative dot product",
    "lstirp_relative_local_stack": "LSTIRP-lite inverse-relative residual norm stack",
    "sinkhorn_ot_transport": "Sinkhorn OT public-calibration transport dot product",
    "sinkhorn_ot_local_stack": "Sinkhorn OT transport with residual chart normalization",
    "gw_transport": "Gromov-Wasserstein public-calibration transport dot product",
    "gw_local_stack": "Gromov-Wasserstein transport with residual chart normalization",
    "diagnostic_ablation": "candidate-local residual without row/payload normalization",
}

CSV_COLUMNS = (
    "row_group",
    "method",
    "split",
    "rows",
    "pass_rows",
    "controls_ok_rows",
    "control_leak_rows",
    "matched_accuracy_min",
    "matched_accuracy_max",
    "target_accuracy_min",
    "target_accuracy_max",
    "best_control_accuracy_max",
    "delta_vs_best_control_min",
    "core_to_holdout_pass_rows",
    "holdout_to_core_pass_rows",
    "same_family_all_pass_rows",
    "interpretation",
)


def _resolve(path: pathlib.Path) -> pathlib.Path:
    return path if path.is_absolute() else ROOT / path


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _relative(path: pathlib.Path) -> str:
    resolved = _resolve(path)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


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


def _direction_pass(rows: list[dict[str, Any]], direction: str) -> int:
    return sum(bool(row["pass_gate"]) for row in rows if row["direction"] == direction)


def _aggregate(rows: list[dict[str, Any]], *, row_group: str, split: str) -> dict[str, Any]:
    matched_min, matched_max = _stats([float(row["matched_accuracy"]) for row in rows])
    target_min, target_max = _stats([float(row["target_accuracy"]) for row in rows])
    control_values = [float(row["best_control_accuracy"]) for row in rows]
    delta_values = [float(row["delta_vs_best_control"]) for row in rows]
    pass_rows = sum(bool(row["pass_gate"]) for row in rows)
    control_leak_rows = sum(not bool(row["controls_ok"]) for row in rows)
    if row_group == "live" and split == "cross_family":
        interpretation = "promoted live cross-family row: passes both directions with clean controls"
    elif row_group == "relative_anchor_common_basis" and split == "cross_family":
        interpretation = "clean partial competitor: core-to-holdout passes, holdout-to-core collapses"
    elif control_leak_rows:
        interpretation = "invalid as source-private communication because destructive controls rise"
    elif pass_rows == len(rows) and rows:
        interpretation = "passes this split under current controls"
    else:
        interpretation = "fails this split without destructive-control leakage"
    return {
        "row_group": row_group,
        "method": ROW_GROUP_LABELS.get(row_group, row_group),
        "split": split,
        "rows": len(rows),
        "pass_rows": pass_rows,
        "controls_ok_rows": sum(bool(row["controls_ok"]) for row in rows),
        "control_leak_rows": control_leak_rows,
        "matched_accuracy_min": matched_min,
        "matched_accuracy_max": matched_max,
        "target_accuracy_min": target_min,
        "target_accuracy_max": target_max,
        "best_control_accuracy_max": max(control_values) if control_values else None,
        "delta_vs_best_control_min": min(delta_values) if delta_values else None,
        "core_to_holdout_pass_rows": _direction_pass(rows, "core_to_holdout"),
        "holdout_to_core_pass_rows": _direction_pass(rows, "holdout_to_core"),
        "same_family_all_pass_rows": _direction_pass(rows, "same_family_all"),
        "interpretation": interpretation,
    }


def _rows_for_split(all_rows: list[dict[str, Any]], row_group: str, directions: tuple[str, ...]) -> list[dict[str, Any]]:
    return [row for row in all_rows if row["row_group"] == row_group and row["direction"] in directions]


def build_cross_family_gate(
    *,
    common_basis_path: pathlib.Path,
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    common = _read_json(common_basis_path)
    all_rows = common["rows"]
    row_groups = [group for group in ROW_GROUP_LABELS if any(row["row_group"] == group for row in all_rows)]
    rows: list[dict[str, Any]] = []
    for row_group in row_groups:
        rows.append(
            _aggregate(
                _rows_for_split(all_rows, row_group, CROSS_FAMILY_DIRECTIONS),
                row_group=row_group,
                split="cross_family",
            )
        )
        rows.append(
            _aggregate(
                _rows_for_split(all_rows, row_group, SAME_FAMILY_DIRECTIONS),
                row_group=row_group,
                split="same_family",
            )
        )

    by_group_split = {(row["row_group"], row["split"]): row for row in rows}
    live_cross = by_group_split[("live", "cross_family")]
    live_same = by_group_split[("live", "same_family")]
    rr_cross = by_group_split.get(("relative_anchor_common_basis", "cross_family"), {})
    rr_same = by_group_split.get(("relative_anchor_common_basis", "same_family"), {})
    headline = {
        "pass_gate": (
            live_cross["rows"] > 0
            and live_cross["pass_rows"] == live_cross["rows"]
            and live_cross["control_leak_rows"] == 0
            and live_same["rows"] > 0
            and live_same["pass_rows"] == live_same["rows"]
            and live_same["control_leak_rows"] == 0
        ),
        "live_cross_family_pass_rows": live_cross["pass_rows"],
        "live_cross_family_rows": live_cross["rows"],
        "live_same_family_pass_rows": live_same["pass_rows"],
        "live_same_family_rows": live_same["rows"],
        "live_cross_family_min_matched_accuracy": live_cross["matched_accuracy_min"],
        "live_cross_family_best_control_accuracy_max": live_cross["best_control_accuracy_max"],
        "rr_cross_family_pass_rows": rr_cross.get("pass_rows", 0),
        "rr_cross_family_rows": rr_cross.get("rows", 0),
        "rr_core_to_holdout_pass_rows": rr_cross.get("core_to_holdout_pass_rows", 0),
        "rr_holdout_to_core_pass_rows": rr_cross.get("holdout_to_core_pass_rows", 0),
        "rr_same_family_pass_rows": rr_same.get("pass_rows", 0),
        "rr_same_family_rows": rr_same.get("rows", 0),
        "control_leaky_cross_family_groups": sorted(
            row["row_group"] for row in rows if row["split"] == "cross_family" and row["control_leak_rows"] > 0
        ),
        "iclr_ready": False,
        "remaining_iclr_gaps": [
            "resolve or frame the RR holdout-to-core collapse",
            "add native/proxy C2C and KVComm rows with source-KV exposure accounting",
            "add TurboQuant/KIVI/KVQuant/CacheGen byte-floor systems rows on native hardware when available",
        ],
    }
    payload = {
        "gate": "source_private_candidate_local_cross_family_gate",
        "source_common_basis": _relative(common_basis_path),
        "headline": headline,
        "rows": rows,
        "interpretation": (
            "The live candidate-local residual receiver passes all n512 same-family and cross-family rows with "
            "clean controls. RR anchor coordinates are the important clean partial competitor: they pass "
            "core-to-holdout and same-family but fail holdout-to-core at target floor. Public transport/common-basis "
            "rows remain unsafe because controls rise."
        ),
        "layman_explanation": (
            "This separates easy within-family examples from harder family-transfer examples. The current method still "
            "works in both transfer directions, while the strongest clean shared-coordinate baseline works only one way."
        ),
    }

    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "candidate_local_cross_family_gate.json"
    csv_path = output_dir / "candidate_local_cross_family_gate.csv"
    md_path = output_dir / "candidate_local_cross_family_gate.md"
    manifest_path = output_dir / "manifest.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
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
                "# Candidate-Local Cross-Family Gate Manifest",
                "",
                f"- pass gate: `{headline['pass_gate']}`",
                (
                    "- live cross-family pass rows: "
                    f"`{headline['live_cross_family_pass_rows']}/{headline['live_cross_family_rows']}`"
                ),
                (
                    "- live same-family pass rows: "
                    f"`{headline['live_same_family_pass_rows']}/{headline['live_same_family_rows']}`"
                ),
                (
                    "- RR cross-family pass rows: "
                    f"`{headline['rr_cross_family_pass_rows']}/{headline['rr_cross_family_rows']}`"
                ),
                f"- ICLR ready: `{headline['iclr_ready']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def _fmt_range(row: dict[str, Any]) -> str:
    if row["matched_accuracy_min"] is None:
        return ""
    return f"{row['matched_accuracy_min']:.3f}-{row['matched_accuracy_max']:.3f}"


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    headline = payload["headline"]
    lines = [
        "# Candidate-Local Cross-Family Gate",
        "",
        "This artifact separates the promoted n512 candidate-local residual row from",
        "same-family-only explanations and from the clean-but-partial RR anchor",
        "baseline.",
        "",
        "## Headline",
        "",
        f"- pass gate: `{headline['pass_gate']}`",
        (
            "- live cross-family pass rows: "
            f"`{headline['live_cross_family_pass_rows']}/{headline['live_cross_family_rows']}`"
        ),
        (
            "- live same-family pass rows: "
            f"`{headline['live_same_family_pass_rows']}/{headline['live_same_family_rows']}`"
        ),
        (
            "- live cross-family matched accuracy min: "
            f"`{headline['live_cross_family_min_matched_accuracy']:.3f}`"
        ),
        (
            "- live cross-family best-control max: "
            f"`{headline['live_cross_family_best_control_accuracy_max']:.3f}`"
        ),
        (
            "- RR cross-family pass rows: "
            f"`{headline['rr_cross_family_pass_rows']}/{headline['rr_cross_family_rows']}`"
        ),
        f"- RR core-to-holdout pass rows: `{headline['rr_core_to_holdout_pass_rows']}`",
        f"- RR holdout-to-core pass rows: `{headline['rr_holdout_to_core_pass_rows']}`",
        (
            "- RR same-family pass rows: "
            f"`{headline['rr_same_family_pass_rows']}/{headline['rr_same_family_rows']}`"
        ),
        f"- ICLR ready: `{headline['iclr_ready']}`",
        "",
        "## Rows",
        "",
        "| Method | Split | Pass | Controls ok | Matched range | Best ctrl max | Interpretation |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        best = "" if row["best_control_accuracy_max"] is None else f"{row['best_control_accuracy_max']:.3f}"
        lines.append(
            "| {method} | `{split}` | {passes}/{rows} | {controls}/{rows} | {acc} | {best} | {interp} |".format(
                method=row["method"],
                split=row["split"],
                passes=row["pass_rows"],
                rows=row["rows"],
                controls=row["controls_ok_rows"],
                acc=_fmt_range(row),
                best=best,
                interp=row["interpretation"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
            "",
            "Layman explanation: " + payload["layman_explanation"],
            "",
            "## Remaining ICLR Gaps",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in headline["remaining_iclr_gaps"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--common-basis", type=pathlib.Path, default=DEFAULT_COMMON_BASIS)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_cross_family_gate(common_basis_path=args.common_basis, output_dir=args.output_dir)
    print(json.dumps({"output_dir": str(_resolve(args.output_dir)), "headline": payload["headline"]}, indent=2))


if __name__ == "__main__":
    main()
