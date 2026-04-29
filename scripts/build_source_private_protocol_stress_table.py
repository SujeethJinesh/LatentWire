from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row(
    *,
    family: str,
    stressor: str,
    surface: str,
    status: str,
    n: int | None,
    bytes_: float | None,
    accuracy: float | None,
    target_accuracy: float | None,
    best_control_accuracy: float | None,
    exact_or_public_parity: bool | None,
    note: str,
) -> dict[str, Any]:
    return {
        "family": family,
        "stressor": stressor,
        "surface": surface,
        "status": status,
        "n": n,
        "mean_payload_bytes": bytes_,
        "accuracy": accuracy,
        "target_accuracy": target_accuracy,
        "best_control_accuracy": best_control_accuracy,
        "delta_vs_target": None if accuracy is None or target_accuracy is None else accuracy - target_accuracy,
        "delta_vs_best_control": None
        if accuracy is None or best_control_accuracy is None
        else accuracy - best_control_accuracy,
        "exact_or_public_parity": exact_or_public_parity,
        "note": note,
    }


def _codebook_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    summary = _read_json(path)
    rows: list[dict[str, Any]] = []
    for seed_row in summary["seed_rows"]:
        for budget in seed_row["budget_summaries"]:
            rows.append(
                _row(
                    family="deterministic diagnostic packet",
                    stressor="diagnostic codebook remap",
                    surface=f"seed {seed_row['seed']} budget {budget['budget_bytes']}",
                    status="pass" if budget["pass_gate"] else "fail",
                    n=seed_row["n"],
                    bytes_=float(budget["budget_bytes"]),
                    accuracy=budget["matched"],
                    target_accuracy=budget["best_no_source"],
                    best_control_accuracy=max(
                        budget["best_source_destroying_control"],
                        budget["best_reviewer_negative_control"],
                        budget["structured_json"],
                        budget["structured_free_text"],
                        budget["diag_masked_full_log"],
                    ),
                    exact_or_public_parity=summary["exact_id_parity_across_seeds"]
                    and summary["public_surface_parity_across_seeds"],
                    note=(
                        f"codebook={seed_row['codebook_sha256'][:8]}; "
                        f"preview={','.join(seed_row['diagnostic_preview'][:4])}"
                    ),
                )
            )
    return rows


def _slot_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    summary = _read_json(path)
    rows = []
    for item in summary["rows"]:
        if item["remap_slot_seed"] is None:
            continue
        rows.append(
            _row(
                family="learned scalar packet",
                stressor="slot-feature remap",
                surface=f"remap {item['remap_slot_seed']}",
                status="pass" if item["pass_gate"] else "fail",
                n=item["n"],
                bytes_=float(summary["budget_bytes"]),
                accuracy=item["scalar_accuracy"],
                target_accuracy=item["target_accuracy"],
                best_control_accuracy=item["best_strict_control_accuracy"],
                exact_or_public_parity=True,
                note=(
                    f"raw_sign={item['raw_sign_accuracy']:.3f}; "
                    f"ci_low_vs_target={item['paired_bootstrap']['target_only']['ci95_low']:.3f}"
                ),
            )
        )
    return rows


def _canonical_rasp_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    summary = _read_json(path)
    rows = []
    for item in summary["rows"]:
        best_control = item["relative_accuracy"] - item["relative_minus_best_strict_control"]
        rows.append(
            _row(
                family="canonical RASP relative-score packet",
                stressor="canonical candidate-order remap",
                surface=f"remap {item['remap_slot_seed']}",
                status="pass" if summary["pass_gate"] and item["pass_gate"] else "near-miss",
                n=512,
                bytes_=item["relative_payload_bytes"],
                accuracy=item["relative_accuracy"],
                target_accuracy=item["target_accuracy"],
                best_control_accuracy=best_control,
                exact_or_public_parity=True,
                note=(
                    f"scalar={item['scalar_accuracy']:.3f}; "
                    f"relative_minus_scalar={item['relative_minus_scalar']:.3f}; "
                    f"ci_low_vs_target={item['paired_bootstrap']['target_only']['ci95_low']:.3f}"
                ),
            )
        )
    return rows


def build_protocol_stress_table(*, output_dir: pathlib.Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    rows.extend(_codebook_rows(ROOT / "results/source_private_codebook_remap_gate_20260428/summary.json"))
    rows.extend(_slot_rows(ROOT / "results/source_private_slot_packet_bootstrap_20260429/summary.json"))
    rows.extend(
        _canonical_rasp_rows(
            ROOT / "results/source_private_relative_canonical_bootstrap_remap7_20260429/summary.json"
        )
    )
    pass_rows = [row for row in rows if row["status"] == "pass"]
    fail_like_rows = [row for row in rows if row["status"] != "pass"]
    payload = {
        "gate": "source_private_protocol_stress_table",
        "rows": rows,
        "headline": {
            "total_rows": len(rows),
            "pass_rows": len(pass_rows),
            "fail_or_near_miss_rows": len(fail_like_rows),
            "min_pass_delta_vs_target": min(row["delta_vs_target"] for row in pass_rows if row["delta_vs_target"] is not None),
            "max_pass_payload_bytes": max(row["mean_payload_bytes"] for row in pass_rows if row["mean_payload_bytes"] is not None),
            "stress_families": sorted({row["stressor"] for row in rows}),
        },
        "pass_gate": bool(pass_rows) and all(row["exact_or_public_parity"] is not False for row in rows),
        "open_gap": (
            "This table covers deterministic codebook remap, learned slot-feature remap, "
            "and canonical candidate-order remap. It does not yet cover learned target-decoder "
            "prompt paraphrases; that remains the next protocol-generalization stress."
        ),
    }
    (output_dir / "protocol_stress_table.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(output_dir / "protocol_stress_table.csv", rows)
    _write_markdown(output_dir / "protocol_stress_table.md", payload)
    manifest = {
        "artifacts": [
            "protocol_stress_table.json",
            "protocol_stress_table.md",
            "protocol_stress_table.csv",
            "manifest.json",
            "manifest.md",
        ],
        "artifact_sha256": {
            name: _sha256_file(output_dir / name)
            for name in ["protocol_stress_table.json", "protocol_stress_table.md", "protocol_stress_table.csv"]
        },
        "pass_gate": payload["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(["# Source-Private Protocol Stress Table Manifest", "", f"- rows: `{len(rows)}`", ""]),
        encoding="utf-8",
    )
    return payload


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# Source-Private Protocol Stress Table",
        "",
        f"- rows: `{h['total_rows']}`",
        f"- pass rows: `{h['pass_rows']}`",
        f"- fail / near-miss rows: `{h['fail_or_near_miss_rows']}`",
        f"- minimum passing delta vs target: `{h['min_pass_delta_vs_target']:.3f}`",
        f"- maximum passing payload bytes: `{h['max_pass_payload_bytes']:.1f}`",
        f"- stress families: `{', '.join(h['stress_families'])}`",
        "",
        "## Rows",
        "",
        "| Family | Stressor | Surface | Status | N | Bytes | Accuracy | Target | Best control | Delta vs target | Note |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['family']} | {row['stressor']} | {row['surface']} | `{row['status']}` | "
            f"{_fmt(row['n'], 0)} | {_fmt(row['mean_payload_bytes'], 1)} | {_fmt(row['accuracy'])} | "
            f"{_fmt(row['target_accuracy'])} | {_fmt(row['best_control_accuracy'])} | "
            f"{_fmt(row['delta_vs_target'])} | {row['note']} |"
        )
    lines.extend(["", "## Open Gap", "", payload["open_gap"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/source_private_protocol_stress_table_20260429"),
    )
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = build_protocol_stress_table(output_dir=output_dir)
    print(json.dumps({"output_dir": str(output_dir), "rows": len(payload["rows"])}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
