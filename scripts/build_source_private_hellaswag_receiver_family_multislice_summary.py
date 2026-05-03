from __future__ import annotations

"""Aggregate generic HellaSwag receiver-family packet gate slices."""

import argparse
import csv
import datetime as dt
import hashlib
import json
import pathlib
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_qwen_strict_packet_to_phi_receiver_multislice_20260503_validation1024_2048"
)
DEFAULT_ARTIFACTS = (
    pathlib.Path(
        "results/source_private_hellaswag_qwen_strict_packet_to_phi_receiver_20260503_validation1024_1536/"
        "receiver_gate/hellaswag_receiver_family_packet_gate.json"
    ),
    pathlib.Path(
        "results/source_private_hellaswag_qwen_strict_packet_to_phi_receiver_20260503_validation1536_2048/"
        "receiver_gate/hellaswag_receiver_family_packet_gate.json"
    ),
)


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path | str) -> str:
    resolved = _resolve(path)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _sha256_file(path: pathlib.Path | str) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: pathlib.Path | str) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _packet_contract(payload: dict[str, Any]) -> dict[str, Any]:
    packet_artifact = payload.get("source_packet", {}).get("artifact_path")
    if not packet_artifact:
        return {}
    artifact = _read_json(packet_artifact)
    return dict(artifact.get("packet_contract", {}))


def _slice_bounds(payload: dict[str, Any]) -> tuple[int | None, int | None]:
    slices = payload.get("target_scores", {}).get("slices", [])
    if slices:
        starts = [item.get("start") for item in slices if item.get("start") is not None]
        ends = [item.get("end") for item in slices if item.get("end") is not None]
        if starts and ends:
            return int(min(starts)), int(max(ends))
    return None, None


def _slice_row(path: pathlib.Path, payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("gate") != "source_private_hellaswag_receiver_family_packet_gate":
        raise ValueError(f"{path} is not a generic HellaSwag receiver-family packet gate")
    headline = payload["headline"]
    contract = _packet_contract(payload)
    start, end = _slice_bounds(payload)
    return {
        "artifact_path": _display_path(path),
        "pass_gate": bool(payload["pass_gate"]),
        "target_family_transfer_gate": bool(payload["target_family_transfer_gate"]),
        "receiver_improvement_gate": bool(payload["receiver_improvement_gate"]),
        "slice_start": start,
        "slice_end_exclusive": end,
        "row_count": int(headline["row_count"]),
        "train_rows": int(headline["train_rows"]),
        "eval_rows": int(headline["eval_rows"]),
        "source_family": str(headline["source_family"]),
        "target_family": str(headline["target_family"]),
        "target_only_eval_accuracy": float(headline["target_only_eval_accuracy"]),
        "packet_only_eval_accuracy": float(headline["packet_only_eval_accuracy"]),
        "receiver_eval_accuracy": float(headline["receiver_eval_accuracy"]),
        "target_or_packet_oracle_eval_accuracy": float(
            headline["target_or_packet_oracle_eval_accuracy"]
        ),
        "receiver_minus_target_only": float(headline["receiver_minus_target_only"]),
        "receiver_minus_packet_only": float(headline["receiver_minus_packet_only"]),
        "receiver_ci95_low_vs_target_only": float(headline["receiver_ci95_low_vs_target_only"]),
        "receiver_ci95_low_vs_packet_only": float(headline["receiver_ci95_low_vs_packet_only"]),
        "selected_receiver_kind": str(headline["selected_receiver_kind"]),
        "selected_receiver_train_accuracy": float(headline["selected_receiver_train_accuracy"]),
        "raw_payload_bytes": int(contract.get("raw_payload_bytes", -1)),
        "framed_record_bytes": int(contract.get("framed_record_bytes", -1)),
        "source_text_exposed": bool(contract.get("source_text_exposed", False)),
        "source_kv_exposed": bool(contract.get("source_kv_exposed", False)),
        "raw_hidden_vector_transmitted": bool(
            contract.get("raw_hidden_vector_transmitted", False)
        ),
        "raw_scores_transmitted": bool(contract.get("raw_scores_transmitted", False)),
        "control_count": len(payload.get("control_rows", [])),
    }


def _weighted(rows: list[dict[str, Any]], key: str) -> float:
    total = sum(int(row["eval_rows"]) for row in rows)
    if total <= 0:
        return 0.0
    return sum(float(row[key]) * int(row["eval_rows"]) for row in rows) / total


def _contiguous(rows: list[dict[str, Any]]) -> bool:
    clean = [row for row in rows if row["slice_start"] is not None and row["slice_end_exclusive"] is not None]
    if len(clean) != len(rows):
        return False
    ordered = sorted(clean, key=lambda row: int(row["slice_start"]))
    return all(
        int(left["slice_end_exclusive"]) == int(right["slice_start"])
        for left, right in zip(ordered, ordered[1:], strict=False)
    )


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Receiver-Family Multi-Slice Summary",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- source -> target: `{h['source_family']} -> {h['target_family']}`",
        f"- slices: `{h['slice_count']}`",
        f"- range: `{h['range_start']}:{h['range_end_exclusive']}`",
        f"- total eval rows: `{h['total_eval_rows']}`",
        f"- weighted target-only accuracy: `{h['weighted_target_only_eval_accuracy']:.6f}`",
        f"- weighted packet-only accuracy: `{h['weighted_packet_only_eval_accuracy']:.6f}`",
        f"- weighted receiver accuracy: `{h['weighted_receiver_eval_accuracy']:.6f}`",
        f"- weighted oracle accuracy: `{h['weighted_target_or_packet_oracle_eval_accuracy']:.6f}`",
        f"- packet minus target-only: `{h['weighted_packet_minus_target_only']:.6f}`",
        f"- receiver minus packet-only: `{h['weighted_receiver_minus_packet_only']:.6f}`",
        f"- oracle minus packet-only: `{h['weighted_oracle_minus_packet_only']:.6f}`",
        f"- receiver-improvement slices: `{h['receiver_improvement_slice_count']}/{h['slice_count']}`",
        f"- min receiver CI95 low vs packet-only: `{h['min_receiver_ci95_low_vs_packet_only']:.6f}`",
        f"- packet: `{h['raw_payload_bytes']}B` raw / `{h['framed_record_bytes']}B` framed",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary(
    *,
    output_dir: pathlib.Path = DEFAULT_OUTPUT,
    artifacts: tuple[pathlib.Path, ...] = DEFAULT_ARTIFACTS,
    run_date: str = "2026-05-03",
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        (_slice_row(path, _read_json(path)) for path in artifacts),
        key=lambda row: (-1 if row["slice_start"] is None else int(row["slice_start"])),
    )
    source_families = sorted({row["source_family"] for row in rows})
    target_families = sorted({row["target_family"] for row in rows})
    if len(source_families) != 1 or len(target_families) != 1:
        raise ValueError("all slices must share source and target families")
    weighted_target = _weighted(rows, "target_only_eval_accuracy")
    weighted_packet = _weighted(rows, "packet_only_eval_accuracy")
    weighted_receiver = _weighted(rows, "receiver_eval_accuracy")
    weighted_oracle = _weighted(rows, "target_or_packet_oracle_eval_accuracy")
    raw_bytes = sorted({row["raw_payload_bytes"] for row in rows})
    framed_bytes = sorted({row["framed_record_bytes"] for row in rows})
    headline = {
        "slice_count": len(rows),
        "range_start": rows[0]["slice_start"],
        "range_end_exclusive": rows[-1]["slice_end_exclusive"],
        "contiguous": _contiguous(rows),
        "total_rows": sum(int(row["row_count"]) for row in rows),
        "total_train_rows": sum(int(row["train_rows"]) for row in rows),
        "total_eval_rows": sum(int(row["eval_rows"]) for row in rows),
        "source_family": source_families[0],
        "target_family": target_families[0],
        "weighted_target_only_eval_accuracy": weighted_target,
        "weighted_packet_only_eval_accuracy": weighted_packet,
        "weighted_receiver_eval_accuracy": weighted_receiver,
        "weighted_target_or_packet_oracle_eval_accuracy": weighted_oracle,
        "weighted_packet_minus_target_only": weighted_packet - weighted_target,
        "weighted_receiver_minus_target_only": weighted_receiver - weighted_target,
        "weighted_receiver_minus_packet_only": weighted_receiver - weighted_packet,
        "weighted_oracle_minus_packet_only": weighted_oracle - weighted_packet,
        "target_family_transfer_slice_count": sum(
            1 for row in rows if row["target_family_transfer_gate"]
        ),
        "receiver_improvement_slice_count": sum(
            1 for row in rows if row["receiver_improvement_gate"]
        ),
        "min_receiver_ci95_low_vs_target_only": min(
            float(row["receiver_ci95_low_vs_target_only"]) for row in rows
        ),
        "min_receiver_ci95_low_vs_packet_only": min(
            float(row["receiver_ci95_low_vs_packet_only"]) for row in rows
        ),
        "raw_payload_bytes": raw_bytes[0] if len(raw_bytes) == 1 else None,
        "framed_record_bytes": framed_bytes[0] if len(framed_bytes) == 1 else None,
        "source_private_packet": all(
            not row["source_text_exposed"]
            and not row["source_kv_exposed"]
            and not row["raw_hidden_vector_transmitted"]
            and not row["raw_scores_transmitted"]
            for row in rows
        ),
    }
    pass_gate = bool(
        rows
        and all(row["target_family_transfer_gate"] for row in rows)
        and all(row["receiver_improvement_gate"] for row in rows)
    )
    interpretation = (
        "This aggregate asks whether a source-family packet can be consumed by a Phi-3 "
        "target receiver on adjacent HellaSwag slices. Packet-only utility over Phi "
        "target-only is useful transfer evidence, but receiver improvement over packet-only "
        "is required for a learned cross-family receiver claim."
    )
    payload = {
        "gate": "source_private_hellaswag_receiver_family_multislice_summary",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "pass_gate": pass_gate,
        "pass_rule": (
            "A strict receiver-family pass requires every slice to clear target-family "
            "transfer and receiver improvement over packet-only. Packet utility alone is "
            "not sufficient for the ICLR receiver-fusion claim."
        ),
        "headline": headline,
        "slice_rows": rows,
        "interpretation": interpretation,
    }
    json_path = output_dir / "hellaswag_receiver_family_multislice_summary.json"
    md_path = output_dir / "hellaswag_receiver_family_multislice_summary.md"
    csv_path = output_dir / "slice_rows.csv"
    manifest_path = output_dir / "manifest.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload)
    _write_csv(csv_path, rows)
    manifest = {
        "gate": payload["gate"],
        "created_utc": payload["created_utc"],
        "headline": headline,
        "source_artifacts": [
            {"path": _display_path(path), "sha256": _sha256_file(path)} for path in artifacts
        ],
        "files": [
            {"path": _display_path(path), "sha256": _sha256_file(path), "bytes": _resolve(path).stat().st_size}
            for path in (json_path, md_path, csv_path, manifest_path)
            if _resolve(path).exists()
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifacts", type=pathlib.Path, nargs="+", default=list(DEFAULT_ARTIFACTS))
    parser.add_argument("--run-date", default="2026-05-03")
    args = parser.parse_args()
    payload = build_summary(
        output_dir=args.output_dir,
        artifacts=tuple(args.artifacts),
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
