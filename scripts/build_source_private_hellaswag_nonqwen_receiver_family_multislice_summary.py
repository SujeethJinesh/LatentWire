from __future__ import annotations

"""Aggregate non-Qwen HellaSwag receiver-family packet scout slices."""

import argparse
import csv
import datetime as dt
import json
import pathlib
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_nonqwen_receiver_family_multislice_summary_20260503_validation1024_2048"
)
DEFAULT_ARTIFACTS = (
    pathlib.Path(
        "results/source_private_hellaswag_nonqwen_receiver_family_packet_gate_20260503_validation1024_1536/"
        "hellaswag_nonqwen_receiver_family_packet_gate.json"
    ),
    pathlib.Path(
        "results/source_private_hellaswag_nonqwen_receiver_family_packet_gate_20260503_validation1536_2048/"
        "hellaswag_nonqwen_receiver_family_packet_gate.json"
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
    import hashlib

    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: pathlib.Path | str) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _slice_row(path: pathlib.Path, payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("gate") != "source_private_hellaswag_nonqwen_receiver_family_packet_gate":
        raise ValueError(f"{path} is not a non-Qwen receiver-family packet gate artifact")
    headline = payload["headline"]
    receiver_gate = payload["receiver_gate"]
    source_packet = payload["source_packet"]
    control_fields = receiver_gate.get("source_packet", {}).get("control_fields")
    if control_fields is None:
        control_fields = [
            item.get("name")
            for item in receiver_gate.get("control_rows", [])
            if item.get("name") is not None
        ]
    row = {
        "artifact_path": _display_path(path),
        "pass_gate": bool(payload["pass_gate"]),
        "slice_start": int(headline["slice_start"]),
        "slice_end_exclusive": int(headline["slice_end_exclusive"]),
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
        "packet_minus_target_only": float(headline["packet_minus_target_only"]),
        "receiver_minus_target_only": float(headline["receiver_minus_target_only"]),
        "receiver_minus_packet_only": float(headline["receiver_minus_packet_only"]),
        "receiver_ci95_low_vs_target_only": float(headline["receiver_ci95_low_vs_target_only"]),
        "receiver_ci95_low_vs_packet_only": float(headline["receiver_ci95_low_vs_packet_only"]),
        "source_utility_gate": bool(headline["source_utility_gate"]),
        "target_family_transfer_gate": bool(headline["target_family_transfer_gate"]),
        "receiver_improvement_gate": bool(headline["receiver_improvement_gate"]),
        "selected_receiver_kind": str(headline["selected_receiver_kind"]),
        "target_score_cache_hit": bool(headline["target_score_cache_hit"]),
        "target_score_latency_s": float(headline["target_score_latency_s"] or 0.0),
        "native_systems_complete": bool(headline["native_systems_complete"]),
        "raw_payload_bytes": int(source_packet["raw_payload_bytes"]),
        "framed_record_bytes": int(source_packet["framed_record_bytes"]),
        "exposes_source_text": bool(source_packet["exposes_source_text"]),
        "exposes_source_kv": bool(source_packet["exposes_source_kv"]),
        "exposes_raw_hidden": bool(source_packet["exposes_raw_hidden"]),
        "exposes_raw_scores": bool(source_packet["exposes_raw_scores"]),
        "control_field_count": len(control_fields),
    }
    row["slice_order_key"] = row["slice_start"]
    return row


def _contiguous(rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    ordered = sorted(rows, key=lambda item: item["slice_start"])
    return all(
        left["slice_end_exclusive"] == right["slice_start"]
        for left, right in zip(ordered, ordered[1:], strict=False)
    )


def _weighted(rows: list[dict[str, Any]], key: str) -> float:
    total = sum(int(row["eval_rows"]) for row in rows)
    if total <= 0:
        return 0.0
    return sum(float(row[key]) * int(row["eval_rows"]) for row in rows) / total


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = [key for key in rows[0].keys() if key != "slice_order_key"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Non-Qwen Receiver-Family Multi-Slice Summary",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- slices: `{h['slice_count']}`",
        f"- contiguous range: `{h['range_start']}:{h['range_end_exclusive']}`",
        f"- total eval rows: `{h['total_eval_rows']}`",
        f"- weighted Phi target-only accuracy: `{h['weighted_target_only_eval_accuracy']:.6f}`",
        f"- weighted TinyLlama packet-only accuracy: `{h['weighted_packet_only_eval_accuracy']:.6f}`",
        f"- weighted receiver accuracy: `{h['weighted_receiver_eval_accuracy']:.6f}`",
        f"- weighted target-or-packet oracle accuracy: `{h['weighted_target_or_packet_oracle_eval_accuracy']:.6f}`",
        f"- packet minus target-only: `{h['weighted_packet_minus_target_only']:.6f}`",
        f"- receiver minus packet-only: `{h['weighted_receiver_minus_packet_only']:.6f}`",
        f"- oracle minus packet-only: `{h['weighted_oracle_minus_packet_only']:.6f}`",
        f"- source utility slices: `{h['source_utility_slice_count']}/{h['slice_count']}`",
        f"- receiver-improvement slices: `{h['receiver_improvement_slice_count']}/{h['slice_count']}`",
        f"- minimum receiver CI95 low vs packet-only: `{h['min_receiver_ci95_low_vs_packet_only']:.6f}`",
        f"- packet contract: `{h['raw_payload_bytes']}B` raw / `{h['framed_record_bytes']}B` framed",
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
        key=lambda item: item["slice_order_key"],
    )
    source_families = sorted({row["source_family"] for row in rows})
    target_families = sorted({row["target_family"] for row in rows})
    raw_payload_bytes = sorted({int(row["raw_payload_bytes"]) for row in rows})
    framed_record_bytes = sorted({int(row["framed_record_bytes"]) for row in rows})
    if len(source_families) != 1 or len(target_families) != 1:
        raise ValueError("all slices must share the same source and target family")
    if len(raw_payload_bytes) != 1 or len(framed_record_bytes) != 1:
        raise ValueError("all slices must share the same packet byte contract")

    weighted_target = _weighted(rows, "target_only_eval_accuracy")
    weighted_packet = _weighted(rows, "packet_only_eval_accuracy")
    weighted_receiver = _weighted(rows, "receiver_eval_accuracy")
    weighted_oracle = _weighted(rows, "target_or_packet_oracle_eval_accuracy")
    pass_gate = bool(
        rows
        and all(row["source_utility_gate"] for row in rows)
        and all(row["target_family_transfer_gate"] for row in rows)
        and all(row["receiver_improvement_gate"] for row in rows)
    )
    headline = {
        "slice_count": len(rows),
        "range_start": int(rows[0]["slice_start"]) if rows else None,
        "range_end_exclusive": int(rows[-1]["slice_end_exclusive"]) if rows else None,
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
        "source_utility_slice_count": sum(1 for row in rows if row["source_utility_gate"]),
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
        "target_score_latency_s_sum": sum(float(row["target_score_latency_s"]) for row in rows),
        "target_score_cache_hit_slice_count": sum(1 for row in rows if row["target_score_cache_hit"]),
        "raw_payload_bytes": raw_payload_bytes[0],
        "framed_record_bytes": framed_record_bytes[0],
        "source_private_packet": all(
            not row["exposes_source_text"]
            and not row["exposes_source_kv"]
            and not row["exposes_raw_hidden"]
            and not row["exposes_raw_scores"]
            for row in rows
        ),
        "native_systems_complete": all(row["native_systems_complete"] for row in rows),
    }
    interpretation = (
        "Two adjacent HellaSwag slices show stable non-Qwen packet utility: the TinyLlama "
        "fixed-byte packet beats Phi-3 target-only by a large margin on both slices. The "
        "selected receiver still fails to beat packet-only, so this strengthens the "
        "receiver-family packet-utility claim but does not close the ICLR cross-model "
        "reasoning or receiver-fusion gate."
    )
    payload = {
        "gate": "source_private_hellaswag_nonqwen_receiver_family_multislice_summary",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "pass_gate": pass_gate,
        "pass_rule": (
            "A pass requires every slice to have source utility, target-family transfer, "
            "and receiver improvement versus packet-only. Packet utility without receiver "
            "improvement is recorded as useful but insufficient for the ICLR receiver gate."
        ),
        "headline": headline,
        "slice_rows": rows,
        "interpretation": interpretation,
    }
    json_path = output_dir / "hellaswag_nonqwen_receiver_family_multislice_summary.json"
    md_path = output_dir / "hellaswag_nonqwen_receiver_family_multislice_summary.md"
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
