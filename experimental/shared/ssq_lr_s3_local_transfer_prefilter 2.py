"""Build an SSQ-LR local two-model S3 transfer preflight packet.

This runner is stricter than the cache-only transfer prefilter because it
combines actual frozen-recipe replay rows from a source S2 packet and one or
more separately run transfer S2 scout packets. It is still deliberately labeled
as local preflight evidence: short same-family replays can clear the executable
S3 contract, but they do not replace the preregistered larger transfer surface
or native GPU validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from experimental.shared.followup_gate_contracts import evaluate_ssq_lr_s3
from experimental.shared.ssq_lr_s3_transfer_prefilter import (
    DEFAULT_PREREGISTRATION,
    _freeze_recipe,
    _load_rows,
    _s3_row,
    _sha256_path,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_S2_PACKET = (
    ROOT / "experimental/shared/results/ssq_lr_s3_source_granite_tiny_12p_layer0_ctx32_20260507"
)
DEFAULT_TRANSFER_S2_PACKETS = (
    ROOT / "experimental/shared/results/ssq_lr_s3_transfer_granite_350m_12p_layer0_20260507",
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "experimental/shared/results/ssq_lr_s3_local_transfer_prefilter_mixed25_granite_tiny_350m_layer0_12p_20260507"
)


def _reset_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _selected_recipe_rows(packet_dir: Path, recipe_id: str) -> list[dict[str, Any]]:
    rows = [
        row
        for row in _load_rows(packet_dir / "raw_rows.jsonl")
        if str(row.get("control_type")) == "candidate_recipe" and str(row.get("recipe_id")) == recipe_id
    ]
    if not rows:
        raise ValueError(f"{packet_dir} has no candidate rows for recipe {recipe_id!r}")
    return rows


def _packet_summary(packet_dir: Path) -> dict[str, Any]:
    with (packet_dir / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if str(summary.get("gate_status")) != "PASS_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY":
        raise ValueError(f"{packet_dir} is not a passing S2 replay packet")
    return summary


def _role_for_packet(packet_dir: Path, *, source_s2_dir: Path) -> str:
    if packet_dir.resolve() == source_s2_dir.resolve():
        return "source_model_frozen_recipe_reference"
    return "transfer_model_no_retune_local_replay"


def _row_boundary_note(packet_dir: Path, *, source_s2_dir: Path) -> str:
    if packet_dir.resolve() == source_s2_dir.resolve():
        return "source-model replay row used only to freeze and audit the recipe"
    return "local no-retuning transfer replay row; short same-family scout, not GPU evidence"


def build_local_transfer_prefilter(
    *,
    source_s2_dir: Path = DEFAULT_SOURCE_S2_PACKET,
    transfer_s2_dirs: tuple[Path, ...] = DEFAULT_TRANSFER_S2_PACKETS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    preregistration: Path = DEFAULT_PREREGISTRATION,
    recipe_id: str = "mixed_int3_mxfp4_low_error_25pct",
    primary_layers: str = "0",
    block_size: int = 256,
    seed: int = 20260507,
) -> dict[str, Any]:
    packet_dirs = (source_s2_dir, *transfer_s2_dirs)
    source_rows = _selected_recipe_rows(source_s2_dir, recipe_id)
    source_summary = _packet_summary(source_s2_dir)
    for packet_dir in transfer_s2_dirs:
        _packet_summary(packet_dir)

    _reset_output_dir(output_dir)
    source_s2_packet_sha256 = _sha256_path(source_s2_dir / "summary.json")
    recipe, frozen_recipe_sha256 = _freeze_recipe(
        source_s2_dir=source_s2_dir,
        selected_rows=source_rows,
        recipe_id=recipe_id,
        primary_layers=primary_layers,
        block_size=block_size,
    )

    transfer_rows: list[dict[str, Any]] = []
    replay_packets: list[dict[str, Any]] = []
    for packet_dir in packet_dirs:
        rows = _selected_recipe_rows(packet_dir, recipe_id)
        summary = _packet_summary(packet_dir)
        packet_hash = _sha256_path(packet_dir / "summary.json")
        row_model_ids = sorted({str(row["model_id"]) for row in rows})
        replay_packets.append(
            {
                "packet_dir": str(packet_dir),
                "packet_sha256": packet_hash,
                "model_ids": row_model_ids,
                "prompt_count": len({str(row["prompt_id"]) for row in rows}),
                "row_count": len(rows),
                "selected_accuracy_delta_abs": float(summary.get("selected_accuracy_delta_abs", 0.0)),
                "selected_nll_delta_abs": float(summary.get("selected_nll_delta_abs", 0.0)),
                "selected_nll_delta_ci_high": float(summary.get("selected_nll_delta_ci_high", 0.0)),
                "role": _role_for_packet(packet_dir, source_s2_dir=source_s2_dir),
            }
        )
        for row in rows:
            s3_row = _s3_row(
                row=row,
                frozen_recipe_sha256=frozen_recipe_sha256,
                source_s2_packet_sha256=source_s2_packet_sha256,
                model_role=_role_for_packet(packet_dir, source_s2_dir=source_s2_dir),
                control_type="transfer_eval",
            )
            s3_row["transfer_replay_packet_sha256"] = packet_hash
            s3_row["claim_boundary_note"] = _row_boundary_note(packet_dir, source_s2_dir=source_s2_dir)
            transfer_rows.append(s3_row)

    retune_probe = dict(transfer_rows[0])
    retune_probe.update(
        {
            "prompt_id": "retune_probe_guard_no_retuning_performed",
            "model_role": "retune_probe_guard",
            "bf16_accuracy": 1.0,
            "quantized_accuracy": 1.0,
            "accuracy_delta_abs": 0.0,
            "paired_ci_high": 0.0,
            "bf16_nll": float(transfer_rows[0]["bf16_nll"]),
            "quantized_nll": float(transfer_rows[0]["bf16_nll"]),
            "nll_delta": 0.0,
            "control_type": "retune_probe",
            "claim_boundary_note": "required no-retuning sentinel; no retuned rows are emitted",
        }
    )
    rows = [*transfer_rows, retune_probe]
    evaluation = evaluate_ssq_lr_s3(rows)
    transfer_model_prompt_counts = {
        model_id: len(
            {
                str(row["prompt_id"])
                for row in transfer_rows
                if str(row["model_id"]) == model_id
            }
        )
        for model_id in sorted({str(row["model_id"]) for row in transfer_rows})
    }
    minimum_model_prompt_count = min(transfer_model_prompt_counts.values(), default=0)
    claim_boundary = [
        "local two-model S3 preflight only; not native GPU evidence",
        "transfer replay includes a short same-family Granite 350M scout, not the full preregistered target roster",
        "source Granite Tiny rows remain part of the local contract packet and should not be counted as independent cross-family validation",
        "retuned=false for every row; no per-model retuning was performed",
    ]
    config = {
        "gate_name": "ssq_lr_s3",
        "project": "ssq_lr",
        "source_gate_packet_sha256": source_s2_packet_sha256,
        "preregistration_sha256": _sha256_path(preregistration),
        "seed_list": [seed],
        "command": "python -m experimental.shared.ssq_lr_s3_local_transfer_prefilter",
        "source_s2_packet": str(source_s2_dir),
        "transfer_s2_packets": [str(path) for path in transfer_s2_dirs],
        "recipe_id": recipe_id,
        "primary_layers": recipe["primary_layers"],
        "block_size": block_size,
        "claim_boundary": claim_boundary,
    }
    summary = {
        "decision": evaluation["gate_status"],
        "resource_limited_decision": f"LOCAL_PREFLIGHT_NOT_PROMOTABLE_{evaluation['gate_status']}",
        **evaluation,
        "row_count": len(rows),
        "source_s2_packet_sha256": source_s2_packet_sha256,
        "source_s2_gate_status": str(source_summary.get("gate_status")),
        "frozen_recipe": recipe,
        "replay_packets": replay_packets,
        "transfer_model_prompt_counts": transfer_model_prompt_counts,
        "minimum_model_prompt_count": minimum_model_prompt_count,
        "claim_boundary": claim_boundary,
    }

    (output_dir / "frozen_recipe.json").write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n")
    (output_dir / "replay_packets.json").write_text(json.dumps(replay_packets, indent=2, sort_keys=True) + "\n")
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (output_dir / "raw_rows.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_dir / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    (output_dir / "decision.md").write_text(
        "# SSQ-LR Local S3 Transfer Prefilter\n\n"
        f"`{summary['decision']}`\n\n"
        f"Resource-limited label: `{summary['resource_limited_decision']}`\n",
        encoding="utf-8",
    )
    print(json.dumps({"decision": summary["decision"], "output_dir": str(output_dir)}, sort_keys=True))
    return summary


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# SSQ-LR Local S3 Transfer Prefilter",
        "",
        f"Decision: `{summary['decision']}`",
        f"Resource-limited label: `{summary['resource_limited_decision']}`",
        "",
        "This packet is local two-model no-retuning evidence, not camera-ready cross-model transfer.",
        "",
        f"- Transfer model count: `{summary['transfer_model_count']}`",
        f"- Passing model count: `{summary['passing_model_count']}`",
        f"- Minimum prompt count per model: `{summary['minimum_model_prompt_count']}`",
        f"- Frozen recipe hash: `{summary['frozen_recipe_sha256']}`",
        f"- Max accuracy delta: `{summary['max_accuracy_delta_abs']:.6f}`",
        f"- Max CI high: `{summary['max_ci_high']:.6f}`",
        f"- Max NLL delta: `{summary['max_nll_delta_abs']:.6f}`",
        "",
        "| Packet | Role | Models | Prompts | NLL delta abs | NLL CI high |",
        "|---|---|---|---:|---:|---:|",
    ]
    for item in summary["replay_packets"]:
        lines.append(
            "| {packet_dir} | {role} | {models} | {prompt_count} | {selected_nll_delta_abs:.6f} | {selected_nll_delta_ci_high:.6f} |".format(
                models=", ".join(item["model_ids"]),
                **item,
            )
        )
    lines.append("")
    return "\n".join(lines)


def _parse_path_list(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(part).expanduser() for part in value.split(",") if part.strip())
    if not paths:
        raise ValueError("path list must contain at least one path")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-s2-dir", type=Path, default=DEFAULT_SOURCE_S2_PACKET)
    parser.add_argument(
        "--transfer-s2-dirs",
        default=",".join(str(path) for path in DEFAULT_TRANSFER_S2_PACKETS),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--recipe-id", default="mixed_int3_mxfp4_low_error_25pct")
    parser.add_argument("--primary-layers", default="0")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260507)
    args = parser.parse_args()
    build_local_transfer_prefilter(
        source_s2_dir=args.source_s2_dir,
        transfer_s2_dirs=_parse_path_list(args.transfer_s2_dirs),
        output_dir=args.output_dir,
        preregistration=args.preregistration,
        recipe_id=args.recipe_id,
        primary_layers=args.primary_layers,
        block_size=args.block_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
