#!/usr/bin/env python3
"""CPU-only candidate-syndrome decoder over existing source/target artifacts.

This is a bounded artifact probe, not a learned method. It asks whether a tiny
source-derived code can disambiguate target-side candidate answers while cheap
source-destroying controls fail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_svamp32_syndrome_sidecar_probe as syndrome


@dataclass(frozen=True)
class Surface:
    label: str
    target_set_path: pathlib.Path
    target_set: dict[str, Any]
    records_by_label: dict[str, dict[str, dict[str, Any]]]


def _resolve(path: str | pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _record_spec(label: str, raw: dict[str, Any]) -> syndrome.RowSpec:
    return syndrome.RowSpec(
        label=label,
        path=_resolve(str(raw["path"])),
        method=str(raw["method"]),
    )


def _load_records(raw: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    specs = [_record_spec("target", raw["artifacts"]["target"])]
    specs.append(_record_spec("source", raw["artifacts"]["source"]))
    for baseline in raw["artifacts"].get("baselines", []):
        specs.append(_record_spec(str(baseline["label"]), baseline))
    for control in raw["artifacts"].get("controls", []):
        specs.append(_record_spec(str(control["label"]), control))

    reference_ids = [str(example_id) for example_id in raw["reference_ids"]]
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in specs:
        rows = syndrome._subset_reference_order(
            syndrome._records_for_method(spec),
            reference_ids,
        )
        out[spec.label] = syndrome._by_id(rows)
    return out


def _load_surface(label: str, path: pathlib.Path) -> Surface:
    raw = _read_json(path)
    return Surface(
        label=label,
        target_set_path=path,
        target_set=raw,
        records_by_label=_load_records(raw),
    )


def _source_ids(raw: dict[str, Any], key: str) -> set[str]:
    return {str(example_id) for example_id in raw.get("ids", {}).get(key, [])}


def _numeric(row: dict[str, Any] | None) -> str | None:
    if row is None:
        return None
    return syndrome._prediction_numeric(row)


def _candidate_pool(
    surface: Surface,
    example_id: str,
) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()
    for label in surface.records_by_label:
        numeric = _numeric(surface.records_by_label[label].get(example_id))
        if numeric is None or numeric in seen:
            continue
        seen.add(numeric)
        candidates.append({"value": numeric, "source": label})
    return candidates


def _candidate_code(value: str, *, bits: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    mask = (1 << bits) - 1
    return int.from_bytes(digest[:8], "big") & mask


def _ordered_ids(surface: Surface) -> list[str]:
    return [str(example_id) for example_id in surface.target_set["reference_ids"]]


def _shuffled_source_value(surface: Surface, example_id: str) -> str | None:
    ids = _ordered_ids(surface)
    idx = ids.index(example_id)
    for offset in range(1, len(ids)):
        other_id = ids[(idx + offset) % len(ids)]
        value = _numeric(surface.records_by_label["source"].get(other_id))
        if value is not None:
            return value
    return None


def _target_value(surface: Surface, example_id: str) -> str | None:
    return _numeric(surface.records_by_label["target"].get(example_id))


def _source_value(surface: Surface, example_id: str) -> str | None:
    return _numeric(surface.records_by_label["source"].get(example_id))


def _t2t_value(surface: Surface, example_id: str) -> str | None:
    if "t2t" not in surface.records_by_label:
        return None
    return _numeric(surface.records_by_label["t2t"].get(example_id))


def _decode(
    *,
    surface: Surface,
    example_id: str,
    mode: str,
    bits: int,
    rng: random.Random,
) -> tuple[str | None, str]:
    target = _target_value(surface, example_id)
    if mode in {"target_only", "zero_source"}:
        return target, "target_fallback"
    if mode == "slots_only":
        return _t2t_value(surface, example_id) or target, "slots_proxy"

    candidates = _candidate_pool(surface, example_id)
    if not candidates:
        return target, "empty_pool"

    if mode == "matched":
        source = _source_value(surface, example_id)
        if source is None:
            return target, "missing_source"
        code = _candidate_code(source, bits=bits)
    elif mode == "shuffled_source":
        source = _shuffled_source_value(surface, example_id)
        if source is None:
            return target, "missing_shuffled_source"
        code = _candidate_code(source, bits=bits)
    elif mode == "random_syndrome":
        code = rng.randrange(1 << bits)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    matches = [
        item for item in candidates if _candidate_code(item["value"], bits=bits) == code
    ]
    if not matches:
        return target, "no_code_match"
    if target is not None:
        for item in matches:
            if item["value"] == target:
                return item["value"], f"code_match:{item['source']}:target_preserve"
    return matches[0]["value"], f"code_match:{matches[0]['source']}"


def _gold_by_id(surface: Surface, example_id: str) -> str:
    return syndrome._gold_numeric(surface.records_by_label["target"][example_id])


def _summarize_condition(
    *,
    surface: Surface,
    mode: str,
    bits: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(f"{surface.label}:{mode}:{seed}:{bits}")
    clean_source_only = _source_ids(surface.target_set, "clean_source_only")
    target_self = _source_ids(surface.target_set, "target_self_repair")
    baseline_union = _source_ids(surface.target_set, "baseline_union")
    control_union = _source_ids(surface.target_set, "control_union")

    correct_ids: set[str] = set()
    clean_source_necessary: set[str] = set()
    harms: set[str] = set()
    rows: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    numeric_coverage = 0

    for example_id in _ordered_ids(surface):
        pred, reason = _decode(
            surface=surface,
            example_id=example_id,
            mode=mode,
            bits=bits,
            rng=rng,
        )
        gold = _gold_by_id(surface, example_id)
        correct = pred is not None and pred == gold
        numeric_coverage += int(pred is not None)
        if correct:
            correct_ids.add(example_id)
        if correct and example_id in clean_source_only:
            clean_source_necessary.add(example_id)
        if (not correct) and example_id in target_self:
            harms.add(example_id)
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        rows.append(
            {
                "example_id": example_id,
                "prediction_numeric": pred,
                "gold_numeric": gold,
                "correct": correct,
                "reason": reason,
            }
        )

    return {
        "mode": mode,
        "bits": bits,
        "bytes": max(1, (bits + 7) // 8),
        "n": len(_ordered_ids(surface)),
        "correct": len(correct_ids),
        "numeric_coverage": numeric_coverage,
        "clean_source_necessary_count": len(clean_source_necessary),
        "clean_source_necessary_ids": sorted(clean_source_necessary),
        "target_self_harm_count": len(harms),
        "target_self_harm_ids": sorted(harms),
        "baseline_correct_count": len(correct_ids & baseline_union),
        "control_correct_count": len(correct_ids & control_union),
        "reason_counts": dict(sorted(reason_counts.items())),
        "rows": rows,
    }


def _control_clean_union(conditions: Iterable[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for condition in conditions:
        out.update(str(x) for x in condition["clean_source_necessary_ids"])
    return out


def _analyze_surface(
    *,
    surface: Surface,
    bits: int,
    seed: int,
    controls: list[str],
    min_clean_source_necessary: int,
) -> dict[str, Any]:
    modes = ["matched", *controls]
    conditions = {
        mode: _summarize_condition(surface=surface, mode=mode, bits=bits, seed=seed)
        for mode in modes
    }
    control_modes = [mode for mode in controls if mode in conditions]
    control_union = _control_clean_union(conditions[mode] for mode in control_modes)
    matched = conditions["matched"]
    pass_rule = {
        "min_clean_source_necessary": (
            matched["clean_source_necessary_count"] >= min_clean_source_necessary
        ),
        "control_clean_union_zero": len(control_union) == 0,
        "target_self_harm_zero": matched["target_self_harm_count"] == 0,
        "numeric_coverage_parity": (
            matched["numeric_coverage"]
            >= int(surface.target_set["provenance"]["source_numeric_coverage"])
        ),
    }
    return {
        "label": surface.label,
        "target_set": _display_path(surface.target_set_path),
        "source_surface_counts": surface.target_set.get("counts", {}),
        "conditions": conditions,
        "control_clean_union": sorted(control_union),
        "pass_rule": pass_rule,
        "status": (
            "candidate_syndrome_decoder_passes_smoke"
            if all(pass_rule.values())
            else "candidate_syndrome_decoder_fails_smoke"
        ),
    }


def _write_md(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Candidate-Syndrome Decoder Probe",
        "",
        f"Date: {payload['date']}",
        "",
        f"Status: `{payload['status']}`",
        "",
        "This is a CPU-only artifact probe, not a learned method. It tests a tiny",
        "hash-syndrome over target-side numeric candidate pools with source-destroying",
        "controls.",
        "",
        "## Surfaces",
        "",
    ]
    for surface in payload["surfaces"]:
        lines.extend(
            [
                f"### {surface['label']}",
                "",
                f"- status: `{surface['status']}`",
                f"- target set: `{surface['target_set']}`",
                f"- matched clean source-necessary: "
                f"`{surface['conditions']['matched']['clean_source_necessary_count']}`",
                f"- matched target-self harms: "
                f"`{surface['conditions']['matched']['target_self_harm_count']}`",
                f"- control clean union: `{len(surface['control_clean_union'])}`",
                f"- pass rule: `{surface['pass_rule']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision",
            "",
            payload["decision"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-target-set", required=True)
    parser.add_argument("--holdout-target-set", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--controls",
        nargs="+",
        default=["zero_source", "shuffled_source", "random_syndrome", "target_only", "slots_only"],
    )
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-clean-source-necessary", type=int, default=2)
    parser.add_argument("--run-date", default=date.today().isoformat())
    args = parser.parse_args()

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    surfaces = [
        _analyze_surface(
            surface=_load_surface("live", _resolve(args.live_target_set)),
            bits=int(args.bits),
            seed=int(args.seed),
            controls=list(args.controls),
            min_clean_source_necessary=int(args.min_clean_source_necessary),
        ),
        _analyze_surface(
            surface=_load_surface("holdout", _resolve(args.holdout_target_set)),
            bits=int(args.bits),
            seed=int(args.seed),
            controls=list(args.controls),
            min_clean_source_necessary=int(args.min_clean_source_necessary),
        ),
    ]
    status = (
        "candidate_syndrome_decoder_passes_smoke"
        if all(surface["status"] == "candidate_syndrome_decoder_passes_smoke" for surface in surfaces)
        else "candidate_syndrome_decoder_fails_smoke"
    )
    decision = (
        "Promote to a stricter learned/source-predicate gate."
        if status == "candidate_syndrome_decoder_passes_smoke"
        else (
            "Do not promote this hash-syndrome artifact probe. If the family is "
            "revived, it needs learned source predicates or a stronger source "
            "surface rather than another numeric-hash sidecar."
        )
    )
    payload = {
        "date": str(args.run_date),
        "status": status,
        "config": {
            "bits": int(args.bits),
            "seed": int(args.seed),
            "controls": list(args.controls),
            "min_clean_source_necessary": int(args.min_clean_source_necessary),
        },
        "surfaces": surfaces,
        "decision": decision,
    }

    json_path = output_dir / "candidate_syndrome_decoder_probe.json"
    md_path = output_dir / "candidate_syndrome_decoder_probe.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_md(md_path, payload)
    print(json.dumps({"status": status, "output_json": _display_path(json_path)}, indent=2))


if __name__ == "__main__":
    main()
