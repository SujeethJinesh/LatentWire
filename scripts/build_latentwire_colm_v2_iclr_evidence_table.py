from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_TRIAGE_PATH = (
    ROOT / "results/iclr_colm_v2_live_branch_triage_20260504/live_branch_triage.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "results/latentwire_colm_v2_iclr_evidence_table_20260504"
DEFAULT_PAPER_PATH = ROOT / "paper/latentwire_colm_v2_iclr_evidence_table_20260504.md"


LITERATURE_BOUNDARIES = [
    {
        "work": "Cache-to-Cache (C2C)",
        "source": "https://openreview.net/forum?id=LeatkxrBCi",
        "role": "closest dense KV-cache communication baseline",
        "boundary": (
            "C2C projects and fuses source KV cache into a target KV cache with a learned "
            "gate; LatentWire must not claim the same contribution unless it reports a "
            "different low-rate, source-private packet regime and utility-per-byte controls."
        ),
    },
    {
        "work": "DroidSpeak",
        "source": "https://arxiv.org/abs/2411.02820",
        "role": "cross-LLM KV-cache sharing / serving baseline",
        "boundary": (
            "DroidSpeak reuses KV caches across distributed nodes running different LLMs with "
            "the same architecture; LatentWire must separate semantic packet transfer from "
            "compatible-cache reuse and report source exposure."
        ),
    },
    {
        "work": "KVCOMM",
        "source": "https://arxiv.org/abs/2510.12872",
        "role": "multi-agent KV-cache communication baseline",
        "boundary": (
            "KVCOMM is training-free cross-context KV-cache reuse for multi-agent inference; "
            "LatentWire should avoid claiming general multi-agent cache communication and "
            "instead stress source-private packet transfer under destructive controls."
        ),
    },
    {
        "work": "RelayCaching",
        "source": "https://arxiv.org/abs/2603.13289",
        "role": "collaborative decoding KV-cache reuse baseline",
        "boundary": (
            "RelayCaching accelerates collaboration by reusing decoding KV caches; LatentWire "
            "differs only if it transmits compact source evidence rather than reusing produced "
            "cache states."
        ),
    },
    {
        "work": "BLIP-2 / Q-Former",
        "source": "https://arxiv.org/abs/2301.12597",
        "role": "learned query bottleneck precedent",
        "boundary": (
            "A lightweight query transformer between frozen encoders and LMs is established; "
            "LatentWire novelty cannot be the existence of a query bottleneck."
        ),
    },
    {
        "work": "Flamingo / Perceiver Resampler",
        "source": "https://arxiv.org/abs/2204.14198",
        "role": "fixed latent-token resampler and gated cross-attention precedent",
        "boundary": (
            "Fixed latent resampling into a frozen LM is established in multimodal models; "
            "LatentWire must distinguish by source-private model-to-model transfer and "
            "destructive controls."
        ),
    },
    {
        "work": "Perceiver IO",
        "source": "https://arxiv.org/abs/2107.14795",
        "role": "general latent query architecture",
        "boundary": (
            "Flexible latent queries for structured inputs/outputs are prior art; use it as "
            "architectural motivation, not a novelty claim."
        ),
    },
    {
        "work": "TurboQuant",
        "source": "https://arxiv.org/abs/2504.19874",
        "role": "strong online vector/KV quantization baseline",
        "boundary": (
            "TurboQuant optimizes vector/KV distortion under low bit widths; LatentWire "
            "should compare against it as a source-state byte floor and avoid unmeasured "
            "throughput claims."
        ),
    },
    {
        "work": "KVQuant",
        "source": "https://arxiv.org/abs/2401.18079",
        "role": "low-bit KV-cache compression baseline",
        "boundary": (
            "KVQuant compresses a model's own cache for long-context serving; LatentWire "
            "moves compact source evidence to another model and must report this access "
            "model difference."
        ),
    },
]


COLM_PROMOTED_STATUSES = {
    "promote_for_colm_v2_only",
    "promote_for_colm_v2_systems_baseline",
}

COLM_SUPPORTING_STATUSES = {
    "capacity_alive_not_source_private_method",
    "headroom_alive_selector_blocked",
    "ruled_out_cached_policy_packet",
    "ruled_out_simple_integrity_threshold",
}


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _row_to_csv(section: str, row: dict[str, Any]) -> dict[str, str]:
    return {
        "section": section,
        "branch": _fmt(row.get("branch")),
        "status": _fmt(row.get("status")),
        "score": _fmt(row.get("score")),
        "baseline": _fmt(row.get("baseline")),
        "delta": _fmt(row.get("delta")),
        "ci95_low": _fmt(row.get("ci95_low")),
        "record_bytes": _fmt(row.get("record_bytes")),
        "artifact": _fmt(row.get("artifact")),
        "decision": _fmt(row.get("decision")),
    }


def _classify_rows(branch_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    colm_core: list[dict[str, Any]] = []
    colm_supporting: list[dict[str, Any]] = []
    iclr_blockers: list[dict[str, Any]] = []

    for row in branch_rows:
        status = str(row.get("status", ""))
        if status in COLM_PROMOTED_STATUSES:
            colm_core.append(row)
        elif status in COLM_SUPPORTING_STATUSES:
            colm_supporting.append(row)
        if (
            status.startswith("ruled_out")
            or status.startswith("weakened")
            or status.endswith("_blocked")
            or status in {"capacity_alive_not_source_private_method", "headroom_alive_selector_blocked"}
        ):
            iclr_blockers.append(row)

    return {
        "colm_core": colm_core,
        "colm_supporting": colm_supporting,
        "iclr_blockers": iclr_blockers,
    }


def build_table(triage: dict[str, Any]) -> dict[str, Any]:
    branch_rows = list(triage["branch_rows"])
    classified = _classify_rows(branch_rows)
    return {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_triage": str(
            pathlib.Path("results/iclr_colm_v2_live_branch_triage_20260504/live_branch_triage.json")
        ),
        "readiness": triage["readiness"],
        "story": triage["story"],
        "submission_gap": triage["submission_gap"],
        "current_contributions": triage["current_contributions"],
        "claim_boundaries": triage["claim_boundaries"],
        "next_exact_gate": triage["next_exact_gate"],
        "colm_v2_core_rows": classified["colm_core"],
        "colm_v2_supporting_rows": classified["colm_supporting"],
        "iclr_blocker_rows": classified["iclr_blockers"],
        "literature_boundaries": LITERATURE_BOUNDARIES,
        "paper_decision": {
            "single_highest_priority": (
                "Backport the live ICLR triage into COLM_v2 tables/figures now, while "
                "the next ICLR method branch is redesigned around a qualitatively new "
                "source-causal interface rather than another deterministic PQ transform, "
                "scalar integrity gate, source-score selector, or target-native soft-prefix decoder."
            ),
            "colm_v2_claim": (
                "LatentWire_v2 demonstrates byte-scale source-private packet transfer with "
                "strict destructive controls, plus explicit negative gates that prevent "
                "overclaiming cross-family latent communication."
            ),
            "iclr_claim_not_yet_supported": (
                "Sparse Resonance Packets as a broad learned cross-model communication method."
            ),
        },
    }


def _markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Branch | Status | Score | Baseline | Delta | CI95 low | Bytes | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row.get("branch")),
                    f"`{_fmt(row.get('status'))}`",
                    _fmt(row.get("score")),
                    _fmt(row.get("baseline")),
                    _fmt(row.get("delta")),
                    _fmt(row.get("ci95_low")),
                    _fmt(row.get("record_bytes")),
                    _fmt(row.get("decision")),
                ]
            )
            + " |"
        )
    return lines


def render_markdown(table: dict[str, Any]) -> str:
    lines: list[str] = [
        "# LatentWire COLM_v2 / ICLR Evidence Table",
        "",
        f"- created UTC: `{table['created_utc']}`",
        f"- source triage: `{table['source_triage']}`",
        f"- COLM_v2 readiness: `{table['readiness']['colm_v2']}`",
        f"- ICLR readiness: `{table['readiness']['iclr']}`",
        "",
        "## Current Story",
        "",
        table["story"],
        "",
        "## Exact Submission Gap",
        "",
        table["submission_gap"],
        "",
        "## Current Technical Contributions",
        "",
    ]
    for item in table["current_contributions"]:
        needs_work = item.get("gap", item.get("needs_work", "needs current evidence update"))
        lines.append(f"- `{item['name']}`: {item['status']}; {needs_work}")

    lines.extend(
        [
            "",
            "## COLM_v2 Core Rows",
            "",
            "These are the rows that can anchor the narrow workshop version without claiming broad latent language.",
            "",
            *_markdown_table(table["colm_v2_core_rows"]),
            "",
            "## COLM_v2 Supporting / Guardrail Rows",
            "",
            "These rows explain headroom, saturation, and why the paper keeps strong claim boundaries.",
            "",
            *_markdown_table(table["colm_v2_supporting_rows"]),
            "",
            "## ICLR Blocker Rows",
            "",
            "These rows prevent an ICLR-scale claim until a new source-causal interface clears the same controls.",
            "",
            *_markdown_table(table["iclr_blocker_rows"]),
            "",
            "## Literature And Novelty Boundaries",
            "",
            "| Work | Role | Boundary for LatentWire | Source |",
            "|---|---|---|---|",
        ]
    )
    for item in table["literature_boundaries"]:
        lines.append(
            f"| {item['work']} | {item['role']} | {item['boundary']} | {item['source']} |"
        )

    lines.extend(
        [
            "",
            "## Paper Decision",
            "",
            f"- single highest priority: {table['paper_decision']['single_highest_priority']}",
            f"- COLM_v2 claim: {table['paper_decision']['colm_v2_claim']}",
            f"- ICLR claim not yet supported: {table['paper_decision']['iclr_claim_not_yet_supported']}",
            "",
            "## Next Exact Gate",
            "",
            f"- name: `{table['next_exact_gate']['name']}`",
            f"- primary path: {table['next_exact_gate']['primary_path']}",
            f"- fallback path: {table['next_exact_gate']['fallback_path']}",
            f"- pass bar: {table['next_exact_gate']['pass_bar']}",
            "",
            "## Claim Boundaries",
            "",
        ]
    )
    for boundary in table["claim_boundaries"]:
        lines.append(f"- {boundary}")
    lines.append("")
    return "\n".join(lines)


def write_outputs(table: dict[str, Any], output_dir: pathlib.Path, paper_path: pathlib.Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "evidence_table.json"
    csv_path = output_dir / "evidence_table.csv"
    md_path = output_dir / "evidence_table.md"

    json_path.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows: list[dict[str, str]] = []
    for section, key in [
        ("colm_v2_core", "colm_v2_core_rows"),
        ("colm_v2_supporting", "colm_v2_supporting_rows"),
        ("iclr_blocker", "iclr_blocker_rows"),
    ]:
        rows.extend(_row_to_csv(section, row) for row in table[key])
    fieldnames = [
        "section",
        "branch",
        "status",
        "score",
        "baseline",
        "delta",
        "ci95_low",
        "record_bytes",
        "artifact",
        "decision",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    markdown = render_markdown(table)
    md_path.write_text(markdown, encoding="utf-8")
    paper_path.write_text(markdown, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triage-path", type=pathlib.Path, default=DEFAULT_TRIAGE_PATH)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--paper-path", type=pathlib.Path, default=DEFAULT_PAPER_PATH)
    args = parser.parse_args()

    triage = _read_json(args.triage_path)
    table = build_table(triage)
    write_outputs(table, args.output_dir, args.paper_path)


if __name__ == "__main__":
    main()
