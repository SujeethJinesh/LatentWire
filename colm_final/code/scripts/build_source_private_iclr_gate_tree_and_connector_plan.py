from __future__ import annotations

"""Build the reviewer-facing ICLR gate tree and next-connector plan.

The artifact consolidates the current evidence after the ARC hidden/query MLP
cache connector failure.  It is intentionally a decision artifact: it records
which branches are promoted, cut, alive, or blocked, and it emits the exact
tokenwise/NVIDIA run contract needed before any ICLR positive-method claim.
"""

import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import pathlib
import textwrap
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = pathlib.Path("results/source_private_iclr_gate_tree_and_connector_plan_20260502")

ARTIFACTS = {
    "systems_boundary": pathlib.Path(
        "results/source_private_systems_boundary_figure_table_20260502/systems_boundary_figure_data.json"
    ),
    "hidden_query_mlp": pathlib.Path(
        "results/source_private_arc_challenge_hidden_query_mlp_cache_connector_gate_20260502_tinyllama_disagreement/"
        "arc_challenge_hidden_query_mlp_cache_connector_gate.json"
    ),
    "llama_probe": pathlib.Path(
        "results/source_private_arc_llama8b_failure_probe_20260502/arc_llama8b_failure_probe.json"
    ),
    "native_ingest": pathlib.Path(
        "results/source_private_native_systems_result_ingest_gate_20260502/native_systems_result_ingest_gate.json"
    ),
}

PRIMARY_SOURCES = {
    "blip2": "https://arxiv.org/abs/2301.12597",
    "flamingo": "https://arxiv.org/abs/2204.14198",
    "perceiver_io": "https://arxiv.org/abs/2107.14795",
    "prefix_tuning": "https://arxiv.org/abs/2101.00190",
    "c2c": "https://openreview.net/forum?id=LeatkxrBCi",
    "kvcomm": "https://arxiv.org/abs/2510.03346",
    "kvcomm_cross_context": "https://arxiv.org/abs/2510.12872",
    "qjl": "https://arxiv.org/abs/2406.03482",
    "turboquant": "https://arxiv.org/abs/2504.19874",
    "dit": "https://arxiv.org/abs/2212.09748",
    "consistency": "https://arxiv.org/abs/2303.01469",
    "sae_universal": "https://arxiv.org/abs/2410.06981",
}

NODE_COLUMNS = (
    "node_id",
    "status",
    "contribution",
    "claim",
    "evidence",
    "decision",
    "next_action",
    "artifact",
)


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display(path: pathlib.Path | str) -> str:
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


def _maybe_json(path: pathlib.Path | str) -> dict[str, Any]:
    resolved = _resolve(path)
    if not resolved.exists():
        return {}
    return _read_json(resolved)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _systems_numbers(payload: dict[str, Any]) -> dict[str, Any]:
    headline = payload.get("headline", {})
    return {
        "packet_rows": headline.get("packet_rows", 4),
        "framed_byte_range": headline.get("packet_framed_byte_range", "4-15B"),
        "min_source_state_floor_bytes": headline.get("min_source_state_floor_bytes", 768),
        "min_source_state_floor_vs_largest_packet": headline.get(
            "min_source_state_floor_vs_largest_packet", 51.2
        ),
        "native_nvidia_complete": headline.get("native_nvidia_complete", False),
    }


def _hidden_mlp_numbers(payload: dict[str, Any]) -> dict[str, Any]:
    h = payload.get("headline", {})
    return {
        "pass_gate": bool(payload.get("pass_gate", h.get("pass_gate", False))),
        "selected": (
            f"{h.get('selected_view', 'query_residual')} / "
            f"{h.get('selected_pca_dim', 16)} / {h.get('selected_hidden_dim', 16)}"
        ),
        "matched": h.get("test_matched_accuracy_mean", 0.231712),
        "qwen": h.get("test_qwen_substituted_accuracy_mean", 0.317125),
        "cached_tiny": h.get("test_cached_tiny_packet_accuracy_mean", 0.269345),
        "ci_low_qwen": h.get("test_paired_ci95_low_vs_qwen_substituted_min", -0.154334),
    }


def _llama_probe_numbers(payload: dict[str, Any]) -> dict[str, Any]:
    headline = payload.get("headline", {})
    split = payload.get("split_summaries", {}).get("test", {})
    return {
        "source_qwen_oracle": headline.get(
            "test_source_qwen_oracle_accuracy", split.get("source_qwen_oracle_accuracy", 0.613)
        ),
        "packet_oracle": headline.get(
            "test_llama_qwen_packet_oracle_accuracy", split.get("llama_qwen_packet_oracle_accuracy", 0.532)
        ),
        "same_byte_shadow": headline.get(
            "test_same_byte_text_minus_llama_packet", split.get("same_byte_text_minus_llama", 0.126)
        ),
        "source_to_packet_loss": headline.get(
            "test_source_to_llama_packet_loss", split.get("source_to_llama_packet_loss", 0.186)
        ),
    }


def _native_ingest_numbers(payload: dict[str, Any]) -> dict[str, Any]:
    headline = payload.get("headline", payload)
    return {
        "native_systems_complete": headline.get("native_systems_complete", False),
        "paper_native_win_allowed": headline.get("paper_native_win_allowed", False),
        "measurement_rows_ingested": headline.get("measurement_rows_ingested", 0),
        "missing_required_rows": headline.get("missing_required_rows", 11),
    }


def _nodes(facts: dict[str, Any]) -> list[dict[str, Any]]:
    systems = facts["systems"]
    mlp = facts["hidden_mlp"]
    llama = facts["llama_probe"]
    native = facts["native_ingest"]
    return [
        {
            "node_id": "fixed_byte_packet_protocol",
            "status": "promote_for_colm",
            "contribution": "C1 fixed-byte source-private packet protocol",
            "claim": "Tiny task packets can be evaluated with source-destroying controls.",
            "evidence": "ARC/OpenBookQA public-basis packet rows plus destructive controls remain the positive core.",
            "decision": "Keep as contribution, but do not call it solved latent reasoning.",
            "next_action": "Tie every method row to target-only, same-byte text, zero-source, wrong-row, and paired CI.",
            "artifact": "paper/source_private_iclr_colm_readiness_update_20260502.md",
        },
        {
            "node_id": "public_basis_benchmark_gates",
            "status": "promote_for_colm",
            "contribution": "C2 public-basis benchmark gate and negative ladder",
            "claim": "ARC/OpenBookQA common-coordinate packets are the strongest benchmark evidence.",
            "evidence": "The live branch now separates positives from HellaSwag and TinyLlama hidden/query failures.",
            "decision": "Use as benchmark/evaluation contribution, not as final cross-family proof.",
            "next_action": "Add gate-tree figure and larger frozen slices when a connector clears validation.",
            "artifact": "paper/source_private_iclr_colm_readiness_update_20260502.md",
        },
        {
            "node_id": "systems_boundary_accounting",
            "status": "promote_for_colm",
            "contribution": "C3 systems byte/exposure accounting",
            "claim": "LatentWire packets are much smaller than source-state/KV transfer floors.",
            "evidence": (
                f"{systems['packet_rows']} packet rows, {systems['framed_byte_range']} framed bytes, "
                f"{_fmt(systems['min_source_state_floor_vs_largest_packet'])}x floor over largest packet."
            ),
            "decision": "Promote as accounting; native GPU systems win remains blocked.",
            "next_action": "Fill native ingest rows for vLLM/SGLang/C2C/KVComm/KVCOMM/QJL/TurboQuant.",
            "artifact": _display(ARTIFACTS["systems_boundary"]),
        },
        {
            "node_id": "tinyllama_mean_cache_connectors",
            "status": "cut",
            "contribution": "negative method ladder",
            "claim": "Mean hidden/query caches are enough for a low-data connector.",
            "evidence": (
                f"MLP matched/Qwen/cached={_fmt(mlp['matched'])}/{_fmt(mlp['qwen'])}/"
                f"{_fmt(mlp['cached_tiny'])}, CI low vs Qwen={_fmt(mlp['ci_low_qwen'])}; "
                "PCA/ridge, transport, and sparse-query variants were also negative."
            ),
            "decision": "Cut this family for ARC.",
            "next_action": "Do not spend more Mac cycles on mean-cache PCA/RFF/MLP variants.",
            "artifact": _display(ARTIFACTS["hidden_query_mlp"]),
        },
        {
            "node_id": "source_choice_scouts",
            "status": "cut_or_reprompt_only",
            "contribution": "negative source-family ladder",
            "claim": "A stronger non-Qwen source-choice sender fixes cross-family transfer.",
            "evidence": (
                f"Llama source/Qwen oracle={_fmt(llama['source_qwen_oracle'])}, packet oracle="
                f"{_fmt(llama['packet_oracle'])}, same-byte shadow={_fmt(llama['same_byte_shadow'])}, "
                f"source-to-packet loss={_fmt(llama['source_to_packet_loss'])}."
            ),
            "decision": "Cut current Llama/Phi source-choice senders; only revive with selected prompt/scoring.",
            "next_action": "If revived, select prompt/scoring on validation before frozen test.",
            "artifact": _display(ARTIFACTS["llama_probe"]),
        },
        {
            "node_id": "tokenwise_query_connector",
            "status": "alive_next_gate",
            "contribution": "candidate C4 positive learned connector",
            "claim": "A per-example source-conditioned tokenwise query/soft-prefix interface can beat packet-only.",
            "evidence": "Literature supports learned frozen-backbone connectors, but current LatentWire evidence has not tested tokenwise target loss on ARC.",
            "decision": "This is the next method gate; mean-cache proxies do not substitute for it.",
            "next_action": "Train frozen-endpoint 16-64 query connector with target loss and strict source-destroying controls.",
            "artifact": "tokenwise_connector_runbook.md",
        },
        {
            "node_id": "native_systems_rows",
            "status": "blocked_required",
            "contribution": "C3 systems native proof",
            "claim": "LatentWire is faster or more memory-efficient than native KV/cache baselines.",
            "evidence": (
                f"native_complete={native['native_systems_complete']}; "
                f"paper_native_win_allowed={native['paper_native_win_allowed']}; "
                f"measurements={native['measurement_rows_ingested']}; "
                f"missing_required={native['missing_required_rows']}."
            ),
            "decision": "Block native speed/HBM/goodput claims until measurements exist.",
            "next_action": "Run NVIDIA vLLM/SGLang/C2C/KVComm/KVCOMM/QJL/TurboQuant rows through the validator.",
            "artifact": _display(ARTIFACTS["native_ingest"]),
        },
    ]


def _checklists() -> dict[str, list[str]]:
    return {
        "iclr_full_paper": [
            "Positive tokenwise query/soft-prefix or cache-fuser connector on a frozen larger slice.",
            "OpenBookQA 3B train-only receiver gate over packet-only with source-destroying controls.",
            "At least one strict true cross-family pair, not just same-family Qwen scaling.",
            "Seed repeats and paired uncertainty versus target-only, same-byte text, Qwen-substituted, wrong-row, and prompt/prefix controls.",
            "Direct comparison against C2C/KVComm/KVCOMM and KV quantization byte floors.",
            "Native NVIDIA rows for TTFT, TPOT, goodput, GPU memory, HBM/PCIe/NVLink traffic, accuracy, bytes, and source exposure.",
            "Interpretability telemetry: rate-distortion curve, effective rank or gate patterns, and source/help/harm decomposition.",
        ],
        "colm_workshop": [
            "Scope as source-private packet protocol plus public benchmark positives.",
            "Use systems boundary table as accounting, not native speed.",
            "Include gate-tree figure showing promoted, cut, alive, and blocked branches.",
            "Cut HellaSwag receiver-improvement and TinyLlama hidden/query connector claims.",
            "Frame negative ladder as evidence of rigor, not as solved cross-model latent reasoning.",
        ],
        "blockers_for_user": [
            "NVIDIA access for target-loss tokenwise connector and native systems rows.",
            "Decision on COLM framing: workshop evidence package now, or hold for positive connector.",
            "Any preferred source/target model pair for the first expensive cross-family connector run.",
        ],
    }


def _runbook() -> list[dict[str, Any]]:
    return [
        {
            "step": 1,
            "name": "Mac ARC smoke",
            "goal": "Verify target-forward soft-prefix training code on 8-16 ARC rows without claiming evidence.",
            "pass_rule": "Training runs, controls are emitted, and no source answer/text/KV is transmitted.",
        },
        {
            "step": 2,
            "name": "ARC soft-prefix preflight",
            "goal": "Train/select on ARC validation disagreement rows with frozen source and target endpoints.",
            "pass_rule": "Matched beats target-only, source-free, zero-source, row-shuffled, same-byte text, and Qwen-substituted controls.",
        },
        {
            "step": 3,
            "name": "OpenBookQA 3B receiver gate",
            "goal": "Train-only receiver/query connector over the strongest OpenBookQA packet-only rows.",
            "pass_rule": "Packet-plus-receiver beats packet-only by >= +0.005 with positive paired CI95 low and source-destroying controls.",
        },
        {
            "step": 4,
            "name": "Frozen ARC/OpenBookQA test gate",
            "goal": "Evaluate once on frozen larger slices after validation selection.",
            "pass_rule": "Mean delta >= 0.02 and CI95 low > 0 versus Qwen-substituted, packet-only, target-only, and same-byte text.",
        },
        {
            "step": 5,
            "name": "Cross-family falsification",
            "goal": "Repeat with one true non-Qwen source-target pair.",
            "pass_rule": "Positive paired CI survives; same-family-only result is marked diagnostic.",
        },
        {
            "step": 6,
            "name": "Systems ingest",
            "goal": "Fill native validator rows for LatentWire and all KV/cache baselines.",
            "pass_rule": "Validator allows native claims only after complete matched measurements.",
        },
    ]


def _payload() -> dict[str, Any]:
    facts = {
        "systems": _systems_numbers(_maybe_json(ARTIFACTS["systems_boundary"])),
        "hidden_mlp": _hidden_mlp_numbers(_maybe_json(ARTIFACTS["hidden_query_mlp"])),
        "llama_probe": _llama_probe_numbers(_maybe_json(ARTIFACTS["llama_probe"])),
        "native_ingest": _native_ingest_numbers(_maybe_json(ARTIFACTS["native_ingest"])),
    }
    nodes = _nodes(facts)
    return {
        "gate": "source_private_iclr_gate_tree_and_connector_plan",
        "date": "2026-05-02",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "paper_readiness": "COLM workshop plausible; ICLR full paper blocked.",
        "current_story": "Fixed-byte source-private packets plus public benchmark gates and systems byte/exposure accounting.",
        "exact_gap": (
            "A positive tokenwise learned/common-language connector and matched native systems rows are still missing."
        ),
        "technical_contributions": [
            "Fixed-byte source-private packet protocol with destructive controls.",
            "Public-basis benchmark gates plus negative/falsification ladder.",
            "Systems byte/exposure accounting against KV/cache transfer baselines.",
        ],
        "needs_more_work": [
            "Tokenwise query/soft-prefix connector trained against target loss.",
            "Strict true cross-family positive result.",
            "Native NVIDIA systems measurements.",
        ],
        "next_exact_gate": (
            "Mac-local ARC n32 target-loss soft-prefix preflight using Qwen source features and a frozen target, "
            "followed by the OpenBookQA 3B train-only receiver gate if controls pass."
        ),
        "nodes": nodes,
        "checklists": _checklists(),
        "tokenwise_connector_runbook": _runbook(),
        "required_connector_controls": [
            "target-only",
            "packet-only",
            "target-cache-only",
            "candidate-only",
            "target-derived packet",
            "row-shuffled source packet",
            "random same-rate packet",
            "label-permutation decoder",
            "candidate derangement",
            "same-byte visible text",
            "source-label-copy audit upper bound",
        ],
        "subagent_synthesis": [
            {
                "agent": "Pasteur",
                "finding": (
                    "Repo already has SVAMP target-loss soft-prefix and tokenwise query diagnostics, "
                    "but ARC/OpenBookQA lack target-loss connector infrastructure."
                ),
                "decision": "Implement a new isolated ARC/OpenBookQA soft-prefix preflight instead of editing core bridge modules.",
            },
            {
                "agent": "Huygens",
                "finding": (
                    "Novelty survives only under per-example fixed-byte source-conditioned messaging, "
                    "frozen/train-only receivers, and source-destroying controls."
                ),
                "decision": (
                    "Use Prefix-Tuning, BLIP-2/Flamingo/Perceiver IO, C2C/KVComm, "
                    "QJL, and TurboQuant as mandatory boundaries."
                ),
            },
        ],
        "claim_boundary": {
            "unique": (
                "Per-example source-conditioned rate-limited connector under source-destroying controls and "
                "explicit source-exposure accounting."
            ),
            "not_unique": (
                "Generic soft prompts, static prefix tuning, and KV/cache fusion are prior art and must be baselines."
            ),
            "lateral_branches": (
                "DiT/consistency-style iterative refinement and SAE universal feature spaces are inspirations, "
                "not current positive evidence."
            ),
        },
        "primary_sources": PRIMARY_SOURCES,
    }


def _write_csv(path: pathlib.Path, nodes: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=NODE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for node in nodes:
            writer.writerow({key: node.get(key, "") for key in NODE_COLUMNS})


def _svg_text(x: float, y: float, text: str, *, size: int = 13, weight: str = "400", anchor: str = "middle") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="#111827">{html.escape(text)}</text>'
    )


def _wrap_svg_text(x: float, y: float, text: str, *, width: int, size: int = 12) -> list[str]:
    lines = textwrap.wrap(text, width=width)
    return [_svg_text(x, y + idx * (size + 4), line, size=size) for idx, line in enumerate(lines[:4])]


def _box(x: float, y: float, w: float, h: float, title: str, body: str, *, fill: str, stroke: str) -> str:
    parts = [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="7" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>',
        _svg_text(x + w / 2, y + 25, title, size=14, weight="700"),
        *_wrap_svg_text(x + w / 2, y + 49, body, width=30, size=11),
    ]
    return "\n".join(parts)


def _arrow(x1: float, y1: float, x2: float, y2: float) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        'stroke="#374151" stroke-width="2" marker-end="url(#arrow)"/>'
    )


def _write_svg(path: pathlib.Path, payload: dict[str, Any]) -> None:
    width, height = 1180, 680
    colors = {
        "promote_for_colm": ("#e0f2fe", "#0369a1"),
        "cut": ("#fee2e2", "#b91c1c"),
        "cut_or_reprompt_only": ("#ffedd5", "#c2410c"),
        "alive_next_gate": ("#dcfce7", "#15803d"),
        "blocked_required": ("#ede9fe", "#6d28d9"),
    }
    node_by_id = {node["node_id"]: node for node in payload["nodes"]}
    placements = {
        "fixed_byte_packet_protocol": (40, 100, 230, 115),
        "public_basis_benchmark_gates": (40, 270, 230, 115),
        "systems_boundary_accounting": (40, 440, 230, 115),
        "tinyllama_mean_cache_connectors": (360, 100, 245, 115),
        "source_choice_scouts": (360, 270, 245, 115),
        "tokenwise_query_connector": (735, 150, 250, 130),
        "native_systems_rows": (735, 380, 250, 130),
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">',
        '<path d="M0,0 L10,4 L0,8 z" fill="#374151"/>',
        "</marker>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _svg_text(width / 2, 34, "LatentWire ICLR gate tree: promoted core, ruled-out branches, next gates", size=20, weight="700"),
        _svg_text(155, 74, "Promote / keep", size=15, weight="700"),
        _svg_text(482, 74, "Ruled out on current surface", size=15, weight="700"),
        _svg_text(860, 74, "Alive / blocked gates", size=15, weight="700"),
    ]
    for node_id, (x, y, w, h) in placements.items():
        node = node_by_id[node_id]
        fill, stroke = colors[node["status"]]
        parts.append(_box(x, y, w, h, node["node_id"].replace("_", " "), node["decision"], fill=fill, stroke=stroke))
    parts.extend(
        [
            _arrow(270, 157, 360, 157),
            _arrow(270, 327, 360, 327),
            _arrow(605, 157, 735, 205),
            _arrow(605, 327, 735, 215),
            _arrow(270, 497, 735, 445),
            _svg_text(548, 610, "ICLR blocker: positive tokenwise connector + native systems rows", size=16, weight="700"),
        ]
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# ICLR Gate Tree And Connector Plan",
        "",
        f"- date: `{payload['date']}`",
        f"- readiness: {payload['paper_readiness']}",
        f"- story: {payload['current_story']}",
        f"- exact gap: {payload['exact_gap']}",
        "",
        "## Technical Contributions",
        "",
    ]
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(payload["technical_contributions"], start=1))
    lines.extend(["", "## Next Exact Gate", "", payload["next_exact_gate"]])
    lines.extend(["", "## Gate Tree", "", "| Node | Status | Decision | Next Action |", "|---|---|---|---|"])
    for node in payload["nodes"]:
        lines.append(
            f"| `{node['node_id']}` | `{node['status']}` | {node['decision']} | {node['next_action']} |"
        )
    lines.extend(["", "## ICLR Needs", ""])
    lines.extend(f"- {item}" for item in payload["checklists"]["iclr_full_paper"])
    lines.extend(["", "## COLM Workshop Needs", ""])
    lines.extend(f"- {item}" for item in payload["checklists"]["colm_workshop"])
    lines.extend(["", "## Blockers For User Help", ""])
    lines.extend(f"- {item}" for item in payload["checklists"]["blockers_for_user"])
    lines.extend(["", "## Tokenwise Connector Runbook", "", "| Step | Name | Goal | Pass Rule |", "|---:|---|---|---|"])
    for step in payload["tokenwise_connector_runbook"]:
        lines.append(f"| {step['step']} | {step['name']} | {step['goal']} | {step['pass_rule']} |")
    lines.extend(["", "## Required Connector Controls", ""])
    lines.extend(f"- {item}" for item in payload["required_connector_controls"])
    lines.extend(["", "## Subagent Synthesis", ""])
    for item in payload["subagent_synthesis"]:
        lines.append(f"- `{item['agent']}`: {item['finding']} Decision: {item['decision']}")
    lines.extend(["", "## Claim Boundary", ""])
    for key, value in payload["claim_boundary"].items():
        lines.append(f"- `{key}`: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_runbook(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Tokenwise Connector Runbook",
        "",
        "## Method Boundary",
        "",
        payload["claim_boundary"]["unique"],
        "",
        "This must be compared against static prefix/prompt tuning and KV/cache fusion; otherwise the novelty claim is not defensible.",
        "",
        "## Architecture",
        "",
        "- Frozen source and frozen target.",
        "- Tokenwise source activations or KV summaries, not per-choice mean caches.",
        "- 16-64 learned queries or soft-prefix tokens.",
        "- Train only the connector under target loss.",
        "- Transmit only the rate-limited connector output or its packetized form.",
        "",
        "## Required Controls",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["required_connector_controls"])
    lines.extend(
        [
            "- static prompt/prefix tuning",
            "- zero-source connector",
            "- wrong-row/shuffled-source connector",
            "- label-shuffled training",
            "- Qwen-substituted or target-family packet",
            "- C2C/KVComm/KVCOMM cache-transfer baseline",
            "- QJL/TurboQuant byte floors for source-state transfer",
        ]
    )
    lines.extend(
        [
        "",
        "## Run Gates",
        "",
        ]
    )
    for step in payload["tokenwise_connector_runbook"]:
        lines.extend(
            [
                f"### {step['step']}. {step['name']}",
                "",
                f"Goal: {step['goal']}",
                "",
                f"Pass rule: {step['pass_rule']}",
                "",
            ]
        )
    lines.extend(["## Primary Sources", ""])
    for key, url in payload["primary_sources"].items():
        lines.append(f"- `{key}`: {url}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_gate_tree(output_dir: pathlib.Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _payload()
    json_path = output_dir / "iclr_gate_tree_and_connector_plan.json"
    csv_path = output_dir / "iclr_gate_tree_nodes.csv"
    md_path = output_dir / "iclr_gate_tree_and_connector_plan.md"
    svg_path = output_dir / "iclr_gate_tree.svg"
    runbook_path = output_dir / "tokenwise_connector_runbook.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(csv_path, payload["nodes"])
    _write_markdown(md_path, payload)
    _write_svg(svg_path, payload)
    _write_runbook(runbook_path, payload)
    manifest = {
        "gate": payload["gate"],
        "created_utc": payload["created_utc"],
        "files": [
            {"path": _display(path), "sha256": _sha256_file(path), "bytes": _resolve(path).stat().st_size}
            for path in (json_path, csv_path, md_path, svg_path, runbook_path)
        ],
        "inputs": {key: _display(path) for key, path in ARTIFACTS.items() if _resolve(path).exists()},
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_gate_tree(args.output_dir)
    print(
        json.dumps(
            {
                "paper_readiness": payload["paper_readiness"],
                "node_count": len(payload["nodes"]),
                "exact_gap": payload["exact_gap"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
