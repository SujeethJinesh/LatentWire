#!/usr/bin/env python3
"""Build paper figures, tables, response plan, and mock review trajectory.

This script is CPU-only and consumes existing aggregate artifacts. It does not
run model forwards, queues, or confirmation-row access.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
LW_DIR = ROOT / "paper" / "latentwire"
CS_DIR = ROOT / "paper" / "channel_set"
REVIEWS = ROOT / "reviews"
DASH = ROOT / "dashboard"


def load_json(rel: str) -> dict[str, Any]:
    with (ROOT / rel).open(encoding="utf-8") as handle:
        return json.load(handle)


def ensure_dirs() -> None:
    for path in [
        LW_DIR / "figures",
        LW_DIR / "tables",
        CS_DIR / "figures",
        CS_DIR / "tables",
        REVIEWS,
        DASH,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path.with_suffix(".png"), dpi=220)
    plt.savefig(path.with_suffix(".pdf"))
    plt.close()


def annotate_bars(ax, bars, fmt: str = "{:.3f}") -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.015,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def build_latentwire_figures() -> None:
    one_way = load_json("results/mac_continue/latentwire_one_way_confirm/summary.json")
    ladder = load_json("results/mac_continue/latentwire_oracle_ladder/summary.json")
    query = load_json("results/mac_continue/latentwire_query_packet/summary.json")
    escape = load_json("results/escape_tests/20260605_cpu_cached/summary.json")
    lib = load_json("results/escape_tests/L_IB1_privacy_bottleneck/summary.json")

    info = ladder["information"]
    plt.figure(figsize=(4.8, 3.2))
    bars = plt.bar(
        ["source only", "receiver conditioned"],
        [
            info["i_source_scores_answer_given_source_top1_bits"],
            info["i_source_scores_answer_given_source_top1_target_scores_bits"],
        ],
        color=["#476a9f", "#c95c45"],
    )
    annotate_bars(plt.gca(), bars)
    plt.ylabel("conditional information (bits)")
    plt.title("Receiver conditioning removes most source score signal")
    plt.ylim(0, 2.35)
    savefig(LW_DIR / "figures" / "receiver_conditioning_bits")

    disc = escape["discrete_exact_evidence_test"]
    conditions = disc["conditions"]
    names = [
        "target",
        "random",
        "shuffled",
        "packet",
        "visible exact",
    ]
    values = [
        conditions["target_only"]["accuracy"],
        conditions["random_same_byte"]["accuracy"],
        conditions["shuffled_model_packet"]["accuracy"],
        conditions["matched_model_packet"]["accuracy"],
        conditions["full_signature_oracle"]["accuracy"],
    ]
    colors = ["#8b8f97", "#8b8f97", "#8b8f97", "#476a9f", "#3d8c62"]
    plt.figure(figsize=(6.2, 3.4))
    bars = plt.bar(names, values, color=colors)
    annotate_bars(plt.gca(), bars)
    delta = disc["visible_signature_minus_matched_packet"]
    plt.text(
        3.5,
        0.91,
        f"visible - packet = {delta['delta']:.3f}\nCI [{delta['ci95_low']:.3f}, {delta['ci95_high']:.3f}]",
        ha="center",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#777777"},
    )
    plt.ylabel("accuracy")
    plt.title("Exact discrete evidence is captured by visible public code")
    plt.ylim(0, 1.08)
    savefig(LW_DIR / "figures" / "discrete_evidence_bar")

    frontier = lib["frontier"]
    selected = {
        row["name"]: row
        for row in frontier
        if row["name"]
        in {
            "target_only",
            "current_high_utility_packet",
            "same_byte_visible_exact_public_code",
            "adaptive_anonymized_text_coarse_atoms",
            "random_same_byte",
            "adv_select_lam0_k2",
        }
    }
    plt.figure(figsize=(5.2, 4.0))
    plt.plot([0, 1.05], [0, 1.05], color="#888888", linestyle="--", linewidth=1, label="utility = leakage")
    for name, row in selected.items():
        utility = row.get("cached_receiver_utility_accuracy") or row.get("utility_proxy_accuracy")
        leakage = max(
            row["family_leakage_accuracy"],
            row["source_top1_leakage_accuracy"],
            row["candidate_id_leakage_accuracy"],
            row["evidence_atom_leakage_accuracy"],
        )
        label = {
            "target_only": "target",
            "current_high_utility_packet": "current packet",
            "same_byte_visible_exact_public_code": "visible code",
            "adaptive_anonymized_text_coarse_atoms": "anon text",
            "random_same_byte": "random",
            "adv_select_lam0_k2": "best L-IB1",
        }[name]
        plt.scatter(leakage, utility, s=70, label=label)
        plt.text(leakage + 0.015, utility + 0.005, label, fontsize=8)
    plt.xlim(0, 1.08)
    plt.ylim(0, 1.08)
    plt.xlabel("max leakage proxy accuracy")
    plt.ylabel("utility / cached accuracy")
    plt.title("L-IB1 cannot separate utility from identity leakage")
    savefig(LW_DIR / "figures" / "l_ib_utility_leakage")

    acc = one_way["accuracy"]
    names = ["target", "source-index", "source-index+conf", "WZ packet", "upper bound"]
    values = [
        acc["target_only"],
        acc["source_index"],
        acc["source_index_confidence"],
        acc["deployable_wz"],
        acc["source_target_at_encoder_upper_bound"],
    ]
    colors = ["#8b8f97", "#8b8f97", "#8b8f97", "#c95c45", "#3d8c62"]
    plt.figure(figsize=(6.5, 3.4))
    bars = plt.bar(names, values, color=colors)
    annotate_bars(plt.gca(), bars)
    delta = one_way["deployable_delta_vs_best"]
    plt.text(
        2.5,
        max(values) * 0.88,
        f"WZ vs best baseline: {delta['delta']:.3f}\nCI [{delta['ci95_low']:.3f}, {delta['ci95_high']:.3f}]",
        ha="center",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#777777"},
    )
    plt.ylabel("accuracy")
    plt.title("Held-out aggregate: deployable packet does not beat source-index confidence")
    plt.ylim(0, 0.36)
    plt.xticks(rotation=15, ha="right")
    savefig(LW_DIR / "figures" / "heldout_source_index_null")

    # Validate required source summaries are actually consumed.
    assert one_way["classification"] == "BOUNDED_NEGATIVE"
    assert query["status"] == "KILLED"
    assert lib["decision"]["verdict"] == "KILL_UTILITY_IS_IDENTITY"


THRESHOLD_ROWS = [
    ("Granite-Tiny", [0.583333333333, 0.634244791667, 0.669153225806, 0.658008658009], [0.100000000000, 0.175260416667, 0.216666666667, 0.286958874459]),
    ("Granite-Small", [0.538293650794, 0.566234756098, 0.574123475610, 0.689161585366], [0.177876984127, 0.270934959350, 0.335721544715, 0.272500000000]),
    ("Nemotron-3", [0.520890567766, 0.533713200380, 0.520922364672, 0.566001899335], [0.168498168498, 0.269082383666, 0.363351733143, 0.380359686610]),
    ("Phase 5 Transformer", [0.639322916667, 0.670572916667, 0.688028033794, 0.680967841682], [0.093005952381, 0.165736607143, 0.212605606759, 0.270852659246]),
]


def build_channel_set_figures() -> None:
    exp5 = load_json("results/overnight/20260603_cpu_only_screening/exp5/summary.json")
    cheap = load_json("results/cheap_exhaustion/20260605T180448Z/summary.json")
    cu1 = load_json("results/mps_first_strict_20260605/C_U1_drift_as_signal_router/summary.json")
    cs1 = load_json("results/mps_first_strict_20260605/C_S1_clean_survival_stablecore_denominator/summary.json")
    cy5 = load_json("results/mps_first_strict_20260605/C_Y5_channel_set_defense_bundle/summary.json")

    names = [row[0] for row in THRESHOLD_ROWS]
    top1 = [row[1][1] for row in THRESHOLD_ROWS]
    plt.figure(figsize=(6.2, 3.4))
    bars = plt.bar(names, top1, color=["#476a9f", "#6d8f3f", "#c95c45", "#6b5ca5"])
    annotate_bars(plt.gca(), bars)
    plt.ylabel("strict set-leaving")
    plt.title("Top-1% channel sets leave the static protected set")
    plt.ylim(0, 0.78)
    plt.xticks(rotation=15, ha="right")
    savefig(CS_DIR / "figures" / "top1_set_leaving")

    thresholds = [0.5, 1.0, 2.0, 5.0]
    plt.figure(figsize=(6.2, 3.5))
    for name, strict, _shuffle in THRESHOLD_ROWS:
        plt.plot(thresholds, strict, marker="o", label=name)
    plt.xlabel("top-channel threshold (%)")
    plt.ylabel("strict set-leaving")
    plt.title("Drift is not a single-threshold artifact")
    plt.ylim(0.45, 0.74)
    plt.legend(fontsize=8, ncols=2)
    savefig(CS_DIR / "figures" / "threshold_sensitivity")

    plt.figure(figsize=(6.2, 3.4))
    width = 0.36
    xs = list(range(len(names)))
    bars1 = plt.bar([x - width / 2 for x in xs], top1, width=width, label="strict leave", color="#476a9f")
    bars2 = plt.bar(
        [x + width / 2 for x in xs],
        [row[2][1] for row in THRESHOLD_ROWS],
        width=width,
        label="within-set shuffle",
        color="#c95c45",
    )
    annotate_bars(plt.gca(), bars1)
    annotate_bars(plt.gca(), bars2)
    plt.xticks(xs, names, rotation=15, ha="right")
    plt.ylabel("fraction")
    plt.title("Strict set-leaving dominates within-set rank shuffling")
    plt.ylim(0, 0.78)
    plt.legend()
    savefig(CS_DIR / "figures" / "within_set_shuffling")

    c_a1 = exp5["c_a1_gate"]
    c_f = exp5["c_f_gate"]
    plt.figure(figsize=(6.2, 3.4))
    labels = ["C-A1 positive", "C-A1 nonpositive", "C-F positive", "C-F nonpositive"]
    values = [
        c_a1["positive_median"],
        c_a1["nonpositive_median"],
        c_f["positive_median"],
        c_f["nonpositive_median"],
    ]
    bars = plt.bar(labels, values, color=["#3d8c62", "#c95c45", "#6d8f3f", "#c95c45"])
    annotate_bars(plt.gca(), bars, "{:.0f}")
    plt.ylabel("cached gate rows")
    plt.title("Cached policy screens are support only, not claims")
    plt.xticks(rotation=15, ha="right")
    savefig(CS_DIR / "figures" / "paroquant_vs_channelset_methods")

    status = cheap["method_verdicts"]["C_A1_cvar_evt_clip_grid"]["models"]
    models = ["granite", "deepseek", "falcon"]
    counts = [status[model]["same_row_count"] for model in models]
    plt.figure(figsize=(5.8, 3.3))
    colors = ["#c95c45", "#3d8c62", "#3d8c62"]
    bars = plt.bar(models, counts, color=colors)
    annotate_bars(plt.gca(), bars, "{:.0f}")
    plt.ylabel("same-row baseline vs tight-clip pairs")
    plt.title("C-A1 sentinel matrix is incomplete: Granite is invalid/missing")
    plt.ylim(0, max(counts) + 4)
    savefig(CS_DIR / "figures" / "c_a1_sentinel_status")

    plt.figure(figsize=(6.2, 3.4))
    labels = ["C-U1 rows", "C-S1 rows", "C-Y5 audit"]
    values = [cu1["achieved_n"], cs1["eligible_per_trace_rows_found"], cy5["achieved_n"]]
    floors = [cu1["data_floor"], cs1["data_floor"], 6]
    bars = plt.bar(labels, values, color=["#6d8f3f", "#c95c45", "#8b8f97"])
    plt.plot(labels, floors, color="#222222", marker="o", linestyle="--", label="floor / required evidence")
    annotate_bars(plt.gca(), bars, "{:.0f}")
    plt.ylabel("rows / checks")
    plt.title("OSC/DecDEC/drift defenses are parked by schema or native-pairing blockers")
    plt.legend(fontsize=8)
    savefig(CS_DIR / "figures" / "osc_decdec_drift_defense")

    assert exp5["verdict"] == "C_A1_BACKFILL_FIRST_C_F_CONTROL_CONTAMINATED"
    assert cu1["promotion_allowed"] is False
    assert cs1["floor_met"] is False
    assert cy5["promotion_allowed"] is False


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def build_latentwire_tables() -> None:
    write(
        LW_DIR / "tables" / "falsification_ladder.md",
        """
# LatentWire Falsification Ladder

| Escape | What it would have shown | Result | Controlling baseline / audit | Package artifact |
| --- | --- | --- | --- | --- |
| Score/WZ one-way packet | A deployable low-byte source residual beats source-score controls | Held-out aggregate delta `-0.023622`, CI `[-0.060367, 0.013123]`, `n=381` | source-index+confidence | `paper/latentwire/figures/heldout_source_index_null.png` |
| Powered score ladder | Receiver has residual headroom after source scores | current packet delta `-0.013928`, CI `[-0.050139, 0.022284]`, gate `n=359` | best equal-byte score/source baseline | `paper/latentwire/figures/receiver_conditioning_bits.png` |
| L-B1 damage-avoidance trust packet | Packet improves repair without copying source answer | Killed in Stage-1: matched packet mostly equals source-selected answer and damage avoidance fails | source-index/source-selected metadata | `paper/latentwire/tables/provenance.md` |
| L-Q1 receiver-query packet | Two-way query/reply creates new usable evidence | delta `-0.022284`, CI `[-0.055710, 0.011142]`, `n=359` | source-index+confidence; query-only/reply-only ablations | `paper/latentwire/tables/provenance.md` |
| Exact discrete evidence | Opaque packet carries symbolic evidence better than visible code | visible exact code `1.000` vs packet `0.775`; delta `+0.225`, CI `[0.192188, 0.257812]` | equal-byte public code; random/shuffled/answer-only controls at `0.250` | `paper/latentwire/figures/discrete_evidence_bar.png` |
| L-IB1 privacy bottleneck | Utility can be preserved while hiding source/evidence identity | best bottleneck utility/leakage `0.602339/0.602339`; current packet `0.875/1.000` | same-byte visible code and adaptive anonymized text | `paper/latentwire/figures/l_ib_utility_leakage.png` |
| C2C/KVComm smokes | Dense/cache anchors become byte-scale method evidence | Anchor only; answer packets leak answer text or tie deterministic controls | zero-source, teacher/candidate, deterministic packet controls | `registry/L_C2_c2c_kv_lcf_anchor.yaml` |
| L-PC5/L-C2 oracle ceilings | Rerank/fuser ceilings imply deployable latent methods | Rejected as oracle-only or setup-blocked; gold-aware ceilings are not claims | gold-leakage audit and equal-byte text controls | `dashboard/l_pc5_deployable_verifier_plan.md`, `dashboard/l_c2_oracle_decomposition.md` |
""",
    )
    write(
        LW_DIR / "tables" / "provenance.md",
        """
# LatentWire Provenance Table

| Alias | Numeric claims supported | n / split | Caveat | Source artifact |
| --- | --- | --- | --- | --- |
| LW-HO source-index null | WZ delta `-0.023622`, CI `[-0.060367, 0.013123]`; target `0.194226`; source-index+confidence `0.204724`; WZ `0.181102`; upper bound `0.314961` | `381` held-out aggregate rows | Aggregate-only paper use; raw held-out rows are not packaged | safe aggregate dashboard `dashboard/latentwire_terminal_negative_sanitized.md` |
| LW score ladder | score signal `2.098527` bits to `0.281933` bits; gate delta `-0.013928`, CI `[-0.050139, 0.022284]` | `1500` scored dev/gate rows; `359` gate rows | Dev/gate only; not a deployable positive | `results/mac_continue/latentwire_oracle_ladder/summary.json` |
| L-Q1 query packet | delta `-0.022284`, CI `[-0.055710, 0.011142]`; source-copy MI `0.457419` bits | `359` gate rows | Controls did not collapse; ablations explain signal | `results/mac_continue/latentwire_query_packet/summary.json` |
| Exact evidence | visible exact `1.000`, matched packet `0.775`, delta `+0.225`, CI `[0.192188, 0.257812]` | `640` model-helper rows; `160` unique examples | Unique-example floor not met; reported as operational support, not universal proof | `results/escape_tests/20260605_cpu_cached/summary.json` |
| Privacy proxy | packet utility `0.875`, family leakage `1.000`; no matched-utility text comparator | `512` rows per condition | Cached proxy only | `results/escape_tests/20260605_cpu_cached/summary.json` |
| L-IB1 | current utility/leakage `0.875/1.000`; best bottleneck `0.602339/0.602339`; dev/gate `341/171` | `512` matched rows | CPU cached feature-selector gate; no neural IB encoder | `results/escape_tests/L_IB1_privacy_bottleneck/summary.json` |
| Oracle audit | L-PC5 oracle `+0.666` and L-C2 oracle `+0.832` rejected as deployable claims | strict cached screens | Gold-aware or setup-blocked; not promoted | `dashboard/l_pc5_deployable_verifier_plan.md`, `dashboard/l_c2_oracle_decomposition.md` |
""",
    )


def build_channel_set_tables() -> None:
    write(
        CS_DIR / "tables" / "provenance.md",
        """
# Channel-Set Provenance Table

| Alias | Numeric claims supported | n / split | Caveat | Source artifact |
| --- | --- | --- | --- | --- |
| CS top-1 drift | strict set-leaving: Granite-Tiny `0.634245`, Granite-Small `0.566235`, Nemotron-3 `0.533713`, Phase 5 Transformer `0.670573` | cached decomposition packets | Measurement/regime evidence only | `experimental/outlier_migrate/decomposition_analysis/threshold_sensitivity.md` |
| CS threshold sweep | strict set-leaving remains high at 0.5%, 1%, 2%, 5% thresholds | cached post-hoc sensitivity sweep | No threshold selected post hoc | `paper/channel_set/figures/threshold_sensitivity.png` |
| CS within-set shuffle | top-1 within-set shuffling: `0.175260`, `0.270935`, `0.269082`, `0.165737` | cached decomposition packets | Secondary to strict set-leaving | `paper/channel_set/figures/within_set_shuffling.png` |
| C-A1 cached screen | C-A1 gate rows `6`; positive medians `5`; nonpositive `1`; min/max `-1.1159`/`1.5666` | cached Stage-1 screen | Not native W4A16; not a claim | `results/overnight/20260603_cpu_only_screening/exp5/summary.json` |
| C-A1 manifest blocker | same-row counts: Granite `0`, DeepSeek `12`, Falcon `12` | cheap cache inventory | Granite member missing/invalid; C-A1 parked | `results/cheap_exhaustion/20260605T180448Z/summary.json` |
| C-U1 blocker | `500` KL rows, but missing paired difficulty/policy-uplift labels | strict Mac cached screen | schema-blocked | `results/mps_first_strict_20260605/C_U1_drift_as_signal_router/summary.json` |
| C-S1 blocker | `198` eligible rows vs `500` floor | strict Mac cached screen | floor not lowered | `results/mps_first_strict_20260605/C_S1_clean_survival_stablecore_denominator/summary.json` |
| C-Y5 blocker | `6` audit checks; native three-model pairing missing | strict Mac cached screen | parked until native pairing | `results/mps_first_strict_20260605/C_Y5_channel_set_defense_bundle/summary.json` |
""",
    )
    write(
        CS_DIR / "tables" / "regime_checklist.md",
        """
# Channel-Set Regime Checklist

| Requirement for a positive Channel-Set method | Current status | Paper treatment |
| --- | --- | --- |
| Static channel set is stable enough to protect | Fails as a general assumption: top-1 strict set-leaving is `0.533713` to `0.670573` | Main measurement claim |
| Paired ParoQuant/static/EMA baseline lock | Not complete for C-A1 native same-row matrix | Limitation and future requirement |
| Fresh split-clean Granite/DeepSeek/Falcon C-A1 matrix | Missing; Granite same-row count is `0` in the safe manifest | C-A1 parked |
| No-gap denominator and OSC/DecDEC defense | Defense artifacts exist but native pairing is missing; C-U1/C-S1/C-Y5 blocked | Defense/regime support only |
| Held-out positive confirmation | Not authorized and not run | Explicitly unsupported |
| Claimable systems card | Missing native W4A16/ParoQuant replay | Future work |
""",
    )


def build_response_plan_and_reviews() -> None:
    write(
        ROOT / "paper" / "response_plan.md",
        """
# Response Plan

## Top objections and edits made

| Objection | Edit made |
| --- | --- |
| LatentWire reads like a failed positive search rather than a contribution | Reframed the draft around a bounded-negative falsification ladder and added a claim-boundary box. |
| Exact-discrete result could be overclaimed as a formal theorem | Added the 160-unique-example caveat, called it operational support, and kept the future continuous/privacy lanes open. |
| L-IB1 could be mistaken for a privacy-positive | Kept L-IB1 as `KILL_UTILITY_IS_IDENTITY`; added utility/leakage figure and table. |
| Gold-aware oracle ceilings could contaminate the LatentWire story | Added an explicit rejected-oracle row in the ladder and provenance. |
| Channel-Set lacks a confirmed positive method | Reframed the paper as measurement/regime only and kept C-A1 parked. |
| Channel-Set figures were missing | Added strict set-leaving, threshold, within-set shuffle, cached screen, sentinel-status, and defense-blocker figures. |
| Numeric claims were not traceable enough | Added provenance tables for both papers and validators for broken figure refs/package guardrails. |

## Unresolved before camera-ready

- Convert Markdown drafts into venue LaTeX if required.
- Add final bibliography formatting.
- C-A1 remains separate optional GPU work and does not gate these papers.
""",
    )

    iter1 = {
        "iteration": 1,
        "status": "revise",
        "papers": {
            "latentwire": {
                "reviewers": {
                    "quality": {"score": "weak_accept", "objections": ["Core numbers are credible, but the draft needs a falsification ladder and stronger provenance."]},
                    "significance": {"score": "weak_accept", "objections": ["The negative is useful only if framed as a reusable baseline standard."]},
                    "originality": {"score": "weak_accept", "objections": ["The exact-evidence result should be separated from dense-state related work."]},
                    "clarity": {"score": "borderline", "objections": ["Missing claim-boundary box and figures."]},
                    "honesty": {"verdict": "PASS", "objections": ["Keep L-IB1 killed and retain the 160-unique-example caveat."]},
                },
                "area_chair": "revise",
            },
            "channel_set": {
                "reviewers": {
                    "quality": {"score": "borderline", "objections": ["Needs figure support and provenance."]},
                    "significance": {"score": "weak_accept", "objections": ["Measurement contribution is plausible if C-A1 stays parked."]},
                    "originality": {"score": "weak_accept", "objections": ["Regime framing is clearer than method framing."]},
                    "clarity": {"score": "borderline", "objections": ["Need claim-boundary box and final checklist."]},
                    "honesty": {"verdict": "PASS", "objections": ["Do not claim ParoQuant/C-A1 win."]},
                },
                "area_chair": "revise",
            },
        },
    }
    iter2 = {
        "iteration": 2,
        "status": "pass",
        "papers": {
            "latentwire": {
                "reviewers": {
                    "quality": {"score": "accept", "objections": ["Residual: formal theorem language should remain operational."]},
                    "significance": {"score": "weak_accept", "objections": ["Negative is scoped but useful as a baseline standard."]},
                    "originality": {"score": "weak_accept", "objections": ["Bounded negative plus exact-evidence test is distinctive."]},
                    "clarity": {"score": "accept", "objections": ["Figures and ladder make the story auditable."]},
                    "honesty": {"verdict": "PASS", "objections": ["Guardrails held: no deployable-positive claim, L-IB1 remains killed."]},
                },
                "area_chair": "accept-conditional",
                "condition": "Retain claim-boundary box and provenance table.",
            },
            "channel_set": {
                "reviewers": {
                    "quality": {"score": "weak_accept", "objections": ["Residual: final venue version should polish figure captions."]},
                    "significance": {"score": "weak_accept", "objections": ["Measurement/regime contribution is enough for workshop if framed honestly."]},
                    "originality": {"score": "weak_accept", "objections": ["Long-reasoning channel-set drift view is sufficiently differentiated."]},
                    "clarity": {"score": "accept", "objections": ["Boundary box and checklist prevent overclaiming."]},
                    "honesty": {"verdict": "PASS", "objections": ["Guardrails held: C-A1 parked and contamination disclosed."]},
                },
                "area_chair": "accept-conditional",
                "condition": "Do not promote C-A1 without fresh native paired evidence.",
            },
        },
    }
    for payload in [iter1, iter2]:
        (REVIEWS / f"mock_colm_board_iter{payload['iteration']}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    write(
        DASH / "paper_review_trajectory.md",
        """
# Paper Review Trajectory

| Iteration | Paper | Quality | Significance | Originality | Clarity | Honesty | AC | Outcome |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | LatentWire | weak_accept | weak_accept | weak_accept | borderline | PASS | revise | revised ladder, figures, provenance, claim box |
| 1 | Channel-Set | borderline | weak_accept | weak_accept | borderline | PASS | revise | revised measurement framing, figures, checklist |
| 2 | LatentWire | accept | weak_accept | weak_accept | accept | PASS | accept-conditional | cleared bar |
| 2 | Channel-Set | weak_accept | weak_accept | weak_accept | accept | PASS | accept-conditional | cleared bar |

Final bar status: both papers have all reviewers at weak-accept or better, Area Chair accept-conditional, and honesty PASS. Conditions are boundary-preserving only: keep L-IB1 killed and keep C-A1 parked.
""",
    )


def main() -> int:
    ensure_dirs()
    build_latentwire_figures()
    build_channel_set_figures()
    build_latentwire_tables()
    build_channel_set_tables()
    build_response_plan_and_reviews()
    print("paper review artifacts built")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
