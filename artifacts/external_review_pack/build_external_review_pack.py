#!/usr/bin/env python3
"""Build a compact external review pack for OutlierMigrate method search."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "artifacts/external_review_pack"


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    title: str
    method: str
    model: str
    status: str
    run_dirs: tuple[str, ...] = ()
    artifact_files: tuple[str, ...] = ()
    hypothesis: str = ""
    implementation: str = ""
    baseline: str = "BF16 and static top-1% W4A16"
    why: str = ""
    caveats: str = ""
    inspect: str = "Read summary.md, metrics.json, per_trace.csv, decision.json."


EXPERIMENTS: list[Experiment] = [
    Experiment(
        "00_four_model_drift",
        "Four-model strict channel-set drift",
        "strict set-leaving",
        "Granite/Nemotron/DeepSeek/Falcon",
        "PASS",
        (
            "experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z",
            "experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z",
            "experimental/outlier_migrate/phase7/results/om_phase7_falcon_h1_20260512T223600Z",
            "experimental/outlier_migrate/phase9/results/om_v2_m11b_deepseek_20260527T1210Z",
        ),
        hypothesis="Static top activation-channel sets persist through long decode.",
        implementation="Measure strict set-leaving / migration between early and late decode channel sets.",
        why="The diagnostic worked: all measured model families show large channel-set movement, which motivates dynamic or regime-aware protection.",
        caveats="The earliest packets report migration fraction; later paper framing reports strict set-leaving with revised normalizations.",
    ),
    Experiment(
        "01_static_top1_static_union",
        "Static top-1 and static union controls",
        "static protection",
        "Granite",
        "KILL",
        ("experimental/outlier_migrate/phase3/results/om_phase3_20260509T212000Z", "experimental/outlier_migrate/phase4/results/om_phase4_20260511T054000Z"),
        hypothesis="Static or union channel protection should recover long-decode W4A16 quality.",
        implementation="Protect channels selected from fixed calibration positions or unions across calibration positions.",
        why="Static protection is not enough under drift; union/static controls either saturate budget or recover inconsistently.",
        caveats="Early Phase 3/4 packets used Granite-Tiny/Small variants; use as design evidence, not final positive method.",
    ),
    Experiment(
        "02_m2_position_switching",
        "M2 position-conditional switching",
        "M2",
        "Granite",
        "KILL",
        ("experimental/outlier_migrate/phase9/results/om_phase9_m2_granite_small_vac12_finalized_20260514T233800Z",),
        hypothesis="Hard position-conditioned protected sets can follow drift.",
        implementation="Switch protected sets by decode-position bin.",
        why="Hard discontinuities lost to random controls; switching boundaries appear more harmful than stale static sets.",
    ),
    Experiment(
        "03_m10_position_bins",
        "M10 position-binned scales",
        "M10",
        "Granite",
        "KILL",
        ("experimental/outlier_migrate/phase9/results/om_phase9_m10_granite_small_vac12_20260515T085800Z",),
        hypothesis="Position-binned protection/scaling can track long-decode drift with lower discontinuity than M2.",
        implementation="Use position bins and precomputed scale/protection tables.",
        why="Coarse bins still induced boundary artifacts and did not beat matched controls.",
    ),
    Experiment(
        "04_m11_ema_top1",
        "M11 EMA top-1%",
        "M11",
        "Granite",
        "KILL",
        ("experimental/outlier_migrate/phase9/results/om_phase9_m11_granite_small_vac12_20260516T010728Z",),
        hypothesis="EMA smoothing removes boundary harm while staying in a 1% budget.",
        implementation="EMA-smoothed top-1% protected set at scoring endpoint.",
        why="Directionally less discontinuous, but the 1% budget cannot cover the union of drifting important channels.",
    ),
    Experiment(
        "05_m11b_granite",
        "M11b Granite budget-tuned EMA",
        "M11b",
        "Granite",
        "AMBIG",
        ("experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z",),
        hypothesis="Increasing EMA budget to 5-10% rescues Granite.",
        implementation="EMA-smoothed protection at top-1/top-5/top-10 budgets.",
        why="Top-5 median is positive but CI is wide; no-gap traces and trace heterogeneity limit confidence.",
    ),
    Experiment(
        "06_m11b_nemotron",
        "M11b Nemotron budget-tuned EMA",
        "M11b",
        "Nemotron",
        "PASS",
        ("experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z",),
        hypothesis="Budget-tuned EMA transfers to Nemotron.",
        implementation="Corrected W4A16 static-1% baseline plus M11b top-1/top-5/top-10.",
        why="Top-10 has strong positive recovery and beats static top-10, supporting a Nemotron-specific positive regime.",
    ),
    Experiment(
        "07_m11b_deepseek",
        "M11b DeepSeek",
        "M11b",
        "DeepSeek",
        "AMBIG",
        ("experimental/outlier_migrate/phase9/results/om_v2_m11b_deepseek_20260527T1210Z",),
        hypothesis="M11b top-10 should generalize to dense Transformer reasoning models.",
        implementation="Same M11b budget-scaling protocol on DeepSeek-R1-Distill-Qwen-1.5B.",
        why="Top-10 median is positive but CI crosses negative and static top-10 is competitive.",
    ),
    Experiment(
        "08_m11b_falcon",
        "M11b Falcon-H1",
        "M11b",
        "Falcon-H1",
        "KILL",
        ("experimental/outlier_migrate/phase9/results/om_v2_m11b_falcon_20260527T1438Z",),
        hypothesis="M11b top-10 should help Falcon-H1 parallel hybrid.",
        implementation="Same M11b budget-scaling protocol on Falcon-H1.",
        why="Median recovery is near zero; Falcon appears to resist simple EMA channel protection.",
    ),
    Experiment(
        "09_paroquant_granite",
        "ParoQuant Granite baseline",
        "ParoQuant",
        "Granite",
        "PASS",
        ("experimental/outlier_migrate/phase9/results/om_paroquant_granite_small_20260520T1555Z",),
        hypothesis="A rotation baseline can recover W4A16 Granite quality even when channel-set methods struggle.",
        implementation="ParoQuant-style rotation plus W4A16 scoring.",
        why="High median recovery with positive CI; supports the regime-aware claim that Granite prefers rotation.",
    ),
    Experiment(
        "10_v1_paroquant_nemotron",
        "V1 ParoQuant on Nemotron",
        "ParoQuant",
        "Nemotron",
        "RUNNING",
        ("experimental/outlier_migrate/phase9/results/om_v1_paroquant_nemotron_20260528T0318Z",),
        hypothesis="Test whether rotation also dominates or matches M11b on Nemotron.",
        implementation="ParoQuant W4A16 on Nemotron using reused BF16/static traces and score caches.",
        why="Still running; final comparison to M11b top-10 not available in this pack.",
    ),
    Experiment(
        "11_e3_paroquant_plus_m11b",
        "E3 ParoQuant plus M11b composition",
        "ParoQuant+M11b",
        "Granite",
        "KILL",
        ("experimental/outlier_migrate/phase9/results/om_stage1_e3_granite_20260526T034442Z",),
        hypothesis="Rotation and dynamic channel protection compose additively.",
        implementation="ParoQuant weights plus M11b top-10 protected columns and random matched controls.",
        why="Sub-additive; random matched composition was stronger, indicating overlap or interference.",
    ),
    Experiment(
        "12_m18_activation_k",
        "M18 activation+K coupling",
        "M18",
        "Granite",
        "KILL",
        ("experimental/outlier_migrate/phase9/results/om_phase9_m18_granite_small_vac12_20260516T193500Z",),
        hypothesis="Coupling activation channels with K-side protection captures cross-tensor sensitivity.",
        implementation="Cross-tensor activation+K protected sets.",
        why="Negative recovery; extra coupling appears to inject instability rather than useful selectivity.",
    ),
    Experiment(
        "13_m26_stable_core",
        "M26 stable core",
        "M26",
        "Granite",
        "AMBIG",
        ("experimental/outlier_migrate/phase9/results/om_phase9_m26_granite_small_vac12_20260518T203000Z",),
        hypothesis="Channels stable across calibration positions form a reliable protected core.",
        implementation="Protect stable-core channels selected by persistence across positions.",
        why="Positive but small/wide signal; stable channels alone do not solve drift.",
    ),
    Experiment(
        "14_decdec_proxy",
        "DecDEC proxy",
        "DecDEC proxy",
        "Granite",
        "KILL",
        ("experimental/outlier_migrate/phase9/results/om_phase9_decdec_granite_small_vac12_20260517T141500Z",),
        hypothesis="DecDEC-style dynamic saliency transfers to long-reasoning W4A16.",
        implementation="Algorithmic proxy for short-horizon dynamic channel identification.",
        why="Near-zero/negative recovery; short-horizon dynamic selection does not directly solve long-decode protection.",
    ),
    Experiment(
        "15_kl_fft_diagnostics",
        "KL/FFT diagnostics",
        "KL+FFT",
        "Granite/DeepSeek/Falcon",
        "PASS",
        (
            "experimental/outlier_migrate/phase9/results/om_phase9_kl_granite_small_dense_20260519T085400Z",
            "experimental/outlier_migrate/phase9/results/om_stage1_e1_deepseek_falcon_20260526T2335Z",
        ),
        hypothesis="Long-decode W4A16 failure is simple compound-error accumulation.",
        implementation="Per-position KL fits and spectral/autocorrelation diagnostics.",
        why="Sublinear/sqrt-like KL and broadband FFT weaken the compounding-error and simple-predictor stories.",
    ),
    Experiment(
        "16_e2_cross_prompt",
        "E2 cross-prompt drift replication",
        "cross-prompt drift",
        "Granite/DeepSeek/Falcon",
        "PASS",
        ("experimental/outlier_migrate/phase9/results/om_stage1_e2_mathonly_20260526T1202Z",),
        hypothesis="Drift is AIME-specific.",
        implementation="Narrowed MATH-only cross-prompt replication.",
        why="Enough replication to keep the drift story from being AIME-only, with honest narrowed scope.",
    ),
    Experiment(
        "17_mpred",
        "M-PRED predictive tracker",
        "M-PRED",
        "Granite/DeepSeek/Falcon",
        "KILL",
        (
            "experimental/outlier_migrate/phase9/results/om_phase9_mpred_granite_reduced_20260527T201251Z",
            "experimental/outlier_migrate/phase9/results/om_phase9_mpred_deepseek_extension_20260528T005045Z",
            "experimental/outlier_migrate/phase9/results/om_phase9_mpred_falcon_extension_20260528T011658Z",
        ),
        hypothesis="One-step predictive correction beats EMA lag.",
        implementation="AR/Kalman-style innovation tracker with alpha variants.",
        why="Broadband drift defeats simple prediction; predictor chased noise and underperformed EMA.",
    ),
    Experiment(
        "18_wjac_prefilter",
        "WJAC offline prefilter",
        "WJAC",
        "all cached slices",
        "KILL",
        (),
        ("artifacts/wjac_prefilter/decision.json", "artifacts/wjac_prefilter/report.md", "artifacts/wjac_prefilter/wjac_scores.parquet"),
        hypothesis="Weight-column norm sensitivity changes top-k enough to justify GPU scoring.",
        implementation="Score q_i * EMA(x_i^2) with q_i=||W[:,i]||^2; compare overlap/rank/churn offline.",
        why="Corrected two-of-four kill diagnostics triggered on every covered model/slice.",
    ),
    Experiment(
        "19_lambda_prefilter",
        "LAMBDA layerwise budget prefilter",
        "LAMBDA",
        "DeepSeek/Falcon",
        "SMOKE_ONLY",
        (),
        ("artifacts/funnel_prefilters/decision.json", "artifacts/funnel_prefilters/report.md", "artifacts/funnel_prefilters/smoke_traces.json"),
        hypothesis="Layerwise budget allocation can rescue regimes where flat M11b underperforms.",
        implementation="CPU layer heterogeneity and dominant-layer allocation analysis.",
        why="Heterogeneity is sufficient to authorize smoke, but not evidence yet.",
    ),
    Experiment(
        "20_hyst_prefilter",
        "HYST churn prefilter",
        "HYST",
        "DeepSeek/Falcon",
        "SMOKE_ONLY",
        (),
        ("artifacts/funnel_prefilters/decision.json", "artifacts/funnel_prefilters/report.md", "artifacts/funnel_prefilters/smoke_traces.json"),
        hypothesis="Hysteresis can reduce harmful protected-set churn while preserving local pool stability.",
        implementation="CPU churn and local-pool stability analysis.",
        why="Churn/local-stability pattern authorizes smoke; no endpoint scoring yet.",
    ),
    Experiment(
        "21_msurface_diagnostic",
        "M-SURFACE hook diagnostic",
        "M-SURFACE",
        "Granite/Falcon",
        "DEFERRED",
        (),
        ("artifacts/msurface/decision.json", "artifacts/msurface/hook_map.md", "artifacts/msurface/surface_drift_table.csv"),
        hypothesis="An internal surface has materially lower drift than block output.",
        implementation="Static hook map and cache search for internal activations.",
        why="Surfaces are hookable but cached internal activations are absent; no GPU before LAMBDA/HYST smoke.",
    ),
    Experiment(
        "22_mbranch_diagnostic",
        "M-BRANCH Falcon diagnostic",
        "M-BRANCH",
        "Falcon-H1",
        "DEFERRED",
        (),
        ("artifacts/mbranch/decision.json", "artifacts/mbranch/hook_map.md", "artifacts/mbranch/branch_drift_table.csv"),
        hypothesis="Falcon branch-local drift is lower than post-mixer drift.",
        implementation="Static Falcon branch hook map and cached evidence audit.",
        why="No branch-local cache and no clear implementation reason to spend GPU before smoke.",
    ),
    Experiment(
        "23_restricted_kllook_if_present",
        "Restricted KLLOOK oracle",
        "KLLOOK",
        "Granite/Nemotron optional",
        "DEFERRED",
        (),
        (),
        hypothesis="Oracle KL lookahead exposes channel-set ceiling if cheap methods fail.",
        implementation="Sampled forward-KL lookahead runner exists; no decisive restricted packet in current queue.",
        why="Reserved for after LAMBDA/HYST smoke if ceiling evidence is needed.",
    ),
]


CODE_SNIPPETS = {
    "recovery_metric.py": '''"""Recovery metric used in the review pack."""\n\ndef recovery(perplexity_bf16, perplexity_static, perplexity_method):\n    gap = perplexity_static - perplexity_bf16\n    if gap <= 0:\n        return None  # no recoverable static gap\n    return 1.0 - (perplexity_method - perplexity_bf16) / gap\n''',
    "bootstrap_ci.py": '''"""Small percentile bootstrap median CI."""\nimport random\nfrom statistics import median\n\ndef bootstrap_median_ci(values, samples=1000, seed=20260508):\n    if not values:\n        return {\"ci95_low\": None, \"ci95_high\": None}\n    rng = random.Random(seed)\n    draws = []\n    for _ in range(samples):\n        draws.append(median(rng.choice(values) for _ in values))\n    draws.sort()\n    return {\"ci95_low\": draws[int(0.025 * samples)], \"ci95_high\": draws[int(0.975 * samples) - 1]}\n''',
    "topk_set_leaving.py": '''"""Strict top-k set leaving."""\n\ndef set_leaving(s0, st):\n    s0, st = set(s0), set(st)\n    return 1.0 - len(s0 & st) / max(1, len(s0))\n''',
    "m11b_ema_policy.py": '''"""M11b endpoint EMA policy sketch."""\n\ndef ema_update(scores, topk_indicator, alpha=0.3):\n    return [alpha * ind + (1 - alpha) * old for old, ind in zip(scores, topk_indicator)]\n''',
    "wjac_scoring.py": '''"""WJAC score approximation."""\n\ndef wjac_score(ema_x2, weight_column_norm_sq):\n    return weight_column_norm_sq * ema_x2\n''',
    "lambda_budget_rule.py": '''"""LAMBDA smoke waterfill rule."""\n\ndef waterfill(global_scores, total_budget):\n    # global_scores: [(score, layer, channel), ...]\n    return sorted(global_scores, reverse=True)[:total_budget]\n''',
    "hyst_policy.py": '''"""Hysteresis protected-set update."""\n\ndef hyst_update(previous, top_k, top_2k, budget):\n    keep = set(previous) & set(top_2k)\n    keep.update(top_k)\n    return set(list(keep)[:budget])\n''',
    "parq_baseline_scoring.py": '''"""ParoQuant baseline is scored with the same recovery metric after rotated W4A16 quantization."""\nfrom recovery_metric import recovery\n''',
    "kllook_oracle_if_present.py": '''"""Restricted KLLOOK oracle sketch."""\n\ndef delta_kl(kl_without_channel, kl_with_channel):\n    return kl_without_channel - kl_with_channel\n''',
    "load_pack.py": '''"""Load top-level summary tables from an unpacked review pack."""\nfrom pathlib import Path\nimport csv\n\ndef read_summary(root=Path('.')):\n    with (root / 'tables/experiment_summary.csv').open() as f:\n        return list(csv.DictReader(f))\n''',
}


IDEATION_ROWS = [
    ("M2 position switching", "loses to random", "hard boundaries are cheap to track", "soft/hysteretic transitions", "churn/local-pool traces", "cheap", "low"),
    ("M10 position bins", "coarse bins underperform", "position is sufficient state", "content/layer-conditioned smooth budgets", "per-layer drift and trace metadata", "medium", "medium"),
    ("M11 EMA top-1%", "weak recovery", "1% budget covers moving support", "budget expansion or waterfilling", "budget curves and no-gap rows", "cheap", "low"),
    ("M11b Granite", "positive but wide CI", "budget alone solves all models", "regime selector or rotation for Granite", "ParoQuant/V1 and no-gap analysis", "cheap", "low"),
    ("M11b DeepSeek", "ambiguous", "EMA transfers to dense Transformer", "LAMBDA/HYST smoke or surface placement", "stratified smoke traces", "cheap", "medium"),
    ("M11b Falcon", "near zero", "parallel hybrid drift is EMA-trackable", "branch/surface-local protection", "branch-local activations", "medium", "medium"),
    ("E3 composition", "sub-additive", "rotation and channel protection address independent errors", "method selector rather than stacking", "paired ParoQuant/M11b/control rows", "cheap", "low"),
    ("M18 activation+K", "negative recovery", "cross-tensor coupling is stabilizing", "loss-sensitive single-surface scoring", "KLLOOK/Fisher proxy", "medium", "medium"),
    ("M26 stable core", "small ambiguous signal", "stable channels are sufficient", "stable core plus dynamic top-up with smoke gate", "budget split curves", "medium", "medium"),
    ("DecDEC proxy", "weak/negative", "short-horizon saliency transfers to long decode", "long-horizon local surface diagnostics", "per-position activations", "medium", "medium"),
    ("M-PRED", "large negative on Granite, weak elsewhere", "drift is predictable by simple AR/innovation", "avoid predictors; use robust selection or hysteresis", "FFT entropy and per-trace recoveries", "cheap", "low"),
    ("WJAC", "offline kill diagnostics", "weight norms change channel ranking materially", "full Fisher only if KLLOOK shows headroom", "WJAC parquet and KLLOOK samples", "expensive", "medium"),
]


def run(cmd: list[str]) -> str:
    out = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    return (out.stdout + out.stderr).strip()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def tail_file(path: Path, lines: int = 80) -> str:
    if not path.is_file():
        return "No log file available.\n"
    data = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(data[-lines:]) + "\n"


def preferred_regime(exp: Experiment, results: dict[str, Any]) -> str | None:
    candidates_by_id = {
        "02_m2_position_switching": ["m2_position_conditional"],
        "03_m10_position_bins": ["m10_position_binned_scales", "m10_position_bins"],
        "04_m11_ema_top1": ["m11_alpha_0_3", "m11_ema_top1", "m11_ema"],
        "05_m11b_granite": ["m11b_top5", "m11b_top10"],
        "06_m11b_nemotron": ["m11b_top10", "m11b_top5"],
        "07_m11b_deepseek": ["m11b_top10"],
        "08_m11b_falcon": ["m11b_top10", "m11b_top5"],
        "09_paroquant_granite": ["paroquant_w4a16"],
        "10_v1_paroquant_nemotron": ["paroquant_w4a16"],
        "11_e3_paroquant_plus_m11b": ["paroquant_m11b_top10"],
        "12_m18_activation_k": ["m18_activation_k", "m18_cross_tensor"],
        "13_m26_stable_core": ["m26_stable_core", "stable_core"],
        "14_decdec_proxy": ["decdec_proxy", "decdec_dynamic"],
        "17_mpred": ["mpred_top10_alpha_0_95", "mpred_top5_alpha_0_95"],
    }
    candidates = candidates_by_id.get(exp.experiment_id, [])
    for candidate in candidates:
        if candidate in results:
            return candidate
    lowered = exp.method.lower().replace("-", "").replace(" ", "")
    for key in results:
        normalized = key.lower().replace("_", "").replace("-", "")
        if lowered and lowered in normalized:
            return key
    return None


def result_summary(exp: Experiment, metrics: dict[str, Any]) -> dict[str, Any]:
    results = metrics.get("results_by_regime", {}) if isinstance(metrics, dict) else {}
    preferred = preferred_regime(exp, results)
    if preferred and isinstance(results.get(preferred), dict):
        summary = results[preferred]
        ci = summary.get("bootstrap_ci95", {}) if isinstance(summary.get("bootstrap_ci95"), dict) else {}
        return {
            "best_method": preferred,
            "median_recovery": summary.get("median_recovery"),
            "ci_low": ci.get("ci95_low"),
            "ci_high": ci.get("ci95_high"),
            "no_gap_fraction": summary.get("no_recoverable_static_gap_fraction"),
            "included": summary.get("included_trace_count"),
        }
    best: tuple[str, float | None, dict[str, Any]] | None = None
    for method, summary in results.items():
        if not isinstance(summary, dict):
            continue
        med = summary.get("median_recovery")
        if med is None:
            continue
        if best is None or float(med) > float(best[1]):
            best = (method, float(med), summary)
    if best is None:
        return {"best_method": "", "median_recovery": "", "ci_low": "", "ci_high": "", "no_gap_fraction": "", "included": ""}
    summary = best[2]
    ci = summary.get("bootstrap_ci95", {}) if isinstance(summary.get("bootstrap_ci95"), dict) else {}
    return {
        "best_method": best[0],
        "median_recovery": best[1],
        "ci_low": ci.get("ci95_low"),
        "ci_high": ci.get("ci95_high"),
        "no_gap_fraction": summary.get("no_recoverable_static_gap_fraction"),
        "included": summary.get("included_trace_count"),
    }


def copy_compact(src: Path, dst: Path, omitted: list[dict[str, Any]], experiment_id: str) -> None:
    if not src.exists():
        return
    size = src.stat().st_size
    large_name = any(token in src.name for token in ["activation_magnitudes", "logits", "weights"])
    if size > 8 * 1024 * 1024 or large_name:
        omitted.append(
            {
                "path": rel(src),
                "size": size,
                "experiment": experiment_id,
                "why": "large raw tensor/logit/activation artifact omitted; compact summaries and per-trace metrics are included",
                "regenerate": "rerun the source experiment command in its command_metadata.json/command.sh where available",
            }
        )
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def collect_per_trace(exp: Experiment, run_dir: Path, method_hint: str) -> list[dict[str, Any]]:
    per_trace = read_json(run_dir / "per_trace_metrics.json")
    if not isinstance(per_trace, dict):
        return []
    rows = per_trace.get("traces", [])
    out: list[dict[str, Any]] = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        recoveries = row.get("recoveries") if isinstance(row.get("recoveries"), dict) else {}
        mean_nll = row.get("mean_nll") if isinstance(row.get("mean_nll"), dict) else {}
        perplexities = row.get("perplexities") if isinstance(row.get("perplexities"), dict) else {}
        methods = list(recoveries) or [method_hint]
        for method in methods:
            out.append(
                {
                    "experiment_id": exp.experiment_id,
                    "model": exp.model,
                    "method": method,
                    "baseline": "static_1pct",
                    "trace_id": row.get("prompt_index"),
                    "prompt_id": row.get("prompt_id"),
                    "task": "AIME-2025",
                    "seed": "",
                    "L_bf16": mean_nll.get("bf16"),
                    "L_static": mean_nll.get("static_1pct"),
                    "L_method": mean_nll.get(method),
                    "recoverable_gap": row.get("static_gap"),
                    "no_gap_flag": row.get("no_recoverable_static_gap"),
                    "recovery_unclipped": recoveries.get(method),
                    "recovery_clipped_optional": None if recoveries.get(method) is None else max(0.0, min(1.0, float(recoveries[method]))),
                    "tokens_generated": row.get("scored_tokens"),
                    "final_decode_position": row.get("score_end"),
                    "notes": f"perplexity_bf16={perplexities.get('bf16')}; perplexity_static={perplexities.get('static_1pct')}; perplexity_method={perplexities.get(method)}",
                }
            )
    return out


def protected_sample(run_dir: Path) -> dict[str, Any]:
    protected = read_json(run_dir / "protected_sets.json")
    if not isinstance(protected, dict):
        return {}
    sample: dict[str, Any] = {"source": rel(run_dir / "protected_sets.json"), "regimes": {}}
    for regime, spec in list(protected.get("regimes", {}).items())[:8]:
        layers = spec.get("layers", {}) if isinstance(spec, dict) else {}
        sample["regimes"][regime] = {}
        for layer_key, layer in list(sorted(layers.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else 0))[:3]:
            channels = layer.get("protected_channels", []) if isinstance(layer, dict) else []
            sample["regimes"][regime][layer_key] = {
                "layer_name": layer.get("layer_name") if isinstance(layer, dict) else "",
                "protected_count": layer.get("protected_count") if isinstance(layer, dict) else "",
                "protected_channels_first_40": channels[:40],
            }
    return sample


def build_pack(timestamp: str) -> tuple[Path, Path]:
    pack_dir = OUT_ROOT / f"om_positive_method_pack_{timestamp}"
    tar_path = OUT_ROOT / f"om_positive_method_pack_{timestamp}.tar.gz"
    if pack_dir.exists():
        shutil.rmtree(pack_dir)
    pack_dir.mkdir(parents=True)

    omitted: list[dict[str, Any]] = []
    experiment_summary: list[dict[str, Any]] = []
    per_trace_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    no_gap_rows: list[dict[str, Any]] = []
    commands: list[str] = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    manifest_rows: list[dict[str, Any]] = []

    for exp in EXPERIMENTS:
        exp_dir = pack_dir / "experiments" / exp.experiment_id
        (exp_dir / "plots").mkdir(parents=True)
        run_paths = [ROOT / d for d in exp.run_dirs if (ROOT / d).exists()]
        artifact_paths = [ROOT / d for d in exp.artifact_files if (ROOT / d).exists()]
        primary = run_paths[0] if run_paths else None
        metrics = read_json(primary / "metrics.json") if primary else None
        checker_result = read_json(primary / "checker_result.json") if primary else None
        if not isinstance(metrics, dict):
            metrics = {
                "schema_version": "external_review_pack_synthetic_metrics",
                "status": exp.status,
                "model": exp.model,
                "method": exp.method,
                "source_files": [rel(p) for p in [*run_paths, *artifact_paths]],
            }
        if not isinstance(checker_result, dict):
            checker_result = {"decision": exp.status, "reasons": [exp.why], "source_files": [rel(p) for p in [*run_paths, *artifact_paths]]}
        summary = result_summary(exp, metrics)
        experiment_summary.append(
            {
                "experiment_id": exp.experiment_id,
                "model": exp.model,
                "method": exp.method,
                "status": exp.status,
                "median_recovery": summary["median_recovery"],
                "ci_low": summary["ci_low"],
                "ci_high": summary["ci_high"],
                "no_gap_fraction": summary["no_gap_fraction"],
                "n_traces": metrics.get("trace_count") or metrics.get("effective_trace_count") or metrics.get("total_trace_count") or "",
                "n_positive_gap_traces": summary["included"],
                "key_comparison": summary["best_method"],
                "decision_reason": exp.why,
                "source_files": ";".join(rel(p) for p in [*run_paths, *artifact_paths]),
            }
        )
        write_json(exp_dir / "metrics.json", metrics)
        write_json(exp_dir / "decision.json", checker_result)
        if primary:
            for rel_name in [
                "config.json",
                "command.sh",
                "command_metadata.json",
                "decoding_config.json",
                "quantization_config.json",
                "random_seed.json",
                "model_provenance.json",
                "prompt_manifest.json",
                "checker_result.json",
                "per_trace_metrics.json",
                "control_metrics.json",
                "bootstrap_ci.json",
                "protected_sets.json",
                "protected_trajectories.json",
                "source_artifacts.json",
            ]:
                copy_compact(primary / rel_name, exp_dir / "source_files" / rel_name, omitted, exp.experiment_id)
            for score in sorted((primary / "score_cache").glob("*.json")) if (primary / "score_cache").is_dir() else []:
                copy_compact(score, exp_dir / "source_files" / "score_cache" / score.name, omitted, exp.experiment_id)
            logs_tail = "\n".join(
                [
                    "## stdout tail",
                    tail_file(primary / "logs/stdout.log"),
                    "## stderr tail",
                    tail_file(primary / "logs/stderr.log"),
                    "## run_events tail",
                    tail_file(primary / "run_events.jsonl"),
                ]
            )
            write_text(exp_dir / "logs_tail.txt", logs_tail)
            cmd_meta = read_json(primary / "command_metadata.json")
            if isinstance(cmd_meta, dict) and cmd_meta.get("argv"):
                command = " ".join(str(x) for x in cmd_meta["argv"])
            elif (primary / "command.sh").is_file():
                command = (primary / "command.sh").read_text(encoding="utf-8", errors="replace")
            else:
                command = f"# See source run directory: {rel(primary)}"
            write_text(exp_dir / "command.sh", "#!/usr/bin/env bash\nset -euo pipefail\n" + command)
            commands.append(f"# {exp.experiment_id}\n{command}\n")
            sample = protected_sample(primary)
            if sample:
                write_json(exp_dir / "protected_sets_sample.json", sample)
        else:
            write_text(exp_dir / "logs_tail.txt", "No run directory; see artifact files.\n")
            write_text(exp_dir / "command.sh", "#!/usr/bin/env bash\nset -euo pipefail\n# CPU artifact-only diagnostic; no GPU command.\n")
        for art in artifact_paths:
            dst = exp_dir / "source_files" / art.name
            copy_compact(art, dst, omitted, exp.experiment_id)
        rows = []
        for run_path in run_paths:
            rows.extend(collect_per_trace(exp, run_path, exp.method))
        per_trace_rows.extend(rows)
        write_csv(exp_dir / "per_trace.csv", rows)
        write_json(exp_dir / "config.json", {"experiment": exp.__dict__, "source_run_dirs": [rel(p) for p in run_paths], "artifact_files": [rel(p) for p in artifact_paths]})
        write_text(
            exp_dir / "summary.md",
            f"""# {exp.title}

- **Hypothesis:** {exp.hypothesis}
- **Method implemented:** {exp.implementation}
- **Baseline/control:** {exp.baseline}
- **Models/traces:** {exp.model}
- **Result/status:** {exp.status}
- **Why it worked/failed:** {exp.why}
- **Supporting artifacts:** {', '.join(rel(p) for p in [*run_paths, *artifact_paths]) or 'No direct source artifact present'}
- **Caveats:** {exp.caveats or 'None beyond the scope notes in the source packet.'}
- **External reviewer should inspect:** {exp.inspect}
""",
        )
        write_text(exp_dir / "code_snippet.py", f"# Minimal pointer for {exp.experiment_id}\n# Full implementation lives in the source files listed in summary.md.\n")
        exp_manifest: list[dict[str, Any]] = []
        for file in sorted(exp_dir.rglob("*")):
            if file.is_file():
                exp_manifest.append({"artifact_path": rel(file), "artifact_type": file.suffix.lstrip(".") or "text", "file_size": file.stat().st_size, "source_path": "", "included_or_omitted": "included", "notes": exp.experiment_id})
                manifest_rows.append({"artifact_path": rel(file), "experiment": exp.experiment_id, "artifact_type": file.suffix.lstrip(".") or "text", "file_size": file.stat().st_size, "created_at": datetime.fromtimestamp(file.stat().st_mtime, timezone.utc).isoformat(), "source_path": "", "included_or_omitted": "included", "notes": ""})
        write_csv(exp_dir / "artifacts_manifest.csv", exp_manifest)

    for row in per_trace_rows:
        if str(row.get("no_gap_flag")).lower() == "true":
            no_gap_rows.append(row)

    for row in experiment_summary:
        bootstrap_rows.append(
            {
                "experiment_id": row["experiment_id"],
                "model": row["model"],
                "method": row["method"],
                "median_recovery": row["median_recovery"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "source_files": row["source_files"],
            }
        )

    write_csv(pack_dir / "tables/experiment_summary.csv", experiment_summary)
    write_csv(
        pack_dir / "tables/per_trace_recovery.csv",
        per_trace_rows,
        [
            "experiment_id",
            "model",
            "method",
            "baseline",
            "trace_id",
            "prompt_id",
            "task",
            "seed",
            "L_bf16",
            "L_static",
            "L_method",
            "recoverable_gap",
            "no_gap_flag",
            "recovery_unclipped",
            "recovery_clipped_optional",
            "tokens_generated",
            "final_decode_position",
            "notes",
        ],
    )
    write_csv(pack_dir / "tables/per_trace_losses.csv", per_trace_rows)
    write_csv(pack_dir / "tables/no_gap_traces.csv", no_gap_rows)
    write_csv(pack_dir / "tables/model_method_matrix.csv", experiment_summary)
    write_csv(pack_dir / "tables/bootstrap_ci_summary.csv", bootstrap_rows)
    write_csv(pack_dir / "tables/wjac_prefilter_summary.csv", wjac_rows())
    write_csv(pack_dir / "tables/lambda_layer_heterogeneity.csv", flatten_json_rows(ROOT / "artifacts/funnel_prefilters/decision.json", "lambda"))
    write_csv(pack_dir / "tables/hyst_prefilter_summary.csv", flatten_json_rows(ROOT / "artifacts/funnel_prefilters/decision.json", "hyst"))
    write_csv(pack_dir / "tables/surface_branch_diagnostics.csv", surface_branch_rows())
    write_csv(pack_dir / "tables/drift_summary_by_model.csv", drift_model_rows())
    write_csv(pack_dir / "tables/drift_summary_by_layer.csv", drift_layer_rows())
    write_csv(pack_dir / "tables/churn_summary.csv", flatten_json_rows(ROOT / "artifacts/funnel_prefilters/decision.json", "churn"))

    write_top_level_docs(pack_dir, experiment_summary, omitted)
    write_code_snippets(pack_dir)
    write_scripts(pack_dir)
    write_plots(pack_dir)
    write_provenance(pack_dir, commands, manifest_rows, omitted)
    write_ideation_map(pack_dir)
    write_omitted(pack_dir, omitted)

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(pack_dir, arcname=pack_dir.name)
    return pack_dir, tar_path


def wjac_rows() -> list[dict[str, Any]]:
    decision = read_json(ROOT / "artifacts/wjac_prefilter/decision.json")
    if not isinstance(decision, dict):
        return []
    rows = []
    for item in decision.get("summaries", []):
        if isinstance(item, dict):
            rows.append(
                {
                    "model_key": item.get("model_key"),
                    "decision": "KILL" if item.get("corrected_kill") else "KEEP",
                    "diagnostic_count": item.get("corrected_kill_diagnostic_count"),
                    "topk_overlap": item.get("median_wjac_vs_m11b_topk_overlap"),
                    "spearman_wjac_ema": item.get("median_spearman_wjac_vs_ema"),
                    "weight_norm_cv": item.get("median_q_cv"),
                    "churn": item.get("median_wjac_churn_vs_m11b"),
                    "source": "artifacts/wjac_prefilter/decision.json",
                }
            )
    return rows


def flatten_json_rows(path: Path, label: str) -> list[dict[str, Any]]:
    data = read_json(path)
    if not isinstance(data, dict):
        return []
    return [{"source": rel(path), "label": label, "json": json.dumps(data.get("decision", data), sort_keys=True)[:30000]}]


def surface_branch_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in [ROOT / "artifacts/msurface/decision.json", ROOT / "artifacts/mbranch/decision.json"]:
        data = read_json(path)
        if isinstance(data, dict):
            rows.append({"source": rel(path), "decision": data.get("decision"), "recommend": data.get("recommend_msurface_now", data.get("recommend_mbranch")), "reason": json.dumps(data.get("decision", data))[:2000]})
    return rows


def drift_model_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in [
        ROOT / "experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z/metrics.json",
        ROOT / "experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z/metrics.json",
        ROOT / "experimental/outlier_migrate/phase7/results/om_phase7_falcon_h1_20260512T223600Z/metrics.json",
    ]:
        data = read_json(path)
        if isinstance(data, dict):
            ci = data.get("bootstrap_ci95", {})
            rows.append({"model": data.get("model_id"), "migration_fraction": data.get("migration_fraction"), "ci_low": ci.get("ci95_low"), "ci_high": ci.get("ci95_high"), "trace_count": data.get("trace_count"), "source": rel(path)})
    return rows


def drift_layer_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in [
        ROOT / "experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z/metrics.json",
        ROOT / "experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z/metrics.json",
        ROOT / "experimental/outlier_migrate/phase7/results/om_phase7_falcon_h1_20260512T223600Z/metrics.json",
    ]:
        data = read_json(path)
        if isinstance(data, dict):
            for layer in data.get("layer_metrics", [])[:200]:
                rows.append({"model": data.get("model_id"), "layer_index": layer.get("layer_index"), "migration_fraction_mean": layer.get("migration_fraction_mean"), "trace_count": layer.get("trace_count"), "source": rel(path)})
    return rows


def write_top_level_docs(pack_dir: Path, summary_rows: list[dict[str, Any]], omitted: list[dict[str, Any]]) -> None:
    status_lines = "\n".join(f"- `{r['experiment_id']}`: {r['status']} ({r['method']} on {r['model']}); median={r['median_recovery']}" for r in summary_rows)
    write_text(
        pack_dir / "INDEX.md",
        """# OutlierMigrate Positive-Method External Review Pack

Recommended read order:

1. `EXECUTIVE_SUMMARY.md`
2. `tables/experiment_summary.csv`
3. `QUESTIONS_FOR_REVIEWER.md`
4. `experiments/18_wjac_prefilter/summary.md`
5. `experiments/19_lambda_prefilter/summary.md`
6. `experiments/20_hyst_prefilter/summary.md`
7. `IDEATION_MAP.md`

Top-level folders:

- `experiments/`: one folder per branch with summary, metrics, per-trace data, decisions, logs, and compact source files.
- `tables/`: cross-experiment CSVs.
- `plots/`: compact visual summaries generated from tables.
- `code_snippets/`: minimal standalone method/metric sketches.
- `scripts/`: GPU-free helpers for inspecting this pack.
- `provenance/`: git/env/model/artifact provenance.

Large activation tensors are intentionally omitted; see `OMITTED_LARGE_ARTIFACTS.md`.
""",
    )
    write_text(
        pack_dir / "EXECUTIVE_SUMMARY.md",
        f"""# Executive Summary

Current framing: long-reasoning W4A16 channel protection is regime-dependent.
Static and hard-switch policies fail under decode-time channel-set drift.
Budgeted EMA succeeds in the Nemotron MoE-hybrid regime, ParoQuant succeeds in
the Granite dense-hybrid regime, and DeepSeek/Falcon remain the live test
surface for cheap positive-method rescue via LAMBDA/HYST smoke.

Current blocker: V1 ParoQuant-on-Nemotron is still running in the source repo.
This pack marks it `RUNNING` and includes progress/logs, not final numbers.
No GPU jobs were launched to create this pack.

Status snapshot:

{status_lines}

Firm results: four-model drift, M11b Nemotron, ParoQuant Granite, E3
sub-additivity, M-PRED KILL, WJAC offline KILL. Smoke-only/pending:
LAMBDA/HYST and V1 ParoQuant-on-Nemotron. Diagnostics only: M-SURFACE and
M-BRANCH.
""",
    )
    write_text(pack_dir / "RUN_LEDGER_SNAPSHOT.md", (ROOT / "RUN_LEDGER.md").read_text(encoding="utf-8"))
    write_text(pack_dir / "DECISIONS_SNAPSHOT.md", (ROOT / "DECISIONS.md").read_text(encoding="utf-8"))
    write_text(
        pack_dir / "QUESTIONS_FOR_REVIEWER.md",
        """# Questions for External Reviewer

- Are LAMBDA/HYST actually promising based on the prefilters?
- Did WJAC really deserve to be killed offline under the corrected 2-of-4 rule?
- Is there hidden signal in per-trace or per-layer data?
- Are no-gap traces distorting median recovery or CI interpretation?
- Are there architecture-local signals suggesting M-SURFACE or M-BRANCH?
- Is the regime-aware framing supported by raw evidence?
- What positive method, if any, should be tried next?
""",
    )
    write_text(
        pack_dir / "GLOBAL_CONFIG.md",
        f"""# Global Config

- Repo commit: `{run(['git', 'rev-parse', 'HEAD'])}`
- Branch: `{run(['git', 'branch', '--show-current'])}`
- Python/env: see `provenance/environment.txt`
- GPU: `{run(['bash', '-lc', "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true"])}`
- Models: see `provenance/model_checkpoints.csv`
- Prompt sets: AIME-2025 deterministic indices 0-11/0-23 depending on packet; smoke traces are fixed in `artifacts/funnel_prefilters/smoke_traces.json`.
- Quantization: W4A16 symmetric per-output-channel int4 unless a packet states ParoQuant rotation.
- Recovery definition: `1 - (PPL_method - PPL_BF16) / (PPL_static_1pct - PPL_BF16)`.
- No-gap handling: traces with `PPL_static_1pct <= PPL_BF16` are marked no-gap and excluded from median recovery.
- Bootstrap: packet-specific percentile/BCa fields are copied as reported; see `tables/bootstrap_ci_summary.csv`.
""",
    )


def write_code_snippets(pack_dir: Path) -> None:
    for name, text in CODE_SNIPPETS.items():
        write_text(pack_dir / "code_snippets" / name, text)


def write_scripts(pack_dir: Path) -> None:
    scripts = {
        "load_pack.py": '''from pathlib import Path\nimport csv\n\ndef read_experiment_summary(root=Path(__file__).resolve().parents[1]):\n    with (root / "tables/experiment_summary.csv").open() as f:\n        return list(csv.DictReader(f))\n''',
        "print_summary.py": '''from pathlib import Path\nimport csv\nroot = Path(__file__).resolve().parents[1]\nwith (root / "tables/experiment_summary.csv").open() as f:\n    rows = list(csv.DictReader(f))\nprint(f"experiments={len(rows)}")\nfor r in rows:\n    print(f"{r['experiment_id']}: {r['status']} {r['method']} {r['model']} median={r['median_recovery']}")\n''',
        "inspect_trace.py": '''import argparse, csv\nfrom pathlib import Path\np=argparse.ArgumentParser(); p.add_argument("--trace_id", required=True); a=p.parse_args()\nroot=Path(__file__).resolve().parents[1]\nwith (root/"tables/per_trace_recovery.csv").open() as f:\n    for r in csv.DictReader(f):\n        if str(r.get("trace_id")) == str(a.trace_id): print(r)\n''',
        "compare_methods.py": '''import argparse, csv\nfrom pathlib import Path\np=argparse.ArgumentParser(); p.add_argument("--model", required=True); p.add_argument("--method_a", required=True); p.add_argument("--method_b", required=True); a=p.parse_args()\nroot=Path(__file__).resolve().parents[1]\nrows=list(csv.DictReader((root/"tables/per_trace_recovery.csv").open()))\nfor method in [a.method_a, a.method_b]:\n    vals=[float(r["recovery_unclipped"]) for r in rows if a.model.lower() in r["model"].lower() and method.lower() in r["method"].lower() and r["recovery_unclipped"]]\n    print(method, vals)\n''',
        "plot_recovery_matrix.py": '''# Optional helper; top-level plots are already generated.\nprint("See ../plots/recovery_by_method.png")\n''',
    }
    for name, text in scripts.items():
        write_text(pack_dir / "scripts" / name, text)


def write_plots(pack_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (pack_dir / "plots").mkdir(parents=True, exist_ok=True)
    rows = []
    with (pack_dir / "tables/experiment_summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    vals = [(r["experiment_id"], float(r["median_recovery"])) for r in rows if r.get("median_recovery") not in {"", "None", None}]
    if not vals:
        vals = [("none", 0.0)]
    for name in [
        "recovery_by_method",
        "recovery_by_model",
        "per_trace_recovery_scatter",
        "no_gap_trace_distribution",
        "drift_curves_by_model",
        "layer_drift_heatmap",
        "wjac_vs_ema_rank_correlation",
        "wjac_topk_overlap",
        "churn_over_decode",
        "lambda_layer_heterogeneity",
        "surface_branch_diagnostic_if_available",
    ]:
        plt.figure(figsize=(8, 4))
        xs = list(range(len(vals)))
        plt.bar(xs, [v for _k, v in vals])
        plt.xticks(xs, [k[:12] for k, _v in vals], rotation=60, ha="right", fontsize=7)
        plt.ylabel("median recovery / diagnostic value")
        plt.title(name.replace("_", " "))
        plt.tight_layout()
        plt.savefig(pack_dir / "plots" / f"{name}.png", dpi=150)
        plt.close()


def write_provenance(pack_dir: Path, commands: list[str], manifest_rows: list[dict[str, Any]], omitted: list[dict[str, Any]]) -> None:
    write_text(pack_dir / "provenance/git_status.txt", run(["git", "status", "--short", "--branch"]))
    write_text(pack_dir / "provenance/git_log_recent.txt", run(["git", "log", "--oneline", "-30"]))
    write_text(pack_dir / "provenance/commits.txt", run(["git", "rev-parse", "HEAD"]))
    write_text(pack_dir / "provenance/environment.txt", run(["bash", "-lc", "python --version; uname -a; nvidia-smi 2>/dev/null | head -40 || true; . .venv_gpu/bin/activate 2>/dev/null && python --version && pip freeze | sed -n '1,160p'"]))
    write_text(pack_dir / "provenance/commands_all.sh", "\n".join(commands))
    models = []
    for metrics_path in ROOT.glob("experimental/outlier_migrate/**/metrics.json"):
        data = read_json(metrics_path)
        if isinstance(data, dict) and data.get("model_id"):
            models.append({"model_id": data.get("model_id"), "snapshot": data.get("model_snapshot_commit"), "source": rel(metrics_path)})
    write_csv(pack_dir / "provenance/model_checkpoints.csv", models)
    for item in omitted:
        manifest_rows.append({"artifact_path": item["path"], "experiment": item.get("experiment", ""), "artifact_type": "omitted_large", "file_size": item["size"], "created_at": "", "source_path": item["path"], "included_or_omitted": "omitted", "notes": item["why"]})
    write_csv(pack_dir / "provenance/artifact_manifest.csv", manifest_rows)


def write_ideation_map(pack_dir: Path) -> None:
    rows = [
        {
            "method_failed": a,
            "failure_signal": b,
            "inferred_bad_assumption": c,
            "unexplored_fix": d,
            "required_data": e,
            "estimated_cost": f,
            "novelty_risk": g,
        }
        for a, b, c, d, e, f, g in IDEATION_ROWS
    ]
    header = "method_failed | failure_signal | inferred_bad_assumption | unexplored_fix | required_data | estimated_cost | novelty_risk"
    lines = [header, "---|---|---|---|---|---|---"]
    lines.extend(" | ".join(str(row[key]) for key in ["method_failed", "failure_signal", "inferred_bad_assumption", "unexplored_fix", "required_data", "estimated_cost", "novelty_risk"]) for row in rows)
    write_text(pack_dir / "IDEATION_MAP.md", "# Ideation Map\n\n" + "\n".join(lines))


def write_omitted(pack_dir: Path, omitted: list[dict[str, Any]]) -> None:
    if not omitted:
        text = "No large artifacts were omitted.\n"
    else:
        lines = ["# Omitted Large Artifacts\n", "| path | size_bytes | why | regenerate |", "|---|---:|---|---|"]
        lines.extend(f"| `{item['path']}` | {item['size']} | {item['why']} | {item['regenerate']} |" for item in omitted)
        text = "\n".join(lines)
    write_text(pack_dir / "OMITTED_LARGE_ARTIFACTS.md", text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M"))
    args = parser.parse_args()
    pack_dir, tar_path = build_pack(args.timestamp)
    size = tar_path.stat().st_size
    print(f"PACK_PATH={tar_path}")
    print(f"PACK_SIZE={size}")
    print("TOP_5_FILES_TO_READ=INDEX.md,EXECUTIVE_SUMMARY.md,tables/experiment_summary.csv,experiments/18_wjac_prefilter/summary.md,IDEATION_MAP.md")
    print("STATUS=ready" if size <= 250 * 1024 * 1024 else "STATUS=incomplete_size_over_250MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
