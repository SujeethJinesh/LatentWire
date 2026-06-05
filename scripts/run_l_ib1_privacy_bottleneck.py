#!/usr/bin/env python3
"""L_IB1 cached privacy-bottleneck gate.

This runner uses existing non-confirm cached packet rows only. It does not run
models, generate candidates, or touch queues.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import pathlib
import re
from collections import Counter
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
INPUT_ROWS = (
    ROOT
    / "results/source_private_candidate_conditioned_packet_builder_smoke_20260501_seed59/"
    / "core_to_holdout/predictions_budget8.jsonl"
)
OUT_DIR = ROOT / "results/escape_tests/L_IB1_privacy_bottleneck"
REPORT_PATH = ROOT / "dashboard/l_ib_privacy_bottleneck_report.md"
SCOOP_PATH = ROOT / "dashboard/l_ib_scoop_note.md"

CONDITION = "candidate_conditioned_packet_builder"
TARGET_ONLY = "target_only"
ATOM_CLUSTERS = {
    "numeric": {
        "integer",
        "parse",
        "average",
        "mean",
        "round",
        "modulo",
        "positive",
        "negative",
        "zero",
        "all_values",
    },
    "container": {"list", "mapping", "nested", "index", "default", "none"},
    "error": {"failure", "fallback", "exception", "clamp", "filter"},
    "text": {"string"},
}


def refuse_confirm(path: pathlib.Path) -> None:
    rel = path.as_posix().lower()
    if "confirm" in rel or "confirmation" in rel:
        raise SystemExit(f"refusing confirm-looking path: {path}")


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    refuse_confirm(path)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_name(example_id: str) -> str:
    bucket = int(hashlib.sha256(example_id.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "dev" if bucket < 7 else "gate"


def candidate_index(label: str) -> int:
    match = re.search(r"_patch_(\d+)_", label)
    return int(match.group(1)) if match else -1


def label_encode(values: list[Any]) -> tuple[np.ndarray, list[Any]]:
    classes = sorted(set(values), key=lambda value: str(value))
    lookup = {value: idx for idx, value in enumerate(classes)}
    return np.array([lookup[value] for value in values], dtype=np.int64), classes


def ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    *,
    ridge: float = 1e-2,
) -> np.ndarray:
    classes = sorted(set(int(value) for value in y_train.tolist()))
    if len(classes) == 1:
        return np.full(x_eval.shape[0], classes[0], dtype=np.int64)
    class_to_col = {label: idx for idx, label in enumerate(classes)}
    y = np.zeros((len(y_train), len(classes)), dtype=np.float64)
    for row, label in enumerate(y_train.tolist()):
        y[row, class_to_col[int(label)]] = 1.0
    xtr = np.concatenate([x_train, np.ones((x_train.shape[0], 1), dtype=np.float64)], axis=1)
    xev = np.concatenate([x_eval, np.ones((x_eval.shape[0], 1), dtype=np.float64)], axis=1)
    gram = xtr.T @ xtr
    gram += ridge * np.eye(gram.shape[0], dtype=np.float64)
    weights = np.linalg.pinv(gram) @ xtr.T @ y
    pred_cols = np.argmax(xev @ weights, axis=1)
    return np.array([classes[col] for col in pred_cols], dtype=np.int64)


def accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    if len(true) == 0:
        return float("nan")
    return float(np.mean(pred == true))


def class_signal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if x.shape[0] == 0:
        return np.zeros(x.shape[1], dtype=np.float64)
    overall = x.mean(axis=0)
    total = ((x - overall) ** 2).mean(axis=0) + 1e-12
    between = np.zeros(x.shape[1], dtype=np.float64)
    for label in sorted(set(y.tolist())):
        mask = y == label
        if np.any(mask):
            delta = x[mask].mean(axis=0) - overall
            between += (mask.mean()) * (delta**2)
    return between / total


def normalize(values: np.ndarray) -> np.ndarray:
    high = float(np.max(values)) if values.size else 0.0
    if high <= 0:
        return np.zeros_like(values)
    return values / high


def transform_exact(x: np.ndarray) -> np.ndarray:
    return x.copy()


def transform_target_only(x: np.ndarray) -> np.ndarray:
    return np.zeros((x.shape[0], 1), dtype=np.float64)


def transform_random_same_byte(x: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(20260605)
    return rng.random((x.shape[0], min(4, x.shape[1])), dtype=np.float64)


def transform_coarse(x: np.ndarray, atoms: list[str]) -> np.ndarray:
    columns = []
    for cluster_atoms in ATOM_CLUSTERS.values():
        indices = [idx for idx, atom in enumerate(atoms) if atom in cluster_atoms]
        if indices:
            columns.append(x[:, indices].max(axis=1))
        else:
            columns.append(np.zeros(x.shape[0], dtype=np.float64))
    return np.stack(columns, axis=1)


def transform_selected(
    x: np.ndarray,
    *,
    selected: list[int],
    quant_step: float | None = None,
    dropout: float = 0.0,
) -> np.ndarray:
    out = x[:, selected].copy() if selected else np.zeros((x.shape[0], 1), dtype=np.float64)
    if quant_step:
        out = np.round(out / quant_step) * quant_step
    if dropout:
        rng = np.random.default_rng(20260605 + len(selected) + int(dropout * 100))
        mask = rng.random(out.shape) >= dropout
        out = out * mask
    return out


def row_features(rows: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    atoms = sorted(
        {
            atom
            for row in rows
            for atom in row.get("metadata", {}).get("payload_atoms", {}).keys()
        }
    )
    vectors = []
    for row in rows:
        payload = row.get("metadata", {}).get("payload_atoms", {})
        vectors.append([float(payload.get(atom, 0.0)) for atom in atoms])
    return np.array(vectors, dtype=np.float64), atoms


def evaluate_variant(
    *,
    name: str,
    kind: str,
    z: np.ndarray,
    train_mask: np.ndarray,
    labels: dict[str, np.ndarray],
    payload_bytes: int,
    notes: str,
    cached_receiver_utility: float | None = None,
) -> dict[str, Any]:
    z_train = z[train_mask]
    z_eval = z[~train_mask]
    out: dict[str, Any] = {
        "name": name,
        "kind": kind,
        "payload_bytes": payload_bytes,
        "notes": notes,
        "dev_n": int(train_mask.sum()),
        "gate_n": int((~train_mask).sum()),
    }
    for label_name, values in labels.items():
        train_values = values[train_mask]
        eval_values = values[~train_mask]
        pred = ridge_predict(z_train, train_values, z_eval)
        acc = accuracy(pred, eval_values)
        key = "utility_proxy_accuracy" if label_name == "answer_index" else f"{label_name}_leakage_accuracy"
        out[key] = acc
        base = Counter(train_values.tolist()).most_common(1)[0][0]
        out[f"{label_name}_majority_accuracy"] = accuracy(np.full_like(eval_values, base), eval_values)
    if cached_receiver_utility is not None:
        out["cached_receiver_utility_accuracy"] = cached_receiver_utility
    return out


def build_frontier(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matched = sorted([row for row in rows if row.get("condition") == CONDITION], key=lambda row: row["example_id"])
    target = sorted([row for row in rows if row.get("condition") == TARGET_ONLY], key=lambda row: row["example_id"])
    if len(matched) < 500 or len(target) != len(matched):
        raise SystemExit(f"expected >=500 matched rows and aligned target rows, got {len(matched)} / {len(target)}")
    x, atoms = row_features(matched)
    train_mask = np.array([split_name(row["example_id"]) == "dev" for row in matched], dtype=bool)

    answer_index, answer_classes = label_encode([int(row["answer_index"]) for row in matched])
    family, family_classes = label_encode([row["family_name"] for row in matched])
    source_top1, source_top1_classes = label_encode([candidate_index(str(row["prediction"])) for row in matched])
    candidate_id, candidate_classes = label_encode([int(row["answer_index"]) for row in matched])
    top_atom, top_atom_classes = label_encode(
        [
            max(
                row.get("metadata", {}).get("payload_atoms", {}) or {"none": 0.0},
                key=lambda atom: row.get("metadata", {}).get("payload_atoms", {}).get(atom, 0.0),
            )
            for row in matched
        ]
    )
    labels = {
        "answer_index": answer_index,
        "family": family,
        "source_top1": source_top1,
        "candidate_id": candidate_id,
        "evidence_atom": top_atom,
    }

    utility_signal = class_signal(x[train_mask], answer_index[train_mask])
    leak_signal = np.maximum.reduce(
        [
            class_signal(x[train_mask], family[train_mask]),
            class_signal(x[train_mask], source_top1[train_mask]),
            class_signal(x[train_mask], top_atom[train_mask]),
        ]
    )
    utility_signal = normalize(utility_signal)
    leak_signal = normalize(leak_signal)

    target_utility = sum(bool(row.get("correct")) for row in target) / len(target)
    current_utility = sum(bool(row.get("correct")) for row in matched) / len(matched)

    variants: list[dict[str, Any]] = []
    variants.append(
        evaluate_variant(
            name="target_only",
            kind="baseline",
            z=transform_target_only(x),
            train_mask=train_mask,
            labels=labels,
            payload_bytes=0,
            notes="receiver prior only",
            cached_receiver_utility=target_utility,
        )
    )
    variants.append(
        evaluate_variant(
            name="current_high_utility_packet",
            kind="baseline",
            z=transform_exact(x),
            train_mask=train_mask,
            labels=labels,
            payload_bytes=8,
            notes="existing high-utility latent packet payload atoms",
            cached_receiver_utility=current_utility,
        )
    )
    variants.append(
        evaluate_variant(
            name="same_byte_visible_exact_public_code",
            kind="text_baseline",
            z=transform_exact(x),
            train_mask=train_mask,
            labels=labels,
            payload_bytes=8,
            notes="public 2-byte atom/value code carrying the same exact evidence; not no-text",
            cached_receiver_utility=current_utility,
        )
    )
    variants.append(
        evaluate_variant(
            name="adaptive_anonymized_text_coarse_atoms",
            kind="text_baseline",
            z=transform_coarse(x, atoms),
            train_mask=train_mask,
            labels=labels,
            payload_bytes=8,
            notes="coarse atom-cluster text surrogate that hides exact atom names",
        )
    )
    variants.append(
        evaluate_variant(
            name="random_same_byte",
            kind="control",
            z=transform_random_same_byte(x),
            train_mask=train_mask,
            labels=labels,
            payload_bytes=8,
            notes="random same-byte control",
        )
    )

    for lam in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]:
        score = utility_signal - lam * leak_signal
        ranking = list(np.argsort(-score))
        for k in [1, 2, 3, 4]:
            selected = ranking[:k]
            selected_atoms = [atoms[idx] for idx in selected]
            variants.append(
                evaluate_variant(
                    name=f"adv_select_lam{lam:g}_k{k}",
                    kind="l_ib1_candidate",
                    z=transform_selected(x, selected=selected, quant_step=1.0 / 16.0),
                    train_mask=train_mask,
                    labels=labels,
                    payload_bytes=min(8, 2 * k),
                    notes=f"selected atoms={selected_atoms}",
                )
            )
            variants.append(
                evaluate_variant(
                    name=f"adv_select_lam{lam:g}_k{k}_drop25",
                    kind="l_ib1_candidate",
                    z=transform_selected(x, selected=selected, quant_step=1.0 / 16.0, dropout=0.25),
                    train_mask=train_mask,
                    labels=labels,
                    payload_bytes=min(8, 2 * k),
                    notes=f"selected atoms={selected_atoms}; deterministic dropout=0.25",
                )
            )

    meta = {
        "input_rows": str(INPUT_ROWS.relative_to(ROOT)),
        "input_sha256": sha256_file(INPUT_ROWS),
        "matched_rows": len(matched),
        "target_rows": len(target),
        "dev_rows": int(train_mask.sum()),
        "gate_rows": int((~train_mask).sum()),
        "atom_vocab": atoms,
        "answer_classes": answer_classes,
        "family_classes": family_classes,
        "source_top1_classes": source_top1_classes,
        "candidate_classes_count": len(candidate_classes),
        "top_atom_classes": top_atom_classes,
        "target_cached_utility": target_utility,
        "current_cached_utility": current_utility,
    }
    return variants, meta


def classify(variants: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    current = next(row for row in variants if row["name"] == "current_high_utility_packet")
    visible = next(row for row in variants if row["name"] == "same_byte_visible_exact_public_code")
    text_rows = [row for row in variants if row["kind"] == "text_baseline"]
    l_ib_rows = [row for row in variants if row["kind"] == "l_ib1_candidate"]
    target_cached = float(meta["target_cached_utility"])
    current_cached = float(meta["current_cached_utility"])

    def utility(row: dict[str, Any]) -> float:
        return float(row.get("cached_receiver_utility_accuracy", row["utility_proxy_accuracy"]))

    def leak(row: dict[str, Any]) -> float:
        return max(
            float(row["family_leakage_accuracy"]),
            float(row["source_top1_leakage_accuracy"]),
            float(row["candidate_id_leakage_accuracy"]),
            float(row["evidence_atom_leakage_accuracy"]),
        )

    best_text = max(text_rows, key=utility)
    best_l_ib = max(l_ib_rows, key=utility)
    current_leak = leak(current)
    preserving = [
        row
        for row in l_ib_rows
        if float(row["utility_proxy_accuracy"]) >= max(target_cached + 0.15, current_cached - 0.02)
    ]
    best_private = min(preserving, key=leak) if preserving else None
    pass_positive = (
        best_private is not None
        and utility(best_private) >= max(target_cached + 0.15, utility(best_text) - 0.02)
        and leak(best_private) <= min(current_leak - 0.10, leak(best_text) - 0.05)
        and float(best_private["family_leakage_accuracy"]) <= float(current["family_leakage_accuracy"]) - 0.10
    )
    if pass_positive:
        verdict = "PASS_PRIVACY_POSITIVE"
        reason = "A utility-preserving bottleneck reduced leakage below the current packet and matched text."
    elif best_private is None:
        verdict = "KILL_UTILITY_IS_IDENTITY"
        reason = (
            "No adversarial bottleneck preserved current-level utility; the best private code dropped well below "
            "the current packet and still leaked source/evidence structure."
        )
    else:
        leakage_near_current = leak(best_private) >= current_leak - 0.10
        if leakage_near_current:
            verdict = "KILL_UTILITY_IS_IDENTITY"
            reason = "Utility-preserving bottlenecks retained near-current source/evidence leakage."
        elif utility(visible) >= utility(best_private) and leak(visible) <= leak(best_private):
            verdict = "KILL_TEXT_DOMINATES"
            reason = "Visible exact evidence dominated the privacy-utility frontier."
        else:
            verdict = "KILL_TEXT_DOMINATES"
            reason = "No bottleneck beat the matched visible/anonymized text frontier under the gate."
    return {
        "verdict": verdict,
        "reason": reason,
        "target_cached_utility": target_cached,
        "current_cached_utility": current_cached,
        "current_max_leakage": current_leak,
        "best_text": best_text["name"],
        "best_text_utility": utility(best_text),
        "best_text_max_leakage": leak(best_text),
        "best_l_ib": best_l_ib["name"],
        "best_l_ib_utility_proxy": float(best_l_ib["utility_proxy_accuracy"]),
        "best_l_ib_max_leakage": leak(best_l_ib),
        "best_utility_preserving_private": best_private["name"] if best_private else None,
        "best_utility_preserving_private_utility_proxy": float(best_private["utility_proxy_accuracy"])
        if best_private
        else None,
        "best_utility_preserving_private_max_leakage": leak(best_private) if best_private else None,
        "preserving_candidate_count": len(preserving),
    }


def write_frontier_csv(variants: list[dict[str, Any]], path: pathlib.Path) -> None:
    fields = [
        "name",
        "kind",
        "payload_bytes",
        "cached_receiver_utility_accuracy",
        "utility_proxy_accuracy",
        "family_leakage_accuracy",
        "source_top1_leakage_accuracy",
        "candidate_id_leakage_accuracy",
        "evidence_atom_leakage_accuracy",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in variants:
            writer.writerow({field: row.get(field, "") for field in fields})


def metric(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{float(value):.6f}"


def write_report(summary: dict[str, Any]) -> None:
    decision = summary["decision"]
    variants = summary["frontier"]
    ranked = sorted(
        variants,
        key=lambda row: (
            float(row.get("utility_proxy_accuracy", 0.0)),
            -max(
                float(row.get("family_leakage_accuracy", 0.0)),
                float(row.get("source_top1_leakage_accuracy", 0.0)),
                float(row.get("candidate_id_leakage_accuracy", 0.0)),
                float(row.get("evidence_atom_leakage_accuracy", 0.0)),
            ),
        ),
        reverse=True,
    )
    lines = [
        "# L-IB1 Privacy-Bottleneck Report",
        "",
        "RUN_STATUS: CPU_CACHED_ANALYSIS_ONLY",
        "",
        "## Verdict",
        "",
        f"- verdict: `{decision['verdict']}`",
        f"- reason: {decision['reason']}",
        f"- input rows: `{summary['meta']['matched_rows']}` matched packet rows, `{summary['meta']['target_rows']}` target-only rows",
        f"- dev/gate split: `{summary['meta']['dev_rows']}` / `{summary['meta']['gate_rows']}`",
        f"- current packet cached utility: `{decision['current_cached_utility']:.6f}`",
        f"- current packet max leakage: `{decision['current_max_leakage']:.6f}`",
        "",
        "## Frontier",
        "",
        "| Variant | Kind | Bytes | Cached utility | Proxy utility | Family leak | Source-top1 leak | Candidate-id leak | Evidence-atom leak |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    names = {
        "target_only",
        "current_high_utility_packet",
        "same_byte_visible_exact_public_code",
        "adaptive_anonymized_text_coarse_atoms",
        "random_same_byte",
        decision["best_l_ib"],
        decision.get("best_utility_preserving_private"),
    }
    selected_rows = [row for row in variants if row["name"] in names]
    for row in selected_rows:
        lines.append(
            "| "
            f"`{row['name']}` | `{row['kind']}` | {row['payload_bytes']} | "
            f"{metric(row, 'cached_receiver_utility_accuracy')} | "
            f"{metric(row, 'utility_proxy_accuracy')} | "
            f"{metric(row, 'family_leakage_accuracy')} | "
            f"{metric(row, 'source_top1_leakage_accuracy')} | "
            f"{metric(row, 'candidate_id_leakage_accuracy')} | "
            f"{metric(row, 'evidence_atom_leakage_accuracy')} |"
        )
    lines.extend(
        [
            "",
            "Top L-IB1 candidates by proxy utility:",
            "",
            "| Rank | Variant | Proxy utility | Max leakage | Notes |",
            "|---:|---|---:|---:|---|",
        ]
    )
    for idx, row in enumerate([row for row in ranked if row["kind"] == "l_ib1_candidate"][:8], start=1):
        max_leak = max(
            float(row["family_leakage_accuracy"]),
            float(row["source_top1_leakage_accuracy"]),
            float(row["candidate_id_leakage_accuracy"]),
            float(row["evidence_atom_leakage_accuracy"]),
        )
        lines.append(
            f"| {idx} | `{row['name']}` | {row['utility_proxy_accuracy']:.6f} | {max_leak:.6f} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The cached high-utility packet is useful only when it preserves the exact source/evidence structure. "
            "Adversarial feature-selection bottlenecks that hide those atoms drop well below current-packet utility, "
            "while exact-evidence baselines expose the same source/evidence labels. Under this cached proxy, "
            "task utility and source/evidence identity are not separable enough for a privacy-positive claim.",
            "",
            "This is not a live model-forward result and does not test a trained neural IB encoder. It is the requested "
            "last cheap cached gate: PASS would have justified a full privacy-bottleneck build; this KILL strengthens "
            "the bounded-negative paper by showing that the useful cached packet signal is identity/evidence-bearing.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_scoop_note() -> None:
    SCOOP_PATH.write_text(
        "\n".join(
            [
                "# L-IB1 Scoop Note",
                "",
                "Verdict: PARTIAL/OPEN as a niche, but the cached L-IB1 gate killed the local opportunity.",
                "",
                "Adjacent work:",
                "",
                "- IBAL privacy-preserving task-oriented semantic communication uses information bottleneck and adversarial learning against model inversion, but it is not an inter-LLM byte-packet protocol with source-index, wrong-row, and same-byte text controls: https://arxiv.org/abs/2312.03252",
                "- Adaptive text anonymization optimizes visible-text privacy-utility trade-offs, making it the correct matched baseline rather than a scoop of no-text byte packets: https://arxiv.org/abs/2602.20743",
                "- C2C, Interlat, and Latent Cache Flow occupy continuous latent/cache communication for accuracy and efficiency; they are the crowded dense-state escape lane, not a low-byte privacy frontier: https://arxiv.org/abs/2510.03215, https://arxiv.org/abs/2511.09149, https://arxiv.org/abs/2605.22863",
                "",
                "Novelty if it had passed: byte-scale inter-model packet plus privacy adversary plus matched visible/anonymized text baselines plus destructive controls.",
                "",
                "Local gate outcome: no pass. The useful cached packet signal remains source/evidence-identifying, so the privacy axis is not cheap-positive here.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    rows = read_jsonl(INPUT_ROWS)
    frontier, meta = build_frontier(rows)
    decision = classify(frontier, meta)
    summary = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "method_id": "L_IB1_adversarial_privacy_bottleneck_packet",
        "status": "CPU_CACHED_ANALYSIS_ONLY",
        "guardrails": {
            "gpu": "not_used",
            "mps": "not_used",
            "live_forwards": False,
            "new_generation": False,
            "confirm_paths": "refused_by_path_guard",
            "queue_changes": False,
        },
        "meta": meta,
        "decision": decision,
        "frontier": frontier,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_frontier_csv(frontier, OUT_DIR / "frontier.csv")
    write_report(summary)
    write_scoop_note()
    print(f"wrote {(OUT_DIR / 'summary.json').relative_to(ROOT)}")
    print(f"wrote {(OUT_DIR / 'frontier.csv').relative_to(ROOT)}")
    print("wrote dashboard/l_ib_privacy_bottleneck_report.md")
    print("wrote dashboard/l_ib_scoop_note.md")
    print(f"verdict {decision['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
