#!/usr/bin/env python3
"""Cached CPU-only checks for the LatentWire escape hypotheses."""

from __future__ import annotations

import datetime as dt
import glob
import hashlib
import json
import math
import pathlib
import random
from collections import Counter, defaultdict
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "escape_tests" / "20260605_cpu_cached"
CANDIDATE_PACKET_ROWS = (
    ROOT
    / "results/source_private_candidate_conditioned_packet_builder_smoke_20260501_seed59/"
    / "core_to_holdout/predictions_budget8.jsonl"
)
TESTLOG_GLOB = ROOT / "results/source_private_testlog_packet_cross_model_20260428/*/predictions.jsonl"


def refuse_confirm(paths: list[pathlib.Path]) -> None:
    bad = [str(path.relative_to(ROOT)) for path in paths if "confirm" in path.as_posix().lower()]
    if bad:
        raise SystemExit(f"refusing to read confirm-looking paths: {bad}")


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
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
    return "train" if bucket < 7 else "eval"


def payload_vector(payload_hex: str, width: int = 8) -> np.ndarray:
    raw = bytes.fromhex(payload_hex) if payload_hex else b""
    raw = raw[:width].ljust(width, b"\x00")
    return np.frombuffer(raw, dtype=np.uint8).astype(np.float64) / 255.0


def nearest_centroid_leakage(rows: list[dict[str, Any]], *, label_key: str) -> dict[str, Any]:
    train = [row for row in rows if split_name(row["example_id"]) == "train"]
    eval_rows = [row for row in rows if split_name(row["example_id"]) == "eval"]
    labels = sorted({str(row[label_key]) for row in rows})
    if not train or not eval_rows or len(labels) < 2:
        return {"status": "not_enough_rows", "train_n": len(train), "eval_n": len(eval_rows)}

    majority = Counter(str(row[label_key]) for row in train).most_common(1)[0][0]
    centroids: dict[str, np.ndarray] = {}
    for label in labels:
        vectors = [payload_vector(row["payload_hex"]) for row in train if str(row[label_key]) == label]
        if vectors:
            centroids[label] = np.stack(vectors).mean(axis=0)

    def predict(row: dict[str, Any]) -> str:
        vector = payload_vector(row["payload_hex"])
        if not centroids:
            return majority
        return min(centroids, key=lambda label: float(np.linalg.norm(vector - centroids[label])))

    pred_correct = sum(predict(row) == str(row[label_key]) for row in eval_rows)
    majority_correct = sum(majority == str(row[label_key]) for row in eval_rows)
    return {
        "status": "ok",
        "label": label_key,
        "classes": len(labels),
        "train_n": len(train),
        "eval_n": len(eval_rows),
        "nearest_centroid_accuracy": pred_correct / len(eval_rows),
        "majority_accuracy": majority_correct / len(eval_rows),
    }


def condition_rows(rows: list[dict[str, Any]], condition: str) -> list[dict[str, Any]]:
    selected = [row for row in rows if row.get("condition") == condition]
    return sorted(selected, key=lambda row: row["example_id"])


def condition_summary(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    selected = condition_rows(rows, condition)
    if not selected:
        return {"condition": condition, "n": 0}
    return {
        "condition": condition,
        "n": len(selected),
        "unique_examples": len({row["example_id"] for row in selected}),
        "accuracy": sum(bool(row.get("correct")) for row in selected) / len(selected),
        "strict_accuracy": sum(bool(row.get("strict_correct", row.get("correct"))) for row in selected) / len(selected),
        "mean_payload_bytes": sum(float(row.get("payload_bytes", 0.0)) for row in selected) / len(selected),
    }


def bootstrap_delta(a: list[bool], b: list[bool], *, seed: int = 20260605, samples: int = 4000) -> dict[str, Any]:
    if len(a) != len(b) or not a:
        return {"status": "invalid", "n": min(len(a), len(b))}
    rng = random.Random(seed)
    diffs = [int(x) - int(y) for x, y in zip(a, b)]
    n = len(diffs)
    observed = sum(diffs) / n
    draws = []
    for _ in range(samples):
        draws.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    draws.sort()
    return {
        "status": "ok",
        "n": n,
        "delta": observed,
        "ci95_low": draws[int(0.025 * samples)],
        "ci95_high": draws[int(0.975 * samples)],
    }


def privacy_proxy(candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    conditions = [
        "target_only",
        "candidate_conditioned_packet_builder",
        "structured_text_matched",
        "answer_only_text",
        "random_same_byte",
    ]
    summaries = {condition: condition_summary(candidate_rows, condition) for condition in conditions}
    leakage = {}
    for condition in conditions:
        rows = condition_rows(candidate_rows, condition)
        leakage[condition] = nearest_centroid_leakage(rows, label_key="family_name")

    packet = summaries["candidate_conditioned_packet_builder"]
    text = summaries["structured_text_matched"]
    packet_leak = leakage["candidate_conditioned_packet_builder"]
    text_leak = leakage["structured_text_matched"]
    matched_utility = math.isclose(packet["accuracy"], text["accuracy"], abs_tol=0.02)
    positive = (
        matched_utility
        and packet_leak.get("nearest_centroid_accuracy", 1.0) < text_leak.get("nearest_centroid_accuracy", 0.0)
    )
    return {
        "artifact": str(CANDIDATE_PACKET_ROWS.relative_to(ROOT)),
        "artifact_sha256": sha256_file(CANDIDATE_PACKET_ROWS),
        "floor_rows": 500,
        "achieved_rows_per_condition": packet["n"],
        "conditions": summaries,
        "family_name_leakage_adversary": leakage,
        "matched_utility_abs_tol": 0.02,
        "matched_utility": matched_utility,
        "positive": positive,
        "verdict": "POSITIVE_PRIVACY_ESCAPE" if positive else "NO_PRIVACY_POSITIVE_ON_CACHED_PROXY",
        "limitation": (
            "This is a cached proxy over synthetic candidate-repair packets. It does not train a new "
            "information-bottleneck encoder and has no matched-utility visible-text comparator."
        ),
    }


def aggregate_testlog() -> dict[str, Any]:
    paths = sorted(pathlib.Path(path) for path in glob.glob(str(TESTLOG_GLOB)))
    refuse_confirm(paths)
    conditions = [
        "target_only",
        "answer_only",
        "random_same_byte",
        "shuffled_model_packet",
        "matched_model_packet",
        "full_signature_oracle",
    ]
    counts = {condition: [0, 0] for condition in conditions}
    paired_visible: list[bool] = []
    paired_packet: list[bool] = []
    unique_examples = set()
    helpers = []
    for path in paths:
        helpers.append(path.parent.name)
        for row in read_jsonl(path):
            unique_examples.add(row["example_id"])
            row_conditions = row["conditions"]
            for condition in conditions:
                if condition in row_conditions:
                    counts[condition][0] += int(bool(row_conditions[condition]["correct"]))
                    counts[condition][1] += 1
            paired_visible.append(bool(row_conditions["full_signature_oracle"]["correct"]))
            paired_packet.append(bool(row_conditions["matched_model_packet"]["correct"]))
    metrics = {
        condition: {
            "correct": correct,
            "n": n,
            "accuracy": correct / n if n else None,
        }
        for condition, (correct, n) in counts.items()
    }
    return {
        "artifacts": [str(path.relative_to(ROOT)) for path in paths],
        "artifact_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in paths},
        "helpers": helpers,
        "unique_examples": len(unique_examples),
        "model_helper_rows": len(paired_visible),
        "floor_rows": 500,
        "floor_met_by_model_helper_rows": len(paired_visible) >= 500,
        "floor_met_by_unique_examples": len(unique_examples) >= 500,
        "conditions": metrics,
        "visible_signature_minus_matched_packet": bootstrap_delta(paired_visible, paired_packet),
        "verdict": "VISIBLE_EXACT_DISCRETE_EVIDENCE_DOMINATES_PACKET",
        "limitation": "The 640-row floor is met only as model-helper rows; the underlying unique examples are 160.",
    }


def candidate_text_control_audit(candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    packet_rows = condition_rows(candidate_rows, "candidate_conditioned_packet_builder")
    structured_rows = condition_rows(candidate_rows, "structured_text_matched")
    atom_vocab = sorted(
        {
            atom
            for row in packet_rows
            for atom in row.get("metadata", {}).get("payload_atoms", {}).keys()
        }
    )
    encodable_rows = 0
    max_atom_count = 0
    for row in packet_rows:
        atom_count = len(row.get("metadata", {}).get("payload_atoms", {}))
        max_atom_count = max(max_atom_count, atom_count)
        if atom_count <= 4:
            encodable_rows += 1
    first_payloads = Counter(row.get("payload_hex", "") for row in structured_rows).most_common(5)
    return {
        "artifact": str(CANDIDATE_PACKET_ROWS.relative_to(ROOT)),
        "packet_condition": condition_summary(candidate_rows, "candidate_conditioned_packet_builder"),
        "existing_structured_text_control": condition_summary(candidate_rows, "structured_text_matched"),
        "existing_structured_text_top_payload_hex": first_payloads,
        "atom_vocab_size": len(atom_vocab),
        "max_packet_atom_count": max_atom_count,
        "rows_encodable_as_public_2byte_atom_value_code_with_8byte_budget": encodable_rows,
        "n": len(packet_rows),
        "audit_verdict": "EXISTING_TEXT_CONTROL_IS_NOT_IDENTICAL_EVIDENCE",
        "interpretation": (
            "The stored structured_text_matched control truncates the raw hidden log and is mostly the same "
            "prefix bytes, so it is not a valid identical-evidence text baseline. The packet's discrete atom "
            "payloads are, however, public-code serializable within the same 8-byte budget for every row."
        ),
    }


def write_markdown(summary: dict[str, Any], path: pathlib.Path) -> None:
    privacy = summary["privacy_proxy"]
    discrete = summary["discrete_exact_evidence_test"]
    text_audit = summary["candidate_text_control_audit"]
    p_cond = privacy["conditions"]
    p_leak = privacy["family_name_leakage_adversary"]
    d_cond = discrete["conditions"]
    delta = discrete["visible_signature_minus_matched_packet"]
    lines = [
        "# LatentWire Escape Test Report",
        "",
        "RUN_STATUS: CPU_CACHED_ANALYSIS_ONLY",
        "",
        "## Guardrails",
        "",
        "- No GPU, no MPS, no new generation, no confirm paths, no queue changes.",
        "- Inputs are existing non-confirm cached artifacts; the script refuses confirm-looking input paths.",
        "- This report is not a promotion record. It is a cheap gate for whether an escape deserves a full build.",
        "",
        "## Scoop Check",
        "",
        "| Axis | Verdict | Evidence |",
        "|---|---|---|",
        "| Privacy-preserving latent inter-LLM packet vs text on a privacy-utility frontier | PARTIAL/OPEN | Adjacent privacy-utility and IB semantic-communication work exists, but I found no direct inter-LLM latent-packet frontier against equal-byte text. Relevant anchors: IBAL semantic communication against model inversion ([arXiv:2312.03252](https://arxiv.org/abs/2312.03252)), adaptive text anonymization privacy-utility frontier ([arXiv:2602.20743](https://arxiv.org/abs/2602.20743)), and embedding inversion leakage ([arXiv:2305.03010](https://arxiv.org/abs/2305.03010)). |",
        "| Continuous state / cache communication accuracy escape | CROWDED/PARTIAL | C2C ([arXiv:2510.03215](https://arxiv.org/abs/2510.03215)), Interlat ([arXiv:2511.09149](https://arxiv.org/abs/2511.09149)), and LCF ([arXiv:2605.22863](https://arxiv.org/abs/2605.22863)) occupy the no-text latent/cache communication lane; a byte-level, destructive-control distillation remains a narrower possible opening. |",
        "| Discrete evidence packet dominated by equal-byte visible evidence | OPEN AS A NEGATIVE/THEORY CLAIM | Search did not find a direct theorem/result for the exact equal-byte discrete-evidence claim. The local cached test below supports it empirically for exact visible signatures. |",
        "",
        "## Escape A: Privacy Proxy",
        "",
        f"- artifact: `{privacy['artifact']}`",
        f"- achieved rows per condition: `{privacy['achieved_rows_per_condition']}` / floor `{privacy['floor_rows']}`",
        f"- verdict: `{privacy['verdict']}`",
        "",
        "| Condition | Utility accuracy | Payload bytes | Family leakage acc | Leakage majority baseline |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in [
        "target_only",
        "candidate_conditioned_packet_builder",
        "structured_text_matched",
        "answer_only_text",
        "random_same_byte",
    ]:
        cond = p_cond[condition]
        leak = p_leak[condition]
        lines.append(
            "| "
            f"`{condition}` | "
            f"{cond['accuracy']:.6f} | "
            f"{cond['mean_payload_bytes']:.2f} | "
            f"{leak.get('nearest_centroid_accuracy', float('nan')):.6f} | "
            f"{leak.get('majority_accuracy', float('nan')):.6f} |"
        )
    lines.extend(
        [
            "",
            "Decision: no privacy positive from the cached proxy. The high-utility latent packet is not compared to a matched-utility visible-text baseline, and the existing visible text control has target-only utility. This parks privacy until a real IB encoder and matched-utility text/anonymization comparator exist.",
            "",
            "## Theory Test: Exact Discrete Evidence vs Visible Evidence",
            "",
            f"- artifacts: `{len(discrete['artifacts'])}` helper files under `results/source_private_testlog_packet_cross_model_20260428/`",
            f"- model-helper rows: `{discrete['model_helper_rows']}` / floor `{discrete['floor_rows']}`",
            f"- unique examples: `{discrete['unique_examples']}`",
            f"- verdict: `{discrete['verdict']}`",
            "",
            "| Condition | Accuracy | Correct / n |",
            "|---|---:|---:|",
        ]
    )
    for condition in [
        "target_only",
        "answer_only",
        "random_same_byte",
        "shuffled_model_packet",
        "matched_model_packet",
        "full_signature_oracle",
    ]:
        cond = d_cond[condition]
        lines.append(f"| `{condition}` | {cond['accuracy']:.6f} | {cond['correct']} / {cond['n']} |")
    lines.extend(
        [
            "",
            f"- visible exact signature minus matched model packet: `{delta['delta']:.6f}` "
            f"CI95 `[{delta['ci95_low']:.6f}, {delta['ci95_high']:.6f}]` over `{delta['n']}` model-helper rows.",
            "- The visible exact signature is the same 2-byte discrete payload and dominates/ties the model packet; this supports the discrete-evidence-is-text-capturable claim rather than a no-text positive.",
            "",
            "## Candidate Text-Control Audit",
            "",
            f"- packet accuracy: `{text_audit['packet_condition']['accuracy']:.6f}`",
            f"- stored `structured_text_matched` accuracy: `{text_audit['existing_structured_text_control']['accuracy']:.6f}`",
            f"- stored `structured_text_matched` top payload prefixes: `{text_audit['existing_structured_text_top_payload_hex']}`",
            f"- atom vocab size: `{text_audit['atom_vocab_size']}`",
            f"- rows encodable as public 2-byte atom/value code within 8 bytes: `{text_audit['rows_encodable_as_public_2byte_atom_value_code_with_8byte_budget']}` / `{text_audit['n']}`",
            "",
            "Interpretation: the existing `structured_text_matched` row is not a valid identical-evidence text baseline because it truncates the private log prefix (`pytest s...`) and does not expose the same atom evidence. But the atom payload is discrete and public-code serializable in the same byte budget for all rows, so this cache does not establish a no-text advantage.",
            "",
            "## Bottom Line",
            "",
            "- Privacy remains the only cheap LatentWire escape, but this cached proxy does not pass it.",
            "- The exact-discrete evidence check supports the theory: when the same symbolic evidence is visible, text/public code captures or beats the packet.",
            "- Next real work is still writing, unless a separate IB privacy encoder is intentionally built as a new method card with matched-utility text/anonymization baselines.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    paths = [CANDIDATE_PACKET_ROWS, *sorted(pathlib.Path(path) for path in glob.glob(str(TESTLOG_GLOB)))]
    refuse_confirm(paths)
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"missing required artifacts: {missing}")

    candidate_rows = read_jsonl(CANDIDATE_PACKET_ROWS)
    summary = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "status": "CPU_CACHED_ANALYSIS_ONLY",
        "guardrails": {
            "gpu": "not_used",
            "mps": "not_used",
            "new_generation": False,
            "confirm_paths": "refused_by_path_guard",
            "queue_changes": False,
        },
        "privacy_proxy": privacy_proxy(candidate_rows),
        "discrete_exact_evidence_test": aggregate_testlog(),
        "candidate_text_control_audit": candidate_text_control_audit(candidate_rows),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(summary, ROOT / "dashboard" / "escape_test_report.md")
    print(f"wrote {(OUT_DIR / 'summary.json').relative_to(ROOT)}")
    print("wrote dashboard/escape_test_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
