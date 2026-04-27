#!/usr/bin/env python3
"""Materialize answer-masked process-trace candidate sidecars for SVAMP.

This producer scores target-side candidate reasoning traces against the matched
source reasoning trace after masking numbers. It is a CPU-only smoke for
source-derived non-answer process signal; it does not add source-only answers to
the candidate pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import shlex
import subprocess
import sys
from collections import Counter
from datetime import date
from typing import Any, Iterable, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_svamp_source_semantic_predicate_decoder as decoder

NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
WORD_RE = re.compile(r"<num>|[a-z]+|[+\-*/=]")


def _resolve(path: str | pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _write_jsonl(path: pathlib.Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _masked_text(text: str) -> str:
    return NUM_RE.sub(" <num> ", text.lower())


def _tokens(text: str, *, max_ngram: int) -> list[str]:
    base = WORD_RE.findall(_masked_text(text))
    out = list(base)
    for ngram in range(2, max_ngram + 1):
        out.extend("_".join(base[index : index + ngram]) for index in range(len(base) - ngram + 1))
    return out


def _tf(tokens: Iterable[str]) -> Counter[str]:
    return Counter(token for token in tokens if token)


def _idf(docs: Sequence[Counter[str]]) -> dict[str, float]:
    df: Counter[str] = Counter()
    for doc in docs:
        df.update(doc.keys())
    n_docs = len(docs)
    return {term: math.log((1 + n_docs) / (1 + count)) + 1.0 for term, count in df.items()}


def _weighted(counter: Counter[str], idf: dict[str, float]) -> dict[str, float]:
    total = sum(counter.values()) or 1
    return {term: (count / total) * idf.get(term, 1.0) for term, count in counter.items()}


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    dot = sum(left[term] * right[term] for term in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _row_text(surface: decoder.Surface, label: str, example_id: str) -> str:
    row = surface.records_by_label.get(label, {}).get(example_id)
    if row is None:
        return ""
    return str(row.get("prediction") or "")


def _candidate_text(surface: decoder.Surface, candidate: decoder.Candidate, example_id: str) -> str:
    labels = [label for label in candidate.labels if label not in {"source", "target"}]
    labels.extend(label for label in candidate.labels if label == "target")
    for label in labels:
        if label == "source":
            continue
        text = _row_text(surface, label, example_id)
        if text:
            return text
    return ""


def _candidate_pool(
    surface: decoder.Surface,
    example_id: str,
    *,
    exclude_labels: set[str],
    prediction_only: bool,
) -> list[decoder.Candidate]:
    if prediction_only:
        ordered: list[str] = []
        labels_by_value: dict[str, list[str]] = {}
        for label, records in surface.records_by_label.items():
            if label == "source" or label in exclude_labels:
                continue
            value = decoder._prediction_numeric(records.get(example_id))
            if value is None:
                continue
            if value not in labels_by_value:
                ordered.append(value)
                labels_by_value[value] = []
            labels_by_value[value].append(label)
        return [
            decoder.Candidate(value=value, labels=tuple(labels_by_value[value]))
            for value in ordered
        ]
    out: list[decoder.Candidate] = []
    for candidate in decoder._candidate_pool(surface, example_id):
        labels = tuple(label for label in candidate.labels if label not in exclude_labels)
        if not labels:
            continue
        out.append(decoder.Candidate(value=candidate.value, labels=labels))
    return out


def _effective_rank(vectors: Sequence[dict[str, float]]) -> float:
    terms = sorted({term for vector in vectors for term in vector})
    if not terms or not vectors:
        return 0.0
    means = [sum(vector.get(term, 0.0) for vector in vectors) / len(vectors) for term in terms]
    variances = []
    for term, mean in zip(terms, means):
        variances.append(sum((vector.get(term, 0.0) - mean) ** 2 for vector in vectors) / len(vectors))
    total = sum(variances)
    if total <= 0.0:
        return 0.0
    probs = [value / total for value in variances if value > 0.0]
    entropy = -sum(prob * math.log(prob) for prob in probs)
    return float(math.exp(entropy))


def _vector_telemetry(vectors: Sequence[dict[str, float]]) -> dict[str, Any]:
    terms = sorted({term for vector in vectors for term in vector})
    if not terms or not vectors:
        return {
            "feature_count": 0,
            "std_min": 0.0,
            "std_mean": 0.0,
            "effective_rank": 0.0,
            "zero_vectors": len(vectors),
        }
    stds: list[float] = []
    zero_vectors = 0
    for vector in vectors:
        if not vector:
            zero_vectors += 1
    for term in terms:
        values = [vector.get(term, 0.0) for vector in vectors]
        mean = sum(values) / len(values)
        stds.append(math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)))
    return {
        "feature_count": len(terms),
        "std_min": float(min(stds)),
        "std_mean": float(sum(stds) / len(stds)),
        "effective_rank": _effective_rank(vectors),
        "zero_vectors": int(zero_vectors),
    }


def _materialize(
    *,
    target_set_path: pathlib.Path,
    sidecar_bits: int,
    max_ngram: int,
    exclude_labels: set[str],
    prediction_only: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    surface = decoder._load_surface("surface", target_set_path)
    source_counters: dict[str, Counter[str]] = {}
    candidate_counters: dict[tuple[str, str], Counter[str]] = {}
    docs: list[Counter[str]] = []

    for example_id in surface.reference_ids:
        source_counter = _tf(_tokens(_row_text(surface, "source", example_id), max_ngram=max_ngram))
        source_counters[example_id] = source_counter
        docs.append(source_counter)
        for candidate in _candidate_pool(
            surface,
            example_id,
            exclude_labels=exclude_labels,
            prediction_only=prediction_only,
        ):
            counter = _tf(_tokens(_candidate_text(surface, candidate, example_id), max_ngram=max_ngram))
            candidate_counters[(example_id, candidate.value)] = counter
            docs.append(counter)

    idf = _idf(docs)
    source_vectors = {example_id: _weighted(counter, idf) for example_id, counter in source_counters.items()}
    rows: list[dict[str, Any]] = []
    top_label_counts: Counter[str] = Counter()
    top_value_in_unmasked_source = 0

    for example_id in surface.reference_ids:
        source_text = _row_text(surface, "source", example_id)
        source_numbers = set(NUM_RE.findall(source_text))
        source_vec = source_vectors[example_id]
        candidate_scores: list[dict[str, Any]] = []
        for candidate in _candidate_pool(
            surface,
            example_id,
            exclude_labels=exclude_labels,
            prediction_only=prediction_only,
        ):
            candidate_vec = _weighted(candidate_counters[(example_id, candidate.value)], idf)
            labels = tuple(label for label in candidate.labels if label != "source")
            sample_labels = tuple(label for label in labels if label != "target")
            label = sample_labels[0] if sample_labels else (labels[0] if labels else "target")
            score = _cosine(source_vec, candidate_vec)
            candidate_scores.append({"label": label, "value": candidate.value, "score": float(score)})
        candidate_scores.sort(key=lambda item: (-float(item["score"]), item["label"], item["value"]))
        top_score = float(candidate_scores[0]["score"]) if candidate_scores else 0.0
        second_score = float(candidate_scores[1]["score"]) if len(candidate_scores) > 1 else top_score
        top_label = str(candidate_scores[0]["label"]) if candidate_scores else None
        top_value = str(candidate_scores[0]["value"]) if candidate_scores else None
        if top_label is not None:
            top_label_counts[top_label] += 1
        if top_value is not None and top_value in source_numbers:
            top_value_in_unmasked_source += 1
        rows.append(
            {
                "example_id": example_id,
                "candidate_scores": candidate_scores,
                "confidence": float(top_score - second_score),
                "sidecar_bits": int(sidecar_bits),
                "profile_mode": "answer_masked_process_trace",
                "top_value_in_unmasked_source_numbers": bool(top_value is not None and top_value in source_numbers),
                "source_number_count": len(source_numbers),
            }
        )

    margins = [float(row["confidence"]) for row in rows]
    summary = {
        "target_set": _display_path(target_set_path),
        "n": len(surface.reference_ids),
        "sidecar_bits": int(sidecar_bits),
        "sidecar_bytes": max(1, (int(sidecar_bits) + 7) // 8),
        "profile_mode": "answer_masked_process_trace",
        "max_ngram": int(max_ngram),
        "exclude_labels": sorted(exclude_labels),
        "prediction_only": bool(prediction_only),
        "top_label_counts": dict(sorted(top_label_counts.items())),
        "top_value_in_unmasked_source_numbers": int(top_value_in_unmasked_source),
        "zero_margin": sum(1 for value in margins if value == 0.0),
        "margin_min": min(margins) if margins else 0.0,
        "margin_mean": sum(margins) / len(margins) if margins else 0.0,
        "source_vector_telemetry": _vector_telemetry(list(source_vectors.values())),
    }
    return rows, summary


def _write_md(path: pathlib.Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    telemetry = summary["source_vector_telemetry"]
    lines = [
        "# SVAMP Process-Trace Sidecars",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- git commit: `{payload.get('git_commit') or 'unknown'}`",
        f"- target set: `{summary['target_set']}`",
        f"- n: `{summary['n']}`",
        f"- sidecar bytes: `{summary['sidecar_bytes']}`",
        f"- max ngram: `{summary['max_ngram']}`",
        f"- exclude labels: `{summary['exclude_labels']}`",
        f"- prediction only: `{summary['prediction_only']}`",
        f"- top label counts: `{summary['top_label_counts']}`",
        f"- top value in unmasked source numbers: `{summary['top_value_in_unmasked_source_numbers']}`",
        f"- zero margin rows: `{summary['zero_margin']}`",
        f"- margin mean: `{summary['margin_mean']:.6f}`",
        "",
        "## Collapse Telemetry",
        "",
        f"- feature count: `{telemetry['feature_count']}`",
        f"- std min: `{telemetry['std_min']:.6f}`",
        f"- std mean: `{telemetry['std_mean']:.6f}`",
        f"- effective rank: `{telemetry['effective_rank']:.6f}`",
        f"- zero vectors: `{telemetry['zero_vectors']}`",
        "",
        "## Command",
        "",
        "```bash",
        payload["command"],
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-set", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sidecar-bits", type=int, default=256)
    parser.add_argument("--max-ngram", type=int, default=2)
    parser.add_argument("--exclude-label", action="append", default=[])
    parser.add_argument("--prediction-only", action="store_true")
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args(argv)
    raw_argv = sys.argv if argv is None else ["scripts/materialize_svamp_process_trace_sidecars.py", *argv]

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sidecar_rows, summary = _materialize(
        target_set_path=_resolve(args.target_set),
        sidecar_bits=int(args.sidecar_bits),
        max_ngram=int(args.max_ngram),
        exclude_labels={str(label) for label in args.exclude_label},
        prediction_only=bool(args.prediction_only),
    )
    sidecar_path = output_dir / "live_candidate_sidecars.jsonl"
    _write_jsonl(sidecar_path, sidecar_rows)
    payload = {
        "date": str(args.date),
        "status": "process_trace_sidecars_materialized",
        "command": shlex.join(raw_argv),
        "git_commit": _git_commit(),
        "summary": summary,
        "outputs": {
            "live_sidecar": _display_path(sidecar_path),
            "manifest_json": _display_path(output_dir / "manifest.json"),
            "manifest_md": _display_path(output_dir / "manifest.md"),
        },
        "hashes": {"live_sidecar_sha256": _sha256_file(sidecar_path)},
    }
    json_path = output_dir / "manifest.json"
    md_path = output_dir / "manifest.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_md(md_path, payload)
    print(json.dumps({"status": payload["status"], "manifest": _display_path(json_path)}, indent=2))
    return payload


if __name__ == "__main__":
    main()
