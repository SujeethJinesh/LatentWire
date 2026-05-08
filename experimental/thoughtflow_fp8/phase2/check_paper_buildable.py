#!/usr/bin/env python3
"""Check the ThoughtFlow-FP8 paper-polish result packet."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PASS_DECISION = "PASS_THOUGHTFLOW_PAPER_BUILDABLE"
KILL_DECISION = "FAIL_THOUGHTFLOW_PAPER_BROKEN"
INFRA_DECISION = "FAIL_INFRA_THOUGHTFLOW_PAPER_POLISH"

REQUIRED = [
    "environment.json",
    "metrics.json",
    "logs/stdout.log",
    "logs/build_stdout.log",
    "logs/build_stderr.log",
    "logs/pytest_stdout.log",
    "logs/pytest_stderr.log",
    "build/thoughtflow_fp8_colm2026.pdf",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def source_relative(text: str) -> Path | None:
    path = Path(text)
    if path.is_absolute() or ".." in path.parts or str(path) in {"", "."}:
        return None
    return path


def evaluate(run_dir: Path) -> dict[str, Any]:
    infra: list[str] = []
    broken: list[str] = []
    for rel in REQUIRED:
        if not (run_dir / rel).is_file():
            infra.append(f"missing required packet file: {rel}")
    if infra:
        return {"decision": INFRA_DECISION, "run_dir": str(run_dir), "reasons": infra}

    try:
        metrics = load_json(run_dir / "metrics.json")
    except Exception as exc:
        return {"decision": INFRA_DECISION, "run_dir": str(run_dir), "reasons": [f"bad metrics JSON: {exc!r}"]}

    if metrics.get("schema_version") != "thoughtflow_paper_polish_metrics_v1":
        infra.append("metrics schema_version mismatch")
    if metrics.get("paper_mode") != "copyedit_only_falsification_methodology":
        infra.append("paper_mode does not preserve ThoughtFlow copyedit-only status")
    if int(metrics.get("build_returncode", -1)) != 0:
        broken.append("paper build command returned nonzero")
    if int(metrics.get("pytest_returncode", -1)) != 0:
        broken.append("ThoughtFlow owned tests returned nonzero")

    paper_tex_rel = source_relative(str(metrics.get("paper_tex", "")))
    reviewer_pack_rel = source_relative(str(metrics.get("reviewer_pack", "")))
    tracked_pdf_rel = source_relative(str(metrics.get("tracked_pdf", "")))
    built_pdf_rel = source_relative(str(metrics.get("built_pdf", "")))
    for label, rel_path in [
        ("paper_tex", paper_tex_rel),
        ("reviewer_pack", reviewer_pack_rel),
        ("tracked_pdf", tracked_pdf_rel),
        ("built_pdf", built_pdf_rel),
    ]:
        if rel_path is None:
            infra.append(f"{label} path is not repo-relative")
        elif not (ROOT / rel_path).is_file():
            infra.append(f"{label} path does not exist: {rel_path}")

    if paper_tex_rel is not None and (ROOT / paper_tex_rel).is_file():
        paper = (ROOT / paper_tex_rel).read_text(encoding="utf-8")
        required_phrases = [
            "The contribution is diagnostic, not a positive method",
            "no real FP8, CUDA, latency, throughput, or live compression-method claim is made",
            "not reasoning-model benchmarks",
            "Anonymous authors",
        ]
        for phrase in required_phrases:
            if phrase not in paper:
                broken.append(f"paper missing required caveat/anonymity phrase: {phrase}")
        forbidden_phrases = ["TODO", "FIXME", "XXX"]
        for phrase in forbidden_phrases:
            if phrase in paper:
                broken.append(f"paper contains forbidden draft marker: {phrase}")

    if reviewer_pack_rel is not None and (ROOT / reviewer_pack_rel).is_file():
        pack = (ROOT / reviewer_pack_rel).read_text(encoding="utf-8")
        required_pack_phrases = [
            "no live positive method branch",
            "falsification-methodology workshop note",
            "Do not claim method novelty",
        ]
        for phrase in required_pack_phrases:
            if phrase not in pack:
                broken.append(f"reviewer pack missing required status phrase: {phrase}")

    # The reviewer pack is allowed to be updated after the runner records a
    # paper-polish packet, because it needs to point reviewers at that packet.
    # Check its existence and required status language above, but avoid a
    # self-referential hash requirement that would make the pack impossible to
    # keep current.
    hash_fields = [
        ("paper_tex_sha256", paper_tex_rel),
        ("tracked_pdf_sha256", tracked_pdf_rel),
        ("built_pdf_sha256", built_pdf_rel),
    ]
    for field, rel_path in hash_fields:
        if rel_path is not None and (ROOT / rel_path).is_file() and metrics.get(field) != sha256(ROOT / rel_path):
            infra.append(f"{field} mismatch")

    if infra:
        return {"decision": INFRA_DECISION, "run_dir": str(run_dir), "reasons": infra}
    if broken:
        return {"decision": KILL_DECISION, "run_dir": str(run_dir), "reasons": broken}
    return {"decision": PASS_DECISION, "run_dir": str(run_dir), "reasons": []}


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) != 1:
        print("usage: check_paper_buildable.py <run_dir>", file=sys.stderr)
        return 2
    result = evaluate(Path(argv[0]).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"] == PASS_DECISION else 1


if __name__ == "__main__":
    raise SystemExit(main())
