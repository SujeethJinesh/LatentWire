#!/usr/bin/env python3
"""Validate submission_packet.zip."""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import zipfile
from pathlib import Path

import yaml
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
HARD_EXTS = {".npz", ".pt", ".bin", ".safetensors", ".npy", ".pkl"}
REQUIRED = {
    "COMMIT.txt",
    "DIFFSTAT.txt",
    "README_INDEX.md",
    "TESTS_RUN.txt",
    "paper/latentwire/draft.md",
    "paper/latentwire/main.tex",
    "paper/latentwire/main.pdf",
    "paper/latentwire/tables/falsification_ladder.md",
    "paper/latentwire/tables/provenance.md",
    "paper/channel_set/draft.md",
    "paper/channel_set/main.tex",
    "paper/channel_set/main.pdf",
    "paper/channel_set/tables/provenance.md",
    "paper/channel_set/tables/regime_checklist.md",
    "paper/references.bib",
    "paper/response_plan.md",
    "reviews/mock_colm_board_iter3.json",
    "dashboard/paper_review_trajectory.md",
    "queues/gpu_foreground.yaml",
}
REQUIRED_FIGURES = {
    "paper/latentwire/figures/receiver_conditioning_bits.png",
    "paper/latentwire/figures/discrete_evidence_bar.png",
    "paper/latentwire/figures/l_ib_utility_leakage.png",
    "paper/latentwire/figures/heldout_source_index_null.png",
    "paper/channel_set/figures/top1_set_leaving.png",
    "paper/channel_set/figures/threshold_sensitivity.png",
    "paper/channel_set/figures/within_set_shuffling.png",
    "paper/channel_set/figures/paroquant_vs_channelset_methods.png",
    "paper/channel_set/figures/c_a1_sentinel_status.png",
    "paper/channel_set/figures/osc_decdec_drift_defense.png",
}


def fail(message: str) -> None:
    raise SystemExit(f"submission packet validation failed: {message}")


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def read_text(zf: zipfile.ZipFile, name: str) -> str:
    return zf.read(name).decode("utf-8", errors="replace")


def validate_commit(zf: zipfile.ZipFile) -> None:
    commit = read_text(zf, "COMMIT.txt")
    match = re.search(r"^commit:\s*([0-9a-f]{40})\s*$", commit, re.M)
    if not match:
        fail("COMMIT.txt lacks full commit line")
    head = git_head()
    if match.group(1) != head:
        fail(f"COMMIT.txt commit {match.group(1)} != HEAD {head}")
    if f"Commit: `{head}`" not in read_text(zf, "README_INDEX.md"):
        fail("README_INDEX.md commit does not match HEAD")

    dirty_tracked = []
    bad_untracked = []
    for line in commit.splitlines():
        if not line or line.startswith(("branch:", "commit:", "created_utc:", "## ")):
            continue
        if line.startswith("?? "):
            path = line[3:].strip()
            if "/" in path or not path.endswith(".zip"):
                bad_untracked.append(path)
        else:
            dirty_tracked.append(line)
    if dirty_tracked:
        fail(f"COMMIT.txt records tracked worktree changes: {dirty_tracked[:10]}")
    if bad_untracked:
        fail(f"COMMIT.txt records disallowed untracked paths: {bad_untracked[:10]}")


def validate_figures(zf: zipfile.ZipFile, names: set[str]) -> None:
    for draft in ["paper/latentwire/draft.md", "paper/channel_set/draft.md"]:
        text = read_text(zf, draft)
        refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
        missing = []
        for ref in refs:
            if ref.startswith("http"):
                continue
            path = (Path(draft).parent / ref).as_posix()
            if path not in names:
                missing.append(path)
        if missing:
            fail(f"{draft} has missing figure refs: {missing}")

    for tex in ["paper/latentwire/main.tex", "paper/channel_set/main.tex"]:
        text = read_text(zf, tex)
        refs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)
        missing = []
        for ref in refs:
            path = (Path(tex).parent / ref).as_posix()
            if path not in names:
                missing.append(path)
        if missing:
            fail(f"{tex} has missing figure refs: {missing}")


def pdf_page_count(zf: zipfile.ZipFile, name: str) -> int:
    return len(PdfReader(io.BytesIO(zf.read(name))).pages)


def validate_pdfs(zf: zipfile.ZipFile) -> None:
    counts = {
        "paper/latentwire/main.pdf": pdf_page_count(zf, "paper/latentwire/main.pdf"),
        "paper/channel_set/main.pdf": pdf_page_count(zf, "paper/channel_set/main.pdf"),
    }
    for name, count in counts.items():
        if not 3 <= count <= 20:
            fail(f"{name} has suspicious page count {count}")


def validate_reviews(zf: zipfile.ZipFile) -> None:
    payload = json.loads(read_text(zf, "reviews/mock_colm_board_iter3.json"))
    if payload.get("status") != "pass":
        fail("iteration 3 board did not pass")
    for paper in ["latentwire", "channel_set"]:
        record = payload["papers"][paper]
        if record.get("area_chair") != "accept":
            fail(f"{paper} area chair did not accept")
        honesty = record["reviewers"]["honesty"]
        if honesty.get("verdict") != "PASS":
            fail(f"{paper} honesty did not pass")
    trajectory = read_text(zf, "dashboard/paper_review_trajectory.md")
    if "Final bar status: both papers" not in trajectory or "honesty PASS" not in trajectory:
        fail("review trajectory lacks final status")


def validate_claim_boundaries(zf: zipfile.ZipFile) -> None:
    latentwire = read_text(zf, "paper/latentwire/draft.md")
    channel = read_text(zf, "paper/channel_set/draft.md")
    checks = [
        ("LatentWire bounded subtitle", "A Controlled Negative for Discrete No-Text Packets", latentwire),
        ("LatentWire L-IB1 killed", "KILL_UTILITY_IS_IDENTITY", latentwire),
        ("LatentWire no deployable positive", "A deployable-positive claim is unsupported", latentwire),
        ("LatentWire discrete caveat", "160` unique examples", latentwire),
        ("Channel-Set subtitle", "A Measurement Regime For Static W4A16 Protection", channel),
        ("Channel-Set C-A1 parked", "C-A1 remains parked", channel),
        ("Channel-Set no positive", "A confirmed-positive claim is unsupported", channel),
        ("Channel-Set no ParoQuant win", "does not claim to beat ParoQuant", channel),
    ]
    missing = [label for label, needle, haystack in checks if needle not in haystack]
    if missing:
        fail(f"missing claim-boundary text: {missing}")


def validate_queue(zf: zipfile.ZipFile) -> None:
    foreground = yaml.safe_load(read_text(zf, "queues/gpu_foreground.yaml")) or {}
    if foreground.get("foreground") != []:
        fail("queues/gpu_foreground.yaml is not empty")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", type=Path)
    args = parser.parse_args()
    zip_path = args.zip_path if args.zip_path.is_absolute() else ROOT / args.zip_path
    if not zip_path.exists():
        fail(f"missing package: {zip_path}")
    if zip_path.stat().st_size > 25 * 1024 * 1024:
        fail(f"package exceeds 25MB: {zip_path.stat().st_size}")
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        missing = sorted((REQUIRED | REQUIRED_FIGURES) - names)
        if missing:
            fail(f"missing required files: {missing}")
        if any(re.search(r"_confirm", name, re.I) for name in names):
            fail("forbidden confirm-looking entry path included")
        hard = [name for name in names if Path(name).suffix in HARD_EXTS]
        if hard:
            fail(f"hard binary artifacts included: {hard}")
        caches = [name for name in names if name.startswith("caches/") or "/caches/" in name]
        if caches:
            fail(f"cache paths included: {caches}")
        too_large = [info.filename for info in zf.infolist() if info.file_size > 5 * 1024 * 1024]
        if too_large:
            fail(f"single file(s) exceed 5MB: {too_large}")
        validate_commit(zf)
        validate_figures(zf, names)
        validate_pdfs(zf)
        validate_reviews(zf)
        validate_claim_boundaries(zf)
        validate_queue(zf)
    print(f"submission packet OK: {zip_path} ({zip_path.stat().st_size} bytes, {len(names)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
