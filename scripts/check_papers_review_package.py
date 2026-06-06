#!/usr/bin/env python3
"""Validate papers_review.zip."""

from __future__ import annotations

import argparse
import re
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HARD_EXTS = {".npz", ".pt", ".bin", ".safetensors", ".npy", ".pkl"}
REQUIRED = {
    "COMMIT.txt",
    "DIFFSTAT.txt",
    "INDEX.md",
    "paper/latentwire/draft.md",
    "paper/channel_set/draft.md",
    "paper/response_plan.md",
    "paper/references.bib",
    "dashboard/paper_review_trajectory.md",
    "reviews/mock_colm_board_iter1.json",
    "reviews/mock_colm_board_iter2.json",
    "reviews/mock_colm_board_iter3.json",
    "paper/latentwire/tables/falsification_ladder.md",
    "paper/latentwire/tables/provenance.md",
    "paper/channel_set/tables/provenance.md",
    "paper/channel_set/tables/regime_checklist.md",
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
    raise SystemExit(f"papers_review validation failed: {message}")


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def read_text(zf: zipfile.ZipFile, name: str) -> str:
    return zf.read(name).decode("utf-8", errors="replace")


def validate_commit(zf: zipfile.ZipFile) -> None:
    commit = read_text(zf, "COMMIT.txt")
    match = re.search(r"^commit:\s*([0-9a-f]{40})\s*$", commit, re.M)
    if not match:
        fail("COMMIT.txt lacks full commit line")
    if match.group(1) != git_head():
        fail(f"COMMIT.txt commit {match.group(1)} != HEAD {git_head()}")
    if f"Commit: `{git_head()}`" not in read_text(zf, "INDEX.md"):
        fail("INDEX.md commit does not match HEAD")


def validate_drafts(zf: zipfile.ZipFile, names: set[str]) -> None:
    for draft in ["paper/latentwire/draft.md", "paper/channel_set/draft.md"]:
        text = read_text(zf, draft)
        refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
        missing = []
        for ref in refs:
            if ref.startswith("http"):
                continue
            path = str((Path(draft).parent / ref).as_posix())
            path = str(Path(path).as_posix())
            if path not in names:
                missing.append(path)
        if missing:
            fail(f"{draft} has missing figure refs: {missing}")
    latentwire = read_text(zf, "paper/latentwire/draft.md")
    channel = read_text(zf, "paper/channel_set/draft.md")
    if "KILL_UTILITY_IS_IDENTITY" not in latentwire:
        fail("LatentWire draft does not keep L-IB1 killed")
    if "deployable-positive claim is unsupported" not in latentwire:
        fail("LatentWire claim boundary is missing deployable-positive exclusion")
    if "C-A1 remains parked" not in channel:
        fail("Channel-Set draft does not keep C-A1 parked")
    if "confirmed-positive claim is unsupported" not in channel:
        fail("Channel-Set claim boundary is missing positive exclusion")


def validate_reviews(zf: zipfile.ZipFile) -> None:
    text = read_text(zf, "dashboard/paper_review_trajectory.md")
    if "Final bar status: both papers" not in text:
        fail("review trajectory lacks final bar status")
    if "honesty PASS" not in text:
        fail("review trajectory lacks honesty PASS")
    iter2 = read_text(zf, "reviews/mock_colm_board_iter2.json")
    if '"status": "pass"' not in iter2:
        fail("iteration 2 did not pass")
    if "accept-conditional" not in iter2:
        fail("iteration 2 lacks AC accept-conditional")
    iter3 = read_text(zf, "reviews/mock_colm_board_iter3.json")
    if '"status": "pass"' not in iter3:
        fail("iteration 3 did not pass")
    if '"area_chair": "accept"' not in iter3:
        fail("iteration 3 lacks AC accept")


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
        bad_confirm = [name for name in names if re.search(r"_confirm", name, re.I)]
        if bad_confirm:
            fail(f"forbidden confirm-looking entry paths: {bad_confirm}")
        hard = [name for name in names if Path(name).suffix in HARD_EXTS]
        if hard:
            fail(f"hard binary artifacts included: {hard}")
        caches = [name for name in names if name.startswith("caches/") or "/caches/" in name]
        if caches:
            fail(f"cache paths included: {caches}")
        validate_commit(zf)
        validate_drafts(zf, names)
        validate_reviews(zf)
    print(f"papers_review OK: {zip_path} ({zip_path.stat().st_size} bytes, {len(names)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
