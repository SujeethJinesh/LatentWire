"""Build the ThoughtFlow diagnostic provenance packet.

This packet is not a new experiment. It hashes the saved falsification artifacts
and writes one compact evidence ladder so the stopped branch can be reviewed
without confusing historical first-surface wins for a live method claim.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


PHASE2 = Path(__file__).resolve().parent
REPO_ROOT = PHASE2.parents[2]
DEFAULT_OUTPUT = PHASE2 / "diagnostic_packets/thoughtflow_diagnostic_packet_20260506"

ARTIFACTS = [
    {
        "id": "frozen_sparse_cache_probe",
        "path": "frozen_sparse_cache_probe.json",
        "role": "stale_positive_first_surface",
        "claim": "The first frozen sparse-cache surface made rdu_topk look alive.",
    },
    {
        "id": "rdu_robustness_diagnostic",
        "path": "rdu_robustness_diagnostic.json",
        "role": "stale_positive_robustness_probe",
        "claim": "Cached deterministic splits preserved positive means but not a fresh-surface claim.",
    },
    {
        "id": "rdu_same_surface_rerun",
        "path": "rdu_no_retune_reproduction_check.json",
        "role": "historical_positive_same_surface",
        "claim": "rdu_topk reproduced the first frozen sparse-cache gate on the same surface.",
    },
    {
        "id": "rdu_alternate_surface",
        "path": "rdu_alt_surface_reproduction_check.json",
        "role": "same_family_falsification",
        "claim": "rdu_topk failed strict same-family separation on an alternate surface.",
    },
    {
        "id": "rdu_independent_surface",
        "path": "rdu_independent_trace_reproduction_check.json",
        "role": "cross_family_falsification",
        "claim": "rdu_topk failed cross-family separation on independent saved traces.",
    },
    {
        "id": "psi_fresh_surface",
        "path": "psi_fresh_sparse_cache_check.json",
        "role": "fresh_successor_kill",
        "claim": "psi_topk failed its preregistered fresh-surface promotion rule.",
    },
    {
        "id": "vwac_fresh_surface",
        "path": "vwac_fresh_sparse_cache_check.json",
        "role": "fresh_successor_kill",
        "claim": "vwac_topk failed its preregistered fresh-surface promotion rule.",
    },
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for package in ["torch", "transformers", "numpy", "pytest", "triton"]:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def _git_metadata() -> dict[str, str | bool]:
    def run_git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()

    try:
        head = run_git("rev-parse", "HEAD")
        status = run_git("status", "--short")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"head_at_generation": "unavailable", "dirty_at_generation": True}
    return {"head_at_generation": head, "dirty_at_generation": bool(status)}


def _artifact_summary(artifact_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    if artifact_id == "frozen_sparse_cache_probe":
        summary = payload["summary"]
        return {
            "status": payload["status"],
            "n_scored_traces": payload["n_scored_traces"],
            "rdu_topk_nll": summary["rdu_topk"]["nll"],
            "rkv_like_nll": summary["rkv_like"]["nll"],
            "thin_kv_like_nll": summary["thin_kv_like"]["nll"],
        }
    if artifact_id == "rdu_robustness_diagnostic":
        return {
            "status": payload["status"],
            "n_scored_traces": payload["n_scored_traces"],
            "split_mean_margin_passes": payload["split_mean_margin_passes"],
            "split_paired_mean_passes": payload["split_paired_mean_passes"],
            "split_promotion_passes": payload["split_promotion_passes"],
        }
    if artifact_id == "rdu_same_surface_rerun":
        measured = payload["measured_decision"]
        return {
            "status": payload["status"],
            "reproduction_pass": payload["reproduction_pass"],
            "best_compressed_policy": measured["best_compressed_policy"],
            "rdu_nll": measured["rdu_nll"],
            "rdu_is_best_compressed": measured["rdu_is_best_compressed"],
        }
    if artifact_id in {"rdu_alternate_surface", "rdu_independent_surface"}:
        measured = payload["measured_decision"]
        strict = payload.get("strict_family_pass") or payload.get("measured_family_separation", {})
        return {
            "status": payload["status"],
            "reproduction_pass": payload["reproduction_pass"],
            "best_compressed_policy": measured["best_compressed_policy"],
            "rdu_nll": measured["rdu_nll"],
            "rdu_is_best_compressed": measured["rdu_is_best_compressed"],
            "strict_family_readout": strict,
        }
    if artifact_id in {"psi_fresh_surface", "vwac_fresh_surface"}:
        decision = payload["decision"]
        policy_name = payload["policy_name"]
        summary = payload["summary"]
        return {
            "status": payload["status"],
            "policy_name": policy_name,
            "n_scored_traces": payload["n_scored_traces"],
            "promotion_pass": decision["promotion_pass"],
            "policy_nll": summary[policy_name]["nll"],
            "rkv_like_nll": summary["rkv_like"]["nll"],
            "thin_kv_like_nll": summary["thin_kv_like"]["nll"],
        }
    raise ValueError(f"unknown artifact id: {artifact_id}")


def build_packet(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for spec in ARTIFACTS:
        path = PHASE2 / str(spec["path"])
        payload = _load_json(path)
        artifacts.append(
            {
                **spec,
                "sha256": _sha256(path),
                "summary": _artifact_summary(str(spec["id"]), payload),
            }
        )

    manifest = {
        "packet_name": "thoughtflow_diagnostic_packet_20260506",
        "claim_boundary": [
            "diagnostic falsification packet",
            "not a positive method claim",
            "not a GPU/FP8/latency claim",
        ],
        "generated_date": "2026-05-06",
        "git": _git_metadata(),
        "environment": _package_versions(),
        "script": {
            "path": str(Path(__file__).relative_to(REPO_ROOT)),
            "sha256": _sha256(Path(__file__)),
            "command": "./venv_arm64/bin/python experimental/thoughtflow_fp8/phase2/build_diagnostic_packet.py",
        },
        "artifacts": artifacts,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    lines = [
        "# ThoughtFlow Diagnostic Provenance Packet",
        "",
        "Status: diagnostic only; not a positive-method packet.",
        "",
        "| Artifact | Role | SHA-256 | Readout |",
        "|---|---|---|---|",
    ]
    for artifact in artifacts:
        summary = artifact["summary"]
        status = str(summary["status"]).replace("|", "\\|")
        lines.append(
            f"| `{artifact['path']}` | {artifact['role']} | `{artifact['sha256']}` | {status} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The same-surface RDU row is historical. It is paired here with the",
            "original stale-positive surface, the cached robustness diagnostic,",
            "the alternate-surface same-family failure, the independent-surface",
            "cross-family failure, and the fresh PSI/VWAC successor kills. Together",
            "these artifacts lock the current branch as diagnostic-only.",
            "",
        ]
    )
    (output_dir / "falsification_table.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# ThoughtFlow Diagnostic Packet",
                "",
                "Run:",
                "",
                "```bash",
                "./venv_arm64/bin/python experimental/thoughtflow_fp8/phase2/build_diagnostic_packet.py",
                "```",
                "",
                "Then inspect `manifest.json` and `falsification_table.md`.",
                "This tracked packet lives outside ignored `results/` directories so",
                "clean checkouts can run the saved-artifact tests.",
                "This packet does not reopen the stopped RDU/PSI/VWAC branches.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    build_packet()


if __name__ == "__main__":
    main()
