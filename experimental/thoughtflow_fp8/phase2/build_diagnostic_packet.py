"""Build the ThoughtFlow diagnostic provenance packet.

This packet is not a new experiment. It hashes the saved falsification artifacts
and writes one compact evidence ladder so the stopped branch can be reviewed
without confusing historical first-surface wins for a live method claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from .run_real_trace_retention import DEFAULT_TRACES
except ImportError:  # pragma: no cover - supports direct script execution.
    from run_real_trace_retention import DEFAULT_TRACES


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

ARTIFACT_COMMANDS = {
    "frozen_sparse_cache_probe": (
        "./venv_arm64/bin/python experimental/thoughtflow_fp8/phase2/frozen_sparse_cache_probe.py"
    ),
    "rdu_robustness_diagnostic": (
        "./venv_arm64/bin/python experimental/thoughtflow_fp8/phase2/rdu_robustness_diagnostic.py"
    ),
    "rdu_same_surface_rerun": (
        "./venv_arm64/bin/python experimental/thoughtflow_fp8/phase2/rdu_no_retune_reproduction_check.py"
    ),
    "rdu_alternate_surface": (
        "./venv_arm64/bin/python experimental/thoughtflow_fp8/phase2/rdu_alt_surface_reproduction_check.py"
    ),
    "rdu_independent_surface": (
        "./venv_arm64/bin/python experimental/thoughtflow_fp8/phase2/rdu_independent_trace_reproduction_check.py"
    ),
    "psi_fresh_surface": (
        "./venv_arm64/bin/python experimental/thoughtflow_fp8/phase2/psi_fresh_sparse_cache_check.py"
    ),
    "vwac_fresh_surface": (
        "./venv_arm64/bin/python experimental/thoughtflow_fp8/phase2/vwac_fresh_sparse_cache_check.py"
    ),
}

LEGACY_DEFAULT_TRACE_ARTIFACTS = {
    "frozen_sparse_cache_probe",
    "rdu_same_surface_rerun",
    "rdu_alternate_surface",
}

LEGACY_CACHED_RDU_ARTIFACTS = {
    "rdu_same_surface_rerun",
    "rdu_alternate_surface",
}

PREREGISTRATIONS = [
    {
        "id": "rdu_preregistration",
        "path": "preregister_recurrence_distance_utility_20260506.md",
        "role": "one_shot_method_preregistration",
    },
    {
        "id": "psi_preregistration",
        "path": "preregister_prefix_surprisal_utility_20260506.md",
        "role": "fresh_successor_preregistration",
    },
    {
        "id": "vwac_preregistration",
        "path": "preregister_value_weighted_attention_contribution_20260506.md",
        "role": "fresh_successor_preregistration",
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
        path_status = run_git("status", "--short", "--", "experimental/thoughtflow_fp8")
        path_tree = run_git("rev-parse", "HEAD:experimental/thoughtflow_fp8")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            "head_at_generation": "unavailable",
            "thoughtflow_path_tree_at_generation": "unavailable",
            "thoughtflow_path_dirty_at_generation": True,
        }
    return {
        "head_at_generation": head,
        "thoughtflow_path_tree_at_generation": path_tree,
        "thoughtflow_path_dirty_at_generation": bool(path_status),
        "thoughtflow_path_status_at_generation": path_status,
    }


def _thoughtflow_path_status() -> str:
    return subprocess.check_output(
        ["git", "status", "--short", "--", "experimental/thoughtflow_fp8"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def _require_clean_thoughtflow_tree() -> None:
    try:
        path_status = _thoughtflow_path_status()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError("refusing to build diagnostic packet: git status unavailable") from exc
    if path_status:
        raise RuntimeError(
            "refusing to build diagnostic packet while experimental/thoughtflow_fp8 is dirty:\n"
            f"{path_status}"
        )


def _resolve_input_path(raw_path: str) -> Path | None:
    path = Path(raw_path)
    candidates = [path] if path.is_absolute() else [REPO_ROOT / path, PHASE2 / path]
    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(REPO_ROOT.resolve())
        except ValueError:
            continue
        if resolved.is_file():
            return resolved
    return None


def _hash_existing_input_paths(paths: list[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for raw_path in paths:
        resolved = _resolve_input_path(raw_path)
        if resolved is None:
            raise ValueError(f"unresolved diagnostic packet input path: {raw_path}")
        hashes[str(resolved.relative_to(REPO_ROOT))] = _sha256(resolved)
    return hashes


def _repo_path_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _default_trace_path_labels() -> list[str]:
    return [_repo_path_label(path) for path in DEFAULT_TRACES]


def _artifact_provenance(artifact_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    metadata_keys = [
        "model_name",
        "model_id",
        "model_revision",
        "tokenizer_name",
        "tokenizer_revision",
        "keep_fraction",
        "continuation_tokens",
        "max_length",
        "max_traces",
        "n_scored_traces",
        "policy_name",
        "source_artifact",
    ]
    source_metadata = {key: payload[key] for key in metadata_keys if key in payload}
    for section in (
        "cached_surface",
        "measured_surface",
        "cached_baseline",
        "measured_reproduction",
    ):
        section_payload = payload.get(section)
        if isinstance(section_payload, dict):
            selected = {key: section_payload[key] for key in metadata_keys if key in section_payload}
            if selected:
                source_metadata[section] = selected
    input_paths: list[str] = []
    for key in ("source_artifact", "input_paths", "trace_input_paths"):
        value = payload.get(key)
        if isinstance(value, str):
            input_paths.append(value)
        elif isinstance(value, list):
            input_paths.extend(str(item) for item in value if isinstance(item, str))
    input_path_inference: dict[str, Any] | None = None
    if not input_paths and artifact_id in LEGACY_DEFAULT_TRACE_ARTIFACTS:
        input_paths = _default_trace_path_labels()
        if artifact_id in LEGACY_CACHED_RDU_ARTIFACTS:
            input_paths.append("frozen_sparse_cache_probe.json")
        input_path_inference = {
            "source": "run_real_trace_retention.DEFAULT_TRACES",
            "reason": (
                "legacy artifact was generated by frozen sparse-cache defaults "
                "before those default trace paths were serialized"
            ),
        }
    provenance = {
        "command": ARTIFACT_COMMANDS.get(artifact_id, "not_recorded"),
        "input_paths": input_paths,
        "source_metadata": source_metadata,
        "input_hashes": _hash_existing_input_paths(input_paths),
    }
    if input_path_inference is not None:
        provenance["input_path_inference"] = input_path_inference
    return provenance


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


def _diagnostic_readout(artifact: dict[str, Any]) -> str:
    """Return a readout that cannot be mistaken for live method evidence."""

    status = str(artifact["summary"]["status"])
    role = str(artifact["role"])
    if role.startswith("stale_positive") or role.startswith("historical_positive"):
        return "SUPERSEDED historical readout; later gates demote or kill this row"
    return status


def build_packet(output_dir: Path = DEFAULT_OUTPUT, *, require_clean_tree: bool = True) -> dict[str, Any]:
    if require_clean_tree:
        _require_clean_thoughtflow_tree()
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
                "provenance": _artifact_provenance(str(spec["id"]), payload),
            }
        )
        historical_positive = str(spec["role"]).startswith(
            ("stale_positive", "historical_positive")
        )
        if historical_positive:
            original_status = str(artifacts[-1]["summary"]["status"])
            artifacts[-1]["historical_status"] = original_status.split(maxsplit=1)[0]
            artifacts[-1]["summary"]["status"] = f"HISTORICAL/SUPERSEDED: {original_status}"
        artifacts[-1]["current_status"] = (
            "superseded_diagnostic_only"
            if historical_positive
            else "current_falsification_evidence"
        )
        artifacts[-1]["current_claim_allowed"] = not historical_positive
        artifacts[-1]["positive_method_claim_allowed"] = False
    preregistrations = []
    for spec in PREREGISTRATIONS:
        path = PHASE2 / str(spec["path"])
        preregistrations.append(
            {
                **spec,
                "sha256": _sha256(path),
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
            "sha256_role": (
                "current builder-file integrity hash for verifier drift detection; "
                "git.head_at_generation records the historical packet generation commit"
            ),
            "command": (
                "./venv_arm64/bin/python experimental/thoughtflow_fp8/phase2/build_diagnostic_packet.py "
                "--output .debug/thoughtflow_diagnostic_packet_check"
            ),
        },
        "artifacts": artifacts,
        "preregistrations": preregistrations,
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
        status = _diagnostic_readout(artifact).replace("|", "\\|")
        lines.append(
            f"| `{artifact['path']}` | {artifact['role']} | `{artifact['sha256']}` | {status} |"
        )
    lines.extend(
        [
            "",
            "## Preregistrations",
            "",
            "| File | Role | SHA-256 |",
            "|---|---|---|",
        ]
    )
    for prereg in preregistrations:
        lines.append(f"| `{prereg['path']}` | {prereg['role']} | `{prereg['sha256']}` |")
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
                "Status: tracked falsification provenance packet, not positive-method evidence.",
                "",
                "Run:",
                "",
                "```bash",
                "./venv_arm64/bin/python experimental/thoughtflow_fp8/phase2/build_diagnostic_packet.py \\",
                "  --output .debug/thoughtflow_diagnostic_packet_check",
                "```",
                "",
                "Expected pass condition: the builder exits successfully only from a clean",
                "`experimental/thoughtflow_fp8/` path and rewrites the same manifest/table shape",
                "with explicit hashes for the saved decision artifacts.",
                "",
                "Then inspect:",
                "",
                "- `manifest.json` for git state, source metadata, saved-artifact hashes, and",
                "  input-hash provenance where available.",
                "- `falsification_table.md` for the consumed signal ladder and stop decisions.",
                "- `../../current_decision_manifest_20260506.md` for the current branch-level",
                "  claim boundary.",
                "",
                "This tracked packet lives outside ignored `results/` directories so",
                "clean checkouts can run the saved-artifact tests.",
                "This packet does not reopen the stopped RDU/PSI/VWAC branches.",
                "",
                "Local correctness command:",
                "",
                "```bash",
                "cd /Users/sujeethjinesh/Desktop/LatentWire",
                "TRITON_CPU_BACKEND=1 TRITON_INTERPRET=1 \\",
                'TRITON_HOME="$PWD/.debug/triton_home" \\',
                "./venv_arm64/bin/python -m pytest \\",
                "  experimental/thoughtflow_fp8/phase2/tests \\",
                "  experimental/thoughtflow_fp8/phase4/tests -q -rs",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Diagnostic packet output directory. Use a .debug path for local verification.",
    )
    args = parser.parse_args()
    build_packet(args.output)


if __name__ == "__main__":
    main()
