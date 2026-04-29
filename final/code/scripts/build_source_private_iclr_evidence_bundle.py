from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import pathlib
import stat
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]


REPRODUCTION_COMMANDS = [
    "./venv_arm64/bin/python scripts/build_source_private_rate_frontier.py --output-dir results/source_private_rate_frontier_20260429",
    "./venv_arm64/bin/python scripts/build_source_private_kv_cache_baseline_table.py --output-dir results/source_private_kv_cache_baseline_table_20260429",
    "./venv_arm64/bin/python scripts/run_source_private_coded_label_risk_gate.py --examples 160 --candidates 4 --family-set all --seeds 29,31,37 --budget 2 --output-dir results/source_private_coded_label_risk_gate_20260429",
    "./venv_arm64/bin/python scripts/build_source_private_pass_fail_ledger.py --output-dir results/source_private_pass_fail_ledger_20260429",
    "find final -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 shasum -a 256 > final/MANIFEST.sha256",
    "shasum -a 256 -c final/MANIFEST.sha256",
    "./venv_arm64/bin/python -m pytest tests/test_build_source_private_rate_frontier.py tests/test_build_source_private_kv_cache_baseline_table.py tests/test_run_source_private_coded_label_risk_gate.py tests/test_build_source_private_pass_fail_ledger.py -q",
]


REQUIRED_ARTIFACTS = {
    "rate_frontier": "results/source_private_rate_frontier_20260429/rate_frontier.json",
    "kv_cache_baseline": "results/source_private_kv_cache_baseline_table_20260429/kv_cache_baseline_table.json",
    "coded_label_risk": "results/source_private_coded_label_risk_gate_20260429/summary.json",
    "pass_fail_ledger": "results/source_private_pass_fail_ledger_20260429/pass_fail_ledger.json",
    "endpoint_uncertainty_core": "results/source_private_endpoint_uncertainty_20260429/core_label_strict_n160/summary.json",
    "endpoint_uncertainty_holdout": "results/source_private_endpoint_uncertainty_20260429/label_strict_n160/summary.json",
    "systems_summary": "results/source_private_systems_summary_20260428/systems_summary.json",
    "final_table_doc": "paper/source_private_tool_trace_final_table_20260429.md",
    "readiness_doc": "paper/repo_readiness_review_20260426.md",
    "final_manifest": "final/MANIFEST.sha256",
}


NOVELTY_MATRIX = [
    {
        "comparison": "LatentWire source-private packet",
        "source": "this work",
        "communicated_object": "rate-capped private evidence packet decoded with target candidate side information",
        "source_private": True,
        "requires_model_internals": False,
        "extreme_byte_rate": True,
        "source_destroying_controls": True,
        "systems_axis": "bytes, local latency, candidate accuracy, controls",
        "paper_role": "headline method",
    },
    {
        "comparison": "C2C cache-to-cache communication",
        "source": "https://arxiv.org/abs/2510.03215",
        "communicated_object": "projected/fused source KV cache",
        "source_private": "partly",
        "requires_model_internals": True,
        "extreme_byte_rate": False,
        "source_destroying_controls": "not same threat model",
        "systems_axis": "cache transfer accuracy and latency",
        "paper_role": "closest high-rate internal-state baseline/framing",
    },
    {
        "comparison": "KVComm selective KV sharing",
        "source": "https://openreview.net/forum?id=F7rUng23nw",
        "communicated_object": "selected KV pairs/layers",
        "source_private": "partly",
        "requires_model_internals": True,
        "extreme_byte_rate": False,
        "source_destroying_controls": "not same threat model",
        "systems_axis": "fraction of KV cache transmitted",
        "paper_role": "high-rate KV communication baseline/framing",
    },
    {
        "comparison": "TurboQuant / vector-KV quantization",
        "source": "https://arxiv.org/abs/2504.19874",
        "communicated_object": "quantized vectors or KV/cache states",
        "source_private": False,
        "requires_model_internals": True,
        "extreme_byte_rate": False,
        "source_destroying_controls": False,
        "systems_axis": "bits per vector/cache element",
        "paper_role": "systems byte-floor comparator and future vector-packet ablation",
    },
    {
        "comparison": "QJL 1-bit sign sketch",
        "source": "https://arxiv.org/abs/2406.03482",
        "communicated_object": "JL-projected sign sketches for inner products/KV",
        "source_private": False,
        "requires_model_internals": True,
        "extreme_byte_rate": "low-bit but high-dimensional",
        "source_destroying_controls": False,
        "systems_axis": "1-bit cache/vector compression",
        "paper_role": "matched-byte vector sketch baseline if latent branch is promoted",
    },
    {
        "comparison": "Prompt/text compression such as LLMLingua-family methods",
        "source": "https://arxiv.org/abs/2310.05736",
        "communicated_object": "compressed visible prompt/context tokens",
        "source_private": False,
        "requires_model_internals": False,
        "extreme_byte_rate": "token-level",
        "source_destroying_controls": False,
        "systems_axis": "prompt tokens, quality, latency",
        "paper_role": "structured text/compression framing; query-aware text relay is the local control",
    },
    {
        "comparison": "Slepian-Wolf / Wyner-Ziv source coding",
        "source": "https://www.itsoc.org/publications/papers/noiseless-coding-of-correlated-information-sources",
        "communicated_object": "syndrome/source code with decoder side information",
        "source_private": True,
        "requires_model_internals": False,
        "extreme_byte_rate": True,
        "source_destroying_controls": "theory, not benchmark controls",
        "systems_axis": "rate-distortion/coding limit",
        "paper_role": "theory framing, not empirical LLM baseline",
    },
    {
        "comparison": "JEPA / diffusion-transformer latent prediction",
        "source": "https://openaccess.thecvf.com/content/CVPR2023/papers/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.pdf",
        "communicated_object": "predicted latent/representation state",
        "source_private": "not primary",
        "requires_model_internals": True,
        "extreme_byte_rate": False,
        "source_destroying_controls": False,
        "systems_axis": "latent prediction quality",
        "paper_role": "inspiration for future learned receiver; not current claim",
    },
]


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_status() -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    for name, relative in REQUIRED_ARTIFACTS.items():
        path = ROOT / relative
        status[name] = {
            "path": relative,
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else None,
            "sha256": _sha256_file(path) if path.exists() and path.is_file() else None,
        }
    return status


def _contribution_rows(
    *,
    rate: dict[str, Any],
    kv: dict[str, Any],
    coded: dict[str, Any],
    ledger: dict[str, Any],
    endpoint_core: dict[str, Any],
    endpoint_holdout: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "contribution": "Source-private evidence-packet benchmark and controls",
            "status": "strong scoped contribution",
            "headline_evidence": (
                f"{coded['examples']} examples x {len(coded['seeds'])} seeds x "
                f"{len(coded['transforms'])} label/code/order stress transforms pass"
            ),
            "main_metric": (
                f"matched={coded['by_transform']['label_code_order_composed']['min_matched_accuracy']:.3f}, "
                f"target={coded['by_transform']['label_code_order_composed']['max_target_accuracy']:.3f}, "
                f"worst_control={coded['by_transform']['label_code_order_composed']['max_best_source_destroying_control']:.3f}"
            ),
            "remaining_gap": "Still a protocol/candidate-decoder task; frame as source-private evidence communication, not universal semantics.",
        },
        {
            "contribution": "Extreme-rate candidate-syndrome packet method",
            "status": "headline method for scoped paper",
            "headline_evidence": (
                f"packet oracle bytes max={rate['headline']['packet_oracle_bytes_max']:.1f}; "
                f"matched-byte text accuracy max={rate['headline']['matched_byte_text_at_packet_accuracy_max']:.3f}"
            ),
            "main_metric": (
                f"packet vs query-aware text >= {rate['headline']['packet_vs_query_aware_oracle_compression_min']:.1f}x; "
                f"packet vs full log >= {rate['headline']['packet_vs_full_log_compression_min']:.1f}x"
            ),
            "remaining_gap": "Text becomes oracle at higher bytes, so claim only the far-left rate frontier.",
        },
        {
            "contribution": "Systems byte/KV-cache accounting frontier",
            "status": "systems contribution with clear caveat",
            "headline_evidence": (
                f"minimum QJL-style 1-bit cache payload is "
                f"{kv['headline']['min_non_packet_qjl_1bit_bytes_vs_packet']:.1f}x packet"
            ),
            "main_metric": (
                f"minimum KIVI-style 2-bit cache payload is "
                f"{kv['headline']['min_non_packet_kivi_2bit_bytes_vs_packet']:.1f}x packet"
            ),
            "remaining_gap": "Derived byte accounting only; no production GPU serving throughput yet.",
        },
        {
            "contribution": "Endpoint paired uncertainty and local target-decoder evidence",
            "status": "paper-ready evidence rows exist, but systems scope is local",
            "headline_evidence": (
                f"endpoint paired rows pass with min packet-vs-target CI lows "
                f"{endpoint_core['min_packet_vs_target_ci95_low']:.3f}/{endpoint_holdout['min_packet_vs_target_ci95_low']:.3f}"
            ),
            "main_metric": f"paper-ready rows in ledger={len(ledger['paper_ready_rows'])}; total audited rows={ledger['total_rows']}",
            "remaining_gap": "Mac-local proxy, not server TTFT/TPOT/throughput.",
        },
        {
            "contribution": "Learned receiver / latent-method diagnostics",
            "status": "bounded diagnostic contribution, not headline cross-family claim",
            "headline_evidence": (
                f"ledger records {ledger['by_contribution'].get('learned target-preserving receiver', {}).get('positive_needs_more_evidence', 0)} "
                "positive learned-receiver rows and explicit failed/pruned rows"
            ),
            "main_metric": "same-distribution positives exist; simple cross-family masked innovation failed",
            "remaining_gap": "Need shared-dictionary/crosscoder-style method with feature knockout before claiming cross-family latent communication.",
        },
    ]


def _pass_checks(
    *,
    artifacts: dict[str, dict[str, Any]],
    rate: dict[str, Any],
    kv: dict[str, Any],
    coded: dict[str, Any],
    ledger: dict[str, Any],
    endpoint_core: dict[str, Any],
    endpoint_holdout: dict[str, Any],
) -> list[dict[str, Any]]:
    checks = [
        ("required_artifacts_exist", all(row["exists"] for row in artifacts.values())),
        ("rate_frontier_passes", bool(rate["pass_gate"])),
        ("matched_byte_text_stays_at_target", rate["headline"]["matched_byte_text_at_packet_accuracy_max"] <= 0.25),
        ("packet_beats_query_aware_text_by_7x", rate["headline"]["packet_vs_query_aware_oracle_compression_min"] >= 7.0),
        ("kv_cache_qjl_lower_bound_above_1000x", kv["headline"]["min_non_packet_qjl_1bit_bytes_vs_packet"] >= 1000.0),
        ("coded_label_risk_passes", bool(coded["pass_gate"])),
        (
            "composed_label_code_order_stress_passes",
            bool(coded["by_transform"]["label_code_order_composed"]["pass_gate"]),
        ),
        ("endpoint_core_uncertainty_passes", bool(endpoint_core["pass_gate"])),
        ("endpoint_holdout_uncertainty_passes", bool(endpoint_holdout["pass_gate"])),
        ("ledger_has_paper_ready_rows", len(ledger["paper_ready_rows"]) >= 3),
    ]
    return [{"check": name, "pass": bool(value)} for name, value in checks]


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Source-Private ICLR Evidence Bundle",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- created UTC: `{payload['created_utc']}`",
        f"- current readiness: `{payload['readiness']}`",
        "",
        "## Technical Contributions",
        "",
        "| Contribution | Status | Headline evidence | Main metric | Remaining gap |",
        "|---|---|---|---|---|",
    ]
    for row in payload["contribution_rows"]:
        lines.append(
            "| "
            f"{row['contribution']} | {row['status']} | {row['headline_evidence']} | "
            f"{row['main_metric']} | {row['remaining_gap']} |"
        )
    lines.extend(
        [
            "",
            "## Pass Checks",
            "",
            "| Check | Pass |",
            "|---|---|",
        ]
    )
    for check in payload["pass_checks"]:
        lines.append(f"| `{check['check']}` | `{check['pass']}` |")
    lines.extend(
        [
            "",
            "## Novelty Matrix",
            "",
            "| Comparison | Source | Communicated object | Source-private | Internals? | Extreme rate? | Controls? | Paper role |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in payload["novelty_matrix"]:
        lines.append(
            "| "
            f"{row['comparison']} | {row['source']} | {row['communicated_object']} | "
            f"{row['source_private']} | {row['requires_model_internals']} | "
            f"{row['extreme_byte_rate']} | {row['source_destroying_controls']} | {row['paper_role']} |"
        )
    lines.extend(
        [
            "",
            "## Reproduction Commands",
            "",
            "```bash",
            *payload["reproduction_commands"],
            "```",
            "",
            "## Remaining ICLR Risks",
            "",
        ]
    )
    for risk in payload["remaining_iclr_risks"]:
        lines.append(f"- {risk}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_bundle(*, output_dir: pathlib.Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = _artifact_status()
    missing = [name for name, row in artifacts.items() if not row["exists"]]
    if missing:
        raise FileNotFoundError(f"missing required artifacts: {', '.join(missing)}")

    rate = _read_json(ROOT / REQUIRED_ARTIFACTS["rate_frontier"])
    kv = _read_json(ROOT / REQUIRED_ARTIFACTS["kv_cache_baseline"])
    coded = _read_json(ROOT / REQUIRED_ARTIFACTS["coded_label_risk"])
    ledger = _read_json(ROOT / REQUIRED_ARTIFACTS["pass_fail_ledger"])
    endpoint_core = _read_json(ROOT / REQUIRED_ARTIFACTS["endpoint_uncertainty_core"])
    endpoint_holdout = _read_json(ROOT / REQUIRED_ARTIFACTS["endpoint_uncertainty_holdout"])

    contribution_rows = _contribution_rows(
        rate=rate,
        kv=kv,
        coded=coded,
        ledger=ledger,
        endpoint_core=endpoint_core,
        endpoint_holdout=endpoint_holdout,
    )
    pass_checks = _pass_checks(
        artifacts=artifacts,
        rate=rate,
        kv=kv,
        coded=coded,
        ledger=ledger,
        endpoint_core=endpoint_core,
        endpoint_holdout=endpoint_holdout,
    )
    payload = {
        "gate": "source_private_iclr_evidence_bundle",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "readiness": "scoped positive-method paper; not broad cross-family latent-transfer ready",
        "pass_gate": all(check["pass"] for check in pass_checks),
        "pass_checks": pass_checks,
        "artifact_status": artifacts,
        "contribution_rows": contribution_rows,
        "novelty_matrix": NOVELTY_MATRIX,
        "reproduction_commands": REPRODUCTION_COMMANDS,
        "remaining_iclr_risks": [
            "Production serving TTFT/TPOT/throughput on NVIDIA GPUs is still missing.",
            "The headline method is protocol/candidate-side-information communication, not universal semantic latent transfer.",
            "Simple learned cross-family masked-innovation receivers failed; a future shared-dictionary/crosscoder method needs feature knockout before promotion.",
            "The final paper must show text relay catches up at higher byte budgets to avoid unfair-baseline criticism.",
        ],
    }

    (output_dir / "iclr_evidence_bundle.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown(output_dir / "iclr_evidence_bundle.md", payload)
    _write_csv(output_dir / "novelty_matrix.csv", NOVELTY_MATRIX)
    _write_csv(output_dir / "contribution_matrix.csv", contribution_rows)
    commands_path = output_dir / "reproduce_iclr_evidence_bundle.sh"
    commands_path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(REPRODUCTION_COMMANDS) + "\n",
        encoding="utf-8",
    )
    commands_path.chmod(commands_path.stat().st_mode | stat.S_IXUSR)

    artifacts_to_hash = [
        "iclr_evidence_bundle.json",
        "iclr_evidence_bundle.md",
        "novelty_matrix.csv",
        "contribution_matrix.csv",
        "reproduce_iclr_evidence_bundle.sh",
        "manifest.json",
        "manifest.md",
    ]
    manifest = {
        "command": "./venv_arm64/bin/python scripts/build_source_private_iclr_evidence_bundle.py --output-dir "
        + str(output_dir.relative_to(ROOT) if output_dir.is_relative_to(ROOT) else output_dir),
        "artifacts": artifacts_to_hash,
        "artifact_sha256": {
            name: _sha256_file(output_dir / name)
            for name in artifacts_to_hash
            if name not in {"manifest.json", "manifest.md"}
        },
        "pass_gate": payload["pass_gate"],
        "python": sys.version,
        "script_sha256": _sha256_file(pathlib.Path(__file__)),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Source-Private ICLR Evidence Bundle Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                f"- contributions: `{len(contribution_rows)}`",
                f"- novelty comparisons: `{len(NOVELTY_MATRIX)}`",
                "",
                "## Artifacts",
                "",
                *[f"- `{name}`" for name in artifacts_to_hash],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = build_bundle(output_dir=output_dir)
    if not payload["pass_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
