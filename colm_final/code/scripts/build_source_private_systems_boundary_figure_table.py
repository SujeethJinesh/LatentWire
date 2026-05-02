from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_COMPARATOR = pathlib.Path(
    "results/source_private_cross_benchmark_systems_comparator_20260502/"
    "cross_benchmark_systems_comparator.json"
)
DEFAULT_OUTPUT = pathlib.Path("results/source_private_systems_boundary_figure_table_20260502")

CSV_COLUMNS = (
    "row_id",
    "row_group",
    "method",
    "communicated_object",
    "source_private",
    "raw_bytes",
    "framed_bytes",
    "cacheline_bytes",
    "batch64_bytes",
    "source_text_exposed",
    "source_kv_exposed",
    "source_hidden_or_score_vector_exposed",
    "native_measured",
    "measurement_status",
    "nvidia_vllm_required",
    "claim_allowed",
    "overclaim_guard",
    "source_url",
    "notes",
)

PRIMARY_SOURCES = {
    "c2c": {
        "url": "https://arxiv.org/abs/2510.03215",
        "summary": "C2C projects and fuses a source model KV cache into the target cache.",
    },
    "kvcomm": {
        "url": "https://arxiv.org/abs/2510.03346",
        "summary": "KVComm selectively shares KV pairs and reports as few as 30% of layers.",
    },
    "kvcomm_cross_context": {
        "url": "https://arxiv.org/abs/2510.12872",
        "summary": "KVCOMM aligns and reuses cross-context KV caches for multi-agent prefill.",
    },
    "q_kvcomm": {
        "url": "https://arxiv.org/abs/2512.17914",
        "summary": "Q-KVComm transmits compressed KV cache representations with reported 5-6x compression.",
    },
    "kivi": {
        "url": "https://arxiv.org/abs/2402.02750",
        "summary": "KIVI is a tuning-free asymmetric 2-bit KV-cache quantization method.",
    },
    "kvquant": {
        "url": "https://arxiv.org/abs/2401.18079",
        "summary": "KVQuant targets accurate sub-4-bit KV-cache quantization.",
    },
    "qjl": {
        "url": "https://arxiv.org/abs/2406.03482",
        "summary": "QJL uses a JL transform and 1-bit quantization for KV-cache compression.",
    },
    "turboquant": {
        "url": "https://arxiv.org/abs/2504.19874",
        "summary": "TurboQuant reports quality-neutral KV-cache quantization at 3.5 bits per channel.",
    },
    "vllm": {
        "url": "https://arxiv.org/abs/2309.06180",
        "summary": "vLLM/PagedAttention is a KV-cache serving substrate.",
    },
    "sglang": {
        "url": "https://arxiv.org/abs/2312.07104",
        "summary": "SGLang uses RadixAttention for KV-cache reuse in structured LM programs.",
    },
}


def _resolve(path: pathlib.Path) -> pathlib.Path:
    return path if path.is_absolute() else ROOT / path


def _rel(path: pathlib.Path) -> str:
    resolved = _resolve(path)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _sha256_file(path: pathlib.Path) -> str:
    resolved = _resolve(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _round_up(value: float, quantum: int) -> float:
    if value <= 0:
        return 0.0
    return float(quantum * math.ceil(value / quantum))


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _fmt_bytes(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}B"
    return f"{value:.1f}B"


def _tex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _record(
    *,
    row_id: str,
    row_group: str,
    method: str,
    communicated_object: str,
    raw_bytes: float,
    framed_bytes: float | None = None,
    source_text_exposed: bool = False,
    source_kv_exposed: bool = False,
    source_hidden_or_score_vector_exposed: bool = False,
    native_measured: bool = False,
    measurement_status: str,
    nvidia_vllm_required: bool = False,
    claim_allowed: str,
    overclaim_guard: str | None = None,
    source_url: str,
    notes: str,
) -> dict[str, Any]:
    framed = float(raw_bytes if framed_bytes is None else framed_bytes)
    source_private = (
        not source_text_exposed
        and not source_kv_exposed
        and not source_hidden_or_score_vector_exposed
    )
    return {
        "row_id": row_id,
        "row_group": row_group,
        "method": method,
        "communicated_object": communicated_object,
        "source_private": source_private,
        "raw_bytes": float(raw_bytes),
        "framed_bytes": framed,
        "cacheline_bytes": _round_up(framed, 64),
        "batch64_bytes": framed * 64.0,
        "source_text_exposed": bool(source_text_exposed),
        "source_kv_exposed": bool(source_kv_exposed),
        "source_hidden_or_score_vector_exposed": bool(source_hidden_or_score_vector_exposed),
        "native_measured": bool(native_measured),
        "measurement_status": measurement_status,
        "nvidia_vllm_required": bool(nvidia_vllm_required),
        "claim_allowed": claim_allowed,
        "overclaim_guard": overclaim_guard
        or "Do not treat this accounting row as a native latency, goodput, HBM, or baseline-defeat claim.",
        "source_url": source_url,
        "notes": notes,
    }


def _kv_bytes(source_config: dict[str, Any], *, bits_per_element: float, layer_fraction: float = 1.0) -> float:
    elements = float(source_config["kv_elements_per_source_token"])
    return elements * float(bits_per_element) * float(layer_fraction) / 8.0


def _packet_rows(comparator: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in comparator["rows"]:
        method = f"LatentWire {row['dataset']} {row['split']} packet"
        rows.append(
            _record(
                row_id=f"latentwire_{row['row_id']}",
                row_group="LatentWire packet",
                method=method,
                communicated_object="task-level candidate evidence packet",
                raw_bytes=float(row["payload_bytes"]),
                framed_bytes=float(row["framed_record_bytes"]),
                source_text_exposed=bool(row["source_text_exposed"]),
                source_kv_exposed=bool(row["source_kv_exposed"]),
                native_measured=True,
                measurement_status="mac_local_artifact",
                claim_allowed=(
                    "source-private byte/exposure accounting with local benchmark accuracy; "
                    "not a native GPU serving row"
                ),
                overclaim_guard=(
                    "Use as a packet byte/exposure row only; native GPU serving speed is still unmeasured."
                ),
                source_url=row["artifact_path"],
                notes=(
                    f"accuracy={row['matched_accuracy_mean']:.3f}; "
                    f"headline_eligible={str(row['headline_eligible']).lower()}"
                ),
            )
        )
    return rows


def _control_rows(comparator: dict[str, Any], source_config: dict[str, Any]) -> list[dict[str, Any]]:
    arc = next(row for row in comparator["rows"] if row["dataset"] == "ARC-Challenge")
    hidden_bytes = float(source_config["hidden_size"]) * 2.0
    return [
        _record(
            row_id="same_byte_text_control_arc",
            row_group="local control",
            method="Same-byte structured text control (ARC)",
            communicated_object="text-form control with same packet budget",
            raw_bytes=float(arc["payload_bytes"]),
            framed_bytes=float(arc["framed_record_bytes"]),
            source_text_exposed=True,
            native_measured=True,
            measurement_status="mac_local_control",
            claim_allowed="negative/control row only; cannot support source-private claim",
            overclaim_guard="This row intentionally exposes source text and is not a source-private method.",
            source_url=arc["artifact_path"],
            notes=f"same-byte text accuracy={arc['same_byte_text_accuracy']:.3f}",
        ),
        _record(
            row_id="hidden_vector_fp16_floor",
            row_group="state relay floor",
            method="One hidden-vector fp16 relay floor",
            communicated_object="one source hidden vector, fp16",
            raw_bytes=hidden_bytes,
            framed_bytes=hidden_bytes,
            source_hidden_or_score_vector_exposed=True,
            measurement_status="byte_floor_from_model_config",
            claim_allowed="state-exposure lower bound only; not a baseline win",
            overclaim_guard=(
                "This row exposes a continuous hidden vector; do not claim it is a LatentWire packet."
            ),
            source_url="local Qwen2.5-0.5B config",
            notes=f"hidden_size={source_config['hidden_size']}",
        ),
    ]


def _external_rows(source_config: dict[str, Any]) -> list[dict[str, Any]]:
    fp16_kv = _kv_bytes(source_config, bits_per_element=16.0)
    kvcomm30 = _kv_bytes(source_config, bits_per_element=16.0, layer_fraction=0.30)
    kivi2 = _kv_bytes(source_config, bits_per_element=2.0)
    kvquant3 = _kv_bytes(source_config, bits_per_element=3.0)
    qjl1 = _kv_bytes(source_config, bits_per_element=1.0)
    turbo35 = _kv_bytes(source_config, bits_per_element=3.5)
    q_kvcomm_6x = fp16_kv / 6.0
    return [
        _record(
            row_id="qjl_1bit_kv_floor",
            row_group="KV/source-state floor",
            method="1-bit/KV-element accounting floor",
            communicated_object="one-token K+V state at 1 bit/element",
            raw_bytes=qjl1,
            source_kv_exposed=True,
            measurement_status="optimistic_byte_floor_not_native",
            nvidia_vllm_required=True,
            claim_allowed="mathematical state-size lower bound only",
            overclaim_guard=(
                "This is an internal one-bit-per-KV-element accounting floor; "
                "do not treat it as a defeated QJL native baseline."
            ),
            source_url=PRIMARY_SOURCES["qjl"]["url"],
            notes=PRIMARY_SOURCES["qjl"]["summary"],
        ),
        _record(
            row_id="kivi_2bit_kv_floor",
            row_group="KV/source-state floor",
            method="KIVI 2-bit KV floor",
            communicated_object="one-token K+V state at 2 bits/element",
            raw_bytes=kivi2,
            source_kv_exposed=True,
            measurement_status="byte_floor_not_native",
            nvidia_vllm_required=True,
            claim_allowed="KV-cache compression comparator only",
            overclaim_guard="KIVI compresses KV cache state and still requires native quality/throughput comparison.",
            source_url=PRIMARY_SOURCES["kivi"]["url"],
            notes=PRIMARY_SOURCES["kivi"]["summary"],
        ),
        _record(
            row_id="q_kvcomm_6x_floor",
            row_group="KV communication floor",
            method="Q-KVComm optimistic 6x floor",
            communicated_object="compressed source KV cache representation",
            raw_bytes=q_kvcomm_6x,
            source_kv_exposed=True,
            measurement_status="ratio_floor_not_native",
            nvidia_vllm_required=True,
            claim_allowed="compressed-KV communication boundary only",
            overclaim_guard=(
                "Q-KVComm transmits compressed KV-derived state; do not claim this is a source-private packet."
            ),
            source_url=PRIMARY_SOURCES["q_kvcomm"]["url"],
            notes=PRIMARY_SOURCES["q_kvcomm"]["summary"],
        ),
        _record(
            row_id="kvquant_3bit_kv_floor",
            row_group="KV/source-state floor",
            method="KVQuant 3-bit proxy floor",
            communicated_object="one-token K+V state at 3 bits/element",
            raw_bytes=kvquant3,
            source_kv_exposed=True,
            measurement_status="proxy_byte_floor_not_native",
            nvidia_vllm_required=True,
            claim_allowed="sub-4-bit KV comparator only",
            overclaim_guard=(
                "KVQuant is a KV-cache quantization comparator; do not claim it is a packet baseline."
            ),
            source_url=PRIMARY_SOURCES["kvquant"]["url"],
            notes=PRIMARY_SOURCES["kvquant"]["summary"],
        ),
        _record(
            row_id="turboquant_3p5bit_kv_floor",
            row_group="KV/source-state floor",
            method="TurboQuant 3.5-bit KV floor",
            communicated_object="one-token K+V state at 3.5 bits/element",
            raw_bytes=turbo35,
            source_kv_exposed=True,
            measurement_status="byte_floor_not_native",
            nvidia_vllm_required=True,
            claim_allowed="KV/vector quantization comparator only",
            overclaim_guard="TurboQuant is a vector/KV quantizer; byte floors alone do not compare native latency.",
            source_url=PRIMARY_SOURCES["turboquant"]["url"],
            notes=PRIMARY_SOURCES["turboquant"]["summary"],
        ),
        _record(
            row_id="kvcomm30_fp16_floor",
            row_group="KV communication floor",
            method="KVComm 30% fp16 KV floor",
            communicated_object="selected source KV layers, fp16",
            raw_bytes=kvcomm30,
            source_kv_exposed=True,
            measurement_status="30pct_layer_floor_not_native",
            nvidia_vllm_required=True,
            claim_allowed="selective-KV communication boundary only",
            overclaim_guard="KVComm shares selected KV pairs; this is only a native-pending 30%-layer byte floor.",
            source_url=PRIMARY_SOURCES["kvcomm"]["url"],
            notes=PRIMARY_SOURCES["kvcomm"]["summary"],
        ),
        _record(
            row_id="c2c_fp16_kv_floor",
            row_group="KV communication floor",
            method="C2C one-token fp16 KV floor",
            communicated_object="projected/fused source KV cache",
            raw_bytes=fp16_kv,
            source_kv_exposed=True,
            measurement_status="one_token_floor_not_native",
            nvidia_vllm_required=True,
            claim_allowed="closest cache-transfer baseline; native run still required",
            overclaim_guard="C2C fuses source KV cache; do not claim superiority without native C2C.",
            source_url=PRIMARY_SOURCES["c2c"]["url"],
            notes=PRIMARY_SOURCES["c2c"]["summary"],
        ),
        _record(
            row_id="kvcomm_cross_context_fp16_floor",
            row_group="KV communication floor",
            method="KVCOMM cross-context fp16 KV floor",
            communicated_object="aligned/reused source KV cache",
            raw_bytes=fp16_kv,
            source_kv_exposed=True,
            measurement_status="one_token_floor_not_native",
            nvidia_vllm_required=True,
            claim_allowed="systems neighbor only; native run still required",
            overclaim_guard="KVCOMM is a cross-context KV reuse system; this row is only a native-pending byte floor.",
            source_url=PRIMARY_SOURCES["kvcomm_cross_context"]["url"],
            notes=PRIMARY_SOURCES["kvcomm_cross_context"]["summary"],
        ),
        _record(
            row_id="vllm_pagedattention_fp16_kv_floor",
            row_group="serving substrate",
            method="vLLM/PagedAttention one-token KV floor",
            communicated_object="paged KV-cache serving substrate",
            raw_bytes=fp16_kv,
            measurement_status="native_nvidia_pending",
            nvidia_vllm_required=True,
            claim_allowed="native TTFT/TPOT/goodput/HBM target, not closed on Mac",
            overclaim_guard="No vLLM serving claim is valid until native NVIDIA rows are measured.",
            source_url=PRIMARY_SOURCES["vllm"]["url"],
            notes=PRIMARY_SOURCES["vllm"]["summary"],
        ),
        _record(
            row_id="sglang_radixattention_fp16_kv_floor",
            row_group="serving substrate",
            method="SGLang/RadixAttention one-token KV floor",
            communicated_object="KV-cache reuse serving substrate",
            raw_bytes=fp16_kv,
            measurement_status="native_nvidia_pending",
            nvidia_vllm_required=True,
            claim_allowed="native TTFT/TPOT/goodput/HBM target, not closed on Mac",
            overclaim_guard="No SGLang serving claim is valid until native NVIDIA rows are measured.",
            source_url=PRIMARY_SOURCES["sglang"]["url"],
            notes=PRIMARY_SOURCES["sglang"]["summary"],
        ),
    ]


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(row.get(key)) for key in CSV_COLUMNS})


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Systems Boundary Figure/Table V3",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- packet framed-byte range: `{payload['headline']['min_packet_framed_bytes']:.0f}-"
        f"{payload['headline']['max_packet_framed_bytes']:.0f}B`",
        f"- minimum source-state floor vs max packet: "
        f"`{payload['headline']['min_source_state_floor_ratio_vs_max_packet']:.1f}x`",
        f"- native NVIDIA systems complete: `{payload['headline']['native_nvidia_complete']}`",
        "",
        "## Rows",
        "",
        "| Method | Object | Raw | Framed | Cacheline | Batch64 | Private | Text | KV | Hidden | Native | Claim |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["method"],
                    row["communicated_object"],
                    _fmt_bytes(row["raw_bytes"]),
                    _fmt_bytes(row["framed_bytes"]),
                    _fmt_bytes(row["cacheline_bytes"]),
                    _fmt_bytes(row["batch64_bytes"]),
                    "`true`" if row["source_private"] else "`false`",
                    "`true`" if row["source_text_exposed"] else "`false`",
                    "`true`" if row["source_kv_exposed"] else "`false`",
                    "`true`" if row["source_hidden_or_score_vector_exposed"] else "`false`",
                    "`true`" if row["native_measured"] else "`false`",
                    row["claim_allowed"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            payload["headline"]["claim_scope"],
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_tex(path: pathlib.Path, payload: dict[str, Any]) -> None:
    selected = [
        row
        for row in payload["rows"]
        if row["row_group"] in {"LatentWire packet", "KV communication floor", "KV/source-state floor"}
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Method & Object & Raw & Framed & Cacheline & Batch64 & Src text & Src KV & Src hidden \\",
        r"\midrule",
    ]
    for row in selected:
        lines.append(
            " & ".join(
                [
                    _tex_escape(row["method"]),
                    _tex_escape(row["communicated_object"]),
                    _tex_escape(_fmt_bytes(row["raw_bytes"])),
                    _tex_escape(_fmt_bytes(row["framed_bytes"])),
                    _tex_escape(_fmt_bytes(row["cacheline_bytes"])),
                    _tex_escape(_fmt_bytes(row["batch64_bytes"])),
                    "yes" if row["source_text_exposed"] else "no",
                    "yes" if row["source_kv_exposed"] else "no",
                    "yes" if row["source_hidden_or_score_vector_exposed"] else "no",
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\caption{Systems boundary table. LatentWire rows are source-private task packets; "
            r"KV/cache rows are source-state byte floors or pending native baselines, not native wins.}",
            r"\label{tab:systems-boundary}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_svg(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: (row["framed_bytes"], row["row_group"], row["method"]))
    left = 330
    top = 32
    row_h = 29
    plot_w = 690
    height = top * 2 + row_h * len(ordered) + 42
    width = left + plot_w + 60
    values = [max(1.0, float(row["framed_bytes"])) for row in ordered]
    min_log = math.floor(math.log10(min(values))) - 0.1
    max_log = math.ceil(math.log10(max(values))) + 0.1

    def x_for(value: float) -> float:
        log_value = math.log10(max(1.0, value))
        return left + (log_value - min_log) / (max_log - min_log) * plot_w

    ticks = [1, 4, 15, 64, 256, 768, 1536, 2688, 3686.4, 12288]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        'viewBox="0 0 {0} {1}">'.format(width, height),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="24" font-family="Arial, sans-serif" font-size="18" font-weight="700">'
        "LatentWire source-private packet boundary vs KV/source-state floors</text>",
    ]
    axis_y = top + row_h * len(ordered) + 8
    parts.append(f'<line x1="{left}" y1="{axis_y}" x2="{left + plot_w}" y2="{axis_y}" stroke="#333"/>')
    for tick in ticks:
        if tick < min(values) or tick > max(values) * 1.1:
            continue
        x = x_for(float(tick))
        parts.append(f'<line x1="{x:.1f}" y1="{top - 4}" x2="{x:.1f}" y2="{axis_y}" stroke="#e1e5ea"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{axis_y + 18}" text-anchor="middle" '
            'font-family="Arial, sans-serif" font-size="11" fill="#333">'
            f"{html.escape(_fmt_bytes(float(tick)))}</text>"
        )
    colors = {
        "LatentWire packet": "#1f77b4",
        "local control": "#8c8c8c",
        "state relay floor": "#d62728",
        "KV/source-state floor": "#ff7f0e",
        "KV communication floor": "#9467bd",
        "serving substrate": "#2ca02c",
    }
    for index, row in enumerate(ordered):
        y = top + index * row_h + 22
        x0 = left
        x1 = x_for(float(row["framed_bytes"]))
        label = html.escape(row["method"])
        object_text = html.escape(row["communicated_object"])
        color = colors.get(row["row_group"], "#444")
        parts.append(
            f'<text x="24" y="{y + 4}" font-family="Arial, sans-serif" font-size="12" fill="#111">'
            f"{label}</text>"
        )
        parts.append(
            f'<text x="24" y="{y + 17}" font-family="Arial, sans-serif" font-size="10" fill="#555">'
            f"{object_text}</text>"
        )
        parts.append(f'<rect x="{x0}" y="{y - 9}" width="{max(2.0, x1 - x0):.1f}" height="14" fill="{color}"/>')
        parts.append(
            f'<text x="{x1 + 6:.1f}" y="{y + 2}" font-family="Arial, sans-serif" font-size="11" fill="#111">'
            f"{html.escape(_fmt_bytes(float(row['framed_bytes'])))}</text>"
        )
    parts.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 8}" text-anchor="middle" '
        'font-family="Arial, sans-serif" font-size="12" fill="#333">'
        "Log-scale framed or state bytes per request; KV rows are byte floors, not native wins</text>"
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _make_manifest(output_dir: pathlib.Path, payload: dict[str, Any]) -> dict[str, Any]:
    files = [
        "systems_boundary_figure_data.json",
        "systems_boundary_figure_data.csv",
        "systems_boundary_table.md",
        "systems_boundary_table.tex",
        "systems_boundary_waterfall.svg",
    ]
    manifest = {
        "gate": payload["gate"],
        "pass_gate": payload["pass_gate"],
        "headline": payload["headline"],
        "files": [
            {
                "path": _rel(output_dir / filename),
                "sha256": _sha256_file(output_dir / filename),
            }
            for filename in files
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_systems_boundary_figure_table(
    *,
    comparator_path: pathlib.Path = DEFAULT_COMPARATOR,
    output_dir: pathlib.Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparator = _read_json(comparator_path)
    source_config = comparator["source_model_config"]
    rows = (
        _packet_rows(comparator)
        + _control_rows(comparator, source_config)
        + _external_rows(source_config)
    )
    packet_rows = [row for row in rows if row["row_group"] == "LatentWire packet"]
    source_state_rows = [
        row
        for row in rows
        if row["source_kv_exposed"] or row["source_hidden_or_score_vector_exposed"]
    ]
    max_packet = max(row["framed_bytes"] for row in packet_rows)
    min_state_floor = min(row["framed_bytes"] for row in source_state_rows)
    checks = [
        {
            "check": "packet_rows_remain_source_private",
            "pass": all(
                row["source_private"]
                and not row["source_text_exposed"]
                and not row["source_kv_exposed"]
                and not row["source_hidden_or_score_vector_exposed"]
                for row in packet_rows
            ),
            "value": len(packet_rows),
        },
        {
            "check": "source_state_floor_exceeds_max_packet_by_50x",
            "pass": min_state_floor / max_packet >= 50.0,
            "value": min_state_floor / max_packet,
        },
        {
            "check": "native_nvidia_rows_marked_pending",
            "pass": all(
                row["measurement_status"] != "native_nvidia_pending"
                or (not row["native_measured"] and row["nvidia_vllm_required"])
                for row in rows
            ),
            "value": "native NVIDIA rows are pending and non-claims",
        },
        {
            "check": "source_state_rows_have_overclaim_guards",
            "pass": all(
                row["overclaim_guard"]
                and ("claim" in row["overclaim_guard"].lower() or "native" in row["overclaim_guard"].lower())
                for row in source_state_rows
            ),
            "value": len(source_state_rows),
        },
    ]
    headline = {
        "pass_gate": all(check["pass"] for check in checks),
        "packet_row_count": len(packet_rows),
        "min_packet_framed_bytes": min(row["framed_bytes"] for row in packet_rows),
        "max_packet_framed_bytes": max_packet,
        "min_source_state_floor_bytes": min_state_floor,
        "min_source_state_floor_ratio_vs_max_packet": min_state_floor / max_packet,
        "native_nvidia_complete": False,
        "claim_scope": (
            "Paper-ready systems boundary artifact: LatentWire rows are fixed-byte source-private task "
            "packets; KV/cache rows are byte floors or pending native serving baselines. The artifact "
            "supports byte/exposure accounting, not a native C2C/KVComm/TurboQuant/QJL/vLLM/SGLang win."
        ),
    }
    payload = {
        "gate": "source_private_systems_boundary_figure_table",
        "pass_gate": headline["pass_gate"],
        "source_comparator": {
            "path": _rel(comparator_path),
            "sha256": _sha256_file(comparator_path),
        },
        "source_model_config": source_config,
        "headline": headline,
        "checks": checks,
        "primary_sources": PRIMARY_SOURCES,
        "rows": rows,
    }
    figure_json = output_dir / "systems_boundary_figure_data.json"
    figure_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output_dir / "systems_boundary_figure_data.csv", rows)
    _write_markdown(output_dir / "systems_boundary_table.md", payload)
    _write_tex(output_dir / "systems_boundary_table.tex", payload)
    _write_svg(output_dir / "systems_boundary_waterfall.svg", rows)
    _make_manifest(output_dir, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparator", type=pathlib.Path, default=DEFAULT_COMPARATOR)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_systems_boundary_figure_table(
        comparator_path=args.comparator,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "pass_gate": payload["pass_gate"],
                "output_dir": str(_resolve(args.output_dir)),
                "min_source_state_floor_ratio_vs_max_packet": payload["headline"][
                    "min_source_state_floor_ratio_vs_max_packet"
                ],
                "native_nvidia_complete": payload["headline"]["native_nvidia_complete"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
