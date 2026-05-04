from __future__ import annotations

"""ARC target-side behavior-transcoder sparse atom feasibility probe.

This diagnostic is intentionally not a source-private communication result. It
asks whether target-native sparse atoms, fit from Qwen hidden public
innovations, can causally steer Qwen ARC answer margins under the same
candidate/atom destructive controls used by the source-packet harness.
"""

import argparse
import datetime as dt
import gc
import json
import math
import pathlib
import statistics
import sys
from typing import Any, Sequence

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_arc_challenge_behavior_atom_decoder_gate as atom_gate  # noqa: E402
from scripts import build_source_private_arc_challenge_behavior_residual_packet_gate as behavior_gate  # noqa: E402
from scripts import build_source_private_arc_challenge_hidden_atom_decoder_gate as hidden_gate  # noqa: E402
from scripts import build_source_private_arc_challenge_soft_prefix_resonance_gate as soft_gate  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402
from scripts import run_source_private_arc_openbookqa_soft_prefix_preflight as preflight  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/arc_challenge_target_behavior_transcoder_probe_20260504_qwen3_disagreement"
)
DEFAULT_VALIDATION = soft_gate.DEFAULT_VALIDATION
DEFAULT_TEST = soft_gate.DEFAULT_TEST
DEFAULT_SOURCE_FAMILY_GATE_DIR = soft_gate.DEFAULT_SOURCE_FAMILY_GATE_DIR
DEFAULT_TARGET_MODEL = behavior_gate.DEFAULT_QWEN3_MODEL

MATCHED_CONDITION = "matched_target_behavior_atom_packet"
CONTROL_CONDITIONS = (
    "target_only",
    "target_derived_packet",
    "zero_packet",
    "row_shuffle",
    "atom_shuffle",
    "coefficient_shuffle",
    "top_atom_knockout",
    "candidate_roll",
    "candidate_derangement",
)
REPORT_CONDITIONS = (MATCHED_CONDITION, *CONTROL_CONDITIONS)


def _target_oracle_scope_metadata() -> dict[str, Any]:
    return {
        "claim_scope": "target_hidden_oracle_feasibility_only",
        "source_model_used": False,
        "source_private": False,
        "source_exposure_scope": "not_applicable_no_source_model",
        "diagnostic_target_native_packet": True,
        "target_hidden_runtime_used": True,
    }


def _same_shape_shuffle_index(
    *,
    row_index: int,
    eval_indices: Sequence[int],
    rows: Sequence[arc_gate.ArcRow],
) -> int:
    row = rows[row_index]
    same_shape = [index for index in eval_indices if index != row_index and len(rows[index].choices) == len(row.choices)]
    if same_shape:
        return same_shape[0]
    other = [index for index in eval_indices if index != row_index]
    return other[0] if other else row_index


def _condition_metrics(
    prediction_rows: list[dict[str, Any]],
    *,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for condition in REPORT_CONDITIONS:
        subset = [row for row in prediction_rows if row["condition"] == condition]
        correct = sum(1 for row in subset if row["correct"])
        fired = sum(1 for row in subset if row.get("packet_fired"))
        helped = sum(1 for row in subset if row.get("packet_helped"))
        harmed = sum(1 for row in subset if row.get("packet_harmed"))
        metrics[condition] = {
            "n": len(subset),
            "correct": int(correct),
            "accuracy": float(correct / len(subset)) if subset else 0.0,
            "mean_margin": float(statistics.fmean(float(row["margin"]) for row in subset)) if subset else 0.0,
            "packet_fired": int(fired),
            "packet_fired_rate": float(fired / len(subset)) if subset else 0.0,
            "packet_helped_vs_target": int(helped),
            "packet_harmed_vs_target": int(harmed),
            "packet_net_help_vs_target": int(helped - harmed),
        }
    by_id: dict[str, dict[str, dict[str, Any]]] = {}
    for row in prediction_rows:
        by_id.setdefault(str(row["content_id"]), {})[str(row["condition"])] = row
    for control in CONTROL_CONDITIONS:
        correct_deltas = [
            float(group[MATCHED_CONDITION]["correct"]) - float(group[control]["correct"])
            for group in by_id.values()
            if MATCHED_CONDITION in group and control in group
        ]
        margin_deltas = [
            float(group[MATCHED_CONDITION]["margin"]) - float(group[control]["margin"])
            for group in by_id.values()
            if MATCHED_CONDITION in group and control in group
        ]
        metrics[MATCHED_CONDITION][f"paired_accuracy_vs_{control}"] = behavior_gate._paired_bootstrap(
            correct_deltas,
            seed=seed + len(control),
            samples=bootstrap_samples,
        )
        metrics[MATCHED_CONDITION][f"mean_margin_delta_vs_{control}"] = (
            float(statistics.fmean(margin_deltas)) if margin_deltas else 0.0
        )
    return metrics


def _fixed_weight_sweep(
    prediction_rows: list[dict[str, Any]],
    *,
    weights: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
) -> dict[str, Any]:
    by_id: dict[str, dict[str, dict[str, Any]]] = {}
    for row in prediction_rows:
        by_id.setdefault(str(row["content_id"]), {})[str(row["condition"])] = row
    diagnostics: dict[str, Any] = {}
    for condition in (MATCHED_CONDITION, *[name for name in CONTROL_CONDITIONS if name != "target_only"]):
        condition_rows: list[dict[str, Any]] = []
        for weight in weights:
            correct = 0
            helped = 0
            harmed = 0
            n = 0
            for group in by_id.values():
                if "target_only" not in group or condition not in group:
                    continue
                target_row = group["target_only"]
                packet_row = group[condition]
                target = np.asarray(target_row["scores"], dtype=np.float64)
                residual = np.asarray(packet_row["packet_residual"], dtype=np.float64)
                scores = target + float(weight) * residual
                pred = int(np.argmax(scores))
                answer_index = int(packet_row["answer_index"])
                is_correct = pred == answer_index
                target_correct = bool(target_row["correct"])
                correct += int(is_correct)
                helped += int(is_correct and not target_correct)
                harmed += int((not is_correct) and target_correct)
                n += 1
            condition_rows.append(
                {
                    "residual_weight": float(weight),
                    "n": int(n),
                    "accuracy": float(correct / n) if n else 0.0,
                    "helped_vs_target": int(helped),
                    "harmed_vs_target": int(harmed),
                    "net_help_vs_target": int(helped - harmed),
                }
            )
        best = max(condition_rows, key=lambda row: (row["accuracy"], row["net_help_vs_target"], -row["harmed_vs_target"]))
        diagnostics[condition] = {"best": best, "rows": condition_rows}
    return diagnostics


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    headline = payload["strict_headline"]
    lines = [
        "# ARC Target Behavior-Transcoder Probe",
        "",
        f"- date: `{payload['date']}`",
        f"- diagnostic pass: `{payload['pass_gate']}`",
        f"- claim scope: `{payload['systems_packet_sideband']['claim_scope']}`",
        f"- slice selection: `{payload['inputs']['slice_selection']}`",
        f"- train/test disagreement rows: `{payload['train_rows']}` / `{payload['test_rows']}`",
        f"- matched accuracy: `{headline['matched_accuracy']:.6f}`",
        f"- target-only accuracy: `{headline['target_only_accuracy']:.6f}`",
        f"- best control: `{headline['best_required_control']}`",
        f"- best control accuracy: `{headline['best_required_control_accuracy']:.6f}`",
        f"- worst CI95 low: `{headline['worst_required_ci95_low']:.6f}`",
        f"- fired rows: `{headline['matched_packet_fired']}`",
        f"- helps/harms: `{headline['matched_packet_helped']}` / `{headline['matched_packet_harmed']}`",
        f"- diagnostic packet bytes/row: `{payload['systems_packet_sideband']['packet_bytes_per_row']:.3f}`",
        "",
        "## Controls",
        "",
        "| Control | Accuracy | Delta | CI95 low |",
        "|---|---:|---:|---:|",
    ]
    for name, row in payload["strict_control_metrics"].items():
        lines.append(
            f"| `{name}` | {row['control_accuracy']:.6f} | {row['delta_accuracy']:.6f} | {row['ci95_low']:.6f} |"
        )
    sweep = payload.get("fixed_weight_sweep", {})
    if sweep:
        lines.extend(
            [
                "",
                "## Fixed-Weight Diagnostic",
                "",
                "| Packet | Best weight | Accuracy | Helps | Harms |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name, row in sweep.items():
            best = row["best"]
            lines.append(
                f"| `{name}` | {best['residual_weight']:.2f} | {best['accuracy']:.6f} | "
                f"{best['helped_vs_target']} | {best['harmed_vs_target']} |"
            )
    lines.extend(["", "## Interpretation", "", payload["interpretation"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_probe(
    *,
    output_dir: pathlib.Path,
    validation_path: pathlib.Path,
    test_path: pathlib.Path,
    source_family_gate_dir: pathlib.Path,
    target_model: str,
    target_device: str,
    target_attn_implementation: str | None,
    dtype: str,
    target_max_length: int,
    target_hidden_layer: int,
    target_feature_dim: int,
    train_disagreement_limit: int,
    test_disagreement_limit: int,
    ridge: float,
    packet_rank: int,
    packet_top_k: int,
    packet_bits: int,
    batchtopk_epochs: int,
    batchtopk_learning_rate: float,
    batchtopk_batch_size: int,
    batchtopk_reconstruction_weight: float,
    batchtopk_l1_weight: float,
    corruption_loss_weight: float,
    local_files_only: bool,
    bootstrap_samples: int,
    min_accuracy_gap: float,
    min_ci_low: float,
    seed: int,
) -> dict[str, Any]:
    output_dir = behavior_gate._resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = output_dir / "strict_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    agreement_path = behavior_gate._resolve(source_family_gate_dir) / "source_cache_agreement.csv"

    validation_rows_all = arc_gate._load_rows(behavior_gate._resolve(validation_path))
    test_rows_all = arc_gate._load_rows(behavior_gate._resolve(test_path))
    train_ids = soft_gate._read_disagreement_content_ids(
        agreement_path=agreement_path,
        split="validation",
        limit=train_disagreement_limit,
    )
    test_ids = soft_gate._read_disagreement_content_ids(
        agreement_path=agreement_path,
        split="test",
        limit=test_disagreement_limit,
    )
    train_rows = soft_gate._filter_rows_by_content_ids(validation_rows_all, train_ids)
    test_rows = soft_gate._filter_rows_by_content_ids(test_rows_all, test_ids)
    rows = [*train_rows, *test_rows]
    fit_row_count = len(train_rows)
    eval_indices = list(range(fit_row_count, fit_row_count + len(test_rows)))
    fit_candidate_indices = preflight._flat_candidate_indices_for_rows(rows, list(range(fit_row_count)))

    behavior_gate._write_jsonl(
        input_dir / "arc_challenge_validation_train_plus_test_disagreement.jsonl",
        [soft_gate._arc_row_payload(row) for row in rows],
    )

    target_scores, target_score_meta = behavior_gate._score_rows_with_prompt_builder(
        rows,
        model_path=target_model,
        device=target_device,
        dtype=dtype,
        max_length=target_max_length,
        local_files_only=local_files_only,
        normalization="mean",
        prompt_builder=preflight._mcq_prompt,
        attn_implementation=target_attn_implementation,
    )
    target_hidden, target_hidden_meta = preflight._hf_choice_hidden_features(
        rows,
        model_path=target_model,
        device=target_device,
        dtype=dtype,
        max_length=target_max_length,
        local_files_only=local_files_only,
        hidden_layer=target_hidden_layer,
    )
    public_flat, public_meta = preflight._public_candidate_hashed_features(rows, feature_dim=target_feature_dim)
    target_innovation, target_innovation_meta = preflight._public_candidate_innovation_features(
        target_hidden,
        public_flat,
        fit_flat_indices=fit_candidate_indices,
        ridge=ridge,
    )
    behavior_targets = atom_gate._behavior_target_matrix(rows, target_scores)
    packet_flat, sparse_meta = atom_gate._fit_batchtopk_behavior_atom_packet_from_features(
        target_innovation,
        behavior_targets,
        fit_flat_indices=fit_candidate_indices,
        rank=packet_rank,
        top_k=packet_top_k,
        quant_bits=packet_bits,
        epochs=batchtopk_epochs,
        learning_rate=batchtopk_learning_rate,
        batch_size=batchtopk_batch_size,
        reconstruction_weight=batchtopk_reconstruction_weight,
        l1_weight=batchtopk_l1_weight,
        seed=seed,
    )
    target_features = hidden_gate._target_score_features(rows, target_scores)
    target_packet_map = hidden_gate._fit_ridge_matrix_map(
        target_features,
        packet_flat,
        fit_indices=fit_candidate_indices,
        ridge=ridge,
    )
    target_derived_packet_flat = target_packet_map.predict(target_features)
    target_selected = [behavior_gate._prediction(scores) for scores in target_scores]
    decoder, receiver_meta = atom_gate._fit_corruption_noop_decoder(
        rows=rows,
        train_indices=list(range(fit_row_count)),
        target_features=target_features,
        source_packet_flat=packet_flat,
        target_derived_packet_flat=target_derived_packet_flat,
        behavior_targets=behavior_gate._candidate_targets(rows, target_scores),
        source_selected=target_selected,
        decoder_mode="target_conditioned",
        corruption_loss_weight=corruption_loss_weight,
        corruption_condition_weights=None,
        ridge=ridge,
    )

    matched_residuals = atom_gate._decode_packet_residual_rows(
        rows,
        target_features=target_features,
        packet_features=packet_flat,
        decoder=decoder,
        decoder_mode="target_conditioned",
    )
    target_derived_residuals = atom_gate._decode_packet_residual_rows(
        rows,
        target_features=target_features,
        packet_features=target_derived_packet_flat,
        decoder=decoder,
        decoder_mode="target_conditioned",
    )
    zero_packet_flat = np.zeros_like(packet_flat)
    zero_residuals = atom_gate._decode_packet_residual_rows(
        rows,
        target_features=target_features,
        packet_features=zero_packet_flat,
        decoder=decoder,
        decoder_mode="target_conditioned",
    )
    gate_rule = hidden_gate._choose_gate_rule(
        train_rows,
        target_scores[:fit_row_count],
        matched_residuals[:fit_row_count],
    )
    row_packets = hidden_gate._row_packet_arrays(rows, packet_flat)
    target_derived_row_packets = hidden_gate._row_packet_arrays(rows, target_derived_packet_flat)

    prediction_rows: list[dict[str, Any]] = []
    offsets = hidden_gate._row_offsets(rows)
    for eval_position, row in enumerate(test_rows):
        row_index = fit_row_count + eval_position
        target = [float(score) for score in target_scores[row_index]]
        shuffled_index = _same_shape_shuffle_index(row_index=row_index, eval_indices=eval_indices, rows=rows)
        candidate_packets = {
            MATCHED_CONDITION: row_packets[row_index],
            "target_derived_packet": target_derived_row_packets[row_index],
            "zero_packet": np.zeros_like(row_packets[row_index]),
            "row_shuffle": row_packets[shuffled_index],
            "atom_shuffle": hidden_gate._atom_shuffle_packet(row_packets[row_index]),
            "coefficient_shuffle": hidden_gate._coefficient_shuffle_packet(row_packets[row_index]),
            "top_atom_knockout": hidden_gate._top_atom_knockout_packet(row_packets[row_index]),
            "candidate_roll": hidden_gate._candidate_roll_packet(row_packets[row_index], shift=1),
            "candidate_derangement": hidden_gate._candidate_roll_packet(row_packets[row_index], shift=-1),
        }
        condition_scores: dict[str, tuple[list[float], bool, np.ndarray]] = {
            "target_only": (target, False, np.zeros(len(row.choices), dtype=np.float64))
        }
        for condition, packet in candidate_packets.items():
            if condition == MATCHED_CONDITION:
                residual = matched_residuals[row_index]
            elif condition == "target_derived_packet":
                residual = target_derived_residuals[row_index]
            elif condition == "zero_packet":
                residual = zero_residuals[row_index]
            else:
                start, end = offsets[row_index]
                residual = atom_gate._decode_packet_residual_rows(
                    [row],
                    target_features=target_features[start:end],
                    packet_features=np.asarray(packet, dtype=np.float64),
                    decoder=decoder,
                    decoder_mode="target_conditioned",
                )[0]
            fused, fired = hidden_gate._fused_scores(target, residual, rule=gate_rule)
            condition_scores[condition] = (fused, fired, np.asarray(residual, dtype=np.float64))
        target_correct = behavior_gate._prediction(target) == int(row.answer_index)
        for condition in REPORT_CONDITIONS:
            scores, fired, residual = condition_scores[condition]
            pred = behavior_gate._prediction(scores)
            correct = pred == int(row.answer_index)
            prediction_rows.append(
                {
                    "row_id": row.row_id,
                    "content_id": row.content_id,
                    "condition": condition,
                    "answer_index": int(row.answer_index),
                    "answer_label": row.answer_label,
                    "prediction_index": int(pred),
                    "prediction_label": row.choice_labels[pred],
                    "correct": bool(correct),
                    "scores": [float(score) for score in scores],
                    "margin": float(behavior_gate._margin(scores, row.answer_index)),
                    "entropy": float(behavior_gate._entropy(scores)),
                    "packet_fired": bool(fired),
                    "packet_helped": bool(fired and correct and not target_correct),
                    "packet_harmed": bool(fired and (not correct) and target_correct),
                    "packet_residual": [float(value) for value in residual],
                    "gate_rule": dict(gate_rule),
                    "control_origin": condition,
                }
            )

    metrics = _condition_metrics(prediction_rows, seed=seed, bootstrap_samples=bootstrap_samples)
    fixed_weight_sweep = _fixed_weight_sweep(prediction_rows)
    matched = metrics[MATCHED_CONDITION]
    strict_control_metrics: dict[str, dict[str, float]] = {}
    for control in CONTROL_CONDITIONS:
        paired = matched[f"paired_accuracy_vs_{control}"]
        strict_control_metrics[control] = {
            "control_accuracy": float(metrics[control]["accuracy"]),
            "delta_accuracy": float(matched["accuracy"] - metrics[control]["accuracy"]),
            "ci95_low": float(paired["ci95_low"]),
            "ci95_high": float(paired["ci95_high"]),
        }
    best_control = max(CONTROL_CONDITIONS, key=lambda name: metrics[name]["accuracy"])
    worst_ci_low = min(row["ci95_low"] for row in strict_control_metrics.values())
    strict_pass = all(
        row["delta_accuracy"] >= float(min_accuracy_gap) and row["ci95_low"] > float(min_ci_low)
        for row in strict_control_metrics.values()
    )
    packet_bytes_per_row = float(sparse_meta["packet_bytes_per_candidate"] * max(len(row.choices) for row in rows))
    framed_packet_bytes = int(math.ceil(packet_bytes_per_row))
    scope_metadata = _target_oracle_scope_metadata()
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    payload = {
        "gate": "arc_challenge_target_behavior_transcoder_probe",
        "date": dt.date.today().isoformat(),
        "created_utc": created,
        "pass_gate": bool(strict_pass),
        "implementation_gate_only": True,
        "train_rows": int(len(train_rows)),
        "test_rows": int(len(test_rows)),
        "strict_required_controls": list(CONTROL_CONDITIONS),
        "strict_control_metrics": strict_control_metrics,
        "fixed_weight_sweep": fixed_weight_sweep,
        "strict_headline": {
            "matched_accuracy": float(matched["accuracy"]),
            "target_only_accuracy": float(metrics["target_only"]["accuracy"]),
            "best_required_control": best_control,
            "best_required_control_accuracy": float(metrics[best_control]["accuracy"]),
            "worst_required_ci95_low": float(worst_ci_low),
            "matched_packet_fired": int(matched["packet_fired"]),
            "matched_packet_fired_rate": float(matched["packet_fired_rate"]),
            "matched_packet_helped": int(matched["packet_helped_vs_target"]),
            "matched_packet_harmed": int(matched["packet_harmed_vs_target"]),
            "matched_packet_net_help": int(matched["packet_net_help_vs_target"]),
        },
        "condition_metrics": metrics,
        "systems_packet_sideband": {
            **scope_metadata,
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "native_serving_throughput_measured": False,
            "packet_bytes_per_row": float(packet_bytes_per_row),
            "framed_packet_bytes_per_row": int(framed_packet_bytes),
            "cache_line_bytes_per_row_64b": int(math.ceil(max(framed_packet_bytes, 1) / 64.0) * 64),
            "dma_bytes_per_row_128b": int(math.ceil(max(framed_packet_bytes, 1) / 128.0) * 128),
            "decode_flops_proxy_per_row": int(max(len(row.choices) for row in rows) * int(sparse_meta["packet_rank"])),
            "sparse_packet_metadata": sparse_meta,
            "note": (
                "This is a target-native feasibility diagnostic, not a source-private communication result. "
                "Byte counts describe the hypothetical sparse target atom code only."
            ),
        },
        "feature_metadata": {
            "target_score_metadata": target_score_meta,
            "target_hidden": target_hidden_meta,
            "public": public_meta,
            "target_public_innovation": target_innovation_meta,
            "receiver_training": receiver_meta,
            "target_derived_packet_map": {
                "ridge": target_packet_map.ridge,
                "fit_mse": target_packet_map.fit_mse,
                "fit_r2": target_packet_map.fit_r2,
            },
            "selected_gate_rule": dict(gate_rule),
        },
        "inputs": {
            "validation_path": behavior_gate._display(validation_path),
            "test_path": behavior_gate._display(test_path),
            "source_family_gate_dir": behavior_gate._display(source_family_gate_dir),
            "agreement_path": behavior_gate._display(agreement_path),
            "slice_selection": "source_family_disagreement_rows",
            "target_model": str(target_model),
            "target_hidden_layer": int(target_hidden_layer),
            "target_max_length": int(target_max_length),
            "train_disagreement_limit": int(train_disagreement_limit),
            "test_disagreement_limit": int(test_disagreement_limit),
            "packet_rank": int(packet_rank),
            "packet_top_k": int(packet_top_k),
            "packet_bits": int(packet_bits),
            "batchtopk_epochs": int(batchtopk_epochs),
            "batchtopk_learning_rate": float(batchtopk_learning_rate),
            "batchtopk_batch_size": int(batchtopk_batch_size),
            "batchtopk_reconstruction_weight": float(batchtopk_reconstruction_weight),
            "batchtopk_l1_weight": float(batchtopk_l1_weight),
            "corruption_loss_weight": float(corruption_loss_weight),
        },
        "interpretation": (
            "This diagnostic asks whether sparse target-native behavior atoms can steer the target model's ARC "
            "candidate margins under destructive controls. A pass would not prove LatentWire source communication, "
            "but would justify trying to make a source transmit target-readable atoms. A fail means the current "
            "sparse atom/receiver path is weak even before cross-model transmission."
        ),
    }
    json_path = output_dir / "arc_challenge_target_behavior_transcoder_probe.json"
    md_path = output_dir / "arc_challenge_target_behavior_transcoder_probe.md"
    audit_path = output_dir / "prediction_audit.jsonl"
    manifest_path = output_dir / "manifest.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload)
    behavior_gate._write_jsonl(audit_path, prediction_rows)
    manifest = {
        "gate": payload["gate"],
        "pass_gate": payload["pass_gate"],
        "created_utc": payload["created_utc"],
        "files": [
            {"path": behavior_gate._display(json_path), "sha256": behavior_gate._sha256_file(json_path), "bytes": json_path.stat().st_size},
            {"path": behavior_gate._display(md_path), "sha256": behavior_gate._sha256_file(md_path), "bytes": md_path.stat().st_size},
            {"path": behavior_gate._display(audit_path), "sha256": behavior_gate._sha256_file(audit_path), "bytes": audit_path.stat().st_size},
        ],
        "inputs": payload["inputs"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    gc.collect()
    print(json.dumps({"headline": payload["strict_headline"], "pass_gate": payload["pass_gate"]}, sort_keys=True))
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-path", type=pathlib.Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--test-path", type=pathlib.Path, default=DEFAULT_TEST)
    parser.add_argument("--source-family-gate-dir", type=pathlib.Path, default=DEFAULT_SOURCE_FAMILY_GATE_DIR)
    parser.add_argument("--target-model", default=str(DEFAULT_TARGET_MODEL))
    parser.add_argument("--target-device", default="mps")
    parser.add_argument("--target-attn-implementation", default="eager")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--target-max-length", type=int, default=192)
    parser.add_argument("--target-hidden-layer", type=int, default=-1)
    parser.add_argument("--target-feature-dim", type=int, default=128)
    parser.add_argument("--train-disagreement-limit", type=int, default=8)
    parser.add_argument("--test-disagreement-limit", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--packet-rank", type=int, default=16)
    parser.add_argument("--packet-top-k", type=int, default=2)
    parser.add_argument("--packet-bits", type=int, default=4)
    parser.add_argument("--batchtopk-epochs", type=int, default=250)
    parser.add_argument("--batchtopk-learning-rate", type=float, default=0.01)
    parser.add_argument("--batchtopk-batch-size", type=int, default=8)
    parser.add_argument("--batchtopk-reconstruction-weight", type=float, default=0.05)
    parser.add_argument("--batchtopk-l1-weight", type=float, default=0.001)
    parser.add_argument("--corruption-loss-weight", type=float, default=0.1)
    parser.add_argument("--local-files-only", choices=("true", "false"), default="true")
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--min-accuracy-gap", type=float, default=0.0)
    parser.add_argument("--min-ci-low", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=37)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build_probe(
        output_dir=args.output_dir,
        validation_path=args.validation_path,
        test_path=args.test_path,
        source_family_gate_dir=args.source_family_gate_dir,
        target_model=str(args.target_model),
        target_device=str(args.target_device),
        target_attn_implementation=str(args.target_attn_implementation),
        dtype=str(args.dtype),
        target_max_length=int(args.target_max_length),
        target_hidden_layer=int(args.target_hidden_layer),
        target_feature_dim=int(args.target_feature_dim),
        train_disagreement_limit=int(args.train_disagreement_limit),
        test_disagreement_limit=int(args.test_disagreement_limit),
        ridge=float(args.ridge),
        packet_rank=int(args.packet_rank),
        packet_top_k=int(args.packet_top_k),
        packet_bits=int(args.packet_bits),
        batchtopk_epochs=int(args.batchtopk_epochs),
        batchtopk_learning_rate=float(args.batchtopk_learning_rate),
        batchtopk_batch_size=int(args.batchtopk_batch_size),
        batchtopk_reconstruction_weight=float(args.batchtopk_reconstruction_weight),
        batchtopk_l1_weight=float(args.batchtopk_l1_weight),
        corruption_loss_weight=float(args.corruption_loss_weight),
        local_files_only=str(args.local_files_only).lower() == "true",
        bootstrap_samples=int(args.bootstrap_samples),
        min_accuracy_gap=float(args.min_accuracy_gap),
        min_ci_low=float(args.min_ci_low),
        seed=int(args.seed),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
