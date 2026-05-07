import json
from pathlib import Path

from experimental.shared.followup_gate_contracts import validate_followup_gate_packet
from experimental.shared.ssq_lr_s3_transfer_prefilter import build_prefilter


def _write_source_s2(tmp_path: Path) -> Path:
    source = tmp_path / "source_s2"
    source.mkdir()
    rows = []
    for prompt_id, accuracy_delta in [("p0", 0.0), ("p1", 0.01)]:
        rows.append(
            {
                "model_id": "ibm-granite/granite-4.0-h-tiny",
                "prompt_id": prompt_id,
                "recipe_id": "int3_primary_state_block_scaled",
                "precision": "int3",
                "scale_granularity": "per_block_absmax",
                "block_size": 256,
                "bf16_state_bytes": 1024.0,
                "quantized_state_bytes": 192.0,
                "scale_bytes": 4.0,
                "metadata_bytes": 0.0,
                "effective_bits": 3.1,
                "bf16_accuracy": 1.0,
                "quantized_accuracy": 1.0 - accuracy_delta,
                "accuracy_delta_abs": accuracy_delta,
                "bf16_nll": 1.0,
                "quantized_nll": 1.01,
                "nll_delta": 0.01,
                "paired_ci_low": 0.0,
                "paired_ci_high": accuracy_delta,
                "bf16_noop_delta": 0.0,
                "control_type": "candidate_recipe",
                "continuation_token_count": 8,
                "bf16_selected_state_bytes": 1024.0,
            }
        )
    (source / "raw_rows.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    (source / "summary.json").write_text(
        json.dumps({"gate_status": "PASS_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY"}) + "\n",
        encoding="utf-8",
    )
    return source


def _write_hf_model_cache(tmp_path: Path, model_id: str, *, complete: bool) -> None:
    repo = tmp_path / "hf" / "hub" / f"models--{model_id.replace('/', '--')}"
    snapshot = repo / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("abc", encoding="utf-8")
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    if complete:
        (snapshot / "model.safetensors").write_bytes(b"weights")


def test_s3_prefilter_writes_validator_clean_fail_packet(tmp_path: Path) -> None:
    source = _write_source_s2(tmp_path)
    prereg = tmp_path / "prereg.md"
    prereg.write_text("S3 prereg\n", encoding="utf-8")
    _write_hf_model_cache(tmp_path, "ibm-granite/granite-4.0-h-tiny", complete=True)
    _write_hf_model_cache(tmp_path, "ibm-granite/granite-4.0-h-small", complete=False)

    output = tmp_path / "packet"
    summary = build_prefilter(
        source_s2_dir=source,
        output_dir=output,
        preregistration=prereg,
        recipe_id="int3_primary_state_block_scaled",
        hf_homes=(tmp_path / "hf",),
        transfer_models=(
            "ibm-granite/granite-4.0-h-tiny",
            "ibm-granite/granite-4.0-h-small",
        ),
    )

    assert summary["gate_pass"] is False
    assert summary["transfer_model_count"] == 1
    assert summary["complete_transfer_model_count"] == 1
    report = validate_followup_gate_packet(output, gate="ssq_lr_s3")
    assert report["ok"], report["errors"]
    rows = [
        json.loads(line)
        for line in (output / "raw_rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert {row["control_type"] for row in rows} == {"transfer_eval", "retune_probe"}
    assert all(row["retuned"] is False for row in rows)
