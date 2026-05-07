import json
from pathlib import Path

from experimental.shared.followup_gate_contracts import validate_followup_gate_packet
from experimental.shared.ssq_lr_s3_local_transfer_prefilter import build_local_transfer_prefilter


def _write_s2_packet(tmp_path: Path, *, name: str, model_id: str, prompt_ids: list[str]) -> Path:
    packet = tmp_path / name
    packet.mkdir()
    rows = []
    for prompt_id in prompt_ids:
        rows.append(
            {
                "model_id": model_id,
                "prompt_id": prompt_id,
                "recipe_id": "mixed_int3_mxfp4_low_error_25pct",
                "precision": "mixed_int3_mxfp4_low_error_25pct",
                "scale_granularity": "per_block_absmax_with_int3_mask",
                "block_size": 256,
                "bf16_state_bytes": 1024.0,
                "quantized_state_bytes": 240.0,
                "scale_bytes": 4.0,
                "metadata_bytes": 1.0,
                "effective_bits": 3.8,
                "bf16_accuracy": 1.0,
                "quantized_accuracy": 1.0,
                "accuracy_delta_abs": 0.0,
                "bf16_nll": 1.0,
                "quantized_nll": 1.004,
                "nll_delta": 0.004,
                "paired_ci_low": 0.0,
                "paired_ci_high": 0.0,
                "bf16_noop_delta": 0.0,
                "control_type": "candidate_recipe",
                "continuation_token_count": 8,
                "bf16_selected_state_bytes": 1024.0,
            }
        )
    (packet / "raw_rows.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    (packet / "summary.json").write_text(
        json.dumps(
            {
                "gate_status": "PASS_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY",
                "selected_accuracy_delta_abs": 0.0,
                "selected_nll_delta_abs": 0.004,
                "selected_nll_delta_ci_high": 0.004,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return packet


def test_local_transfer_prefilter_combines_source_and_transfer_packets(tmp_path: Path) -> None:
    source = _write_s2_packet(
        tmp_path,
        name="source_s2",
        model_id="ibm-granite/granite-4.0-h-tiny",
        prompt_ids=["s0", "s1"],
    )
    transfer = _write_s2_packet(
        tmp_path,
        name="transfer_s2",
        model_id="ibm-granite/granite-4.0-h-350m",
        prompt_ids=["t0", "t1"],
    )
    prereg = tmp_path / "prereg.md"
    prereg.write_text("S3 prereg\n", encoding="utf-8")

    output = tmp_path / "packet"
    summary = build_local_transfer_prefilter(
        source_s2_dir=source,
        transfer_s2_dirs=(transfer,),
        output_dir=output,
        preregistration=prereg,
    )

    assert summary["gate_pass"] is True
    assert summary["transfer_model_count"] == 2
    assert summary["passing_model_count"] == 2
    assert summary["minimum_model_prompt_count"] == 2
    assert summary["resource_limited_decision"].startswith("LOCAL_PREFLIGHT_NOT_PROMOTABLE_")
    report = validate_followup_gate_packet(output, gate="ssq_lr_s3")
    assert report["ok"], report["errors"]
    rows = [
        json.loads(line)
        for line in (output / "raw_rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert {row["model_id"] for row in rows if row["control_type"] == "transfer_eval"} == {
        "ibm-granite/granite-4.0-h-tiny",
        "ibm-granite/granite-4.0-h-350m",
    }
    assert all(row["retuned"] is False for row in rows)
    assert len({row["frozen_recipe_sha256"] for row in rows}) == 1
    assert len({row["source_s2_packet_sha256"] for row in rows}) == 1
