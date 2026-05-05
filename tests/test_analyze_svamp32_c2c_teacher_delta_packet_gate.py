from __future__ import annotations

import json
import pathlib
import random

import torch

from scripts import analyze_svamp32_c2c_teacher_delta_packet_gate as gate


class FakeTokenizer:
    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(str(int(token_id)) for token_id in token_ids)


def _row(example_id: str, answer: str, target_logits: torch.Tensor, teacher_logits: torch.Tensor) -> gate.RowCapture:
    packet = gate.build_step_packet(
        target_logits,
        teacher_logits,
        top_k=2,
        coeff_bits=4,
        mode="positive",
    )
    return gate.RowCapture(
        index=int(answer),
        example_id=example_id,
        answers=[answer],
        c2c_tokens=[int(answer)],
        c2c_text=answer,
        target_logits=target_logits.view(1, -1),
        packets=[packet],
    )


def test_step_packet_recovers_teacher_token_and_destructive_controls_break_it() -> None:
    target_logits = torch.tensor([0.0, 0.0, 0.0, 0.0])
    teacher_logits = torch.tensor([0.0, 3.0, 1.0, 0.0])
    packet = gate.build_step_packet(
        target_logits,
        teacher_logits,
        top_k=2,
        coeff_bits=4,
        mode="positive",
    )

    assert packet.token_ids[0] == 1
    assert gate.apply_packet(target_logits, packet) == 1
    assert gate.apply_packet(target_logits, None) == 0
    assert gate.apply_packet(
        target_logits,
        gate.transform_packet(packet, condition="atom_shuffle", rng=random.Random(1)),
    ) == 2
    assert gate.apply_packet(
        target_logits,
        gate.transform_packet(packet, condition="coeff_shuffle", rng=random.Random(1)),
    ) == 2


def test_evaluate_conditions_identifies_source_necessary_clean_rows() -> None:
    rows = [
        _row("row-a", "1", torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.0, 4.0, 1.0])),
        _row("row-b", "2", torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 4.0])),
    ]

    payload = gate.evaluate_conditions(
        rows,
        tokenizer=FakeTokenizer(),
        target_ids={"teacher_only": {"row-a", "row-b"}, "clean_residual_targets": {"row-a", "row-b"}},
        conditions=("matched", "target_only", "row_shuffle", "atom_shuffle", "coeff_shuffle"),
        rng_seed=7,
    )

    assert payload["condition_summaries"]["matched"]["correct_count"] == 2
    assert payload["condition_summaries"]["target_only"]["correct_count"] == 0
    assert payload["condition_summaries"]["row_shuffle"]["correct_count"] == 0
    assert payload["condition_summaries"]["atom_shuffle"]["correct_count"] == 0
    assert payload["condition_summaries"]["coeff_shuffle"]["correct_count"] == 0
    assert payload["source_necessary_clean_ids"] == ["row-a", "row-b"]


def test_analyze_writes_manifest_and_capacity_pass(tmp_path: pathlib.Path) -> None:
    rows = [
        _row("row-a", "1", torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.0, 4.0, 1.0])),
        _row("row-b", "2", torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 4.0])),
    ]
    target_set = tmp_path / "target_set.json"
    target_set.write_text(
        json.dumps({"ids": {"teacher_only": ["row-a", "row-b"], "clean_residual_targets": ["row-a", "row-b"]}}),
        encoding="utf-8",
    )

    output_json = tmp_path / "gate.json"
    output_md = tmp_path / "gate.md"
    payload = gate.analyze(
        rows=rows,
        tokenizer=FakeTokenizer(),
        target_set_path=target_set,
        run_config={
            "source_model": "source",
            "target_model": "target",
            "top_k": 2,
            "coeff_bits": 4,
            "packet_mode": "positive",
        },
        run_date="2026-05-05",
        output_json=output_json,
        output_md=output_md,
    )

    assert payload["status"] == "teacher_delta_packet_capacity_clears_controls_not_deployable"
    assert payload["packet_contract"]["source_private"] is False
    assert output_json.exists()
    assert output_md.exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == payload["status"]
