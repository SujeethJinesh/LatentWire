from __future__ import annotations

import torch

from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate
from scripts import run_source_private_arc_openbookqa_soft_prefix_preflight as preflight


def _rows() -> list[arc_gate.ArcRow]:
    return [
        arc_gate.ArcRow(
            row_id="r0",
            content_id="c0",
            question="Which object is magnetic?",
            choices=("wood", "iron", "paper"),
            choice_labels=("A", "B", "C"),
            answer_index=1,
            answer_label="B",
        ),
        arc_gate.ArcRow(
            row_id="r1",
            content_id="c1",
            question="What do plants need?",
            choices=("sunlight", "sand"),
            choice_labels=("A", "B"),
            answer_index=0,
            answer_label="A",
        ),
    ]


def test_prompt_and_source_choice_texts_are_answer_key_forbidden() -> None:
    row = _rows()[0]

    prompt = preflight._mcq_prompt(row)
    source_texts = preflight._source_choice_texts([row])

    assert "Choices:" in prompt
    assert "B. iron" in prompt
    assert "Best answer:" in prompt
    assert len(source_texts) == 3
    assert "Candidate under consideration: B. iron" in source_texts[1]
    assert row.answer_label not in preflight._source_prompt(row).split("Useful evidence:")[-1]


def test_select_rows_with_cache_requires_valid_source_predictions() -> None:
    rows = _rows()
    selected, predictions = preflight._select_rows_with_cache(
        rows,
        {"c0": 1, "c1": 0, "missing": 0},
        row_limit=2,
    )

    assert [row.content_id for row in selected] == ["c0", "c1"]
    assert predictions == [1, 0]


def test_soft_prefix_connector_shapes() -> None:
    source = torch.randn(5)
    target = torch.randn(3)
    matched = preflight.SourceSoftPrefixConnector(
        source_dim=5,
        target_dim=3,
        target_embed_dim=7,
        hidden_dim=11,
        prefix_len=4,
        use_source=True,
        use_target=True,
    )
    slots = preflight.SourceSoftPrefixConnector(
        source_dim=5,
        target_dim=3,
        target_embed_dim=7,
        hidden_dim=11,
        prefix_len=2,
        use_source=False,
        use_target=False,
    )

    assert matched(source, target).shape == (4, 7)
    assert slots(source, target).shape == (2, 7)


def test_condition_metrics_adds_paired_controls() -> None:
    rows = [
        {
            "content_id": "c0",
            "condition": preflight.MATCHED_CONDITION,
            "correct": True,
            "margin": 0.8,
        },
        {"content_id": "c0", "condition": "target_only", "correct": False, "margin": -0.1},
        {
            "content_id": "c1",
            "condition": preflight.MATCHED_CONDITION,
            "correct": False,
            "margin": -0.2,
        },
        {"content_id": "c1", "condition": "target_only", "correct": False, "margin": -0.4},
    ]
    for condition in preflight.REPORT_CONDITIONS:
        if condition in {preflight.MATCHED_CONDITION, "target_only"}:
            continue
        rows.extend(
            [
                {"content_id": "c0", "condition": condition, "correct": False, "margin": 0.0},
                {"content_id": "c1", "condition": condition, "correct": False, "margin": 0.0},
            ]
        )

    metrics = preflight._condition_metrics(rows, seed=3, bootstrap_samples=20)

    assert metrics[preflight.MATCHED_CONDITION]["accuracy"] == 0.5
    assert metrics["target_only"]["accuracy"] == 0.0
    paired = metrics[preflight.MATCHED_CONDITION]["paired_accuracy_vs_target_only"]
    assert paired["mean"] == 0.5
    assert metrics[preflight.MATCHED_CONDITION]["mean_margin_delta_vs_target_only"] > 0.0
