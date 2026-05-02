from __future__ import annotations

import json
import pathlib

import torch

from scripts import analyze_svamp32_source_latent_syndrome_probe as probe
from scripts import analyze_svamp32_syndrome_sidecar_probe as syndrome


def _row(example_id: str, answer: str, pred: str, method: str, correct: bool) -> dict:
    return {
        "example_id": example_id,
        "method": method,
        "answer": answer,
        "prediction": f"candidate mentions {answer}; final answer: {pred}",
        "normalized_prediction": pred,
        "correct": correct,
    }


def test_source_latent_evaluator_requires_matched_beyond_controls() -> None:
    reference_ids = ["clean_a", "clean_b", "self"]
    target_by_id = {
        "clean_a": _row("clean_a", "6", "1", "target_alone", False),
        "clean_b": _row("clean_b", "7", "2", "target_alone", False),
        "self": _row("self", "8", "0", "target_alone", False),
    }
    teacher_by_id = {
        "clean_a": _row("clean_a", "6", "6", "c2c", True),
        "clean_b": _row("clean_b", "7", "7", "c2c", True),
        "self": _row("self", "8", "8", "c2c", True),
    }
    target_self = {
        "clean_a": _row("clean_a", "6", "0", "target_self_repair", False),
        "clean_b": _row("clean_b", "7", "0", "target_self_repair", False),
        "self": _row("self", "8", "8", "target_self_repair", True),
    }
    candidate_by_label = {
        "target_alone": target_by_id,
        "target_self_repair": target_self,
    }

    run = probe._evaluate_source_latent_probe(
        reference_ids=reference_ids,
        target_by_id=target_by_id,
        teacher_by_id=teacher_by_id,
        candidate_by_label=candidate_by_label,
        target_label="target_alone",
        fallback_label="target_self_repair",
        target_ids={
            "teacher_only": set(reference_ids),
            "clean_residual_targets": {"clean_a", "clean_b"},
            "target_self_repair": {"self"},
        },
        residue_predictions={
            "matched": [(6,), (7,), (8,)],
            "zero_source": [(0,), (0,), (0,)],
            "shuffled_source": [(7,), (8,), (6,)],
            "label_shuffled": [(1,), (2,), (3,)],
        },
        config=probe.ProbeConfig(moduli=(11,), min_correct=3, min_clean_source_necessary=2),
    )

    assert run["status"] == "source_latent_syndrome_probe_clears_gate"
    assert run["condition_summaries"]["matched"]["correct_count"] == 3
    assert run["condition_summaries"]["target_only"]["clean_correct_count"] == 0
    assert run["source_necessary_clean_ids"] == ["clean_a", "clean_b"]


def test_loocv_residue_predictions_returns_signatures() -> None:
    features = torch.eye(4)
    labels = {3: torch.tensor([0, 1, 0, 1])}
    predictions = probe._loocv_residue_predictions(
        features,
        [{"example_id": str(idx), "feature_layers": [0]} for idx in range(4)],
        labels,
        config=probe.ProbeConfig(moduli=(3,), ridge_lambda=0.1),
    )

    assert sorted(predictions) == [
        "label_shuffled",
        "matched",
        "shuffled_source",
        "zero_source",
    ]
    assert len(predictions["matched"]) == 4
    assert all(len(signature) == 1 for signature in predictions["matched"])


def test_query_bottleneck_predictions_returns_control_signatures() -> None:
    features = torch.tensor(
        [
            [2.0, 0.0, 2.0, 0.0],
            [0.0, 2.0, 0.0, 2.0],
            [1.8, 0.1, 1.8, 0.1],
            [0.1, 1.8, 0.1, 1.8],
        ]
    )
    labels = {2: torch.tensor([0, 1, 0, 1])}
    predictions = probe._loocv_residue_predictions(
        features,
        [{"example_id": str(idx), "feature_layers": [0, 1]} for idx in range(4)],
        labels,
        config=probe.ProbeConfig(
            moduli=(2,),
            probe_model="query_bottleneck",
            query_slots=2,
            query_epochs=2,
            query_seed=7,
        ),
    )

    assert sorted(predictions) == [
        "label_shuffled",
        "matched",
        "shuffled_source",
        "zero_source",
    ]
    assert len(predictions["matched"]) == 4
    assert all(len(signature) == 1 for signature in predictions["zero_source"])


def test_high_dimensional_ridge_uses_dual_unregularized_intercept() -> None:
    train_x = torch.tensor(
        [
            [1.0, 0.0, 2.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.0],
        ]
    )
    train_y = torch.tensor([0, 1, 2])
    weights = probe._fit_ridge_classifier(
        train_x,
        train_y,
        num_classes=3,
        ridge_lambda=0.5,
    )

    y = torch.nn.functional.one_hot(train_y, num_classes=3).float()
    mean_x = train_x.mean(dim=0, keepdim=True)
    mean_y = y.mean(dim=0, keepdim=True)
    x_centered = train_x - mean_x
    y_centered = y - mean_y
    primal = torch.linalg.solve(
        x_centered.T @ x_centered + 0.5 * torch.eye(train_x.shape[1]),
        x_centered.T @ y_centered,
    )
    bias = mean_y - mean_x @ primal
    expected = torch.cat([primal, bias], dim=0)

    assert train_x.shape[1] > train_x.shape[0]
    assert torch.allclose(weights, expected, atol=1e-5)


def test_analyze_with_features_validates_feature_order(tmp_path: pathlib.Path) -> None:
    target_path = tmp_path / "target.jsonl"
    teacher_path = tmp_path / "teacher.jsonl"
    target_set_path = tmp_path / "target_set.json"
    target_path.write_text(
        json.dumps(_row("a", "1", "0", "target_alone", False)) + "\n",
        encoding="utf-8",
    )
    teacher_path.write_text(
        json.dumps(_row("a", "1", "1", "c2c", True)) + "\n",
        encoding="utf-8",
    )
    target_set_path.write_text(
        json.dumps(
            {
                "ids": {
                    "teacher_only": ["a"],
                    "clean_residual_targets": ["a"],
                    "target_self_repair": [],
                }
            }
        ),
        encoding="utf-8",
    )

    try:
        probe.analyze_with_features(
            features=torch.zeros((1, 2)),
            feature_metadata=[{"example_id": "wrong"}],
            target_spec=syndrome.RowSpec("target_alone", target_path, "target_alone"),
            teacher_spec=syndrome.RowSpec("c2c", teacher_path, "c2c_generate"),
            candidate_specs=[],
            target_set_path=target_set_path,
            fallback_label="target_alone",
            config=probe.ProbeConfig(moduli=(3,)),
            min_numeric_coverage=1,
            run_date="2026-04-24",
        )
    except ValueError as exc:
        assert "feature metadata IDs" in str(exc)
    else:
        raise AssertionError("feature ID mismatch should raise")
