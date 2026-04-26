from __future__ import annotations

import json
import pathlib

import torch

from latent_bridge import c2c_eval
from scripts import analyze_svamp32_c2c_mechanism_syndrome_probe as probe
from scripts import analyze_svamp32_source_latent_syndrome_probe as source_probe
from scripts import analyze_svamp32_syndrome_sidecar_probe as syndrome


class _Projector:
    def __init__(self) -> None:
        self.last_norm_key_scalar = torch.tensor([[[[0.25], [0.75]]]])
        self.last_norm_value_scalar = torch.tensor([[[[0.5], [1.0]]]])
        self.last_key_gate_logit = 0.2
        self.last_value_gate_logit = -0.1


class _Model:
    def __init__(self) -> None:
        self.projector_list = [_Projector()]


def _row(example_id: str, answer: str, pred: str, method: str, correct: bool) -> dict:
    return {
        "example_id": example_id,
        "method": method,
        "answer": answer,
        "prediction": f"candidate mentions {answer}; final answer: {pred}",
        "normalized_prediction": pred,
        "correct": correct,
    }


def test_summarize_c2c_projector_trace_schema() -> None:
    features, metadata = c2c_eval.summarize_c2c_projector_trace(_Model())

    assert features.shape == (32,)
    names = c2c_eval.c2c_trace_feature_names(1)
    assert names[:12] == [
        "projector_00.key_scalar.mean",
        "projector_00.key_scalar.std",
        "projector_00.key_scalar.min",
        "projector_00.key_scalar.max",
        "projector_00.value_scalar.mean",
        "projector_00.value_scalar.std",
        "projector_00.value_scalar.min",
        "projector_00.value_scalar.max",
        "projector_00.key_gate_logit",
        "projector_00.value_gate_logit",
        "projector_00.key_gate_active",
        "projector_00.value_gate_active",
    ]
    assert names[-1] == "projector_00.value_residual.tail_delta_to_target_ratio"
    assert metadata[0]["has_trace"] is True
    assert metadata[0]["key_gate_active"] is True
    assert metadata[0]["value_gate_active"] is False


def test_c2c_mechanism_probe_relabels_status(tmp_path: pathlib.Path) -> None:
    target_path = tmp_path / "target.jsonl"
    teacher_path = tmp_path / "teacher.jsonl"
    target_set_path = tmp_path / "target_set.json"
    target_rows = [
        _row("a", "1", "0", "target_alone", False),
        _row("b", "2", "0", "target_alone", False),
        _row("c", "3", "0", "target_alone", False),
    ]
    teacher_rows = [
        _row("a", "1", "1", "c2c_generate", True),
        _row("b", "2", "2", "c2c_generate", True),
        _row("c", "3", "3", "c2c_generate", True),
    ]
    target_path.write_text(
        "".join(json.dumps(row) + "\n" for row in target_rows),
        encoding="utf-8",
    )
    teacher_path.write_text(
        "".join(json.dumps(row) + "\n" for row in teacher_rows),
        encoding="utf-8",
    )
    target_set_path.write_text(
        json.dumps(
            {
                "ids": {
                    "teacher_only": ["a", "b", "c"],
                    "clean_residual_targets": ["a"],
                    "target_self_repair": [],
                }
            }
        ),
        encoding="utf-8",
    )

    payload = probe.analyze_with_c2c_features(
        features=torch.eye(3),
        feature_metadata=[
            {"example_id": "a", "feature_family": "test"},
            {"example_id": "b", "feature_family": "test"},
            {"example_id": "c", "feature_family": "test"},
        ],
        c2c_run_config={"source_model": "source", "target_model": "target"},
        target_spec=syndrome.RowSpec("target_alone", target_path, "target_alone"),
        teacher_spec=syndrome.RowSpec("c2c", teacher_path, "c2c_generate"),
        candidate_specs=[],
        target_set_path=target_set_path,
        fallback_label="target_alone",
        config=source_probe.ProbeConfig(
            moduli=(3,),
            min_correct=1,
            min_clean_source_necessary=1,
        ),
        min_numeric_coverage=1,
        run_date="2026-04-26",
    )

    assert payload["status"].startswith("c2c_mechanism_syndrome_probe_")
    assert payload["run"]["status"].startswith("c2c_mechanism_syndrome_probe_")
    assert payload["config"]["feature_family"] == "c2c_prefill_projector_residual_trace"
    assert "final answers" in payload["interpretation"]
