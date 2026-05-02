from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_sparse_projection_packet_selects_matching_candidate() -> None:
    row = arc_gate.ArcRow(
        row_id="r0",
        content_id="c0",
        question="Which option is the source choice?",
        choices=("alpha", "beta"),
        choice_labels=("A", "B"),
        answer_index=0,
        answer_label="A",
    )
    projection = np.eye(2, dtype=np.float64)
    payload, meta = arc_gate._encode_packet(np.asarray([1.0, 0.0]), projection, budget_bytes=2)
    prediction, decode_meta = arc_gate._predict_from_code(
        row=row,
        residuals=np.asarray([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float64),
        payload=payload,
        projection=projection,
        index_prior=[0.5, 0.5],
    )

    assert prediction == 0
    assert meta["packet_nonzero_dims"] == 1
    assert decode_meta["decoder"] == "sparse_projection_candidate_residual"


def test_label_permutation_changes_labels_without_changing_answer_index() -> None:
    row = arc_gate.ArcRow(
        row_id="r0",
        content_id="c0",
        question="Pick the right option.",
        choices=("one", "two", "three", "four"),
        choice_labels=("A", "B", "C", "D"),
        answer_index=2,
        answer_label="C",
    )

    permuted = arc_gate._permuted_row(row)

    assert permuted.choices == row.choices
    assert permuted.answer_index == row.answer_index
    assert sorted(permuted.choice_labels) == sorted(row.choice_labels)
    assert permuted.choice_labels != row.choice_labels
    assert permuted.answer_label == permuted.choice_labels[row.answer_index]


def test_anchor_relative_hashed_features_use_public_anchor_basis() -> None:
    features = arc_gate._features(
        ["Question: Which answer is hot?\nCandidate answer: bright fire"],
        dim=8,
        feature_mode="anchor_relative_hashed",
        feature_model="BAAI/bge-small-en",
        feature_device="cpu",
        feature_dtype="float32",
        feature_max_length=64,
        local_files_only=True,
        anchor_texts=[
            "Question: Which choice is hot?\nCandidate answer: bright fire",
            "Question: Which choice is cold?\nCandidate answer: frozen ice",
            "Question: Which choice is wet?\nCandidate answer: falling rain",
        ],
    )

    assert features.shape == (1, 8)
    assert np.isfinite(features).all()
    assert np.linalg.norm(features[0]) > 0.0


def test_lm_choice_scoring_normalizes_default_rope_and_disables_cache(monkeypatch) -> None:
    import sys

    import torch

    config = SimpleNamespace(rope_scaling={"rope_type": "default", "factor": 1.0})
    captured = {}

    class FakeTokenizer:
        pad_token_id = None
        eos_token = "<eos>"
        pad_token = None

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def __call__(self, texts, **kwargs):
            if isinstance(texts, str):
                return SimpleNamespace(input_ids=torch.tensor([[1, 2]], dtype=torch.long))
            if kwargs.get("padding") is False:
                return {"input_ids": [[1, 2, 3] for _ in texts]}
            return {
                "input_ids": torch.tensor([[1, 2, 3] for _ in texts], dtype=torch.long),
                "attention_mask": torch.ones((len(texts), 3), dtype=torch.long),
            }

    class FakeConfig:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return config

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            captured["config"] = kwargs["config"]
            captured["torch_dtype"] = kwargs["torch_dtype"]
            captured["attn_implementation"] = kwargs.get("attn_implementation")
            captured["model_calls"] = 0
            return cls()

        def to(self, device):
            captured["device"] = device
            return self

        def eval(self):
            return None

        def __call__(self, **kwargs):
            captured["model_calls"] += 1
            captured["use_cache"] = kwargs.get("use_cache")
            batch, seq = kwargs["input_ids"].shape
            return SimpleNamespace(logits=torch.zeros((batch, seq, 8), dtype=torch.float32))

    transformers = sys.modules["transformers"]
    monkeypatch.setattr(transformers, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(transformers, "AutoConfig", FakeConfig, raising=False)
    monkeypatch.setattr(transformers, "AutoModelForCausalLM", FakeModel)

    row = arc_gate.ArcRow(
        row_id="r0",
        content_id="c0",
        question="Which choice fits?",
        choices=("alpha", "beta"),
        choice_labels=("A", "B"),
        answer_index=0,
        answer_label="A",
    )
    scores, predictions, state = arc_gate._lm_choice_loglikelihood_scores(
        [row],
        model_path="fake-phi",
        device="auto_cpu",
        dtype="float32",
        max_length=16,
        local_files_only=True,
        normalization="mean",
        prompt_mode="qa",
        attn_implementation="eager",
        choice_batch_size=1,
    )

    assert config.rope_scaling is None
    assert captured["config"] is config
    assert captured["use_cache"] is False
    assert captured["device"] == "cpu"
    assert captured["torch_dtype"] is torch.float32
    assert captured["attn_implementation"] == "eager"
    assert captured["model_calls"] == 2
    assert predictions == [0]
    assert len(scores) == 1
    assert state["kind"] == "local_causal_lm_choice_loglikelihood"
    assert state["attn_implementation"] == "eager"
    assert state["choice_batch_size"] == 1


def test_run_gate_writes_arc_control_artifacts(tmp_path) -> None:
    train_rows = [
        {
            "id": "train_hot",
            "question": "Which choice is hottest?",
            "choices": {"text": ["ice cube", "bright fire", "wet rain"], "label": ["A", "B", "C"]},
            "answerKey": "B",
        },
        {
            "id": "train_cold",
            "question": "Which choice is coldest?",
            "choices": {"text": ["warm soup", "frozen ice", "open flame"], "label": ["A", "B", "C"]},
            "answerKey": "B",
        },
        {
            "id": "train_water",
            "question": "Which choice is wet?",
            "choices": {"text": ["dry sand", "falling rain", "hot coal"], "label": ["A", "B", "C"]},
            "answerKey": "B",
        },
    ]
    eval_rows = [
        {
            "id": "eval_hot",
            "question": "Which answer is hottest?",
            "choices": {"text": ["cold snow", "small fire", "blue water"], "label": ["A", "B", "C"]},
            "answerKey": "B",
        },
        {
            "id": "eval_wet",
            "question": "Which answer is wet?",
            "choices": {"text": ["falling rain", "dry rock", "burning fire"], "label": ["A", "B", "C"]},
            "answerKey": "A",
        },
    ]
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)

    payload = arc_gate.run_gate(
        output_dir=tmp_path / "out",
        train_path=train_path,
        eval_path=eval_path,
        train_limit=None,
        eval_limit=None,
        budget_bytes=12,
        feature_dim=64,
        code_dim=32,
        feature_mode="hashed",
        feature_model="BAAI/bge-small-en",
        feature_device="cpu",
        feature_dtype="float32",
        feature_max_length=64,
        local_files_only=True,
        source_score_mode="pair_ridge",
        source_lm_model="unused",
        source_lm_device="cpu",
        source_lm_dtype="float32",
        source_lm_max_length=64,
        source_lm_normalization="mean",
        ridge=0.1,
        seed=7,
        bootstrap_samples=25,
        min_lift_over_target=0.0,
        min_gap_over_control=0.0,
        min_gap_over_text=0.0,
    )

    assert payload["gate"] == "source_private_arc_challenge_fixed_packet_gate"
    assert payload["train_eval_content_overlap_count"] == 0
    assert payload["budget_bytes"] == 12
    assert payload["systems_trace"]["record_bytes_with_header_crc"] == 15
    assert payload["systems_trace"]["single_request_cacheline_bytes"] == 64.0
    assert payload["systems_trace"]["batch64_dma_bytes_per_request"] == 16.0
    assert payload["systems_trace"]["source_private"] is True
    assert payload["systems_trace"]["source_kv_exposed"] is False
    assert arc_gate.MATCHED_CONDITION in payload["condition_metrics"]
    assert "candidate_derangement" in payload["condition_metrics"]
    assert (tmp_path / "out" / "arc_challenge_fixed_packet_gate.json").exists()
    assert (tmp_path / "out" / "predictions.jsonl").exists()
