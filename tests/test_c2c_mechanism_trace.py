from __future__ import annotations

import json
import pathlib
import sys
import types

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


class _ForwardProjector(_Projector):
    def __init__(self) -> None:
        super().__init__()
        self.offset = 0.0

    def forward(self, source_kv, target_kv):
        source_key, source_value = source_kv
        target_key, target_value = target_kv
        self.offset += 1.0
        self.last_norm_key_scalar = torch.tensor([[[[self.offset]]]])
        self.last_norm_value_scalar = torch.tensor([[[[self.offset + 1.0]]]])
        self.last_key_gate_logit = self.offset
        self.last_value_gate_logit = -self.offset
        return target_key + source_key + self.offset, target_value + source_value + self.offset


class _Model:
    def __init__(self) -> None:
        self.projector_list = [_Projector()]


def test_dynamic_cache_compatibility_shim_clones_new_cache_api(monkeypatch) -> None:
    class _StubDynamicCache:
        def __init__(self, ddp_cache_data=None):
            self.layers = []
            for key, value in ddp_cache_data or []:
                self.layers.append(types.SimpleNamespace(keys=key, values=value))

    key = torch.arange(12, dtype=torch.float32).view(1, 1, 3, 4)
    value = key + 100
    cache = _StubDynamicCache([(key, value)])
    wrapper_module = types.SimpleNamespace()

    monkeypatch.setitem(
        sys.modules,
        "transformers.cache_utils",
        types.SimpleNamespace(DynamicCache=_StubDynamicCache),
    )

    c2c_eval.install_c2c_dynamic_cache_compatibility_shim(wrapper_module)

    assert torch.equal(cache.key_cache[0], key)
    assert torch.equal(cache.value_cache[0], value)
    assert torch.equal(cache[0][0], key)

    cache.key_cache[0][:, :, 0, :] = -1
    assert torch.equal(cache[0][0][:, :, 0, :], torch.full((1, 1, 4), -1.0))

    clone = wrapper_module.clone_kv_cache(cache)
    assert isinstance(clone, _StubDynamicCache)
    assert torch.equal(clone.key_cache[0], cache.key_cache[0])

    cache.key_cache[0].fill_(7)
    assert not torch.equal(clone.key_cache[0], cache.key_cache[0])
    assert wrapper_module.hybrid_to_dynamic(clone) is clone


def test_force_c2c_mps_eager_attention_only_for_mps() -> None:
    cpu_model = types.SimpleNamespace(
        model_list=[types.SimpleNamespace(config=types.SimpleNamespace(_attn_implementation="sdpa"))]
    )
    mps_model = types.SimpleNamespace(
        model_list=[types.SimpleNamespace(config=types.SimpleNamespace(_attn_implementation="sdpa"))]
    )

    c2c_eval.force_c2c_mps_eager_attention(cpu_model, device="cpu")
    c2c_eval.force_c2c_mps_eager_attention(mps_model, device="mps")

    assert cpu_model.model_list[0].config._attn_implementation == "sdpa"
    assert mps_model.model_list[0].config._attn_implementation == "eager"


def test_install_c2c_decode_attention_mask_source_patch_is_idempotent(tmp_path: pathlib.Path) -> None:
    wrapper_path = tmp_path / "rosetta" / "model" / "wrapper.py"
    wrapper_path.parent.mkdir(parents=True)
    wrapper_path.write_text(
        "\n".join(
            [
                "            prefill_attention_mask = base_attention_mask[:, :end] if base_attention_mask is not None else None",
                "            prefill_position_ids = position_ids[:, start:end] if position_ids is not None else None",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    c2c_eval.install_c2c_decode_attention_mask_source_patch(tmp_path)
    once = wrapper_path.read_text(encoding="utf-8")
    c2c_eval.install_c2c_decode_attention_mask_source_patch(tmp_path)

    assert wrapper_path.read_text(encoding="utf-8") == once
    assert "past_key_values is not None" in once
    assert "prefill_attention_mask = base_attention_mask\n" in once


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


def test_c2c_trace_residual_projection_schema_is_deterministic() -> None:
    source = torch.zeros(1, 1, 3, 2)
    target = torch.ones(1, 1, 3, 2)
    output = torch.arange(6, dtype=torch.float32).view(1, 1, 3, 2)

    first = c2c_eval._trace_tensor_stats(
        source=source,
        target=target,
        output=output,
        projection_dim=4,
        projection_salt=7,
    )
    second = c2c_eval._trace_tensor_stats(
        source=source,
        target=target,
        output=output,
        projection_dim=4,
        projection_salt=7,
    )
    features, _ = c2c_eval.summarize_c2c_projector_trace(
        _Model(),
        residual_projection_dim=4,
    )
    names = c2c_eval.c2c_trace_feature_names(1, residual_projection_dim=4)

    assert first["delta_projection"] == second["delta_projection"]
    assert len(first["delta_projection"]) == 4
    assert len(first["tail_delta_projection"]) == 4
    assert features.shape == (48,)
    assert len(names) == 48
    assert names[-1] == "projector_00.value_residual.tail_delta_projection_003"


def test_c2c_local_tail_tokens_are_query_probe_compatible() -> None:
    source = torch.zeros(1, 2, 3, 4)
    target = torch.ones(1, 2, 3, 4)
    output = torch.arange(24, dtype=torch.float32).view(1, 2, 3, 4)
    projector = _Projector()
    projector.last_latentwire_local_tokens = {
        "key": c2c_eval._tail_local_tokens(source=source, target=target, output=output),
        "value": c2c_eval._tail_local_tokens(source=source + 1, target=target, output=output),
    }
    model = _Model()
    model.projector_list = [projector]

    features, metadata = c2c_eval.summarize_c2c_projector_local_tokens(model)
    tokens = source_probe._feature_summary_tokens(
        features.unsqueeze(0),
        [{"feature_token_shape": metadata["feature_token_shape"]}],
    )

    assert metadata["feature_family"] == "c2c_prefill_token_layer_tail_residual"
    assert metadata["feature_token_shape"] == [8, 8]
    assert features.shape == (64,)
    assert tokens.shape == (1, 8, 8)
    assert metadata["token_names"][0] == "projector_00.key.source.tail"
    assert metadata["token_names"][-1] == "projector_00.value.delta.tail"


def test_generation_trace_history_records_multiple_calls() -> None:
    model = _Model()
    projector = _ForwardProjector()
    model.projector_list = [projector]
    c2c_eval.install_c2c_projector_trace_hooks(model, residual_projection_dim=2)
    c2c_eval.reset_c2c_projector_trace_history(model, enabled=True)
    source = (torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2) * 2)
    target = (torch.ones(1, 1, 2, 2) * 3, torch.ones(1, 1, 2, 2) * 4)

    projector.forward(source, target)
    projector.forward(source, target)
    c2c_eval.stop_c2c_projector_trace_history(model)
    features, metadata = c2c_eval.summarize_c2c_projector_generation_history(model)

    assert metadata["feature_family"] == "c2c_generation_projector_trace_history"
    assert metadata["projectors"][0]["history_length"] == 2
    assert features.numel() == len(metadata["feature_names"])
    assert any(name.endswith("key_gate_logit.last") for name in metadata["feature_names"])
    assert any(name.endswith("key_residual.delta_projection_000.mean") for name in metadata["feature_names"])


def test_generation_score_history_summarizes_decode_logits() -> None:
    scores = [
        torch.tensor([[0.0, 2.0, 1.0]], dtype=torch.float32),
        torch.tensor([[3.0, 1.0, 0.0]], dtype=torch.float32),
    ]
    generated = torch.tensor([1, 0], dtype=torch.long)

    features, metadata = c2c_eval.summarize_c2c_generation_score_history(scores, generated)

    assert metadata["feature_family"] == "c2c_generation_target_logit_history"
    assert metadata["step_count"] == 2
    assert features.numel() == len(metadata["feature_names"])
    assert "generation_logits.top_margin.mean" in metadata["feature_names"]
    assert "generation_logits.generated_rank_frac.max" in metadata["feature_names"]


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
