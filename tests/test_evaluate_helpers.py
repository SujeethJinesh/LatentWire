from __future__ import annotations

import json
import math
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn
import pytest

from latent_bridge import RotAlignKVTranslator, TranslatorConfig
import latent_bridge.evaluate as evaluate
import latent_bridge.translator as translator_mod


class FakeTokenizer:
    def __init__(self) -> None:
        self.vocab: dict[str, int] = {
            "<pad>": 0,
            "<eos>": 1,
            "Question:": 2,
            "Answer:": 3,
            "A": 4,
            "B": 5,
            "because": 6,
            "hint": 7,
        }
        self.inverse = {v: k for k, v in self.vocab.items()}
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.eos_token = "<eos>"

    def _encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for token in text.replace("\n", " ").split():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                self.inverse[self.vocab[token]] = token
            ids.append(self.vocab[token])
        return ids or [self.eos_token_id]

    def __call__(self, text, return_tensors="pt", **kwargs):
        if isinstance(text, list):
            encoded = [self._encode(t) for t in text]
            max_len = kwargs.get("max_length")
            if max_len is None:
                max_len = max(len(seq) for seq in encoded)
            padded = []
            for seq in encoded:
                seq = seq[:max_len]
                seq = seq + [self.pad_token_id] * (max_len - len(seq))
                padded.append(seq)
            return SimpleNamespace(input_ids=torch.tensor(padded, dtype=torch.long))
        return SimpleNamespace(input_ids=torch.tensor([self._encode(text)], dtype=torch.long))

    def decode(self, ids, skip_special_tokens=True):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        tokens = [self.inverse.get(int(i), f"tok{i}") for i in ids]
        return " ".join(tokens)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=None):
        assert tokenize is False
        assert add_generation_prompt is True
        content = messages[0]["content"]
        suffix = "" if enable_thinking is None else f" thinking={str(enable_thinking).lower()}"
        return f"<chat{suffix}> {content}"


class FakeCache:
    def __init__(self, layers):
        self._layers = layers

    def to_legacy_cache(self):
        return self._layers

    def __getitem__(self, index):
        return self._layers[index]

    def get_seq_length(self):
        return self._layers[0][0].shape[2]


class FakeCausalLM(nn.Module):
    def __init__(self, *, preferred_token_id: int = 4, n_layers: int = 2, vocab_size: int = 32):
        super().__init__()
        self.preferred_token_id = preferred_token_id
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.dummy = nn.Parameter(torch.zeros(1))
        self.last_call = None
        self.last_generate_call = None
        self.attention_pattern: torch.Tensor | None = None
        self.model = SimpleNamespace(
            layers=[
                SimpleNamespace(self_attn=SimpleNamespace(q_proj=nn.Identity(), num_heads=1))
                for _ in range(n_layers)
            ]
        )

    def _make_pkv(self, batch: int, seq: int, device: torch.device):
        layers = []
        for layer_idx in range(self.n_layers):
            base = float(layer_idx + 1)
            K = torch.full((batch, 1, seq, 1), base, device=device)
            V = torch.full((batch, 1, seq, 1), base + 100.0, device=device)
            layers.append((K, V))
        return FakeCache(layers)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        labels=None,
        output_attentions=False,
        output_hidden_states=False,
        **kwargs,
    ):
        self.last_call = {
            "input_shape": tuple(input_ids.shape),
            "attention_mask_shape": None if attention_mask is None else tuple(attention_mask.shape),
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "output_attentions": output_attentions,
        }
        batch, seq = input_ids.shape
        logits = torch.full((batch, seq, self.vocab_size), -20.0, device=input_ids.device)
        logits[..., self.preferred_token_id] = 20.0
        attentions = None
        hidden_states = None
        if output_attentions:
            total_len = seq + (
                past_key_values.get_seq_length()
                if hasattr(past_key_values, "get_seq_length")
                else (past_key_values[0][0].shape[2] if past_key_values is not None else 0)
            )
            pattern = self.attention_pattern
            if pattern is None:
                pattern = torch.ones(total_len, dtype=torch.float32)
            pattern = pattern.to(device=input_ids.device, dtype=torch.float32)
            pattern = pattern / pattern.sum().clamp_min(1e-8)
            attentions = tuple(
                pattern.view(1, 1, 1, total_len).expand(batch, 1, seq, total_len).clone()
                for _ in range(self.n_layers)
            )
        if output_hidden_states:
            hidden = input_ids.to(dtype=torch.float32).unsqueeze(-1)
            hidden_states = tuple(hidden.clone() for _ in range(self.n_layers + 1))
        return SimpleNamespace(
            logits=logits,
            past_key_values=self._make_pkv(batch, seq, input_ids.device) if use_cache else None,
            attentions=attentions,
            hidden_states=hidden_states,
        )

    def generate(self, input_ids, max_new_tokens, do_sample, pad_token_id, attention_mask=None):
        self.last_generate_call = {
            "input_shape": tuple(input_ids.shape),
            "attention_mask_shape": None if attention_mask is None else tuple(attention_mask.shape),
        }
        suffix = torch.tensor([[6]], dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, suffix], dim=1)


class _TinyQuantizer:
    def __init__(self, bits: int = 4) -> None:
        self.bits = bits

    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def to(self, *args, **kwargs):
        return self


def _make_identity_translator(monkeypatch, *, layers: int = 2) -> RotAlignKVTranslator:
    monkeypatch.setattr(translator_mod, "GaussianQuantizer", _TinyQuantizer)
    monkeypatch.setattr(translator_mod, "make_rotation", lambda d, **_: torch.eye(d))
    cfg = TranslatorConfig(
        src_head_dim=1,
        src_num_heads=1,
        num_src_layers=layers,
        tgt_head_dim=1,
        tgt_num_heads=1,
        num_tgt_layers=layers,
        quant_bits=2,
        layer_pairing=list(range(layers)),
    )
    tr = RotAlignKVTranslator(cfg)
    with torch.no_grad():
        tr.R_s.copy_(torch.eye(1))
        tr.R_t.copy_(torch.eye(1))
        for w in list(tr.W_K) + list(tr.W_V):
            w.copy_(torch.eye(1))
        tr._fitted = True
    return tr


def test_load_mcq_reads_examples(tmp_path) -> None:
    path = tmp_path / "mcq.jsonl"
    path.write_text('{"question": "q1", "choices": ["A", "B"], "answer": 0}\n', encoding="utf-8")

    examples = evaluate.load_mcq(str(path))
    assert len(examples) == 1
    assert examples[0].question == "q1"
    assert examples[0].choices == ["A", "B"]
    assert examples[0].answer == 0


def test_generation_match_keeps_generic_alias_matching() -> None:
    assert evaluate._generation_match("  Blue whale.  ", ["blue whale", "whale"]) is True
    assert evaluate._generation_match("humpback whale", ["blue whale"]) is False


def test_generation_match_extracts_gsm8k_final_number() -> None:
    prediction = "We compute 18 + 24 = 42. Therefore, the answer is 42."
    assert evaluate._generation_match(prediction, ["42"]) is True


def test_generation_match_accepts_gsm8k_hash_style_reference() -> None:
    assert evaluate._generation_match("The final answer is $1,234.00.", ["#### 1234"]) is True


def test_generation_match_avoids_numeric_false_positive_for_textual_alias() -> None:
    assert evaluate._generation_match("51", ["Area 51"]) is False


def test_source_reasoning_prompt_variants() -> None:
    assert evaluate._source_reasoning_prompt("solve this", "plain") == "solve this"
    assert "Let's think step by step." in evaluate._source_reasoning_prompt("solve this", "cot")
    assert "scratchpad" in evaluate._source_reasoning_prompt("solve this", "scratchpad").lower()


def test_format_prompt_for_tokenizer_supports_chat_template_and_enable_thinking() -> None:
    tok = FakeTokenizer()

    formatted = evaluate._format_prompt_for_tokenizer(
        tok,
        "solve this",
        use_chat_template=True,
        enable_thinking=False,
    )

    assert formatted == "<chat thinking=false> solve this"


def test_score_helpers_rank_the_preferred_choice() -> None:
    tok = FakeTokenizer()
    model = FakeCausalLM(preferred_token_id=4)

    score_a = evaluate.score_choice_loglik(model, tok, "Question: x Answer:", "A", "cpu")
    score_b = evaluate.score_choice_loglik(model, tok, "Question: x Answer:", "B", "cpu")

    assert score_a > score_b


def test_prefix_mcq_scoring_matches_space_prefixed_choice_boundary() -> None:
    class SpyTokenizer(FakeTokenizer):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[str] = []

        def __call__(self, text, return_tensors="pt", **kwargs):
            if isinstance(text, str):
                self.calls.append(text)
            return super().__call__(text, return_tensors=return_tensors, **kwargs)

    tok = SpyTokenizer()
    model = FakeCausalLM(preferred_token_id=4)
    prefix = evaluate.PrefixState(None, torch.tensor([[3]], dtype=torch.long), 1)

    evaluate._score_mcq_with_prefix_state(model, tok, prefix, "A", "cpu")

    assert tok.calls[-1] == " A"


def test_score_with_injected_kv_passes_cache_and_attention_mask() -> None:
    tok = FakeTokenizer()
    model = FakeCausalLM()
    injected_pkv = ((torch.zeros(1, 1, 2, 1), torch.zeros(1, 1, 2, 1)),)
    injected_mask = torch.ones(1, 2, dtype=torch.long)

    score = evaluate.score_with_injected_kv(
        model,
        tok,
        injected_pkv,
        injected_mask,
        "A",
        "cpu",
    )

    assert torch.isfinite(torch.tensor(score))
    assert model.last_call["past_key_values"] == injected_pkv
    assert model.last_call["attention_mask_shape"] == (1, 3)
    assert model.last_call["input_shape"] == (1, 1)


def test_cache_for_model_converts_legacy_cache_for_modern_transformers(monkeypatch) -> None:
    legacy_cache = ((torch.zeros(1, 1, 2, 1), torch.ones(1, 1, 2, 1)),)

    class _StubDynamicCache:
        def __init__(self, layers):
            self._layers = tuple(layers)

        @classmethod
        def from_legacy_cache(cls, pkv):
            return cls(pkv)

        def get_seq_length(self):
            return self._layers[0][0].shape[2]

        def to_legacy_cache(self):
            return self._layers

    monkeypatch.setitem(sys.modules, "transformers.cache_utils", SimpleNamespace(DynamicCache=_StubDynamicCache))
    model_cache = evaluate._cache_for_model(SimpleNamespace(config=SimpleNamespace()), legacy_cache)

    assert hasattr(model_cache, "get_seq_length")
    assert model_cache.get_seq_length() == 2
    assert tuple(model_cache.to_legacy_cache()) == legacy_cache


def test_translator_set_layer_gates_only_updates_one_layer(monkeypatch) -> None:
    translator = _make_identity_translator(monkeypatch, layers=2)

    translator.set_layer_gates(1, alpha_k=0.25, alpha_v=0.75)

    assert translator.gate_value(0) == pytest.approx((0.5, 0.5))
    assert translator.gate_value(1)[0] == pytest.approx(0.25)
    assert translator.gate_value(1)[1] == pytest.approx(0.75)


def test_evaluate_parse_args_supports_gate_search(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluate.sys,
        "argv",
        [
            "evaluate.py",
            "--translator",
            "translator.pt",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--eval-file",
            "eval.jsonl",
            "--gate-mode",
            "search",
            "--gate-search-file",
            "dev.jsonl",
            "--gate-search-limit",
            "8",
            "--source-kv-control",
            "shuffle_positions",
            "--quantization-control",
            "matched_noise",
            "--translated-kv-control",
            "zero",
            "--fusion-rule",
            "kalman_tokenwise",
            "--kv-transport",
            "k_only",
            "--position-selection-metric",
            "attention_disagreement",
            "--position-selection-prior-file",
            "data/calibration.txt",
            "--position-selection-prior-bins",
            "64",
            "--runtime-head-selection-ratio",
            "0.5",
            "--runtime-head-selection-metric",
            "attention_blend",
            "--runtime-head-prior-file",
            "data/head_calibration.txt",
            "--runtime-head-prior-load",
            "data/head_prior.pt",
            "--runtime-head-prior-save",
            "artifacts/head_prior.pt",
            "--runtime-head-prior-metric",
            "attention_entropy",
            "--runtime-head-prior-alpha",
            "0.25",
            "--runtime-head-prior-shrinkage",
            "0.3",
            "--runtime-head-prior-shrink-target",
            "global",
            "--per-head-position-budget-mode",
            "attention_peak",
        ],
    )

    args = evaluate.parse_args()
    assert args.gate_mode == "search"
    assert args.gate_search_file == "dev.jsonl"
    assert args.gate_search_limit == 8
    assert args.source_kv_control == "shuffle_positions"
    assert args.quantization_control == "matched_noise"
    assert args.translated_kv_control == "zero"
    assert args.fusion_rule == "kalman_tokenwise"
    assert args.kv_transport == "k_only"
    assert args.position_selection_metric == "attention_disagreement"
    assert args.position_selection_prior_file == "data/calibration.txt"
    assert args.position_selection_prior_source == "calibration_mean_attention"
    assert args.position_selection_prior_bins == 64
    assert args.runtime_head_selection_ratio == 0.5
    assert args.runtime_head_selection_metric == "attention_blend"
    assert args.runtime_head_prior_file == "data/head_calibration.txt"
    assert args.runtime_head_prior_load == "data/head_prior.pt"
    assert args.runtime_head_prior_save == "artifacts/head_prior.pt"
    assert args.runtime_head_prior_metric == "attention_entropy"
    assert args.runtime_head_prior_alpha == 0.25
    assert args.runtime_head_prior_shrinkage == 0.3
    assert args.runtime_head_prior_shrink_target == "global"
    assert args.per_head_position_budget_mode == "attention_peak"


def test_evaluate_parse_args_accepts_headwise_route_atom(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluate.sys,
        "argv",
        [
            "evaluate.py",
            "--translator",
            "translator.pt",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--eval-file",
            "eval.jsonl",
            "--runtime-head-selection-ratio",
            "0.5",
            "--runtime-head-selection-metric",
            "headwise_route_atom",
            "--runtime-head-gate-metric",
            "headwise_route_atom",
        ],
    )

    args = evaluate.parse_args()

    assert args.runtime_head_selection_metric == "headwise_route_atom"
    assert args.runtime_head_gate_metric == "headwise_route_atom"


def test_evaluate_parse_args_accepts_asymmetric_kv_position_ratios(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluate.sys,
        "argv",
        [
            "evaluate.py",
            "--translator",
            "translator.pt",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--eval-file",
            "eval.jsonl",
            "--kv-transport",
            "both",
            "--position-selection-metric",
            "attention",
            "--kv-route-selection-ratio",
            "0.25",
            "--kv-value-selection-ratio",
            "0.75",
            "--kv-route-selection-metric",
            "attention",
            "--kv-value-selection-metric",
            "energy",
        ],
    )

    args = evaluate.parse_args()

    assert args.kv_route_selection_ratio == 0.25
    assert args.kv_value_selection_ratio == 0.75
    assert args.kv_route_selection_metric == "attention"
    assert args.kv_value_selection_metric == "energy"


def test_evaluate_parse_args_accepts_chat_template_and_thinking_flags(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluate.sys,
        "argv",
        [
            "evaluate.py",
            "--translator",
            "translator.pt",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--eval-file",
            "eval.jsonl",
            "--source-use-chat-template",
            "--target-use-chat-template",
            "--source-enable-thinking",
            "false",
            "--target-enable-thinking",
            "false",
        ],
    )

    args = evaluate.parse_args()
    assert args.source_use_chat_template is True
    assert args.target_use_chat_template is True
    assert args.source_enable_thinking == "false"
    assert args.target_enable_thinking == "false"


def test_evaluate_parse_args_accepts_stratified_position_metric(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluate.sys,
        "argv",
        [
            "evaluate.py",
            "--translator",
            "translator.pt",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--eval-file",
            "eval.jsonl",
            "--position-selection-metric",
            "attention_stratified",
        ],
    )

    args = evaluate.parse_args()

    assert args.position_selection_metric == "attention_stratified"


def test_evaluate_parse_args_accepts_query_pool_transport_metric(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluate.sys,
        "argv",
        [
            "evaluate.py",
            "--translator",
            "translator.pt",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--eval-file",
            "eval.jsonl",
            "--position-selection-metric",
            "query_pool_transport",
        ],
    )

    args = evaluate.parse_args()

    assert args.position_selection_metric == "query_pool_transport"


def test_evaluate_parse_args_accepts_attention_qk_bank_transport_metrics(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluate.sys,
        "argv",
        [
            "evaluate.py",
            "--translator",
            "translator.pt",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--eval-file",
            "eval.jsonl",
            "--runtime-head-selection-metric",
            "attention_qk_bank_transport",
            "--runtime-head-gate-metric",
            "attention_qk_bank_transport_shuffled",
            "--per-head-position-budget-mode",
            "attention_qk_bank_transport",
        ],
    )

    args = evaluate.parse_args()
    assert args.runtime_head_selection_metric == "attention_qk_bank_transport"
    assert args.runtime_head_gate_metric == "attention_qk_bank_transport_shuffled"
    assert args.per_head_position_budget_mode == "attention_qk_bank_transport"


def test_evaluate_parse_args_accepts_attention_qk_fidelity_tokenwise_gate(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluate.sys,
        "argv",
        [
            "evaluate.py",
            "--translator",
            "translator.pt",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--eval-file",
            "eval.jsonl",
            "--runtime-head-gate-metric",
            "attention_qk_fidelity_tokenwise",
        ],
    )

    args = evaluate.parse_args()
    assert args.runtime_head_gate_metric == "attention_qk_fidelity_tokenwise"


def test_query_pool_transport_pools_bins_without_changing_cache_shape() -> None:
    K_t = torch.zeros(1, 1, 4, 1)
    V_t = torch.zeros(1, 1, 4, 1)
    K_hat = torch.tensor([[[[1.0], [3.0], [10.0], [14.0]]]])
    V_hat = torch.tensor([[[[101.0], [103.0], [110.0], [114.0]]]])
    scores = torch.tensor([1.0, 3.0, 2.0, 6.0])

    selected_k, selected_v, trace = evaluate._apply_position_selection(
        K_t,
        V_t,
        K_hat,
        V_hat,
        protocol="translated_only",
        kv_transport="both",
        position_selection_ratio=0.5,
        position_selection_metric="query_pool_transport",
        position_scores=scores,
        return_trace=True,
    )

    assert selected_k.shape == K_hat.shape
    assert selected_v.shape == V_hat.shape
    assert selected_k[0, 0, :, 0].tolist() == pytest.approx([0.0, 2.5, 0.0, 13.0])
    assert selected_v[0, 0, :, 0].tolist() == pytest.approx([0.0, 102.5, 0.0, 113.0])
    assert trace["selection_policy"] == "query_pool_transport_bins"
    assert trace["selected_positions_full"] == [1, 3]
    assert trace["query_pool_slots"] == 2
    assert trace["query_pool_mean_bin_span"] == pytest.approx(2.0)


def test_evaluate_parse_args_accepts_layer_knockout(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluate.sys,
        "argv",
        [
            "evaluate.py",
            "--translator",
            "translator.pt",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--eval-file",
            "eval.jsonl",
            "--drop-target-layers",
            "27,5,23",
            "--drop-target-layer-mode",
            "target",
        ],
    )

    args = evaluate.parse_args()
    assert args.drop_target_layers == "27,5,23"
    assert args.drop_target_layer_mode == "target"
    assert evaluate._parse_target_layer_set(args.drop_target_layers) == {5, 23, 27}


def test_source_kv_controls_are_negative_controls() -> None:
    K = torch.arange(12, dtype=torch.float32).view(1, 2, 3, 2)
    V = K + 100.0

    K_zero, V_zero = evaluate._apply_source_kv_control(K, V, "zero", 0)
    assert torch.equal(K_zero, torch.zeros_like(K))
    assert torch.equal(V_zero, torch.zeros_like(V))

    K_flip, V_flip = evaluate._apply_source_kv_control(K, V, "shuffle_positions", 0)
    assert torch.equal(K_flip, K.flip(dims=[2]))
    assert torch.equal(V_flip, V.flip(dims=[2]))

    K_rand, V_rand = evaluate._apply_source_kv_control(K, V, "random", 0)
    assert K_rand.shape == K.shape
    assert V_rand.shape == V.shape
    assert not torch.equal(K_rand, K)


def test_translated_kv_controls_are_target_space_controls() -> None:
    K = torch.arange(12, dtype=torch.float32).view(1, 2, 3, 2)
    V = K + 100.0

    K_zero, V_zero = evaluate._apply_translated_kv_control(K, V, "zero", 0)
    assert torch.equal(K_zero, torch.zeros_like(K))
    assert torch.equal(V_zero, torch.zeros_like(V))

    K_flip, V_flip = evaluate._apply_translated_kv_control(K, V, "shuffle_positions", 0)
    assert torch.equal(K_flip, K.flip(dims=[2]))
    assert torch.equal(V_flip, V.flip(dims=[2]))

    K_rand, V_rand = evaluate._apply_translated_kv_control(K, V, "random", 0)
    assert K_rand.shape == K.shape
    assert V_rand.shape == V.shape
    assert not torch.equal(K_rand, K)


def test_paired_prediction_metrics_report_directional_flips() -> None:
    records = [
        {"index": 0, "method": "target_alone", "correct": False},
        {"index": 0, "method": "rotalign_kv", "correct": True},
        {"index": 1, "method": "target_alone", "correct": True},
        {"index": 1, "method": "rotalign_kv", "correct": False},
        {"index": 2, "method": "target_alone", "correct": True},
        {"index": 2, "method": "rotalign_kv", "correct": True},
    ]

    stats = evaluate.paired_prediction_metrics(records, "rotalign_kv", "target_alone", n_bootstrap=32)

    assert stats["paired_n"] == 3.0
    assert stats["method_only"] == 1.0
    assert stats["baseline_only"] == 1.0
    assert stats["both_correct"] == 1.0


def test_search_per_layer_gates_updates_k_and_v_independently(monkeypatch) -> None:
    tok_s = FakeTokenizer()
    tok_t = FakeTokenizer()
    src = FakeCausalLM(preferred_token_id=4, n_layers=2)
    tgt = FakeCausalLM(preferred_token_id=4, n_layers=2)
    translator = _make_identity_translator(monkeypatch, layers=2)
    examples = [evaluate.MCQExample(question="q", choices=["A", "B"], answer=0)]

    target = {
        0: (0.25, 0.75),
        1: (0.50, 0.00),
    }

    def fake_eval_rotalign_kv(
        source_model,
        source_tokenizer,
        target_model,
        target_tokenizer,
        translator_obj,
        examples_arg,
        device,
        quantize=True,
        protocol="fused",
        source_reasoning_mode="brief_analysis",
        source_kv_control="real",
        quantization_control="real",
        translated_kv_control="real",
        fusion_rule="static",
        kv_transport="both",
        position_selection_ratio=1.0,
        position_selection_metric="energy",
        runtime_head_selection_ratio=1.0,
        runtime_head_selection_metric="attention_peak",
        fixed_head_profiles=None,
        runtime_head_prior_alpha=0.5,
        per_head_position_budget_mode="none",
        records=None,
        method_name="rotalign_kv",
        **kwargs,
    ) -> float:
        score = 0.0
        for layer_idx, (goal_k, goal_v) in target.items():
            cur_k, cur_v = translator_obj.gate_value(layer_idx)
            score -= abs(cur_k - goal_k) + abs(cur_v - goal_v)
        return score

    monkeypatch.setattr(evaluate, "eval_rotalign_kv", fake_eval_rotalign_kv)

    stats = evaluate._search_per_layer_gates(
        src,
        tok_s,
        tgt,
        tok_t,
        translator,
        examples,
        "cpu",
        quantize=False,
        protocol="fused",
        source_reasoning_mode="plain",
        gate_values=[0.0, 0.25, 0.5, 0.75],
    )

    assert stats["gate_K"][0] == pytest.approx(0.25)
    assert stats["gate_K"][1] == pytest.approx(0.5)
    assert stats["gate_V"][0] == pytest.approx(0.75)
    assert stats["gate_V"][1] == pytest.approx(0.0, abs=1e-5)
    gate0 = translator.gate_value(0)
    gate1 = translator.gate_value(1)
    assert gate0[0] == pytest.approx(0.25)
    assert gate0[1] == pytest.approx(0.75)
    assert gate1[0] == pytest.approx(0.5)
    assert gate1[1] == pytest.approx(0.0, abs=1e-5)


def test_search_per_layer_gates_uses_generation_objective_for_generation_examples(monkeypatch) -> None:
    tok_s = FakeTokenizer()
    tok_t = FakeTokenizer()
    src = FakeCausalLM(preferred_token_id=4, n_layers=2)
    tgt = FakeCausalLM(preferred_token_id=4, n_layers=2)
    translator = _make_identity_translator(monkeypatch, layers=2)
    examples = [evaluate.GenerationExample(prompt="q", answers=["a"])]
    calls: list[str] = []

    def fake_eval_generation_rotalign(
        source_model,
        source_tokenizer,
        target_model,
        target_tokenizer,
        translator_obj,
        examples_arg,
        device,
        max_new_tokens,
        quantize=True,
        protocol="fused",
        source_reasoning_mode="brief_analysis",
        source_kv_control="real",
        quantization_control="real",
        translated_kv_control="real",
        fusion_rule="static",
        kv_transport="both",
        position_selection_ratio=1.0,
        position_selection_metric="energy",
        runtime_head_selection_ratio=1.0,
        runtime_head_selection_metric="attention_peak",
        fixed_head_profiles=None,
        runtime_head_prior_alpha=0.5,
        per_head_position_budget_mode="none",
        **kwargs,
    ):
        calls.append(protocol)
        return (0.25, 16.0, 1.0)

    def fail_mcq(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("MCQ evaluator should not be used for generation gate search")

    monkeypatch.setattr(evaluate, "_eval_generation_rotalign", fake_eval_generation_rotalign)
    monkeypatch.setattr(evaluate, "eval_rotalign_kv", fail_mcq)

    stats = evaluate._search_per_layer_gates(
        src,
        tok_s,
        tgt,
        tok_t,
        translator,
        examples,
        "cpu",
        quantize=False,
        protocol="translated_only",
        source_reasoning_mode="plain",
        gate_values=[0.15, 0.25],
    )

    assert calls
    assert set(calls) == {"translated_only"}
    assert len(stats["gate_K"]) == 2
    assert len(stats["gate_V"]) == 2


@pytest.mark.parametrize(
    ("kv_transport", "expected_calls", "preserved_index"),
    [
        ("k_only", 4, 1),
        ("v_only", 4, 0),
    ],
)
def test_search_per_layer_gates_skips_irrelevant_transport_branch(
    monkeypatch,
    kv_transport: str,
    expected_calls: int,
    preserved_index: int,
) -> None:
    tok_s = FakeTokenizer()
    tok_t = FakeTokenizer()
    src = FakeCausalLM(preferred_token_id=4, n_layers=2)
    tgt = FakeCausalLM(preferred_token_id=4, n_layers=2)
    translator = _make_identity_translator(monkeypatch, layers=2)
    examples = [evaluate.MCQExample(question="q", choices=["A", "B"], answer=0)]
    calls = 0
    initial_gates = [translator.gate_value(layer_idx) for layer_idx in range(2)]

    def fake_eval_rotalign_kv(
        source_model,
        source_tokenizer,
        target_model,
        target_tokenizer,
        translator_obj,
        examples_arg,
        device,
        quantize=True,
        protocol="fused",
        source_reasoning_mode="brief_analysis",
        source_kv_control="real",
        quantization_control="real",
        translated_kv_control="real",
        fusion_rule="static",
        kv_transport="both",
        position_selection_ratio=1.0,
        position_selection_metric="energy",
        runtime_head_selection_ratio=1.0,
        runtime_head_selection_metric="attention_peak",
        fixed_head_profiles=None,
        runtime_head_prior_alpha=0.5,
        per_head_position_budget_mode="none",
        records=None,
        method_name="rotalign_kv",
        **kwargs,
    ) -> float:
        nonlocal calls
        calls += 1
        return 0.0

    monkeypatch.setattr(evaluate, "eval_rotalign_kv", fake_eval_rotalign_kv)

    evaluate._search_per_layer_gates(
        src,
        tok_s,
        tgt,
        tok_t,
        translator,
        examples,
        "cpu",
        quantize=False,
        protocol="fused",
        source_reasoning_mode="plain",
        gate_values=[0.0, 0.25],
        kv_transport=kv_transport,
    )

    assert calls == expected_calls
    for layer_idx in range(2):
        assert translator.gate_value(layer_idx)[preserved_index] == pytest.approx(
            initial_gates[layer_idx][preserved_index]
        )


def test_restore_gate_values_restores_each_layer(monkeypatch) -> None:
    translator = _make_identity_translator(monkeypatch, layers=2)
    translator.set_layer_gates(0, alpha_k=0.25, alpha_v=0.75)
    translator.set_layer_gates(1, alpha_k=0.5, alpha_v=0.0)

    evaluate._restore_gate_values(translator, [(0.1, 0.2), (0.3, 0.4)])

    assert translator.gate_value(0) == pytest.approx((0.1, 0.2))
    assert translator.gate_value(1) == pytest.approx((0.3, 0.4))


def test_main_gate_search_uses_protocol_specific_objective(monkeypatch, tmp_path, capsys) -> None:
    tok_s = FakeTokenizer()
    tok_t = FakeTokenizer()
    src = FakeCausalLM(preferred_token_id=4, n_layers=2)
    tgt = FakeCausalLM(preferred_token_id=4, n_layers=2)
    translator = _make_identity_translator(monkeypatch, layers=2)
    search_protocols: list[str] = []

    class _Factory:
        def __init__(self, models):
            self._models = list(models)

        def from_pretrained(self, *args, **kwargs):
            return self._models.pop(0)

    monkeypatch.setattr(evaluate, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: tok_s if "src" in a[0] else tok_t))
    monkeypatch.setattr(evaluate, "AutoModelForCausalLM", _Factory([src, tgt]))
    monkeypatch.setattr(evaluate.RotAlignKVTranslator, "load", classmethod(lambda cls, path, map_location="cpu": translator))
    monkeypatch.setattr(evaluate, "infer_task_type", lambda path: "generation")
    monkeypatch.setattr(evaluate, "load_generation", lambda path: [evaluate.GenerationExample(prompt="q", answers=["a"])])

    def fake_search(
        source_model,
        source_tokenizer,
        target_model,
        target_tokenizer,
        translator_obj,
        examples,
        device,
        quantize,
        protocol,
        source_reasoning_mode,
        gate_values,
        source_kv_control="real",
        quantization_control="real",
        translated_kv_control="real",
        fusion_rule="static",
        kv_transport="both",
        position_selection_ratio=1.0,
        position_selection_metric="energy",
        fixed_position_profiles=None,
        runtime_head_selection_ratio=1.0,
        runtime_head_selection_metric="attention_peak",
        fixed_head_profiles=None,
        runtime_head_prior_alpha=0.5,
        per_head_position_budget_mode="none",
        **kwargs,
    ):
        search_protocols.append(protocol)
        return {"gate_K": [0.15, 0.15], "gate_V": [0.25, 0.25]}

    monkeypatch.setattr(evaluate, "_search_per_layer_gates", fake_search)
    monkeypatch.setattr(
        evaluate,
        "_eval_generation_rotalign_with_stats",
        lambda *args, **kwargs: {
            "accuracy": 0.2,
            "bits": 16.0,
            "bytes": 2.0,
            "ttft_sec": 0.1,
            "tokens_per_sec": 10.0,
            "examples_per_sec": 1.0,
            "latency_sec": 1.0,
        },
    )
    monkeypatch.setattr(
        evaluate,
        "_eval_generation_text_to_text_with_stats",
        lambda *args, **kwargs: {
            "accuracy": 0.1,
            "ttft_sec": 0.1,
            "tokens_per_sec": 8.0,
            "examples_per_sec": 1.0,
            "latency_sec": 1.0,
        },
    )
    monkeypatch.setattr(
        evaluate,
        "_eval_generation_target_alone_with_stats",
        lambda *args, **kwargs: {
            "accuracy": 0.05,
            "ttft_sec": 0.1,
            "tokens_per_sec": 12.0,
            "examples_per_sec": 1.0,
            "latency_sec": 1.0,
        },
    )
    monkeypatch.setattr(
        evaluate,
        "parse_args",
        lambda: SimpleNamespace(
            translator=str(tmp_path / "translator.pt"),
            source_model="src",
            target_model="tgt",
            eval_file=str(tmp_path / "eval.jsonl"),
            task_type="generation",
            device="cpu",
            dtype="float32",
            max_new_tokens=8,
            source_reasoning_mode="brief_analysis",
            no_quantize=False,
            methods=["target", "t2t", "rotalign_translated"],
            gate_mode="search",
            fixed_gate=0.5,
            gate_values=[0.15, 0.25, 0.30],
            gate_search_file=str(tmp_path / "dev.jsonl"),
            gate_search_limit=4,
            source_kv_control="real",
            quantization_control="real",
            translated_kv_control="real",
            prediction_output=None,
        ),
    )

    evaluate.main()

    assert search_protocols == ["translated_only"]
    out = capsys.readouterr().out
    assert "protocol=translated_only" in out


def test_eval_helpers_work_with_fakes(monkeypatch) -> None:
    tok_s = FakeTokenizer()
    tok_t = FakeTokenizer()
    src = FakeCausalLM(preferred_token_id=4, n_layers=2)
    tgt = FakeCausalLM(preferred_token_id=4, n_layers=2)
    translator = _make_identity_translator(monkeypatch, layers=2)
    translate_calls: list[tuple[int, bool]] = []
    hint_calls: list[tuple[str, str]] = []

    def fake_translate_layer(K_s, V_s, tgt_layer_idx, quantize=True, quantization_control="real"):
        translate_calls.append((tgt_layer_idx, quantize))
        return K_s.clone(), V_s.clone()

    def fake_generate_source_hint(source_model, source_tokenizer, prompt, device, source_reasoning_mode):
        hint_calls.append((prompt, source_reasoning_mode))
        return "hint"

    translator.translate_layer = fake_translate_layer  # type: ignore[method-assign]
    monkeypatch.setattr(evaluate, "_generate_source_hint", fake_generate_source_hint)

    examples = [evaluate.MCQExample(question="q", choices=["A", "B"], answer=0)]

    assert evaluate.eval_target_alone(tgt, tok_t, examples, "cpu") == 1.0
    assert (
        evaluate.eval_text_to_text(
            src,
            tok_s,
            tgt,
            tok_t,
            examples,
            "cpu",
            source_reasoning_mode="cot",
        )
        == 1.0
    )
    assert (
        evaluate.eval_rotalign_kv(
            src,
            tok_s,
            tgt,
            tok_t,
            translator,
            examples,
            "cpu",
            quantize=False,
            source_reasoning_mode="cot",
        )
        == 1.0
    )
    assert hint_calls == [("Question: q\nAnswer:", "cot")]
    assert translate_calls == [(0, False), (1, False)]


def test_build_rotalign_prefix_state_uses_matching_source_and_target_prefix_lengths(monkeypatch) -> None:
    tok_s = FakeTokenizer()
    tok_t = FakeTokenizer()
    src = FakeCausalLM(preferred_token_id=4, n_layers=2)
    tgt = FakeCausalLM(preferred_token_id=4, n_layers=2)
    translator = _make_identity_translator(monkeypatch, layers=2)
    translator.translate_layer = lambda K_s, V_s, tgt_layer_idx, quantize=True, quantization_control="real": (  # type: ignore[method-assign]
        K_s.clone(),
        V_s.clone(),
    )

    state, stats = evaluate._build_rotalign_prefix_state(
        src,
        tok_s,
        tgt,
        tok_t,
        translator,
        source_prompt="alpha beta gamma",
        target_prompt="alpha beta gamma",
        device="cpu",
        quantize=False,
        protocol="fused",
    )

    assert src.last_call["input_shape"] == (1, 2)
    assert tgt.last_call["input_shape"] == (1, 2)
    assert state.prefix_len == 3
    assert stats["bits"] > 0.0


def test_build_rotalign_prefix_state_layer_knockout_removes_communication_bits(monkeypatch) -> None:
    tok_s = FakeTokenizer()
    tok_t = FakeTokenizer()
    src = FakeCausalLM(preferred_token_id=4, n_layers=2)
    tgt = FakeCausalLM(preferred_token_id=4, n_layers=2)
    translator = _make_identity_translator(monkeypatch, layers=2)
    translator.translate_layer = lambda K_s, V_s, tgt_layer_idx, quantize=True, quantization_control="real": (  # type: ignore[method-assign]
        K_s.clone(),
        V_s.clone(),
    )

    _, full_stats = evaluate._build_rotalign_prefix_state(
        src,
        tok_s,
        tgt,
        tok_t,
        translator,
        source_prompt="alpha beta gamma",
        target_prompt="alpha beta gamma",
        device="cpu",
        quantize=False,
        protocol="fused",
    )
    _, dropped_stats = evaluate._build_rotalign_prefix_state(
        src,
        tok_s,
        tgt,
        tok_t,
        translator,
        source_prompt="alpha beta gamma",
        target_prompt="alpha beta gamma",
        device="cpu",
        quantize=False,
        protocol="fused",
        drop_target_layers={1},
        drop_target_layer_mode="target",
    )

    assert dropped_stats["bits"] < full_stats["bits"]
    assert dropped_stats["dropped_target_layers"] == [1]
    assert dropped_stats["drop_target_layer_mode"] == "target"
    assert dropped_stats["selector_trace"][1]["target_layer_drop_mode"] == "target"


def test_target_space_translated_controls_report_zero_communication(monkeypatch) -> None:
    translator = _make_identity_translator(monkeypatch, layers=2)

    assert (
        evaluate._communication_bits(
            translator,
            seq_len=2,
            quantize=True,
            translated_kv_control="real",
        )
        > 0.0
    )
    assert (
        evaluate._communication_bits(
            translator,
            seq_len=2,
            quantize=True,
            translated_kv_control="zero",
        )
        == 0.0
    )
    assert (
        evaluate._communication_bits(
            translator,
            seq_len=2,
            quantize=True,
            translated_kv_control="random",
        )
        == 0.0
    )


def test_zero_fusion_gate_reports_zero_real_kv_communication(monkeypatch) -> None:
    translator = _make_identity_translator(monkeypatch, layers=2)
    translator.set_fixed_gates(0.0)

    assert (
        evaluate._communication_bits(
            translator,
            seq_len=2,
            quantize=True,
            translated_kv_control="real",
            protocol="fused",
        )
        == 0.0
    )
    assert (
        evaluate._communication_bits(
            translator,
            seq_len=2,
            quantize=True,
            translated_kv_control="real",
            protocol="translated_only",
        )
        > 0.0
    )


def test_head_selection_reduces_reported_real_kv_communication(monkeypatch) -> None:
    translator = _make_identity_translator(monkeypatch, layers=2)

    full_bits = evaluate._communication_bits(
        translator,
        seq_len=2,
        quantize=True,
        translated_kv_control="real",
        protocol="translated_only",
    )
    translator.head_selected_mask.fill_(False)
    translator.head_selected_mask[0].copy_(torch.tensor([1], dtype=torch.bool))
    sparse_bits = evaluate._communication_bits(
        translator,
        seq_len=2,
        quantize=True,
        translated_kv_control="real",
        protocol="translated_only",
    )

    assert sparse_bits < full_bits


def test_position_selection_reduces_reported_real_kv_communication(monkeypatch) -> None:
    translator = _make_identity_translator(monkeypatch, layers=2)

    full_bits = evaluate._communication_bits(
        translator,
        seq_len=4,
        quantize=True,
        translated_kv_control="real",
        protocol="translated_only",
        position_selection_ratio=1.0,
    )
    sparse_bits = evaluate._communication_bits(
        translator,
        seq_len=4,
        quantize=True,
        translated_kv_control="real",
        protocol="translated_only",
        position_selection_ratio=0.5,
    )

    assert sparse_bits < full_bits


def test_asymmetric_kv_position_selection_uses_separate_route_and_value_masks() -> None:
    K_t = torch.zeros(1, 1, 4, 1)
    V_t = torch.zeros(1, 1, 4, 1)
    K_hat = torch.tensor([[[[9.0], [1.0], [0.5], [0.1]]]])
    V_hat = torch.tensor([[[[0.1], [0.5], [1.0], [8.0]]]])

    selected_k, selected_v, trace = evaluate._apply_asymmetric_kv_position_selection(
        K_t,
        V_t,
        K_hat,
        V_hat,
        protocol="fused",
        route_selection_ratio=0.25,
        value_selection_ratio=0.50,
        route_selection_metric="energy",
        value_selection_metric="energy",
    )

    assert selected_k[0, 0, 0, 0] == pytest.approx(9.0)
    assert selected_k[0, 0, 3, 0] == pytest.approx(0.0)
    assert selected_v[0, 0, 3, 0] == pytest.approx(8.0)
    assert selected_v[0, 0, 2, 0] == pytest.approx(1.0)
    assert trace["selection_policy"] == "asymmetric_kv_position"
    assert trace["kv_route_keep"] == 1
    assert trace["kv_value_keep"] == 2
    assert trace["kv_route_selection_metric"] == "energy"
    assert trace["kv_value_selection_metric"] == "energy"
    assert trace["kv_route_value_jaccard"] == 0.0


def test_equal_kv_position_ratio_override_keeps_shared_selector_bits(monkeypatch) -> None:
    tok_s = FakeTokenizer()
    tok_t = FakeTokenizer()
    src = FakeCausalLM(preferred_token_id=4, n_layers=2)
    tgt = FakeCausalLM(preferred_token_id=4, n_layers=2)
    translator = _make_identity_translator(monkeypatch, layers=2)
    translator.translate_layer = lambda K_s, V_s, tgt_layer_idx, quantize=True, quantization_control="real": (  # type: ignore[method-assign]
        K_s.clone(),
        V_s.clone(),
    )

    state, stats = evaluate._build_rotalign_prefix_state(
        src,
        tok_s,
        tgt,
        tok_t,
        translator,
        source_prompt="alpha beta gamma",
        target_prompt="alpha beta gamma",
        device="cpu",
        quantize=False,
        protocol="fused",
        position_selection_ratio=1.0,
        kv_route_selection_ratio=0.5,
        kv_value_selection_ratio=0.5,
    )
    expected_bits = evaluate._communication_bits(
        translator,
        seq_len=state.prefix_len - 1,
        quantize=False,
        translated_kv_control="real",
        protocol="fused",
        kv_transport="both",
        position_selection_ratio=0.5,
    )

    assert stats["kv_asymmetric_position_selection"] is False
    assert stats["bits"] == pytest.approx(expected_bits)


def test_kv_transport_reduces_reported_real_kv_communication(monkeypatch) -> None:
    translator = _make_identity_translator(monkeypatch, layers=2)

    both_bits = evaluate._communication_bits(
        translator,
        seq_len=2,
        quantize=True,
        translated_kv_control="real",
        protocol="translated_only",
        kv_transport="both",
    )
    k_only_bits = evaluate._communication_bits(
        translator,
        seq_len=2,
        quantize=True,
        translated_kv_control="real",
        protocol="translated_only",
        kv_transport="k_only",
    )
    v_only_bits = evaluate._communication_bits(
        translator,
        seq_len=2,
        quantize=True,
        translated_kv_control="real",
        protocol="translated_only",
        kv_transport="v_only",
    )

    assert k_only_bits == v_only_bits
    assert both_bits == k_only_bits + v_only_bits


def test_build_rotalign_prefix_state_supports_k_only_and_v_only_transport(monkeypatch) -> None:
    tok_s = FakeTokenizer()
    tok_t = FakeTokenizer()
    src = FakeCausalLM(preferred_token_id=4, n_layers=2)
    tgt = FakeCausalLM(preferred_token_id=4, n_layers=2)
    translator = _make_identity_translator(monkeypatch, layers=2)
    translator.set_fixed_gates(0.5)
    translator.translate_layer = lambda K_s, V_s, tgt_layer_idx, quantize=True, quantization_control="real": (  # type: ignore[method-assign]
        K_s.clone() + 10.0,
        V_s.clone() + 20.0,
    )

    k_state, k_stats = evaluate._build_rotalign_prefix_state(
        src,
        tok_s,
        tgt,
        tok_t,
        translator,
        source_prompt="alpha beta gamma",
        target_prompt="alpha beta gamma",
        device="cpu",
        quantize=False,
        protocol="fused",
        kv_transport="k_only",
    )
    v_state, v_stats = evaluate._build_rotalign_prefix_state(
        src,
        tok_s,
        tgt,
        tok_t,
        translator,
        source_prompt="alpha beta gamma",
        target_prompt="alpha beta gamma",
        device="cpu",
        quantize=False,
        protocol="fused",
        kv_transport="v_only",
    )

    k_layer = k_state.past_key_values[0]
    v_layer = v_state.past_key_values[0]
    k_tensor, k_value = k_layer
    v_tensor, v_value = v_layer

    assert torch.allclose(k_tensor, torch.full_like(k_tensor, 6.0))
    assert torch.allclose(k_value, torch.full_like(k_value, 101.0))
    assert torch.allclose(v_tensor, torch.full_like(v_tensor, 1.0))
    assert torch.allclose(v_value, torch.full_like(v_value, 111.0))
    assert k_stats["bits"] == v_stats["bits"]


def test_build_rotalign_prefix_state_supports_position_sparse_transport(monkeypatch) -> None:
    tok_s = FakeTokenizer()
    tok_t = FakeTokenizer()
    src = FakeCausalLM(preferred_token_id=4, n_layers=2)
    tgt = FakeCausalLM(preferred_token_id=4, n_layers=2)
    translator = _make_identity_translator(monkeypatch, layers=2)
    translator.set_fixed_gates(0.5)

    def translated(K_s, V_s, tgt_layer_idx, quantize=True, quantization_control="real"):
        K_hat = torch.zeros_like(K_s)
        K_hat[:, :, 0, :] = 11.0
        K_hat[:, :, 1, :] = 21.0
        return K_hat, torch.zeros_like(V_s)

    translator.translate_layer = translated  # type: ignore[method-assign]

    sparse_state, sparse_stats = evaluate._build_rotalign_prefix_state(
        src,
        tok_s,
        tgt,
        tok_t,
        translator,
        source_prompt="alpha beta gamma",
        target_prompt="alpha beta gamma",
        device="cpu",
        quantize=False,
        protocol="fused",
        kv_transport="k_only",
        position_selection_ratio=0.5,
    )
    full_state, full_stats = evaluate._build_rotalign_prefix_state(
        src,
        tok_s,
        tgt,
        tok_t,
        translator,
        source_prompt="alpha beta gamma",
        target_prompt="alpha beta gamma",
        device="cpu",
        quantize=False,
        protocol="fused",
        kv_transport="k_only",
        position_selection_ratio=1.0,
    )

    sparse_k, sparse_v = sparse_state.past_key_values[0]
    full_k, full_v = full_state.past_key_values[0]

    assert torch.allclose(sparse_k[:, :, 0, :], torch.full_like(sparse_k[:, :, 0, :], 1.0))
    assert torch.allclose(sparse_k[:, :, 1, :], torch.full_like(sparse_k[:, :, 1, :], 11.0))
    assert torch.allclose(full_k[:, :, 0, :], torch.full_like(full_k[:, :, 0, :], 6.0))
    assert torch.allclose(full_k[:, :, 1, :], torch.full_like(full_k[:, :, 1, :], 11.0))
    assert torch.allclose(sparse_v, full_v)
    assert sparse_stats["bits"] < full_stats["bits"]


def test_build_rotalign_prefix_state_supports_disagreement_position_selection(monkeypatch) -> None:
    tok_s = FakeTokenizer()
    tok_t = FakeTokenizer()
    src = FakeCausalLM(preferred_token_id=4, n_layers=2)
    tgt = FakeCausalLM(preferred_token_id=4, n_layers=2)
    translator = _make_identity_translator(monkeypatch, layers=2)
    translator.set_fixed_gates(0.5)

    def translated(K_s, V_s, tgt_layer_idx, quantize=True, quantization_control="real"):
        K_hat = torch.zeros_like(K_s)
        K_hat[:, :, 0, :] = 1.5
        K_hat[:, :, 1, :] = 11.0
        return K_hat, torch.zeros_like(V_s)

    translator.translate_layer = translated  # type: ignore[method-assign]

    sparse_state, stats = evaluate._build_rotalign_prefix_state(
        src,
        tok_s,
        tgt,
        tok_t,
        translator,
        source_prompt="alpha beta gamma",
        target_prompt="alpha beta gamma",
        device="cpu",
        quantize=False,
        protocol="fused",
        kv_transport="k_only",
        position_selection_ratio=0.5,
        position_selection_metric="disagreement",
    )

    sparse_k, _ = sparse_state.past_key_values[0]

    assert torch.allclose(sparse_k[:, :, 0, :], torch.full_like(sparse_k[:, :, 0, :], 1.0))
    assert torch.allclose(sparse_k[:, :, 1, :], torch.full_like(sparse_k[:, :, 1, :], 6.0))


def test_build_rotalign_prefix_state_supports_attention_position_selection(monkeypatch) -> None:
    tok_s = FakeTokenizer()
    tok_t = FakeTokenizer()
    src = FakeCausalLM(preferred_token_id=4, n_layers=2)
    tgt = FakeCausalLM(preferred_token_id=4, n_layers=2)
    tgt.attention_pattern = torch.tensor([0.9, 0.1, 0.0], dtype=torch.float32)
    translator = _make_identity_translator(monkeypatch, layers=2)
    translator.set_fixed_gates(0.5)

    def translated(K_s, V_s, tgt_layer_idx, quantize=True, quantization_control="real"):
        K_hat = torch.zeros_like(K_s)
        K_hat[:, :, 0, :] = 11.0
        K_hat[:, :, 1, :] = 21.0
        return K_hat, torch.zeros_like(V_s)

    translator.translate_layer = translated  # type: ignore[method-assign]

    sparse_state, stats = evaluate._build_rotalign_prefix_state(
        src,
        tok_s,
        tgt,
        tok_t,
        translator,
        source_prompt="alpha beta gamma",
        target_prompt="alpha beta gamma",
        device="cpu",
        quantize=False,
        protocol="fused",
        kv_transport="k_only",
        position_selection_ratio=0.5,
        position_selection_metric="attention",
    )

    sparse_k, _ = sparse_state.past_key_values[0]

    assert torch.allclose(sparse_k[:, :, 0, :], torch.full_like(sparse_k[:, :, 0, :], 6.0))
    assert torch.allclose(sparse_k[:, :, 1, :], torch.full_like(sparse_k[:, :, 1, :], 1.0))
    assert stats["selector_trace"][0]["selected_positions"] == [0]
    assert stats["selector_trace"][0]["metric"] == "attention"


def test_build_rotalign_prefix_state_supports_source_attention_position_selection(monkeypatch) -> None:
    tok_s = FakeTokenizer()
    tok_t = FakeTokenizer()
    src = FakeCausalLM(preferred_token_id=4, n_layers=2)
    tgt = FakeCausalLM(preferred_token_id=4, n_layers=2)
    src.attention_pattern = torch.tensor([0.9, 0.1, 0.0], dtype=torch.float32)
    translator = _make_identity_translator(monkeypatch, layers=2)
    translator.set_fixed_gates(0.5)

    def translated(K_s, V_s, tgt_layer_idx, quantize=True, quantization_control="real"):
        K_hat = torch.zeros_like(K_s)
        K_hat[:, :, 0, :] = 11.0
        K_hat[:, :, 1, :] = 21.0
        return K_hat, torch.zeros_like(V_s)

    translator.translate_layer = translated  # type: ignore[method-assign]

    sparse_state, _ = evaluate._build_rotalign_prefix_state(
        src,
        tok_s,
        tgt,
        tok_t,
        translator,
        source_prompt="alpha beta gamma",
        target_prompt="alpha beta gamma",
        device="cpu",
        quantize=False,
        protocol="fused",
        kv_transport="k_only",
        position_selection_ratio=0.5,
        position_selection_metric="source_attention",
    )

    sparse_k, _ = sparse_state.past_key_values[0]

    assert torch.allclose(sparse_k[:, :, 0, :], torch.full_like(sparse_k[:, :, 0, :], 6.0))
    assert torch.allclose(sparse_k[:, :, 1, :], torch.full_like(sparse_k[:, :, 1, :], 1.0))


def test_mean_attention_prior_from_prompts_averages_last_token_attention(monkeypatch) -> None:
    tok = FakeTokenizer()
    tgt = FakeCausalLM(preferred_token_id=4, n_layers=2)
    tgt.attention_pattern = torch.tensor([0.9, 0.1, 0.0], dtype=torch.float32)
    translator = _make_identity_translator(monkeypatch, layers=2)

    priors = evaluate._mean_attention_prior_from_prompts(
        tgt,
        tok,
        ["alpha beta gamma", "delta epsilon zeta"],
        "cpu",
        translator=translator,
        bins=2,
    )

    assert len(priors) == 2
    for prior in priors:
        assert torch.allclose(prior, torch.tensor([0.9, 0.1], dtype=torch.float32))


def test_uniform_attention_prior_builds_flat_profiles() -> None:
    priors = evaluate._uniform_attention_prior(layer_count=3, bins=4)

    assert len(priors) == 3
    for prior in priors:
        assert torch.allclose(prior, torch.full((4,), 0.25, dtype=torch.float32))


def test_mean_head_prior_from_prompts_averages_runtime_head_scores(monkeypatch) -> None:
    tok = FakeTokenizer()
    tgt = FakeCausalLM(preferred_token_id=4, n_layers=2)
    translator = _make_identity_translator(monkeypatch, layers=2)

    def fake_maps(*args, **kwargs):
        return [
            torch.tensor([[0.9, 0.1], [0.4, 0.6]], dtype=torch.float32),
            torch.tensor([[0.2, 0.8], [0.7, 0.3]], dtype=torch.float32),
        ]

    monkeypatch.setattr(evaluate, "_last_token_attention_maps", fake_maps)

    priors = evaluate._mean_head_prior_from_prompts(
        tgt,
        tok,
        ["alpha beta gamma", "delta epsilon zeta"],
        "cpu",
        translator=translator,
        metric="attention_peak",
    )

    assert len(priors) == 2
    assert torch.allclose(priors[0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(priors[1], torch.tensor([1.0, 0.0]))


def test_build_rotalign_prefix_state_supports_attention_prior_selection(monkeypatch) -> None:
    tok_s = FakeTokenizer()
    tok_t = FakeTokenizer()
    src = FakeCausalLM(preferred_token_id=4, n_layers=2)
    tgt = FakeCausalLM(preferred_token_id=4, n_layers=2)
    translator = _make_identity_translator(monkeypatch, layers=2)
    translator.set_fixed_gates(0.5)

    def translated(K_s, V_s, tgt_layer_idx, quantize=True, quantization_control="real"):
        K_hat = torch.zeros_like(K_s)
        K_hat[:, :, 0, :] = 11.0
        K_hat[:, :, 1, :] = 21.0
        return K_hat, torch.zeros_like(V_s)

    translator.translate_layer = translated  # type: ignore[method-assign]

    sparse_state, _ = evaluate._build_rotalign_prefix_state(
        src,
        tok_s,
        tgt,
        tok_t,
        translator,
        source_prompt="alpha beta gamma",
        target_prompt="alpha beta gamma",
        device="cpu",
        quantize=False,
        protocol="fused",
        kv_transport="k_only",
        position_selection_ratio=0.5,
        position_selection_metric="attention_prior",
        fixed_position_profiles=[torch.tensor([0.9, 0.1]), torch.tensor([0.9, 0.1])],
    )

    sparse_k, _ = sparse_state.past_key_values[0]

    assert torch.allclose(sparse_k[:, :, 0, :], torch.full_like(sparse_k[:, :, 0, :], 6.0))
    assert torch.allclose(sparse_k[:, :, 1, :], torch.full_like(sparse_k[:, :, 1, :], 1.0))


def test_position_selection_random_metric_is_deterministic() -> None:
    K_t = torch.zeros(1, 1, 2, 1)
    V_t = torch.zeros(1, 1, 2, 1)
    K_hat = torch.tensor([[[[1.0], [2.0]]]])
    V_hat = torch.zeros_like(K_hat)

    first = evaluate._position_selection_scores(
        K_t,
        V_t,
        K_hat,
        V_hat,
        kv_transport="k_only",
        position_selection_metric="random",
    )
    second = evaluate._position_selection_scores(
        K_t,
        V_t,
        K_hat,
        V_hat,
        kv_transport="k_only",
        position_selection_metric="random",
    )

    assert first.shape == (2,)
    assert torch.allclose(first, second)


def test_position_selection_recency_metric_prefers_latest_positions() -> None:
    K_t = torch.zeros(1, 1, 3, 1)
    V_t = torch.zeros(1, 1, 3, 1)
    K_hat = torch.tensor([[[[1.0], [2.0], [3.0]]]])
    V_hat = torch.zeros_like(K_hat)

    scores = evaluate._position_selection_scores(
        K_t,
        V_t,
        K_hat,
        V_hat,
        kv_transport="k_only",
        position_selection_metric="recency",
    )

    assert torch.allclose(scores, torch.tensor([1.0, 2.0, 3.0]))


def test_runtime_head_scores_support_attention_peak_margin_retrieval_and_random() -> None:
    attention_map = torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float32)

    peak = evaluate._runtime_head_scores(
        attention_map,
        metric="attention_peak",
        layer_idx=0,
    )
    margin = evaluate._runtime_head_scores(
        attention_map,
        metric="attention_margin",
        layer_idx=0,
    )
    retrieval = evaluate._runtime_head_scores(
        attention_map,
        metric="retrieval_peak",
        layer_idx=0,
    )
    first = evaluate._runtime_head_scores(
        attention_map,
        metric="random",
        layer_idx=2,
    )
    second = evaluate._runtime_head_scores(
        attention_map,
        metric="random",
        layer_idx=2,
    )

    assert torch.allclose(peak, torch.tensor([0.9, 0.9]))
    assert torch.allclose(margin, torch.full_like(margin, math.log(9.0)))
    assert float(retrieval[1]) > float(retrieval[0])
    assert torch.allclose(first, second)


def test_runtime_head_scores_support_headwise_route_atom() -> None:
    attention_map = torch.tensor(
        [
            [0.34, 0.33, 0.33],
            [0.95, 0.04, 0.01],
            [0.05, 0.90, 0.05],
        ],
        dtype=torch.float32,
    )

    scores = evaluate._runtime_head_scores(
        attention_map,
        metric="headwise_route_atom",
        layer_idx=0,
    )
    trace = evaluate._headwise_route_atom_trace_fields(
        attention_map,
        scores,
        torch.tensor([1], dtype=torch.long),
    )

    assert scores.shape == torch.Size([3])
    assert float(scores[1]) > float(scores[0])
    assert trace["route_atom_keep_fraction"] == pytest.approx(1 / 3)
    assert trace["route_atom_score_entropy"] >= 0.0
    assert trace["route_atom_js_divergence_mean"] >= 0.0


def test_runtime_head_scores_attention_margin_prefers_sharper_head() -> None:
    attention_map = torch.tensor([[0.95, 0.05], [0.60, 0.40]], dtype=torch.float32)

    margin = evaluate._runtime_head_scores(
        attention_map,
        metric="attention_margin",
        layer_idx=0,
    )

    assert float(margin[0]) > float(margin[1])


def test_runtime_head_scores_with_prior_supports_prior_and_blend() -> None:
    attention_map = torch.tensor([[0.9, 0.1], [0.4, 0.6]], dtype=torch.float32)
    prior = torch.tensor([0.2, 0.8], dtype=torch.float32)

    prior_only, _ = evaluate._runtime_head_scores_with_prior(
        attention_map,
        metric="attention_prior",
        layer_idx=0,
        prior_scores=prior,
        prior_alpha=0.5,
    )
    blended, _ = evaluate._runtime_head_scores_with_prior(
        attention_map,
        metric="attention_blend",
        layer_idx=0,
        prior_scores=prior,
        prior_alpha=0.5,
    )

    assert torch.allclose(prior_only, torch.tensor([0.0, 1.0]))
    assert blended.shape == torch.Size([2])
    assert torch.allclose(blended, torch.tensor([0.5, 0.5]))


def test_runtime_head_scores_with_prior_supports_expected_attention() -> None:
    attention_map = torch.tensor(
        [[0.9, 0.05, 0.05], [0.05, 0.05, 0.9]],
        dtype=torch.float32,
    )
    position_prior = torch.tensor([0.8, 0.1, 0.1], dtype=torch.float32)

    expected_scores, _ = evaluate._runtime_head_scores_with_prior(
        attention_map,
        metric="attention_expected",
        layer_idx=0,
        position_prior=position_prior,
    )
    shuffled_scores, _ = evaluate._runtime_head_scores_with_prior(
        attention_map,
        metric="attention_expected_shuffled",
        layer_idx=0,
        position_prior=position_prior,
    )

    assert float(expected_scores[0]) > float(expected_scores[1])
    assert shuffled_scores.shape == torch.Size([2])
    assert not torch.allclose(expected_scores, shuffled_scores)


def test_attention_fidelity_head_scores_prefers_geometry_preserving_head() -> None:
    attention_map = torch.tensor(
        [[0.9, 0.1], [0.1, 0.9]],
        dtype=torch.float32,
    )
    target_keys = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    translated_keys = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    scores = evaluate._attention_fidelity_head_scores(attention_map, target_keys, translated_keys)

    assert float(scores[0]) > float(scores[1])


def test_runtime_head_scores_with_prior_supports_attention_fidelity() -> None:
    attention_map = torch.tensor(
        [[0.9, 0.1], [0.1, 0.9]],
        dtype=torch.float32,
    )
    target_keys = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    translated_keys = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    scores, _ = evaluate._runtime_head_scores_with_prior(
        attention_map,
        metric="attention_fidelity",
        layer_idx=0,
        target_keys=target_keys,
        translated_keys=translated_keys,
    )
    shuffled_scores, _ = evaluate._runtime_head_scores_with_prior(
        attention_map,
        metric="attention_fidelity_shuffled",
        layer_idx=0,
        target_keys=target_keys,
        translated_keys=translated_keys,
    )

    assert float(scores[0]) > float(scores[1])
    assert shuffled_scores.shape == scores.shape
    assert not torch.allclose(scores, shuffled_scores)


def test_attention_procrustes_head_scores_prefers_rotationally_aligned_head() -> None:
    attention_map = torch.tensor(
        [[0.9, 0.1], [0.1, 0.9]],
        dtype=torch.float32,
    )
    target_keys = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    translated_keys = torch.tensor(
        [
            [[0.0, 1.0], [-1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    scores = evaluate._attention_procrustes_head_scores(attention_map, target_keys, translated_keys)

    assert float(scores[0]) > float(scores[1])


def test_runtime_head_scores_with_prior_supports_attention_procrustes() -> None:
    attention_map = torch.tensor(
        [[0.9, 0.1], [0.1, 0.9]],
        dtype=torch.float32,
    )
    target_keys = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    translated_keys = torch.tensor(
        [
            [[0.0, 1.0], [-1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    scores, _ = evaluate._runtime_head_scores_with_prior(
        attention_map,
        metric="attention_procrustes",
        layer_idx=0,
        target_keys=target_keys,
        translated_keys=translated_keys,
    )
    shuffled_scores, _ = evaluate._runtime_head_scores_with_prior(
        attention_map,
        metric="attention_procrustes_shuffled",
        layer_idx=0,
        target_keys=target_keys,
        translated_keys=translated_keys,
    )

    assert float(scores[0]) > float(scores[1])
    assert shuffled_scores.shape == scores.shape
    assert not torch.allclose(scores, shuffled_scores)


def test_attention_qk_fidelity_head_scores_prefers_query_preserving_head() -> None:
    query_heads = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    target_keys = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    translated_keys = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=torch.float32,
    )

    scores = evaluate._attention_qk_fidelity_head_scores(query_heads, target_keys, translated_keys)

    assert float(scores[0]) > float(scores[1])


def test_runtime_head_scores_with_prior_supports_attention_qk_fidelity() -> None:
    query_heads = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    target_keys = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    translated_keys = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=torch.float32,
    )

    scores, _ = evaluate._runtime_head_scores_with_prior(
        None,
        metric="attention_qk_fidelity",
        layer_idx=0,
        query_heads=query_heads,
        target_keys=target_keys,
        translated_keys=translated_keys,
    )
    shuffled_scores, _ = evaluate._runtime_head_scores_with_prior(
        None,
        metric="attention_qk_fidelity_shuffled",
        layer_idx=0,
        query_heads=query_heads,
        target_keys=target_keys,
        translated_keys=translated_keys,
    )

    assert float(scores[0]) > float(scores[1])
    assert shuffled_scores.shape == scores.shape
    assert not torch.allclose(scores, shuffled_scores)


def test_attention_qk_template_transport_scores_prefers_matching_template() -> None:
    query_heads = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    target_keys = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 2.0]],
        ],
        dtype=torch.float32,
    )
    qk_templates = torch.tensor(
        [[0.85, 0.15], [0.15, 0.85]],
        dtype=torch.float32,
    )
    prior_scores = torch.tensor([0.8, 0.2], dtype=torch.float32)

    scores = evaluate._attention_qk_template_transport_scores(
        query_heads,
        target_keys,
        qk_templates,
        prior_scores,
        layer_idx=0,
    )
    shuffled_scores = evaluate._attention_qk_template_transport_scores(
        query_heads,
        target_keys,
        qk_templates,
        prior_scores,
        layer_idx=0,
        shuffled=True,
    )

    assert float(scores[0]) > float(scores[1])
    assert shuffled_scores.shape == scores.shape
    assert not torch.allclose(scores, shuffled_scores)


def test_runtime_head_scores_with_prior_supports_attention_qk_template_transport() -> None:
    query_heads = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    target_keys = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 2.0]],
        ],
        dtype=torch.float32,
    )
    qk_templates = torch.tensor(
        [[0.85, 0.15], [0.15, 0.85]],
        dtype=torch.float32,
    )
    prior_scores = torch.tensor([0.8, 0.2], dtype=torch.float32)

    scores, _ = evaluate._runtime_head_scores_with_prior(
        None,
        metric="attention_qk_template_transport",
        layer_idx=0,
        query_heads=query_heads,
        target_keys=target_keys,
        prior_scores=prior_scores,
        qk_templates=qk_templates,
    )
    shuffled_scores, _ = evaluate._runtime_head_scores_with_prior(
        None,
        metric="attention_qk_template_transport_shuffled",
        layer_idx=0,
        query_heads=query_heads,
        target_keys=target_keys,
        prior_scores=prior_scores,
        qk_templates=qk_templates,
    )

    assert float(scores[0]) > float(scores[1])
    assert shuffled_scores.shape == scores.shape
    assert not torch.allclose(scores, shuffled_scores)


def test_attention_qk_bank_transport_scores_prefers_matching_prompt_bank() -> None:
    query_heads = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    target_keys = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 2.0]],
        ],
        dtype=torch.float32,
    )
    qk_template_bank = torch.tensor(
        [
            [[0.85, 0.15], [0.15, 0.85]],
            [[0.15, 0.85], [0.85, 0.15]],
        ],
        dtype=torch.float32,
    )
    prior_scores = torch.tensor([0.8, 0.2], dtype=torch.float32)

    scores = evaluate._attention_qk_bank_transport_scores(
        query_heads,
        target_keys,
        qk_template_bank,
        prior_scores,
        layer_idx=0,
    )
    shuffled_scores = evaluate._attention_qk_bank_transport_scores(
        query_heads,
        target_keys,
        qk_template_bank,
        prior_scores,
        layer_idx=0,
        shuffled=True,
    )

    assert float(scores[0]) > float(scores[1])
    assert shuffled_scores.shape == scores.shape
    assert not torch.allclose(scores, shuffled_scores)


def test_runtime_head_scores_with_prior_supports_attention_qk_bank_transport() -> None:
    query_heads = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    target_keys = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 2.0]],
        ],
        dtype=torch.float32,
    )
    qk_template_bank = torch.tensor(
        [
            [[0.85, 0.15], [0.15, 0.85]],
            [[0.15, 0.85], [0.85, 0.15]],
        ],
        dtype=torch.float32,
    )
    prior_scores = torch.tensor([0.8, 0.2], dtype=torch.float32)

    scores, _ = evaluate._runtime_head_scores_with_prior(
        None,
        metric="attention_qk_bank_transport",
        layer_idx=0,
        query_heads=query_heads,
        target_keys=target_keys,
        prior_scores=prior_scores,
        qk_template_bank=qk_template_bank,
    )
    shuffled_scores, _ = evaluate._runtime_head_scores_with_prior(
        None,
        metric="attention_qk_bank_transport_shuffled",
        layer_idx=0,
        query_heads=query_heads,
        target_keys=target_keys,
        prior_scores=prior_scores,
        qk_template_bank=qk_template_bank,
    )

    assert float(scores[0]) > float(scores[1])
    assert shuffled_scores.shape == scores.shape
    assert not torch.allclose(scores, shuffled_scores)


def test_per_head_gate_override_from_scores_preserves_mean_gate() -> None:
    scores = torch.tensor([0.0, 2.0], dtype=torch.float32)

    gate = evaluate._per_head_gate_override_from_scores(0.2, scores, strength=1.0)

    assert float(gate[1]) > float(gate[0])
    assert float(gate.mean()) == pytest.approx(0.2, abs=1e-6)


def test_attention_qk_fidelity_token_scores_focus_on_shared_query_mass() -> None:
    query_heads = torch.tensor(
        [[1.0, 0.0]],
        dtype=torch.float32,
    )
    target_keys = torch.tensor(
        [[[2.0, 0.0], [0.0, 0.0]]],
        dtype=torch.float32,
    )
    translated_keys = torch.tensor(
        [[[1.8, 0.0], [0.1, 0.0]]],
        dtype=torch.float32,
    )

    scores = evaluate._attention_qk_fidelity_token_scores(
        query_heads,
        target_keys,
        translated_keys,
    )

    assert scores.shape == (1, 2)
    assert float(scores[0, 0]) > float(scores[0, 1])


def test_tokenwise_gate_override_from_scores_preserves_per_head_mean() -> None:
    base_gate = torch.tensor([0.25, 0.5], dtype=torch.float32)
    scores = torch.tensor(
        [[2.0, 1.0], [1.0, 3.0]],
        dtype=torch.float32,
    )

    gate = evaluate._tokenwise_gate_override_from_scores(base_gate, scores, strength=1.0)

    assert gate.shape == (1, 2, 2, 1)
    assert float(gate[0, 0, 0, 0]) > float(gate[0, 0, 1, 0])
    assert float(gate[0, 1, 1, 0]) > float(gate[0, 1, 0, 0])
    assert float(gate[0, 0, :, 0].mean()) == pytest.approx(0.25, abs=1e-6)
    assert float(gate[0, 1, :, 0].mean()) == pytest.approx(0.5, abs=1e-6)


def test_attention_template_transport_scores_prefers_matching_template() -> None:
    attention_map = torch.tensor(
        [[0.9, 0.1], [0.1, 0.9]],
        dtype=torch.float32,
    )
    head_templates = torch.tensor(
        [[0.85, 0.15], [0.15, 0.85]],
        dtype=torch.float32,
    )
    prior_scores = torch.tensor([0.8, 0.2], dtype=torch.float32)

    scores = evaluate._attention_template_transport_scores(
        attention_map,
        head_templates,
        prior_scores,
        layer_idx=0,
    )
    shuffled_scores = evaluate._attention_template_transport_scores(
        attention_map,
        head_templates,
        prior_scores,
        layer_idx=0,
        shuffled=True,
    )

    assert float(scores[0]) > float(scores[1])
    assert shuffled_scores.shape == scores.shape
    assert not torch.allclose(scores, shuffled_scores)


def test_runtime_head_scores_with_prior_supports_attention_template_transport() -> None:
    attention_map = torch.tensor(
        [[0.9, 0.1], [0.1, 0.9]],
        dtype=torch.float32,
    )
    head_templates = torch.tensor(
        [[0.85, 0.15], [0.15, 0.85]],
        dtype=torch.float32,
    )
    prior_scores = torch.tensor([0.8, 0.2], dtype=torch.float32)

    scores, _ = evaluate._runtime_head_scores_with_prior(
        attention_map,
        metric="attention_template_transport",
        layer_idx=0,
        prior_scores=prior_scores,
        head_templates=head_templates,
    )
    shuffled_scores, _ = evaluate._runtime_head_scores_with_prior(
        attention_map,
        metric="attention_template_transport_shuffled",
        layer_idx=0,
        prior_scores=prior_scores,
        head_templates=head_templates,
    )

    assert float(scores[0]) > float(scores[1])
    assert shuffled_scores.shape == scores.shape
    assert not torch.allclose(scores, shuffled_scores)


def test_attention_sinkhorn_transport_scores_prefers_matching_template() -> None:
    attention_map = torch.tensor(
        [[0.9, 0.1], [0.1, 0.9]],
        dtype=torch.float32,
    )
    head_templates = torch.tensor(
        [[0.85, 0.15], [0.15, 0.85]],
        dtype=torch.float32,
    )
    prior_scores = torch.tensor([0.8, 0.2], dtype=torch.float32)

    scores = evaluate._attention_sinkhorn_transport_scores(
        attention_map,
        head_templates,
        prior_scores,
        layer_idx=0,
    )
    shuffled_scores = evaluate._attention_sinkhorn_transport_scores(
        attention_map,
        head_templates,
        prior_scores,
        layer_idx=0,
        shuffled=True,
    )

    assert float(scores[0]) > float(scores[1])
    assert shuffled_scores.shape == scores.shape
    assert not torch.allclose(scores, shuffled_scores)


def test_runtime_head_scores_with_prior_supports_attention_sinkhorn() -> None:
    attention_map = torch.tensor(
        [[0.9, 0.1], [0.1, 0.9]],
        dtype=torch.float32,
    )
    head_templates = torch.tensor(
        [[0.85, 0.15], [0.15, 0.85]],
        dtype=torch.float32,
    )
    prior_scores = torch.tensor([0.8, 0.2], dtype=torch.float32)

    scores, _ = evaluate._runtime_head_scores_with_prior(
        attention_map,
        metric="attention_sinkhorn",
        layer_idx=0,
        prior_scores=prior_scores,
        head_templates=head_templates,
    )
    shuffled_scores, _ = evaluate._runtime_head_scores_with_prior(
        attention_map,
        metric="attention_sinkhorn_shuffled",
        layer_idx=0,
        prior_scores=prior_scores,
        head_templates=head_templates,
    )

    assert float(scores[0]) > float(scores[1])
    assert shuffled_scores.shape == scores.shape
    assert not torch.allclose(scores, shuffled_scores)


def test_match_prior_scores_to_live_order_is_permutation_invariant() -> None:
    live_scores = torch.tensor([0.2, 0.7, 0.9], dtype=torch.float32)
    prior_scores = torch.tensor([0.1, 0.8, 0.4], dtype=torch.float32)

    matched = evaluate._match_prior_scores_to_live_order(
        live_scores,
        prior_scores,
        layer_idx=0,
    )
    shuffled = evaluate._match_prior_scores_to_live_order(
        live_scores,
        prior_scores,
        layer_idx=0,
        shuffled=True,
    )

    assert torch.argsort(matched, descending=True).tolist() == torch.argsort(live_scores, descending=True).tolist()
    assert shuffled.shape == matched.shape
    assert not torch.allclose(matched, shuffled)


def test_runtime_head_scores_with_prior_supports_attention_match() -> None:
    attention_map = torch.tensor(
        [[0.2, 0.8], [0.9, 0.1], [0.6, 0.4]],
        dtype=torch.float32,
    )
    prior = torch.tensor([0.1, 0.9, 0.4], dtype=torch.float32)

    matched_scores, _ = evaluate._runtime_head_scores_with_prior(
        attention_map,
        metric="attention_match",
        layer_idx=0,
        prior_scores=prior,
    )
    shuffled_scores, _ = evaluate._runtime_head_scores_with_prior(
        attention_map,
        metric="attention_match_shuffled",
        layer_idx=0,
        prior_scores=prior,
    )

    keep = torch.topk(matched_scores, k=1, largest=True).indices.item()
    assert keep == 1
    assert shuffled_scores.shape == matched_scores.shape
    assert not torch.allclose(matched_scores, shuffled_scores)


def test_resample_head_profile_preserves_distribution() -> None:
    resampled = evaluate._resample_head_profile(torch.tensor([1.0, 3.0]), 4)
    assert resampled.shape == torch.Size([4])
    assert torch.allclose(resampled.sum(), torch.tensor(1.0))


def test_allocate_per_head_token_budgets_conserves_total_budget() -> None:
    keep = evaluate._allocate_per_head_token_budgets(
        torch.tensor([0.9, 0.1], dtype=torch.float32),
        seq_len=4,
        position_selection_ratio=0.5,
    )

    assert keep.tolist() == [4, 0]


def test_translated_bit_breakdown_splits_payload_and_selector_bits() -> None:
    translator = SimpleNamespace(
        config=SimpleNamespace(tgt_num_heads=2, tgt_head_dim=4, quant_bits=4),
        selected_layer_indices=lambda: [0],
        selected_head_count=lambda layer_idx: 2,
    )

    payload_bits, selector_bits = evaluate._translated_bit_breakdown(
        translator,
        seq_len=4,
        quantize=True,
        active_k_head_counts=[2],
        active_v_head_counts=[],
        kv_transport="k_only",
        position_selection_ratio=0.5,
    )

    assert payload_bits > 0.0
    assert selector_bits > 0.0
    assert evaluate._translated_bits(
        translator,
        seq_len=4,
        quantize=True,
        active_k_head_counts=[2],
        active_v_head_counts=[],
        kv_transport="k_only",
        position_selection_ratio=0.5,
    ) == pytest.approx(payload_bits + selector_bits)


def test_save_and_load_head_profile_bundle_resamples_layers(tmp_path) -> None:
    bundle_path = tmp_path / "head_prior.pt"
    profiles = [
        torch.tensor([1.0, 3.0], dtype=torch.float32),
        torch.tensor([2.0, 2.0], dtype=torch.float32),
    ]

    evaluate._save_head_profile_bundle(
        str(bundle_path),
        profiles,
        metadata={"source_model": "src", "target_model": "tgt"},
    )
    loaded, metadata = evaluate._load_head_profile_bundle(str(bundle_path), target_layers=4)

    assert len(loaded) == 4
    assert all(profile.ndim == 1 for profile in loaded)
    assert metadata["source_model"] == "src"
    assert metadata["stored_layer_count"] == 2
    assert metadata["resampled_to_layer_count"] == 4


def test_shrink_head_profiles_blends_toward_uniform_and_global() -> None:
    profiles = [
        torch.tensor([0.0, 1.0, 0.2], dtype=torch.float32),
        torch.tensor([1.0, 0.0, 0.5], dtype=torch.float32),
    ]

    uniform = evaluate._shrink_head_profiles(profiles, strength=0.5, target="uniform")
    global_profiles = evaluate._shrink_head_profiles(profiles, strength=0.5, target="global")

    assert len(uniform) == len(profiles)
    assert len(global_profiles) == len(profiles)
    assert uniform[0].shape == profiles[0].shape
    assert global_profiles[1].shape == profiles[1].shape
    assert float(uniform[0][2]) > float(profiles[0][2])
    assert all(torch.isfinite(profile).all() for profile in uniform)
    assert all(torch.isfinite(profile).all() for profile in global_profiles)


def test_allocate_per_head_token_budgets_sum_normalization_preserves_shrinkage_effect() -> None:
    keep = evaluate._allocate_per_head_token_budgets(
        torch.tensor([0.4, 0.35, 0.25], dtype=torch.float32),
        seq_len=4,
        position_selection_ratio=0.5,
        score_normalization="sum",
    )

    assert keep.tolist() == [2, 2, 2]


def test_apply_per_head_position_selection_prefers_top_positions_per_head() -> None:
    K_t = torch.zeros(1, 2, 4, 1)
    V_t = torch.zeros_like(K_t)
    K_hat = torch.tensor([[[[1.0], [2.0], [3.0], [4.0]], [[5.0], [6.0], [7.0], [8.0]]]])
    V_hat = torch.zeros_like(K_hat)
    selected_k, _, trace, keep_counts = evaluate._apply_per_head_position_selection(
        K_t,
        V_t,
        K_hat,
        V_hat,
        protocol="translated_only",
        kv_transport="k_only",
        position_selection_ratio=0.5,
        head_scores=torch.tensor([0.8, 0.2], dtype=torch.float32),
        per_head_position_scores=torch.tensor(
            [[0.9, 0.8, 0.1, 0.0], [0.0, 0.2, 0.7, 0.6]],
            dtype=torch.float32,
        ),
        active_head_indices=torch.tensor([0, 1], dtype=torch.long),
        return_trace=True,
    )

    assert keep_counts.tolist() == [4, 0]
    assert torch.allclose(selected_k[:, 0], K_hat[:, 0])
    assert torch.allclose(selected_k[:, 1], torch.zeros_like(selected_k[:, 1]))
    assert trace["head_budget_nonzero_heads"] == 1


def test_runtime_head_scores_with_prior_supports_shuffled_prior() -> None:
    prior_scores = torch.tensor([0.1, 0.9, 0.3, 0.7], dtype=torch.float32)

    shuffled, _ = evaluate._runtime_head_scores_with_prior(
        None,
        metric="attention_prior_shuffled",
        layer_idx=2,
        prior_scores=prior_scores,
        prior_alpha=0.5,
    )

    direct = evaluate._normalize_selection_scores(prior_scores)
    assert sorted(round(float(v), 6) for v in shuffled.tolist()) == sorted(round(float(v), 6) for v in direct.tolist())
    assert not torch.allclose(shuffled, direct)


def test_apply_runtime_head_selection_fills_with_target_on_fused() -> None:
    translated = torch.tensor([[[[10.0]], [[20.0]]]])
    target = torch.tensor([[[[1.0]], [[2.0]]]])
    mask = torch.tensor([True, False])

    fused = evaluate._apply_runtime_head_selection(
        translated,
        target,
        mask,
        protocol="fused",
    )
    translated_only = evaluate._apply_runtime_head_selection(
        translated,
        target,
        mask,
        protocol="translated_only",
    )

    assert torch.allclose(fused, torch.tensor([[[[10.0]], [[2.0]]]]))
    assert torch.allclose(translated_only, torch.tensor([[[[10.0]], [[0.0]]]]))


def test_position_selection_attention_shuffled_metric_is_deterministic() -> None:
    K_t = torch.zeros(1, 1, 3, 1)
    V_t = torch.zeros(1, 1, 3, 1)
    K_hat = torch.tensor([[[[1.0], [2.0], [3.0]]]])
    V_hat = torch.zeros_like(K_hat)
    scores = torch.tensor([0.9, 0.1, 0.0], dtype=torch.float32)

    first = evaluate._position_selection_scores(
        K_t,
        V_t,
        K_hat,
        V_hat,
        kv_transport="k_only",
        position_selection_metric="attention_shuffled",
        position_scores=scores,
    )
    second = evaluate._position_selection_scores(
        K_t,
        V_t,
        K_hat,
        V_hat,
        kv_transport="k_only",
        position_selection_metric="attention_shuffled",
        position_scores=scores,
    )

    assert first.shape == (3,)
    assert torch.allclose(first, second)
    assert torch.allclose(first.sort().values, scores.sort().values)
    assert not torch.allclose(first, scores)


def test_position_selection_attention_disagreement_combines_query_and_delta() -> None:
    K_t = torch.zeros(1, 1, 2, 1)
    V_t = torch.zeros(1, 1, 2, 1)
    K_hat = torch.tensor([[[[1.0], [6.0]]]])
    V_hat = torch.zeros_like(K_hat)
    scores = torch.tensor([0.9, 0.1], dtype=torch.float32)

    combined = evaluate._position_selection_scores(
        K_t,
        V_t,
        K_hat,
        V_hat,
        kv_transport="k_only",
        position_selection_metric="attention_disagreement",
        position_scores=scores,
    )

    expected = torch.tensor([0.2, 0.8], dtype=torch.float32)
    assert combined.shape == (2,)
    assert torch.allclose(combined, expected, atol=1e-6)


def test_stratified_topk_spreads_positions_across_regions() -> None:
    scores = torch.tensor([9.0, 8.0, 7.0, 6.0, 0.0, 0.0, 0.0, 5.0])

    selected = evaluate._stratified_topk_indices(scores, keep=4, bins=4)

    assert set(selected.tolist()) == {0, 2, 4, 7}


def test_apply_position_selection_attention_stratified_changes_selected_region_coverage() -> None:
    K_t = torch.zeros(1, 1, 8, 1)
    V_t = torch.zeros_like(K_t)
    K_hat = torch.arange(1, 9, dtype=torch.float32).view(1, 1, 8, 1)
    V_hat = K_hat + 100.0
    scores = torch.tensor([9.0, 8.0, 7.0, 6.0, 0.0, 0.0, 0.0, 5.0])

    selected_k, _, trace = evaluate._apply_position_selection(
        K_t,
        V_t,
        K_hat,
        V_hat,
        protocol="fused",
        kv_transport="k_only",
        position_selection_ratio=0.5,
        position_selection_metric="attention_stratified",
        position_scores=scores,
        return_trace=True,
    )

    kept = torch.nonzero(selected_k.view(-1), as_tuple=False).flatten().tolist()
    assert kept == [0, 2, 4, 7]
    assert trace["selected_positions"] == [0, 2, 4, 7]
    assert trace["selection_policy"] == "stratified_topk_4bins"


def test_generation_rotalign_uses_source_reasoning_prompt(monkeypatch) -> None:
    tok_s = FakeTokenizer()
    tok_t = FakeTokenizer()
    src = FakeCausalLM(preferred_token_id=4, n_layers=2)
    tgt = FakeCausalLM(preferred_token_id=4, n_layers=2)
    translator = _make_identity_translator(monkeypatch, layers=2)
    captured: dict[str, str] = {}

    def fake_build_rotalign_prefix_state(
        source_model,
        source_tokenizer,
        target_model,
        target_tokenizer,
        translator_obj,
        source_prompt,
        target_prompt,
        device,
        quantize,
        protocol,
        source_kv_control="real",
        quantization_control="real",
        translated_kv_control="real",
        fusion_rule="static",
        kv_transport="both",
        position_selection_ratio=1.0,
        position_selection_metric="energy",
        fixed_position_profiles=None,
        runtime_head_selection_ratio=1.0,
        runtime_head_selection_metric="attention_peak",
        fixed_head_profiles=None,
        runtime_head_prior_alpha=0.5,
        per_head_position_budget_mode="none",
        **kwargs,
    ):
        captured["source_prompt"] = source_prompt
        captured["target_prompt"] = target_prompt
        return evaluate.PrefixState(None, torch.tensor([[2]], dtype=torch.long), 1), {"bits": 0.0}

    monkeypatch.setattr(evaluate, "_build_rotalign_prefix_state", fake_build_rotalign_prefix_state)

    examples = [evaluate.GenerationExample(prompt="solve this", answers=["A"])]

    score, bits, latency = evaluate._eval_generation_rotalign(
        src,
        tok_s,
        tgt,
        tok_t,
        translator,
        examples,
        "cpu",
        max_new_tokens=1,
        quantize=False,
        protocol="fused",
        source_reasoning_mode="scratchpad",
        source_kv_control="real",
        quantization_control="real",
        translated_kv_control="real",
    )

    assert score == 1.0
    assert bits == 0.0
    assert latency >= 0.0
    assert "scratchpad" in captured["source_prompt"].lower()
    assert captured["target_prompt"] == "solve this"


def test_generation_stats_helpers_report_system_metrics(monkeypatch) -> None:
    tok_s = FakeTokenizer()
    tok_t = FakeTokenizer()
    src = FakeCausalLM(preferred_token_id=4, n_layers=2)
    tgt = FakeCausalLM(preferred_token_id=4, n_layers=2)
    translator = _make_identity_translator(monkeypatch, layers=2)

    def fake_build_rotalign_prefix_state(
        source_model,
        source_tokenizer,
        target_model,
        target_tokenizer,
        translator_obj,
        source_prompt,
        target_prompt,
        device,
        quantize,
        protocol,
        source_kv_control="real",
        quantization_control="real",
        translated_kv_control="real",
        fusion_rule="static",
        kv_transport="both",
        position_selection_ratio=1.0,
        position_selection_metric="energy",
        fixed_position_profiles=None,
        runtime_head_selection_ratio=1.0,
        runtime_head_selection_metric="attention_peak",
        fixed_head_profiles=None,
        runtime_head_prior_alpha=0.5,
        per_head_position_budget_mode="none",
        **kwargs,
    ):
        return evaluate.PrefixState(None, torch.tensor([[2]], dtype=torch.long), 1), {"bits": 32.0}

    monkeypatch.setattr(evaluate, "_build_rotalign_prefix_state", fake_build_rotalign_prefix_state)
    examples = [evaluate.GenerationExample(prompt="solve this", answers=["A"])]

    target_stats = evaluate._eval_generation_target_alone_with_stats(
        tgt,
        tok_t,
        examples,
        "cpu",
        max_new_tokens=1,
    )
    rotalign_stats = evaluate._eval_generation_rotalign_with_stats(
        src,
        tok_s,
        tgt,
        tok_t,
        translator,
        examples,
        "cpu",
        max_new_tokens=1,
        quantize=False,
        protocol="fused",
        source_reasoning_mode="plain",
    )

    assert target_stats["accuracy"] == 1.0
    assert target_stats["ttft_sec"] >= 0.0
    assert target_stats["tokens_per_sec"] > 0.0
    assert target_stats["examples_per_sec"] > 0.0
    assert target_stats["latency_sec"] >= 0.0
    assert target_stats["generated_tokens_avg"] == 1.0

    assert rotalign_stats["accuracy"] == 1.0
    assert rotalign_stats["bits"] == 32.0
    assert rotalign_stats["bytes"] == 4.0
    assert rotalign_stats["ttft_sec"] >= 0.0
    assert rotalign_stats["tokens_per_sec"] > 0.0
    assert rotalign_stats["examples_per_sec"] > 0.0
    assert rotalign_stats["latency_sec"] >= 0.0


def test_generation_text_to_text_passes_attention_mask_to_source_generate() -> None:
    tok_s = FakeTokenizer()
    tok_t = FakeTokenizer()
    src = FakeCausalLM(preferred_token_id=4, n_layers=2)
    tgt = FakeCausalLM(preferred_token_id=4, n_layers=2)
    examples = [evaluate.GenerationExample(prompt="solve this", answers=["because"])]

    score = evaluate._eval_generation_text_to_text(
        src,
        tok_s,
        tgt,
        tok_t,
        examples,
        device="cpu",
        max_new_tokens=4,
    )

    assert 0.0 <= score <= 1.0
    assert src.last_generate_call["attention_mask_shape"] == src.last_generate_call["input_shape"]


def test_write_prediction_sidecar_writes_run_and_method_summary(tmp_path) -> None:
    pred_path = tmp_path / "predictions.jsonl"
    records = [
        {
            "index": 0,
            "method": "target_alone",
            "prediction": "42",
            "answer": ["42"],
            "correct": True,
        },
        {
            "index": 0,
            "method": "rotalign_kv_gate_0.10",
            "prediction": "42",
            "answer": ["42"],
            "correct": True,
            "bits": 16.0,
            "payload_bits": 12.0,
            "selector_bits": 4.0,
            "metadata_bits": 4.0,
            "kv_route_selection_ratio": 0.25,
            "kv_value_selection_ratio": 0.75,
            "selector_trace": [
                {
                    "keep_fraction": 0.5,
                    "score_entropy": 0.7,
                    "selected_positions": [1],
                    "kv_route_keep_fraction": 0.25,
                    "kv_value_keep_fraction": 0.75,
                    "kv_route_value_overlap": 0.5,
                    "kv_route_value_jaccard": 0.25,
                    "kv_route_score_entropy": 0.2,
                    "kv_value_score_entropy": 0.8,
                }
            ],
            "head_trace": [
                {
                    "head_keep_fraction": 0.5,
                    "head_score_entropy": 0.2,
                    "head_prior_overlap_jaccard": 0.25,
                    "route_atom_keep_fraction": 0.5,
                    "route_atom_score_entropy": 0.4,
                    "route_atom_score_gap": 0.1,
                    "route_atom_sharpness_mean": 0.8,
                    "route_atom_js_divergence_mean": 0.3,
                    "route_atom_orientation_span": 0.6,
                    "selected_head_ids": [0],
                }
            ],
            "head_budget_trace": [
                {
                    "head_budget_keep_fraction": 0.5,
                    "head_budget_nonzero_fraction": 0.25,
                }
            ],
        },
    ]
    results = {
        "target_alone": 1.0,
        "rotalign_kv_gate_0.10": 1.0,
        "paired_rotalign_kv_gate_0_10_vs_target_alone_delta_accuracy": 0.0,
    }

    evaluate.write_prediction_sidecar(
        str(pred_path),
        records,
        results,
        {"source_model": "src", "target_model": "tgt"},
    )

    payload = json.loads((tmp_path / "predictions.jsonl.meta.json").read_text())
    assert payload["run_config"]["source_model"] == "src"
    assert payload["method_summary"]["rotalign_kv_gate_0.10"]["avg_bits"] == 16.0
    assert payload["method_summary"]["rotalign_kv_gate_0.10"]["payload_bits_avg"] == 12.0
    assert payload["method_summary"]["rotalign_kv_gate_0.10"]["selector_bits_avg"] == 4.0
    assert payload["method_summary"]["rotalign_kv_gate_0.10"]["kv_route_selection_ratio_avg"] == 0.25
    assert payload["method_summary"]["rotalign_kv_gate_0.10"]["kv_value_selection_ratio_avg"] == 0.75
    assert payload["method_summary"]["rotalign_kv_gate_0.10"]["selector_keep_fraction_avg"] == 0.5
    assert payload["method_summary"]["rotalign_kv_gate_0.10"]["kv_route_value_jaccard_avg"] == 0.25
    assert payload["method_summary"]["rotalign_kv_gate_0.10"]["kv_value_score_entropy_avg"] == 0.8
    assert payload["method_summary"]["rotalign_kv_gate_0.10"]["head_keep_fraction_avg"] == 0.5
    assert payload["method_summary"]["rotalign_kv_gate_0.10"]["head_prior_overlap_jaccard_avg"] == 0.25
    assert payload["method_summary"]["rotalign_kv_gate_0.10"]["route_atom_score_entropy_avg"] == 0.4
    assert payload["method_summary"]["rotalign_kv_gate_0.10"]["route_atom_orientation_span_avg"] == 0.6
    assert payload["method_summary"]["rotalign_kv_gate_0.10"]["head_budget_keep_fraction_avg"] == 0.5
    assert "paired_rotalign_kv_gate_0_10_vs_target_alone_delta_accuracy" in payload["paired_summary"]


def test_append_prediction_record_preserves_stable_example_id() -> None:
    records: list[dict[str, object]] = []
    example = evaluate.GenerationExample(prompt="p", answers=["1"])
    example_id = evaluate._generation_example_id(example)

    evaluate._append_prediction_record(
        records,
        index=0,
        example_id=example_id,
        method="target_alone",
        prediction="1",
        answer=["1"],
        correct=True,
    )

    assert records[0]["example_id"] == example_id
    assert example_id == evaluate._generation_example_id(example)
