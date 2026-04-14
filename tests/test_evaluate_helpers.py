from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

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

    def _make_pkv(self, batch: int, seq: int, device: torch.device):
        layers = []
        for layer_idx in range(self.n_layers):
            base = float(layer_idx + 1)
            K = torch.full((batch, 1, seq, 1), base, device=device)
            V = torch.full((batch, 1, seq, 1), base + 100.0, device=device)
            layers.append((K, V))
        return FakeCache(layers)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, labels=None):
        self.last_call = {
            "input_shape": tuple(input_ids.shape),
            "attention_mask_shape": None if attention_mask is None else tuple(attention_mask.shape),
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }
        batch, seq = input_ids.shape
        logits = torch.full((batch, seq, self.vocab_size), -20.0, device=input_ids.device)
        logits[..., self.preferred_token_id] = 20.0
        return SimpleNamespace(
            logits=logits,
            past_key_values=self._make_pkv(batch, seq, input_ids.device) if use_cache else None,
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


def test_score_helpers_rank_the_preferred_choice() -> None:
    tok = FakeTokenizer()
    model = FakeCausalLM(preferred_token_id=4)

    score_a = evaluate.score_choice_loglik(model, tok, "Question: x Answer:", "A", "cpu")
    score_b = evaluate.score_choice_loglik(model, tok, "Question: x Answer:", "B", "cpu")

    assert score_a > score_b


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


def test_eval_helpers_work_with_fakes(monkeypatch) -> None:
    tok_s = FakeTokenizer()
    tok_t = FakeTokenizer()
    src = FakeCausalLM(preferred_token_id=4, n_layers=2)
    tgt = FakeCausalLM(preferred_token_id=4, n_layers=2)
    translator = _make_identity_translator(monkeypatch, layers=2)
    translate_calls: list[tuple[int, bool]] = []

    def fake_translate_layer(K_s, V_s, tgt_layer_idx, quantize=True):
        translate_calls.append((tgt_layer_idx, quantize))
        return K_s.clone(), V_s.clone()

    translator.translate_layer = fake_translate_layer  # type: ignore[method-assign]

    examples = [evaluate.MCQExample(question="q", choices=["A", "B"], answer=0)]

    assert evaluate.eval_target_alone(tgt, tok_t, examples, "cpu") == 1.0
    assert evaluate.eval_text_to_text(src, tok_s, tgt, tok_t, examples, "cpu") == 1.0
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
        )
        == 1.0
    )
    assert translate_calls == [(0, False), (1, False)]


def test_build_rotalign_prefix_state_uses_matching_source_and_target_prefix_lengths(monkeypatch) -> None:
    tok_s = FakeTokenizer()
    tok_t = FakeTokenizer()
    src = FakeCausalLM(preferred_token_id=4, n_layers=2)
    tgt = FakeCausalLM(preferred_token_id=4, n_layers=2)
    translator = _make_identity_translator(monkeypatch, layers=2)
    translator.translate_layer = lambda K_s, V_s, tgt_layer_idx, quantize=True: (  # type: ignore[method-assign]
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
