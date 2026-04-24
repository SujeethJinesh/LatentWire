from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import json
import torch

import latent_bridge.ablation_sweep as sweep
import latent_bridge.calibrate as calibrate


class _FakeTokenizer:
    def __init__(self, lengths_by_prompt: dict[str, int]) -> None:
        self.pad_token_id = None
        self.eos_token = "<eos>"
        self.lengths_by_prompt = lengths_by_prompt

    def __call__(self, batch, return_tensors="pt", padding=None, truncation=None, max_length=None):
        if isinstance(batch, str):
            batch = [batch]
        seq_len = max_length or max(self.lengths_by_prompt.values())
        data = []
        masks = []
        for prompt in batch:
            real_len = min(self.lengths_by_prompt[prompt], seq_len)
            data.append(list(range(1, real_len + 1)) + [0] * (seq_len - real_len))
            masks.append([1] * real_len + [0] * (seq_len - real_len))
        tensor = torch.tensor(data, dtype=torch.long)
        mask = torch.tensor(masks, dtype=torch.long)

        class _Encoded(dict):
            def __init__(self, input_ids, attention_mask):
                super().__init__(input_ids=input_ids, attention_mask=attention_mask)
                self.input_ids = input_ids
                self.attention_mask = attention_mask

            def to(self, device):
                self.input_ids = self.input_ids.to(device)
                self["input_ids"] = self.input_ids
                self.attention_mask = self.attention_mask.to(device)
                self["attention_mask"] = self.attention_mask
                return self

        return _Encoded(tensor, mask)


class _FakeChatTokenizer(_FakeTokenizer):
    def __init__(self, lengths_by_prompt: dict[str, int]) -> None:
        super().__init__(lengths_by_prompt)
        self.chat_calls: list[tuple[str, bool | None]] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=None):
        assert tokenize is False
        assert add_generation_prompt is True
        content = messages[0]["content"]
        self.chat_calls.append((content, enable_thinking))
        return f"chat::{enable_thinking}::{content}"


class _FakeOffsetTokenizer(_FakeTokenizer):
    def __init__(self) -> None:
        super().__init__({})

    def __call__(
        self,
        batch,
        return_tensors="pt",
        padding=None,
        truncation=None,
        max_length=None,
        return_offsets_mapping=False,
    ):
        if isinstance(batch, str):
            texts = [batch]
            single = True
        else:
            texts = list(batch)
            single = False
        if return_offsets_mapping:
            encoded = {
                "input_ids": [],
                "offset_mapping": [],
            }
            for text in texts:
                if truncation and max_length is not None:
                    text = text[:max_length]
                encoded["input_ids"].append(list(range(len(text))))
                encoded["offset_mapping"].append([(idx, idx + 1) for idx in range(len(text))])
            if single:
                return {
                    "input_ids": encoded["input_ids"][0],
                    "offset_mapping": encoded["offset_mapping"][0],
                }
            return encoded
        lengths = {text: min(len(text), max_length or len(text)) for text in texts}
        self.lengths_by_prompt = lengths
        return super().__call__(
            texts if not single else texts[0],
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )


class _ScriptedOffsetTokenizer(_FakeTokenizer):
    def __init__(self, offsets_by_text: dict[str, list[tuple[int, int]]]) -> None:
        super().__init__({})
        self.offsets_by_text = offsets_by_text

    def __call__(
        self,
        batch,
        return_tensors="pt",
        padding=None,
        truncation=None,
        max_length=None,
        return_offsets_mapping=False,
    ):
        if isinstance(batch, str):
            texts = [batch]
            single = True
        else:
            texts = list(batch)
            single = False
        if return_offsets_mapping:
            encoded = {"input_ids": [], "offset_mapping": []}
            for text in texts:
                offsets = list(self.offsets_by_text[text])
                encoded["input_ids"].append(list(range(len(offsets))))
                encoded["offset_mapping"].append(offsets)
            if single:
                return {
                    "input_ids": encoded["input_ids"][0],
                    "offset_mapping": encoded["offset_mapping"][0],
                }
            return encoded
        self.lengths_by_prompt = {text: len(self.offsets_by_text[text]) for text in texts}
        return super().__call__(
            texts if not single else texts[0],
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )


def test_calibrate_load_prompt_records_extracts_stable_generation_ids(tmp_path) -> None:
    path = tmp_path / "calibration.jsonl"
    row = {
        "question": (
            "Tiffany was collecting cans for recycling. On monday she had 7 bags of cans. "
            "The next day she found 12 more bags worth of cans. How many more bags did she "
            "find on the next day than she had on monday?"
        ),
        "answer": "5",
        "aliases": ["5.0", "#### 5"],
        "metadata": {"id": "chal-31", "dataset": "SVAMP"},
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    records = calibrate.load_prompt_records(str(path))

    assert records == [(row["question"], {"chal-31", "013133cdef4f637c"})]


def test_calibrate_build_innovation_prompt_weights_marks_target_ids() -> None:
    weights, matched = calibrate.build_innovation_prompt_weights(
        [{"a"}, {"target", "metadata-id"}, set()],
        {"target"},
        positive_weight=8.0,
        default_weight=0.5,
    )

    assert matched == 1
    assert torch.allclose(weights, torch.tensor([0.5, 8.0, 0.5]))


def test_calibrate_build_innovation_prompt_weight_plan_marks_target_self_preserve_ids() -> None:
    weights, matched, preserve_mask, preserve_matched = calibrate.build_innovation_prompt_weight_plan(
        [{"a"}, {"target", "metadata-id"}, {"self-repair"}, set()],
        {"target"},
        positive_weight=8.0,
        default_weight=0.5,
        preserve_ids={"self-repair"},
        preserve_weight=4.0,
    )

    assert matched == 1
    assert preserve_matched == 1
    assert torch.allclose(weights, torch.tensor([0.5, 8.0, 4.0, 0.5]))
    assert torch.equal(preserve_mask, torch.tensor([False, False, True, False]))


class _FakeCache:
    def __init__(self, layers):
        self.layers = layers

    def to_legacy_cache(self):
        return self.layers


class _FakeModel:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, **enc):
        self.calls += 1
        input_ids = enc["input_ids"]
        batch, seq = input_ids.shape
        layers = []
        for layer_idx in range(2):
            values = input_ids.to(torch.float32).view(batch, 1, seq, 1)
            K = values + float(layer_idx * 100)
            V = values + float(layer_idx * 1000)
            layers.append((K, V, None))
        return SimpleNamespace(past_key_values=_FakeCache(layers))


class _FakeAttentionModel:
    def __init__(self) -> None:
        self.model = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    self_attn=SimpleNamespace(
                        q_proj=lambda hidden: hidden
                    )
                )
            ]
        )

    def __call__(self, **enc):
        input_ids = enc["input_ids"]
        batch, seq = input_ids.shape
        attn = torch.zeros(batch, 2, seq, seq, dtype=torch.float32)
        layers = []
        keys = torch.zeros(batch, 2, seq, 2, dtype=torch.float32)
        hidden = torch.zeros(batch, seq, 4, dtype=torch.float32)
        for batch_idx in range(batch):
            valid_len = int(enc["attention_mask"][batch_idx].sum().item())
            if valid_len <= 1:
                continue
            attn[batch_idx, 0, valid_len - 1, 0] = 1.0
            attn[batch_idx, 1, valid_len - 1, valid_len - 2] = 1.0
            positions = torch.arange(1, valid_len + 1, dtype=torch.float32)
            keys[batch_idx, 0, :valid_len, 0] = positions
            keys[batch_idx, 1, :valid_len, 1] = positions
            hidden[batch_idx, valid_len - 1] = torch.tensor([1.0, 0.0, 0.0, 2.0], dtype=torch.float32)
        layers.append((keys, keys + 0.5, None))
        return SimpleNamespace(attentions=(attn,), past_key_values=_FakeCache(layers), hidden_states=(hidden, hidden + 0.1))


def test_collect_kvs_masks_padding_tokens_and_concatenates_valid_positions() -> None:
    model = _FakeModel()
    tokenizer = _FakeTokenizer({"a": 4, "b": 2, "c": 1})
    prompts = ["a", "b", "c"]

    kvs = calibrate.collect_kvs(
        model,
        tokenizer,
        prompts,
        max_length=4,
        batch_size=2,
        device="cpu",
    )

    assert len(kvs) == 2
    for K, V in kvs:
        assert K.shape == (7, 1, 1, 1)
        assert V.shape == (7, 1, 1, 1)
        assert K.dtype == torch.float32
        assert V.dtype == torch.float32

    # The padded positions for prompts "b" and "c" are skipped entirely.
    assert kvs[0][0].flatten().tolist() == [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 1.0]
    assert kvs[0][1].flatten().tolist() == [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 1.0]


def test_collect_aligned_kv_pairs_truncates_to_shorter_valid_length() -> None:
    src_model = _FakeModel()
    tgt_model = _FakeModel()
    src_tokenizer = _FakeTokenizer({"a": 4, "b": 3})
    tgt_tokenizer = _FakeTokenizer({"a": 2, "b": 5})
    prompts = ["a", "b"]

    src_kvs, tgt_kvs = calibrate.collect_aligned_kv_pairs(
        src_model,
        src_tokenizer,
        tgt_model,
        tgt_tokenizer,
        prompts,
        max_length=5,
        batch_size=2,
        device="cpu",
    )

    assert len(src_kvs) == 2
    assert len(tgt_kvs) == 2
    for pair in src_kvs + tgt_kvs:
        K, V = pair
        assert K.shape == (5, 1, 1, 1)
        assert V.shape == (5, 1, 1, 1)

    assert src_kvs[0][0].flatten().tolist() == [1.0, 2.0, 1.0, 2.0, 3.0]
    assert tgt_kvs[0][0].flatten().tolist() == [1.0, 2.0, 1.0, 2.0, 3.0]


def test_collect_aligned_kv_pairs_uses_source_reasoning_prompt() -> None:
    src_model = _FakeModel()
    tgt_model = _FakeModel()
    prompt = "a"
    source_prompt = calibrate._source_reasoning_prompt(prompt, "cot")
    src_tokenizer = _FakeTokenizer({source_prompt: 4})
    tgt_tokenizer = _FakeTokenizer({prompt: 2})

    src_kvs, tgt_kvs = calibrate.collect_aligned_kv_pairs(
        src_model,
        src_tokenizer,
        tgt_model,
        tgt_tokenizer,
        [prompt],
        max_length=5,
        batch_size=1,
        device="cpu",
        source_reasoning_mode="cot",
    )

    assert src_kvs[0][0].shape == (2, 1, 1, 1)
    assert tgt_kvs[0][0].shape == (2, 1, 1, 1)


def test_collect_aligned_kv_pairs_uses_weighted_target_mixtures() -> None:
    src_model = _FakeModel()
    tgt_model = _FakeModel()
    src_tokenizer = _FakeTokenizer({"abc": 3})
    tgt_tokenizer = _FakeTokenizer({"abc": 3})

    src_kvs, tgt_kvs = calibrate.collect_aligned_kv_pairs(
        src_model,
        src_tokenizer,
        tgt_model,
        tgt_tokenizer,
        ["abc"],
        max_length=4,
        batch_size=1,
        device="cpu",
        aligned_position_mixtures=[[(1, (0, 2), (0.25, 0.75))]],
    )

    assert src_kvs[0][0].shape == (1, 1, 1, 1)
    assert tgt_kvs[0][0].shape == (1, 1, 1, 1)
    assert src_kvs[0][0].flatten().tolist() == [2.0]
    assert tgt_kvs[0][0].flatten().tolist() == [2.5]


def test_collect_aligned_prompt_valid_lengths_matches_pair_truncation() -> None:
    prompt = "a"
    source_prompt = calibrate._source_reasoning_prompt(prompt, "cot")
    source_prompt_b = calibrate._source_reasoning_prompt("b", "cot")
    src_tokenizer = _FakeTokenizer({source_prompt: 4, source_prompt_b: 3})
    tgt_tokenizer = _FakeTokenizer({prompt: 2, "b": 5})

    lengths = calibrate.collect_aligned_prompt_valid_lengths(
        src_tokenizer,
        tgt_tokenizer,
        [prompt, "b"],
        max_length=5,
        batch_size=2,
        source_reasoning_mode="cot",
    )

    assert lengths == [2, 3]


def test_prepare_prompt_text_supports_chat_template_and_enable_thinking() -> None:
    prompt = "a"
    source_prompt = calibrate._source_reasoning_prompt(prompt, "cot")
    tokenizer = _FakeChatTokenizer({f"chat::False::{source_prompt}": 4})

    formatted = calibrate._prepare_prompt_text(
        prompt,
        reasoning_mode="cot",
        tokenizer=tokenizer,
        use_chat_template=True,
        enable_thinking=False,
    )

    assert formatted == f"chat::False::{source_prompt}"
    assert tokenizer.chat_calls == [(source_prompt, False)]


def test_collect_group_attention_templates_supports_peak_mode() -> None:
    model = _FakeAttentionModel()
    tokenizer = _FakeTokenizer({"a": 4, "b": 4})

    templates = calibrate.collect_group_attention_templates(
        model,
        tokenizer,
        ["a", "b"],
        max_length=4,
        batch_size=2,
        device="cpu",
        kv_heads=2,
        group_count=2,
        bins=4,
        template_mode="peak",
    )

    assert len(templates) == 1
    layer0 = templates[0]
    assert layer0.shape == (2, 4)
    assert torch.allclose(layer0[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(layer0[1], torch.tensor([0.0, 0.0, 1.0, 0.0]))


def test_collect_group_attention_template_bank_preserves_prompt_axis() -> None:
    model = _FakeAttentionModel()
    tokenizer = _FakeTokenizer({"a": 4, "b": 3})

    banks = calibrate.collect_group_attention_template_bank(
        model,
        tokenizer,
        ["a", "b"],
        max_length=4,
        batch_size=2,
        device="cpu",
        kv_heads=2,
        group_count=2,
        bins=3,
        template_mode="peak",
    )

    assert len(banks) == 1
    layer0 = banks[0]
    assert layer0.shape == (2, 2, 3)
    assert torch.allclose(layer0[0, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(layer0[0, 1], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(layer0[1, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(layer0[1, 1], torch.tensor([0.0, 1.0, 0.0]))


def test_collect_group_key_signatures_returns_normalized_spectra() -> None:
    model = _FakeAttentionModel()
    tokenizer = _FakeTokenizer({"a": 4, "b": 4})

    signatures = calibrate.collect_group_key_signatures(
        model,
        tokenizer,
        ["a", "b"],
        max_length=4,
        batch_size=2,
        device="cpu",
        kv_heads=2,
        group_count=2,
        rank=2,
    )

    assert len(signatures) == 1
    layer0 = signatures[0]
    assert layer0.shape == (2, 2)
    assert torch.allclose(layer0.sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.all(layer0 >= 0.0)


def test_collect_group_qk_templates_returns_finite_profiles() -> None:
    model = _FakeAttentionModel()
    tokenizer = _FakeTokenizer({"a": 4, "b": 4})

    templates = calibrate.collect_group_qk_templates(
        model,
        tokenizer,
        ["a", "b"],
        max_length=4,
        batch_size=2,
        device="cpu",
        kv_heads=2,
        group_count=2,
        bins=4,
    )

    assert len(templates) == 1
    layer0 = templates[0]
    assert layer0.shape == (2, 4)
    assert torch.isfinite(layer0).all()
    assert layer0.abs().sum() > 0


def test_collect_group_qk_template_bank_preserves_prompt_axis() -> None:
    model = _FakeAttentionModel()
    tokenizer = _FakeTokenizer({"a": 4, "b": 4})

    bank = calibrate.collect_group_qk_template_bank(
        model,
        tokenizer,
        ["a", "b"],
        max_length=4,
        batch_size=2,
        device="cpu",
        kv_heads=2,
        group_count=2,
        bins=4,
    )

    assert len(bank) == 1
    layer0 = bank[0]
    assert layer0.shape == (2, 2, 4)
    assert torch.isfinite(layer0).all()
    assert layer0[0].abs().sum() > 0


def test_collect_aligned_qk_position_weights_matches_aligned_lengths() -> None:
    model = _FakeAttentionModel()
    tokenizer = _FakeTokenizer({"a": 4, "b": 3})

    weights = calibrate.collect_aligned_qk_position_weights(
        model,
        tokenizer,
        ["a", "b"],
        aligned_lengths=[3, 2],
        max_length=4,
        batch_size=2,
        device="cpu",
        kv_heads=2,
    )

    assert len(weights) == 1
    layer0 = weights[0]
    assert layer0.shape == (5,)
    assert torch.isfinite(layer0).all()
    assert bool((layer0 > 0).all())
    assert abs(float(layer0.mean().item()) - 1.0) < 1e-5


def test_collect_aligned_query_features_matches_aligned_lengths() -> None:
    model = _FakeAttentionModel()
    tokenizer = _FakeTokenizer({"a": 4, "b": 3})

    features = calibrate.collect_aligned_query_features(
        model,
        tokenizer,
        ["a", "b"],
        aligned_lengths=[4, 3],
        max_length=4,
        batch_size=2,
        device="cpu",
        kv_heads=2,
        head_dim=2,
    )

    assert len(features) == 1
    layer0 = features[0]
    assert layer0.shape == (7, 4)
    assert torch.isfinite(layer0).all()
    assert layer0.abs().sum() > 0


def test_calibrate_config_helpers_parse_and_batch() -> None:
    assert calibrate.torch_dtype("float32") is torch.float32
    assert calibrate.torch_dtype("float16") is torch.float16
    assert list(calibrate.batched([1, 2, 3, 4], 3)) == [[1, 2, 3], [4]]


def test_calibrate_parse_args_supports_unit_tested_ablation_flags(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--alignment",
            "reduced_rank",
            "--alignment-rank",
            "2",
            "--rotation",
            "dct",
            "--layer-pairing",
            "random",
            "--whitening",
            "--target-whitening",
            "--fit-ridge-override-lambda",
            "0.01",
            "--fit-ridge-override-streams",
            "v",
            "--fit-ridge-override-layer",
            "8",
            "--fit-ridge-override-layer",
            "10",
            "--fit-ridge-protected-rank",
            "2",
            "--source-reasoning-mode",
            "cot",
        ],
    )

    args = calibrate.parse_args()
    assert args.alignment == "reduced_rank"
    assert args.alignment_rank == 2
    assert args.rotation == "dct"
    assert args.layer_pairing == "random"
    assert args.whitening is True
    assert args.target_whitening is True
    assert args.fit_ridge_override_lambda == 0.01
    assert args.fit_ridge_override_streams == "v"
    assert args.fit_ridge_override_layers == [8, 10]
    assert args.fit_ridge_protected_rank == 2
    assert args.source_reasoning_mode == "cot"


def test_calibrate_parse_args_accepts_chat_template_and_thinking_flags(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--source-use-chat-template",
            "--target-use-chat-template",
            "--source-enable-thinking",
            "false",
            "--target-enable-thinking",
            "true",
        ],
    )

    args = calibrate.parse_args()
    assert args.source_use_chat_template is True
    assert args.target_use_chat_template is True
    assert args.source_enable_thinking == "false"
    assert args.target_enable_thinking == "true"


def test_calibrate_parse_args_accepts_grouped_alignment(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--alignment",
            "grouped_auto",
        ],
    )

    args = calibrate.parse_args()
    assert args.alignment == "grouped_auto"


def test_calibrate_parse_args_accepts_grouped_transport(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--alignment",
            "grouped_transport",
            "--transport-residual-rank",
            "8",
            "--transport-temperature",
            "0.5",
            "--transport-sinkhorn-iters",
            "12",
        ],
    )

    args = calibrate.parse_args()
    assert args.alignment == "grouped_transport"
    assert args.transport_residual_rank == 8
    assert args.transport_temperature == 0.5
    assert args.transport_sinkhorn_iters == 12


def test_calibrate_parse_args_accepts_grouped_permutation(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--alignment",
            "grouped_permutation",
        ],
    )

    args = calibrate.parse_args()
    assert args.alignment == "grouped_permutation"


def test_calibrate_parse_args_accepts_grouped_signature_transport(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--alignment",
            "grouped_signature_transport",
            "--transport-signature-rank",
            "6",
            "--transport-signature-weight",
            "0.25",
        ],
    )

    args = calibrate.parse_args()
    assert args.alignment == "grouped_signature_transport"
    assert args.transport_signature_rank == 6
    assert args.transport_signature_weight == 0.25


def test_calibrate_parse_args_accepts_grouped_subspace_transport(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--alignment",
            "grouped_subspace_transport",
            "--transport-signature-rank",
            "4",
            "--transport-signature-weight",
            "0.1",
        ],
    )

    args = calibrate.parse_args()
    assert args.alignment == "grouped_subspace_transport"
    assert args.transport_signature_rank == 4
    assert args.transport_signature_weight == 0.1


def test_calibrate_parse_args_accepts_grouped_canonical_transport(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--alignment",
            "grouped_canonical_transport",
            "--canonical-subspace-rank",
            "6",
        ],
    )

    args = calibrate.parse_args()
    assert args.alignment == "grouped_canonical_transport"
    assert args.canonical_subspace_rank == 6


def test_calibrate_parse_args_accepts_grouped_adaptive_canonical_transport(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--alignment",
            "grouped_adaptive_canonical_transport",
            "--canonical-subspace-rank",
            "6",
        ],
    )

    args = calibrate.parse_args()
    assert args.alignment == "grouped_adaptive_canonical_transport"
    assert args.canonical_subspace_rank == 6


def test_calibrate_parse_args_accepts_grouped_fitted_rotation_transport(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--alignment",
            "grouped_fitted_rotation_transport",
        ],
    )

    args = calibrate.parse_args()
    assert args.alignment == "grouped_fitted_rotation_transport"


def test_calibrate_parse_args_accepts_grouped_shared_basis_transport(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--alignment",
            "grouped_shared_basis_transport",
        ],
    )

    args = calibrate.parse_args()
    assert args.alignment == "grouped_shared_basis_transport"


def test_calibrate_parse_args_accepts_grouped_covariance_transport(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--alignment",
            "grouped_covariance_transport",
            "--transport-signature-weight",
            "0.1",
        ],
    )

    args = calibrate.parse_args()
    assert args.alignment == "grouped_covariance_transport"


def test_calibrate_parse_args_accepts_grouped_template_transport(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--alignment",
            "grouped_template_transport",
            "--transport-template-bins",
            "32",
        ],
    )

    args = calibrate.parse_args()
    assert args.alignment == "grouped_template_transport"
    assert args.transport_template_bins == 32
    assert args.transport_signature_weight == 0.0


def test_calibrate_parse_args_accepts_grouped_qk_retrieval_transport(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--alignment",
            "grouped_qk_retrieval_transport",
            "--transport-template-bins",
            "28",
            "--transport-signature-weight",
            "0.15",
        ],
    )

    args = calibrate.parse_args()
    assert args.alignment == "grouped_qk_retrieval_transport"
    assert args.transport_template_bins == 28
    assert args.transport_signature_weight == 0.15


def test_calibrate_parse_args_accepts_grouped_contrastive_template_transport(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--alignment",
            "grouped_contrastive_template_transport",
            "--transport-template-bins",
            "24",
            "--transport-signature-weight",
            "0.2",
        ],
    )

    args = calibrate.parse_args()
    assert args.alignment == "grouped_contrastive_template_transport"
    assert args.transport_template_bins == 24
    assert args.transport_signature_weight == 0.2


def test_calibrate_parse_args_accepts_grouped_template_subspace_transport(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--alignment",
            "grouped_template_subspace_transport",
            "--transport-template-bins",
            "48",
        ],
    )

    args = calibrate.parse_args()
    assert args.alignment == "grouped_template_subspace_transport"
    assert args.transport_template_bins == 48


def test_calibrate_parse_args_accepts_broadcast_template_transport(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--alignment",
            "broadcast_template_transport",
            "--transport-template-bins",
            "40",
        ],
    )

    args = calibrate.parse_args()
    assert args.alignment == "broadcast_template_transport"
    assert args.transport_template_bins == 40


def test_calibrate_parse_args_accepts_broadcast_template_ot_transport(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--alignment",
            "broadcast_template_ot_transport",
            "--transport-template-bins",
            "32",
        ],
    )

    args = calibrate.parse_args()
    assert args.alignment == "broadcast_template_ot_transport"
    assert args.transport_template_bins == 32


def test_calibrate_parse_args_accepts_broadcast_peak_template_ot_transport(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--alignment",
            "broadcast_peak_template_ot_transport",
            "--transport-template-bins",
            "24",
        ],
    )

    args = calibrate.parse_args()
    assert args.alignment == "broadcast_peak_template_ot_transport"


def test_calibrate_parse_args_accepts_broadcast_retrieval_spectrum_ot_transport(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--alignment",
            "broadcast_retrieval_spectrum_ot_transport",
            "--transport-signature-rank",
            "6",
            "--transport-signature-weight",
            "0.2",
        ],
    )

    args = calibrate.parse_args()
    assert args.alignment == "broadcast_retrieval_spectrum_ot_transport"
    assert args.transport_signature_rank == 6
    assert args.transport_signature_weight == 0.2


def test_calibrate_parse_args_accepts_broadcast_qk_template_ot_transport(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--alignment",
            "broadcast_qk_template_ot_transport",
            "--transport-template-bins",
            "16",
            "--transport-signature-weight",
            "0.15",
        ],
    )

    args = calibrate.parse_args()
    assert args.alignment == "broadcast_qk_template_ot_transport"
    assert args.transport_template_bins == 16
    assert args.transport_signature_weight == 0.15


def test_calibrate_parse_args_supports_head_and_prequant_flags(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--head-selection-topk",
            "1",
            "--head-selection-ratio",
            "0.5",
            "--head-selection-metric",
            "negative_error",
            "--pre-quant-rank",
            "64",
            "--pre-quant-shrinkage",
            "0.25",
            "--quantization-correction",
            "affine",
        ],
    )

    args = calibrate.parse_args()
    assert args.head_selection_topk == 1
    assert args.head_selection_ratio == 0.5
    assert args.head_selection_metric == "negative_error"
    assert args.pre_quant_rank == 64
    assert args.pre_quant_shrinkage == 0.25
    assert args.quantization_correction == "affine"


def test_calibrate_parse_args_accepts_ridge_quantization_correction(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "ridge",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "ridge"


def test_calibrate_parse_args_accepts_bridge_affine_quantization_correction(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_affine",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_affine"


def test_calibrate_parse_args_accepts_bridge_ridge_quantization_correction(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_ridge"


def test_calibrate_parse_args_accepts_bridge_ridge_query_quantization_correction(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_query",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_ridge_query"


def test_calibrate_parse_args_accepts_bridge_ridge_qk_readout_adapter(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_readout_adapter",
            "--quantization-correction-rank",
            "8",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_ridge_qk_readout_adapter"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_bridge_ridge_qk_predkl_adapter(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_predkl_adapter",
            "--quantization-correction-rank",
            "8",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_ridge_qk_predkl_adapter"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_bridge_ridge_qk_asym_adapter(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_asym_adapter",
            "--quantization-correction-rank",
            "8",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_ridge_qk_asym_adapter"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_bridge_ridge_qk_asym_projector(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_asym_projector",
            "--quantization-correction-rank",
            "8",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_ridge_qk_asym_projector"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_bridge_ridge_qk_asym_predkl_adapter(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_asym_predkl_adapter",
            "--quantization-correction-rank",
            "8",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_ridge_qk_asym_predkl_adapter"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_bridge_ridge_qk_asym_dynmap_adapter(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_asym_dynmap_adapter",
            "--quantization-correction-rank",
            "8",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_ridge_qk_asym_dynmap_adapter"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_bridge_ridge_qk_xattn_adapter(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_xattn_adapter",
            "--quantization-correction-rank",
            "8",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_ridge_qk_xattn_adapter"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_bridge_ridge_qk_xattn_dynmap_adapter(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_xattn_dynmap_adapter",
            "--quantization-correction-rank",
            "8",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_ridge_qk_xattn_dynmap_adapter"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_bridge_ridge_qk_module_adapter(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_module_adapter",
            "--quantization-correction-rank",
            "8",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_ridge_qk_module_adapter"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_bridge_ridge_qk_module_replace(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_module_replace",
            "--quantization-correction-rank",
            "8",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_ridge_qk_module_replace"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_bridge_ridge_qk_spanalign_module_replace(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_spanalign_module_replace",
            "--quantization-correction-rank",
            "8",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_ridge_qk_spanalign_module_replace"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_bridge_ridge_qk_bytespan_module_replace(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_bytespan_module_replace",
            "--quantization-correction-rank",
            "8",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_ridge_qk_bytespan_module_replace"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_bridge_ridge_qk_tokenbasis_replace(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_tokenbasis_replace",
            "--quantization-correction-rank",
            "8",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_ridge_qk_tokenbasis_replace"
    assert args.quantization_correction_rank == 8


def test_collect_aligned_prompt_position_pairs_aligns_raw_prompt_content() -> None:
    tok = _FakeOffsetTokenizer()
    prompt = "abc"
    pairs = calibrate.collect_aligned_prompt_position_pairs(
        tok,
        tok,
        [prompt],
        max_length=128,
        batch_size=1,
        source_reasoning_mode="brief_analysis",
        source_use_chat_template=False,
        source_enable_thinking=None,
        target_use_chat_template=False,
        target_enable_thinking=None,
    )

    src_text = calibrate._prepare_prompt_text(
        prompt,
        reasoning_mode="brief_analysis",
        tokenizer=tok,
        use_chat_template=False,
        enable_thinking=None,
    )
    src_start = src_text.find(prompt)

    assert pairs == [[(src_start + 0, 0), (src_start + 1, 1), (src_start + 2, 2)]]


def test_collect_byte_aligned_prompt_position_pairs_prefers_utf8_byte_mass() -> None:
    prompt = "aé"
    src_tok = _ScriptedOffsetTokenizer({prompt: [(0, 2)]})
    tgt_tok = _ScriptedOffsetTokenizer({prompt: [(0, 1), (1, 2)]})

    pairs = calibrate.collect_byte_aligned_prompt_position_pairs(
        src_tok,
        tgt_tok,
        [prompt],
        max_length=128,
        batch_size=1,
        source_reasoning_mode="plain",
        source_use_chat_template=False,
        source_enable_thinking=None,
        target_use_chat_template=False,
        target_enable_thinking=None,
    )

    assert pairs == [[(0, 1)]]


def test_collect_contextual_prompt_position_mixtures_tracks_split_target_tokens() -> None:
    prompt = "abcd"
    src_tok = _ScriptedOffsetTokenizer({prompt: [(0, 2), (2, 4)]})
    tgt_tok = _ScriptedOffsetTokenizer({prompt: [(0, 1), (1, 2), (2, 3), (3, 4)]})

    mixtures = calibrate.collect_contextual_prompt_position_mixtures(
        src_tok,
        tgt_tok,
        [prompt],
        max_length=128,
        batch_size=1,
        source_reasoning_mode="plain",
        source_use_chat_template=False,
        source_enable_thinking=None,
        target_use_chat_template=False,
        target_enable_thinking=None,
        max_targets_per_source=2,
    )

    assert len(mixtures) == 1
    assert len(mixtures[0]) == 2
    first_src_pos, first_targets, first_weights = mixtures[0][0]
    second_src_pos, second_targets, second_weights = mixtures[0][1]
    assert first_src_pos == 0
    assert second_src_pos == 1
    assert set(first_targets) == {0, 1}
    assert set(second_targets) == {2, 3}
    assert abs(sum(first_weights) - 1.0) < 1e-6
    assert abs(sum(second_weights) - 1.0) < 1e-6


def test_prediction_token_overlap_score_prefers_matching_text() -> None:
    match = calibrate._prediction_token_overlap_score(
        ("ab", "zz"),
        (-0.01, -4.0),
        ("ab", "yy"),
        (-0.02, -4.0),
    )
    mismatch = calibrate._prediction_token_overlap_score(
        ("ab", "zz"),
        (-0.01, -4.0),
        ("cd", "yy"),
        (-0.02, -4.0),
    )

    assert match > mismatch


def test_collect_dynamic_prompt_position_mixtures_uses_prediction_overlap(monkeypatch) -> None:
    prompt = "abcd"
    src_tok = _ScriptedOffsetTokenizer({prompt: [(0, 2), (2, 4)]})
    tgt_tok = _ScriptedOffsetTokenizer({prompt: [(0, 1), (1, 2), (2, 3), (3, 4)]})

    def fake_signatures(model, tokenizer, prompt_text, *, max_length, device, topk):
        del model, max_length, device, topk
        if tokenizer is src_tok:
            return [
                (("ab",), (0.0,)),
                (("cd",), (0.0,)),
            ]
        return [
            (("a",), (0.0,)),
            (("ab",), (0.0,)),
            (("c",), (0.0,)),
            (("cd",), (0.0,)),
        ]

    monkeypatch.setattr(calibrate, "_collect_prompt_prediction_signatures", fake_signatures)

    mixtures = calibrate.collect_dynamic_prompt_position_mixtures(
        object(),
        src_tok,
        object(),
        tgt_tok,
        [prompt],
        max_length=128,
        batch_size=1,
        device="cpu",
        source_reasoning_mode="plain",
        source_use_chat_template=False,
        source_enable_thinking=None,
        target_use_chat_template=False,
        target_enable_thinking=None,
        max_targets_per_source=1,
    )

    assert mixtures == [[(0, (1,), (1.0,)), (1, (3,), (1.0,))]]


def test_collect_alignment_confidence_weights_prefers_concentrated_mixtures() -> None:
    weights = calibrate.collect_alignment_confidence_weights(
        [
            [(0, (1,), (1.0,))],
            [(0, (1, 2), (0.5, 0.5))],
        ]
    )

    assert weights.shape == (2,)
    assert float(weights[0].item()) > float(weights[1].item())


def test_collect_prediction_confidence_weights_prefers_low_entropy_teacher() -> None:
    teacher_log_probs = torch.log(
        torch.tensor(
            [
                [0.95, 0.05],
                [0.5, 0.5],
            ],
            dtype=torch.float32,
        )
    )

    weights = calibrate.collect_prediction_confidence_weights(teacher_log_probs)

    assert weights.shape == (2,)
    assert float(weights[0].item()) > float(weights[1].item())


def test_align_score_matrix_monotone_prefers_global_monotone_path() -> None:
    score_matrix = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [0.0, 0.5, 2.0],
        ],
        dtype=torch.float32,
    )

    pairs = calibrate._align_score_matrix_monotone(score_matrix, skip_penalty=0.25)

    assert pairs == [(0, 0), (1, 2)]


def test_collect_dynamic_program_prompt_position_pairs_uses_prediction_overlap(monkeypatch) -> None:
    prompt = "abcd"
    src_tok = _ScriptedOffsetTokenizer({prompt: [(0, 2), (2, 4)]})
    tgt_tok = _ScriptedOffsetTokenizer({prompt: [(0, 1), (1, 2), (2, 3), (3, 4)]})

    def fake_signatures(model, tokenizer, prompt_text, *, max_length, device, topk):
        del model, max_length, device, topk
        if tokenizer is src_tok:
            return [
                (("ab",), (0.0,)),
                (("cd",), (0.0,)),
            ]
        return [
            (("a",), (0.0,)),
            (("ab",), (0.0,)),
            (("c",), (0.0,)),
            (("cd",), (0.0,)),
        ]

    monkeypatch.setattr(calibrate, "_collect_prompt_prediction_signatures", fake_signatures)

    pairs = calibrate.collect_dynamic_program_prompt_position_pairs(
        object(),
        src_tok,
        object(),
        tgt_tok,
        [prompt],
        max_length=128,
        batch_size=1,
        device="cpu",
        source_reasoning_mode="plain",
        source_use_chat_template=False,
        source_enable_thinking=None,
        target_use_chat_template=False,
        target_enable_thinking=None,
    )

    assert pairs == [[(0, 1), (1, 3)]]


def test_calibrate_parse_args_accepts_bridge_ridge_qk_ctxalign_module_replace(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_ctxalign_module_replace",
            "--quantization-correction-rank",
            "8",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_ridge_qk_ctxalign_module_replace"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_bridge_ridge_qk_dynalign_module_replace(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_dynalign_module_replace",
            "--quantization-correction-rank",
            "8",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_ridge_qk_dynalign_module_replace"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_bridge_ridge_qk_dynalign_preserve_module_replace(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_dynalign_preserve_module_replace",
            "--quantization-correction-rank",
            "8",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_ridge_qk_dynalign_preserve_module_replace"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_bridge_ridge_qk_dynalign_eigenspace_module_replace(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_dynalign_eigenspace_module_replace",
            "--quantization-correction-rank",
            "8",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_ridge_qk_dynalign_eigenspace_module_replace"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_bridge_ridge_qk_dynalign_saliency_module_replace(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_dynalign_saliency_module_replace",
            "--quantization-correction-rank",
            "8",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_ridge_qk_dynalign_saliency_module_replace"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_bridge_ridge_qk_dynalign_saliency_preserve_module_replace(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_dynalign_saliency_preserve_module_replace",
            "--quantization-correction-rank",
            "8",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_ridge_qk_dynalign_saliency_preserve_module_replace"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_bridge_ridge_qk_dynalign_anchor_tail_module_replace(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_dynalign_anchor_tail_module_replace",
            "--quantization-correction-rank",
            "8",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_ridge_qk_dynalign_anchor_tail_module_replace"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_bridge_ridge_qk_dynalign_v8_outlier_escrow_module_replace(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_dynalign_v8_outlier_escrow_module_replace",
            "--quantization-correction-rank",
            "8",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_ridge_qk_dynalign_v8_outlier_escrow_module_replace"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_bridge_ridge_qk_dynalign_routed_module_replace(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_dynalign_routed_module_replace",
            "--quantization-correction-rank",
            "8",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_ridge_qk_dynalign_routed_module_replace"


def test_calibrate_parse_args_accepts_bridge_ridge_qk_dynalign_value_routed_module_replace(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_dynalign_value_routed_module_replace",
            "--quantization-correction-rank",
            "8",
        ],
    )
    args = calibrate.parse_args()
    assert args.quantization_correction == "bridge_ridge_qk_dynalign_value_routed_module_replace"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_bridge_ridge_qk_dynalign_query_resampler_replace(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_dynalign_query_resampler_replace",
            "--quantization-correction-rank",
            "16",
        ],
    )
    args = calibrate.parse_args()
    assert args.quantization_correction == "bridge_ridge_qk_dynalign_query_resampler_replace"
    assert args.quantization_correction_rank == 16


def test_calibrate_parse_args_accepts_bridge_ridge_qk_dynalign_query_innovation_resampler_replace(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_dynalign_query_innovation_resampler_replace",
            "--quantization-correction-rank",
            "16",
            "--innovation-target-set-json",
            "targets.json",
            "--innovation-positive-weight",
            "12",
            "--innovation-default-weight",
            "0.75",
            "--innovation-control-weight",
            "0.5",
            "--innovation-control-mode",
            "zero_and_shuffle",
            "--innovation-contrastive-margin",
            "0.01",
            "--innovation-target-self-preserve-weight",
            "6",
            "--innovation-value-loss-weight",
            "0.25",
            "--innovation-conditional-target-memory",
            "--innovation-conditional-delta-memory",
        ],
    )
    args = calibrate.parse_args()
    assert args.quantization_correction == "bridge_ridge_qk_dynalign_query_innovation_resampler_replace"
    assert args.quantization_correction_rank == 16
    assert args.innovation_target_set_json == "targets.json"
    assert args.innovation_positive_weight == 12
    assert args.innovation_default_weight == 0.75
    assert args.innovation_control_weight == 0.5
    assert args.innovation_control_mode == "zero_and_shuffle"
    assert args.innovation_contrastive_margin == 0.01
    assert args.innovation_target_self_preserve_weight == 6
    assert args.innovation_value_loss_weight == 0.25
    assert args.innovation_conditional_target_memory is True
    assert args.innovation_conditional_delta_memory is True


def test_calibrate_parse_args_accepts_bridge_ridge_qk_dynalign_value_bank_module_replace(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_dynalign_value_bank_module_replace",
            "--quantization-correction-rank",
            "8",
        ],
    )
    args = calibrate.parse_args()
    assert args.quantization_correction == "bridge_ridge_qk_dynalign_value_bank_module_replace"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_bridge_ridge_qk_dynalign_value_query_bank_module_replace(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_dynalign_value_query_bank_module_replace",
            "--quantization-correction-rank",
            "8",
        ],
    )
    args = calibrate.parse_args()
    assert args.quantization_correction == "bridge_ridge_qk_dynalign_value_query_bank_module_replace"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_bridge_ridge_qk_dynalign_value_routed_bank_module_replace(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_dynalign_value_routed_bank_module_replace",
            "--quantization-correction-rank",
            "8",
        ],
    )
    args = calibrate.parse_args()
    assert args.quantization_correction == "bridge_ridge_qk_dynalign_value_routed_bank_module_replace"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_bridge_ridge_qk_dynalign_value_verifier_sidecar_module_replace(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_dynalign_value_verifier_sidecar_module_replace",
            "--quantization-correction-rank",
            "8",
        ],
    )
    args = calibrate.parse_args()
    assert args.quantization_correction == "bridge_ridge_qk_dynalign_value_verifier_sidecar_module_replace"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_bridge_ridge_qk_dynalign_ctxonly_module_replace(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_dynalign_ctxonly_module_replace",
            "--quantization-correction-rank",
            "8",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_ridge_qk_dynalign_ctxonly_module_replace"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_bridge_ridge_qk_dynalign_dwakd_module_replace(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_dynalign_dwakd_module_replace",
            "--quantization-correction-rank",
            "8",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_ridge_qk_dynalign_dwakd_module_replace"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_bridge_ridge_qk_dynalign_likelihood_module_replace(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_dynalign_likelihood_module_replace",
            "--quantization-correction-rank",
            "8",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_ridge_qk_dynalign_likelihood_module_replace"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_bridge_ridge_qk_dynalign_spanalm_module_replace(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_dynalign_spanalm_module_replace",
            "--quantization-correction-rank",
            "8",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_ridge_qk_dynalign_spanalm_module_replace"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_bridge_ridge_qk_dynalign_prefdist_module_replace(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_dynalign_prefdist_module_replace",
            "--quantization-correction-rank",
            "8",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_ridge_qk_dynalign_prefdist_module_replace"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_bridge_ridge_qk_dynalign_dwainteract_module_replace(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_dynalign_dwainteract_module_replace",
            "--quantization-correction-rank",
            "8",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_ridge_qk_dynalign_dwainteract_module_replace"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_bridge_ridge_qk_dynalign_interact_module_replace(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_dynalign_interact_module_replace",
            "--quantization-correction-rank",
            "8",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_ridge_qk_dynalign_interact_module_replace"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_bridge_ridge_qk_dpalign_module_replace(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_dpalign_module_replace",
            "--quantization-correction-rank",
            "8",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_ridge_qk_dpalign_module_replace"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_bridge_ridge_qk_sae_adapter(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "calibration.txt",
            "--output",
            "translator.pt",
            "--quantization-correction",
            "bridge_ridge_qk_sae_adapter",
            "--quantization-correction-rank",
            "8",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_ridge_qk_sae_adapter"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_bridge_ridge_qk_generated_adapter(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "calibration.txt",
            "--output",
            "translator.pt",
            "--quantization-correction",
            "bridge_ridge_qk_generated_adapter",
            "--quantization-correction-rank",
            "8",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_ridge_qk_generated_adapter"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_bridge_ridge_qk_predkl_bank(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "bridge_ridge_qk_predkl_bank",
            "--quantization-correction-rank",
            "8",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "bridge_ridge_qk_predkl_bank"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_low_rank_quantization_correction(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "low_rank",
            "--quantization-correction-rank",
            "8",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "low_rank"
    assert args.quantization_correction_rank == 8


def test_calibrate_parse_args_accepts_learned_fusion_dropout(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--learned-fusion-dropout",
            "0.5",
        ],
    )

    args = calibrate.parse_args()
    assert args.learned_fusion_dropout == 0.5


def test_ablation_sweep_parse_accuracies_and_main_plumbing(monkeypatch, tmp_path) -> None:
    summary = """
    noise before
    === Summary ===
      target_alone: 0.250
      text_to_text: 0.500
      rotalign_kv: 0.750
    trailing noise
    """
    parsed = sweep.parse_accuracies(summary)
    assert parsed == {
        "target_alone": 0.25,
        "text_to_text": 0.5,
        "rotalign_kv": 0.75,
    }

    out_path = tmp_path / "results.jsonl"
    ckpt_dir = tmp_path / "ckpts"
    args = Namespace(
        source_model="src-model",
        target_model="tgt-model",
        calibration_file="cal.txt",
        eval_file="eval.jsonl",
        output=str(out_path),
        checkpoint_dir=str(ckpt_dir),
        rotations=["orthogonal"],
        alignments=["procrustes"],
        bits=[4],
        whiten=["on"],
        device="cpu",
        dtype="float32",
        head_selection_topks=[1],
        head_selection_ratios=[0.5],
        head_selection_metrics=["negative_error"],
        pre_quant_ranks=[64],
        pre_quant_shrinkages=[0.25],
        quantization_corrections=["affine"],
        source_reasoning_modes=["plain", "cot"],
        kv_transports=["k_only"],
    )
    monkeypatch.setattr(sweep, "parse_args", lambda: args)

    commands = []

    def fake_run_cmd(cmd):
        commands.append(cmd)
        if any(part.endswith("evaluate.py") for part in cmd):
            return """
            === Summary ===
              target_alone: 0.100
              text_to_text: 0.200
              rotalign_kv: 0.300
            """
        return "calibrated"

    monkeypatch.setattr(sweep, "run_cmd", fake_run_cmd)
    times = iter([10.0, 16.0, 20.0, 26.0, 30.0, 36.0, 40.0, 46.0, 50.0, 56.0])
    monkeypatch.setattr(sweep.time, "time", lambda: next(times))

    sweep.main()

    assert len(commands) == 4
    assert any(part.endswith("scripts/calibrate.py") for part in commands[0])
    assert any(part == "--whitening" for part in commands[0])
    assert any(part == "--head-selection-topk" for part in commands[0])
    assert any(part == "--head-selection-ratio" for part in commands[0])
    assert any(part == "--head-selection-metric" for part in commands[0])
    assert any(part == "--pre-quant-rank" for part in commands[0])
    assert any(part == "--quantization-correction" for part in commands[0])
    assert any(part == "--source-reasoning-mode" for part in commands[0])
    assert any(part.endswith("scripts/evaluate.py") for part in commands[1])
    assert any(part == "--kv-transport" for part in commands[1])
    assert any(part == "--source-reasoning-mode" for part in commands[1])

    records = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert records[0]["rotation"] == "orthogonal"
    assert records[0]["alignment"] == "procrustes"
    assert records[0]["bits"] == 4
    assert records[0]["whitening"] is True
    assert records[0]["head_selection_topk"] == 1
    assert records[0]["head_selection_ratio"] == 0.5
    assert records[0]["head_selection_metric"] == "negative_error"
    assert records[0]["pre_quant_rank"] == 64
    assert records[0]["quantization_correction"] == "affine"
    assert records[0]["kv_transport"] == "k_only"
    assert records[0]["rotalign_kv"] == 0.3
    assert {record["source_reasoning_mode"] for record in records} == {"plain", "cot"}
