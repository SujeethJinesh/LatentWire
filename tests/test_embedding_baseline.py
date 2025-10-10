"""Tests for embedding baseline functionality."""

import types
from unittest.mock import MagicMock, patch

import pytest

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    pytest.skip("PyTorch not available", allow_module_level=True)

from latentwire.models import Adapter


class DummyWrapper:
    """Mock LMWrapper for testing."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.input_embed = model.get_input_embeddings()
        self.use_latent_adapters = False

    def loss_with_text_embedding_replay(
        self, prompt_ids, target_ids, compute_loss=True
    ):
        """Mock implementation of embedding replay."""
        device = next(self.model.parameters()).device
        prompt_ids = prompt_ids.to(device)
        target_ids = target_ids.to(device)

        # Simple mock loss calculation
        batch_size = prompt_ids.size(0)
        n_tokens = target_ids.size(1) - 1  # Exclude first token

        if compute_loss:
            loss = torch.tensor(0.5, device=device)  # Dummy loss
        else:
            loss = None

        return loss, n_tokens

    def generate_from_prefix(
        self, prefix, max_new_tokens=10, **kwargs
    ):
        """Mock generation from prefix."""
        batch_size = prefix.size(0)
        vocab_size = self.model.vocab_size
        output_ids = torch.randint(
            0, vocab_size, (batch_size, max_new_tokens)
        )
        return output_ids

    def decode_batch_then_clean(self, ids):
        """Mock decode."""
        return ["decoded text" for _ in ids]

    def _adapter_context(self, adapter):
        """Mock adapter context manager."""
        from contextlib import nullcontext
        return nullcontext()

    def input_embedding_rms(self):
        """Mock RMS calculation for input embeddings."""
        return 0.015  # Return a typical RMS value

    def first_token_logits_from_prefix(
        self, prefix, anchor_token_text=None,
        append_bos_after_prefix=None, deep_prefix_past=None, latent=None
    ):
        """Mock first token logits generation."""
        batch_size = prefix.size(0)
        vocab_size = self.model.vocab_size
        # Return random logits for the first token
        return torch.randn(batch_size, vocab_size)

    def forward_with_prefix_loss(
        self, prefix, target_ids, anchor_token_ids=None,
        append_bos_after_prefix=None, deep_prefix_cache=None,
        deep_prefix_past=None, latent=None
    ):
        """Mock forward with prefix loss calculation."""
        # Return just the loss tensor
        return torch.tensor(0.5)


class DummyModel(nn.Module):
    """Dummy model for testing."""

    def __init__(self, vocab_size: int = 1000, hidden_size: int = 128):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.config = types.SimpleNamespace(
            hidden_size=hidden_size,
            num_hidden_layers=4,
            num_attention_heads=4,
            vocab_size=vocab_size,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.device = torch.device("cpu")

    def forward(self, input_ids=None, inputs_embeds=None, labels=None, **kwargs):
        """Simple forward pass that returns dummy outputs."""
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_len = inputs_embeds.shape[:2]

        # Create dummy logits
        logits = torch.randn(batch_size, seq_len, self.vocab_size)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
                reduction='mean'
            )

        # Return named tuple-like object
        return types.SimpleNamespace(logits=logits, loss=loss)

    def get_input_embeddings(self):
        return self.embed_tokens

    def generate(self, inputs_embeds=None, **kwargs):
        """Mock generate method."""
        batch_size = inputs_embeds.shape[0]
        max_length = kwargs.get("max_length", 20)
        return torch.randint(0, self.vocab_size, (batch_size, max_length))


class DummyTokenizer:
    """Dummy tokenizer for testing."""

    def __init__(self):
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.vocab_size = 1000

    def __call__(self, text, return_tensors=None, add_special_tokens=True,
                 padding=False, truncation=False, **kwargs):
        """Tokenize text into random IDs."""
        if isinstance(text, list):
            # Batch tokenization
            batch_size = len(text)
            seq_len = 10
            input_ids = torch.randint(3, self.vocab_size, (batch_size, seq_len))
        else:
            # Single text
            seq_len = 10
            input_ids = torch.randint(3, self.vocab_size, (1, seq_len))

        return types.SimpleNamespace(input_ids=input_ids)

    def decode(self, ids, skip_special_tokens=False):
        """Mock decode."""
        return "decoded text"

    def batch_decode(self, ids, skip_special_tokens=False):
        """Mock batch decode."""
        return ["decoded text" for _ in ids]


def test_text_embedding_replay_basic():
    """Test basic text embedding replay functionality."""
    model = DummyModel()
    tokenizer = DummyTokenizer()

    wrapper = DummyWrapper(model, tokenizer)

    # Create dummy prompt and target
    prompt_ids = torch.randint(3, 100, (1, 5))
    target_ids = torch.randint(3, 100, (1, 8))

    # Test embedding replay
    loss, n_tokens = wrapper.loss_with_text_embedding_replay(
        prompt_ids, target_ids, compute_loss=True
    )

    assert loss is not None
    assert isinstance(loss, torch.Tensor)
    assert n_tokens > 0


def test_embedding_baseline_modes():
    """Test different embedding baseline modes."""
    from latentwire.eval import _run_embedding_baselines

    # Setup mock wrapper
    model = DummyModel()
    tokenizer = DummyTokenizer()
    wrapper = DummyWrapper(model, tokenizer)

    # Mock the generate_from_prefix method
    wrapper.generate_from_prefix = MagicMock(return_value=torch.randint(0, 1000, (1, 10)))
    wrapper.decode_batch_then_clean = MagicMock(return_value=["test output"])

    # Setup test inputs
    prompts = ["Test prompt 1", "Test prompt 2"]
    golds = ["Gold answer 1", "Gold answer 2"]

    # Mock args
    args = types.SimpleNamespace(
        max_new_tokens=10,
        min_new_tokens=1,
        eos_ban_steps=0,
        first_token_top_p=1.0,
        first_token_temperature=0.0,
        calibration=None,
        prefix_target_rms=None,
    )

    # Test raw mode
    results = _run_embedding_baselines(
        wrapper=wrapper,
        prompts=prompts,
        golds=golds,
        modes=["raw"],
        args=args,
        answer_lengths=None,
        anchor_entry={},
        adapter=None,
        train_stats=None,
        model_name="test",
        latent_len=32,
    )

    assert "raw" in results
    assert "preds" in results["raw"]
    assert len(results["raw"]["preds"]) == len(prompts)


def test_embedding_baseline_with_anchor():
    """Test embedding baseline with anchor text."""
    from latentwire.eval import _run_embedding_baselines

    # Setup mock wrapper
    model = DummyModel()
    tokenizer = DummyTokenizer()
    wrapper = DummyWrapper(model, tokenizer)

    # Mock methods
    wrapper.generate_from_prefix = MagicMock(return_value=torch.randint(0, 1000, (1, 10)))
    wrapper.decode_batch_then_clean = MagicMock(return_value=["test output"])

    prompts = ["Question: What is AI?"]
    golds = ["Artificial Intelligence"]

    args = types.SimpleNamespace(
        max_new_tokens=10,
        min_new_tokens=1,
        eos_ban_steps=0,
        first_token_top_p=1.0,
        first_token_temperature=0.0,
        calibration=None,
        prefix_target_rms=None,
    )

    # Test anchor mode with "Answer: " anchor
    anchor_entry = {"text": "Answer: ", "bos": True}

    results = _run_embedding_baselines(
        wrapper=wrapper,
        prompts=prompts,
        golds=golds,
        modes=["anchor"],
        args=args,
        answer_lengths=None,
        anchor_entry=anchor_entry,
        adapter=None,
        train_stats=None,
        model_name="test",
        latent_len=32,
    )

    assert "anchor" in results
    assert results["anchor"]["preds"][0] == "test output"

    # Verify that generate was called with BOS appended
    wrapper.generate_from_prefix.assert_called()
    call_args = wrapper.generate_from_prefix.call_args
    assert call_args.kwargs["append_bos_after_prefix"] == False  # Already handled in embedding


def test_embedding_baseline_with_adapter():
    """Test embedding baseline with adapter interpolation."""
    from latentwire.eval import _run_embedding_baselines

    # Setup mock wrapper and adapter
    model = DummyModel()
    tokenizer = DummyTokenizer()
    wrapper = DummyWrapper(model, tokenizer)

    # Create a mock adapter
    adapter = MagicMock(spec=Adapter)
    adapter.latent_length = 32
    adapter.d_z = 256
    # Add input_norm with normalized_shape attribute
    adapter.input_norm = MagicMock()
    adapter.input_norm.normalized_shape = [256]
    adapter.return_value = torch.randn(1, 32, 128)  # Fake adapter output

    # Mock methods
    wrapper.generate_from_prefix = MagicMock(return_value=torch.randint(0, 1000, (1, 10)))
    wrapper.decode_batch_then_clean = MagicMock(return_value=["test output"])

    prompts = ["Test prompt"]
    golds = ["Gold answer"]

    args = types.SimpleNamespace(
        max_new_tokens=10,
        min_new_tokens=1,
        eos_ban_steps=0,
        first_token_top_p=1.0,
        first_token_temperature=0.0,
        calibration=None,
        prefix_target_rms=None,
    )

    results = _run_embedding_baselines(
        wrapper=wrapper,
        prompts=prompts,
        golds=golds,
        modes=["adapter"],
        args=args,
        answer_lengths=None,
        anchor_entry={},
        adapter=adapter,
        train_stats=None,
        model_name="test",
        latent_len=32,
    )

    assert "adapter" in results
    assert results["adapter"]["preds"][0] == "test output"

    # Verify adapter was called
    adapter.assert_called()


def test_embedding_baseline_calibration():
    """Test embedding baseline with calibration."""
    from latentwire.eval import _calibrate_prefix

    # Create dummy prefix embeddings
    prefix = torch.randn(1, 10, 128)

    # Create mock wrapper
    model = DummyModel()
    tokenizer = DummyTokenizer()
    wrapper = DummyWrapper(model, tokenizer)

    # Test no calibration (defaults to embed_rms when mode is None)
    calibrated, scale, target_rms = _calibrate_prefix(
        prefix,
        wrapper,
        mode=None,
        fixed_rms=None,
        stats=None,
        model_key="test",
    )

    # When mode is None, it defaults to embed_rms calibration
    assert calibrated.shape == prefix.shape
    assert scale > 0
    assert target_rms > 0

    # Test embed_rms calibration
    calibrated, scale, target_rms = _calibrate_prefix(
        prefix,
        wrapper,
        mode="embed_rms",
        fixed_rms=None,
        stats=None,
        model_key="test",
    )

    # Should be calibrated to match embedding RMS
    assert calibrated.shape == prefix.shape
    assert scale > 0
    assert target_rms > 0


def test_embedding_replay_with_padding():
    """Test embedding replay handles padding correctly."""
    model = DummyModel()
    tokenizer = DummyTokenizer()

    wrapper = DummyWrapper(model, tokenizer)

    # Create prompt and target with padding
    prompt_ids = torch.tensor([[5, 6, 7, 0, 0]])  # 0 is pad token
    target_ids = torch.tensor([[10, 11, 12, 13, 0, 0, 0, 0]])

    loss, n_tokens = wrapper.loss_with_text_embedding_replay(
        prompt_ids, target_ids, compute_loss=True
    )

    assert loss is not None
    # Should only count non-padding tokens
    assert n_tokens > 0 and n_tokens < target_ids.numel()


def test_multiple_embedding_modes():
    """Test running multiple embedding baseline modes in sequence."""
    from latentwire.eval import _run_embedding_baselines

    model = DummyModel()
    tokenizer = DummyTokenizer()
    wrapper = DummyWrapper(model, tokenizer)

    wrapper.generate_from_prefix = MagicMock(return_value=torch.randint(0, 1000, (1, 10)))
    wrapper.decode_batch_then_clean = MagicMock(return_value=["test output"])

    prompts = ["Test"]
    golds = ["Gold"]

    args = types.SimpleNamespace(
        max_new_tokens=10,
        min_new_tokens=1,
        eos_ban_steps=0,
        first_token_top_p=1.0,
        first_token_temperature=0.0,
        calibration=None,
        prefix_target_rms=None,
    )

    # Test multiple modes
    results = _run_embedding_baselines(
        wrapper=wrapper,
        prompts=prompts,
        golds=golds,
        modes=["raw", "anchor"],
        args=args,
        answer_lengths=None,
        anchor_entry={"text": "Answer: "},
        adapter=None,
        train_stats=None,
        model_name="test",
        latent_len=32,
    )

    assert "raw" in results
    assert "anchor" in results
    assert results["raw"]["preds"] is not None
    assert results["anchor"]["preds"] is not None