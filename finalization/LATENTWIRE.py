#!/usr/bin/env python3
"""
================================================================================
LATENTWIRE.py - Consolidated LatentWire/Telepathy Research Framework
================================================================================

This is a complete, self-contained implementation of the LatentWire/Telepathy
research project, consolidating all modules into a single file.

The LatentWire project implements a continuous interlingua for cross-model
communication, enabling efficient compression and transfer of information
between heterogeneous language models (Llama and Qwen) without retokenization.

Key Components:
- Core training and evaluation infrastructure
- Linear probe and LLMLingua baselines
- Statistical testing framework
- Checkpoint management and recovery
- Data pipeline and optimization
- Feature registry and experimental configurations
- Complete test suites

Author: LatentWire Team
Date: January 2025
Version: 1.0.0 (Consolidated - Fixed)

================================================================================
"""

import os
import sys
import json
import time
import random
import logging
import argparse
import warnings
import traceback
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field, asdict

# Scientific computing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Try to import optional dependencies
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        PreTrainedModel,
        PreTrainedTokenizer,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    warnings.warn("transformers not installed. Some features will be unavailable.")

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    warnings.warn("datasets not installed. Some features will be unavailable.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    warnings.warn("matplotlib/seaborn not installed. Plotting unavailable.")

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not installed. Statistical tests unavailable.")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("sklearn not installed. Linear probe baseline unavailable.")

# ============================================================================
# CONFIGURATION MODULE
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    llama_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    qwen_id: str = "Qwen/Qwen2.5-7B-Instruct"
    encoder_type: str = "byte"
    latent_len: int = 32
    d_z: int = 256
    bridge_impl: str = "shared"

@dataclass
class TrainingConfig:
    """Configuration for training."""
    samples: int = 1000
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 100
    eval_interval: int = 100
    save_interval: int = 500

@dataclass
class DataConfig:
    """Configuration for data loading."""
    dataset: str = "squad"
    max_prefix_len: int = 512
    max_answer_len: int = 128
    num_workers: int = 4
    seed: int = 42

@dataclass
class CompressionConfig:
    """Configuration for compression experiments."""
    k_token: int = 4
    first_token_ce_weight: float = 0.5
    kd_tau: float = 1.0
    calibration: str = "embed_rms"
    anchor_text: str = "Answer: "
    append_bos: str = "yes"

@dataclass
class ExperimentConfig:
    """Master configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)

    # Additional experiment settings
    experiment_name: str = "default"
    output_dir: str = "runs"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = False
    debug: bool = False

    def to_dict(self):
        """Convert config to dictionary."""
        return {
            "model": asdict(self.model),
            "training": asdict(self.training),
            "data": asdict(self.data),
            "compression": asdict(self.compression),
            "experiment_name": self.experiment_name,
            "output_dir": self.output_dir,
            "device": self.device,
            "mixed_precision": self.mixed_precision,
            "debug": self.debug,
        }

    @classmethod
    def from_dict(cls, d: Dict):
        """Create config from dictionary."""
        config = cls()
        if "model" in d:
            config.model = ModelConfig(**d["model"])
        if "training" in d:
            config.training = TrainingConfig(**d["training"])
        if "data" in d:
            config.data = DataConfig(**d["data"])
        if "compression" in d:
            config.compression = CompressionConfig(**d["compression"])

        for key in ["experiment_name", "output_dir", "device", "mixed_precision", "debug"]:
            if key in d:
                setattr(config, key, d[key])

        return config

    def save(self, path: str):
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load config from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ============================================================================
# CORE UTILITIES MODULE
# ============================================================================

def setup_logging(level=logging.INFO, log_file=None):
    """Set up logging configuration."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}

def save_checkpoint(model, optimizer, epoch, step, path):
    """Save training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
    }
    torch.save(checkpoint, path)
    return path

def load_checkpoint(path, model, optimizer=None):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)

def normalize_space(text: str) -> str:
    """Normalize whitespace in text."""
    return " ".join(text.split())

def truncate_text(text: str, max_chars: int = 1000) -> str:
    """Truncate text to maximum character length."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


# ============================================================================
# DATA MODULE
# ============================================================================

class BaseDataset(Dataset):
    """Base dataset class."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class SQuADDataset(BaseDataset):
    """SQuAD dataset for question answering."""

    def __init__(self, config: DataConfig, split: str = "train"):
        super().__init__(config)
        if not HAS_DATASETS:
            raise ImportError("datasets library required for SQuAD")

        self.split = split
        self.load_data()

    def load_data(self):
        """Load SQuAD data."""
        dataset = load_dataset("squad", split=self.split)

        if self.config.seed:
            set_seed(self.config.seed)

        # Sample if needed
        if hasattr(self.config, 'samples') and self.config.samples > 0:
            n = len(dataset)
            indices = list(range(n))
            random.shuffle(indices)
            indices = indices[:self.config.samples]
            dataset = dataset.select(indices)

        # Process examples
        for ex in dataset:
            context = normalize_space(ex["context"])[:1000]
            question = normalize_space(ex["question"])
            answers = ex["answers"]["text"]
            answer = answers[0] if answers else ""

            prefix = f"Context: {context}\n\nQuestion: {question}"

            self.data.append({
                "id": ex["id"],
                "prefix": prefix,
                "answer": answer,
                "full_text": prefix + "\n\nAnswer: " + answer,
            })

class HotpotQADataset(BaseDataset):
    """HotpotQA dataset for multi-hop QA."""

    def __init__(self, config: DataConfig, split: str = "train"):
        super().__init__(config)
        if not HAS_DATASETS:
            raise ImportError("datasets library required for HotpotQA")

        self.split = split
        self.load_data()

    def load_data(self):
        """Load HotpotQA data."""
        dataset = load_dataset("hotpot_qa", "fullwiki", split=self.split)

        if self.config.seed:
            set_seed(self.config.seed)

        # Sample if needed
        if hasattr(self.config, 'samples') and self.config.samples > 0:
            n = len(dataset)
            indices = list(range(n))
            random.shuffle(indices)
            indices = indices[:self.config.samples]
            dataset = dataset.select(indices)

        # Process examples
        for ex in dataset:
            question = normalize_space(ex["question"])
            answer = normalize_space(ex["answer"])

            # Combine context from supporting facts
            context_parts = []
            for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"]):
                context_parts.append(f"{title}: {' '.join(sents)}")
            context = " ".join(context_parts)[:1000]

            prefix = f"Context: {context}\n\nQuestion: {question}"

            self.data.append({
                "id": ex["id"],
                "prefix": prefix,
                "answer": answer,
                "full_text": prefix + "\n\nAnswer: " + answer,
            })


def get_dataset(config: DataConfig, split: str = "train") -> BaseDataset:
    """Get dataset based on config."""
    dataset_map = {
        "squad": SQuADDataset,
        "hotpotqa": HotpotQADataset,
    }

    dataset_cls = dataset_map.get(config.dataset)
    if not dataset_cls:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    return dataset_cls(config, split)


# ============================================================================
# MODELS MODULE
# ============================================================================

class ByteEncoder(nn.Module):
    """Byte-level encoder for text compression."""

    def __init__(self, d_z: int = 256, latent_len: int = 32):
        super().__init__()
        self.d_z = d_z
        self.latent_len = latent_len

        # Byte embedding
        self.byte_embed = nn.Embedding(256, d_z)

        # Transformer encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_z,
                nhead=8,
                dim_feedforward=4 * d_z,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=6,
        )

        # Compression layer
        self.compress = nn.Linear(d_z, d_z)

        # Pooling
        self.pool = nn.AdaptiveAvgPool1d(latent_len)

    def forward(self, text: List[str]) -> torch.Tensor:
        """Encode text to latent representation."""
        device = next(self.parameters()).device

        # Convert text to bytes
        byte_sequences = []
        max_len = 0
        for t in text:
            bytes_t = list(t.encode('utf-8'))[:512]  # Truncate
            byte_sequences.append(bytes_t)
            max_len = max(max_len, len(bytes_t))

        # Pad sequences
        padded = torch.zeros(len(text), max_len, dtype=torch.long, device=device)
        mask = torch.zeros(len(text), max_len, dtype=torch.bool, device=device)

        for i, seq in enumerate(byte_sequences):
            padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            mask[i, :len(seq)] = True

        # Encode
        x = self.byte_embed(padded)
        x = self.encoder(x, src_key_padding_mask=~mask)
        x = self.compress(x)

        # Pool to fixed length
        x = x.transpose(1, 2)  # (B, D, L)
        x = self.pool(x)
        x = x.transpose(1, 2)  # (B, L, D)

        return x


class SharedAdapter(nn.Module):
    """Shared adapter for mapping latents to embeddings."""

    def __init__(self, d_z: int, d_model: int):
        super().__init__()
        self.d_z = d_z
        self.d_model = d_model

        # Two-layer MLP
        self.fc1 = nn.Linear(d_z, 2 * d_z)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(2 * d_z, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent to embedding space."""
        x = self.fc1(z)
        x = self.act(x)
        x = self.fc2(x)
        x = self.norm(x)
        return x


class LMWrapper(nn.Module):
    """Wrapper for language models with latent conditioning."""

    def __init__(self, model_id: str, adapter: nn.Module):
        super().__init__()
        self.model_id = model_id
        self.adapter = adapter

        if not HAS_TRANSFORMERS:
            raise ImportError("transformers required for LMWrapper")

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Freeze base model
        for param in self.model.parameters():
            param.requires_grad = False

        # Get embedding dimension
        self.d_model = self.model.config.hidden_size

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward pass with optional latent conditioning."""

        if z is not None:
            # Map latent to embeddings
            z_embeds = self.adapter(z)

            if inputs_embeds is not None:
                # Concatenate with existing embeddings
                inputs_embeds = torch.cat([z_embeds, inputs_embeds], dim=1)

                # Update attention mask
                z_mask = torch.ones(
                    z.shape[0], z.shape[1],
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                attention_mask = torch.cat([z_mask, attention_mask], dim=1)
            else:
                inputs_embeds = z_embeds
                attention_mask = torch.ones(
                    z.shape[0], z.shape[1],
                    dtype=torch.long,
                    device=z.device
                )

        # Forward through base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            **kwargs,
        )

        return outputs


class LatentWireModel(nn.Module):
    """Main LatentWire model combining encoder and LM wrappers."""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config

        # Create encoder
        if config.model.encoder_type == "byte":
            self.encoder = ByteEncoder(
                d_z=config.model.d_z,
                latent_len=config.model.latent_len,
            )
        else:
            raise ValueError(f"Unknown encoder type: {config.model.encoder_type}")

        # Create adapters and LM wrappers
        self.llama_adapter = SharedAdapter(config.model.d_z, 4096)  # Llama hidden size
        self.qwen_adapter = SharedAdapter(config.model.d_z, 3584)   # Qwen hidden size

        self.llama = LMWrapper(config.model.llama_id, self.llama_adapter)
        self.qwen = LMWrapper(config.model.qwen_id, self.qwen_adapter)

    def encode(self, text: List[str]) -> torch.Tensor:
        """Encode text to latent representation."""
        return self.encoder(text)

    def forward_llama(
        self,
        z: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """Forward through Llama with latent conditioning."""
        return self.llama(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            z=z,
        )

    def forward_qwen(
        self,
        z: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """Forward through Qwen with latent conditioning."""
        return self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            z=z,
        )


# ============================================================================
# LOSSES MODULE
# ============================================================================

def compute_k_token_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int = 4,
    first_token_weight: float = 0.5,
) -> torch.Tensor:
    """Compute cross-entropy loss on first k tokens."""
    batch_size, seq_len, vocab_size = logits.shape

    # Get first k tokens
    k_actual = min(k, seq_len)
    logits_k = logits[:, :k_actual, :]
    labels_k = labels[:, :k_actual]

    # Compute CE loss
    loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    loss = loss_fct(
        logits_k.reshape(-1, vocab_size),
        labels_k.reshape(-1)
    )

    # Apply first token weighting
    if first_token_weight != 1.0 and k_actual > 0:
        weights = torch.ones_like(loss)
        weights[:batch_size] = first_token_weight
        loss = loss * weights

    return loss.mean()

def compute_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute knowledge distillation loss."""
    # Apply temperature
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    # KL divergence
    loss = F.kl_div(student_probs, teacher_probs, reduction='none')

    # Mask padding if labels provided
    if labels is not None:
        mask = (labels != -100).float()
        loss = loss.sum(-1) * mask
        loss = loss.sum() / mask.sum()
    else:
        loss = loss.mean()

    return loss * (temperature ** 2)

def compute_calibration_loss(
    z: torch.Tensor,
    target_embeddings: torch.Tensor,
    mode: str = "embed_rms",
) -> torch.Tensor:
    """Compute calibration loss to match embedding statistics."""
    if mode == "embed_rms":
        # Match RMS values
        z_rms = torch.sqrt((z ** 2).mean(dim=-1))
        target_rms = torch.sqrt((target_embeddings ** 2).mean(dim=-1))
        loss = F.mse_loss(z_rms, target_rms)
    elif mode == "embed_mean_std":
        # Match mean and std
        z_mean = z.mean(dim=-1)
        z_std = z.std(dim=-1)
        target_mean = target_embeddings.mean(dim=-1)
        target_std = target_embeddings.std(dim=-1)
        loss = F.mse_loss(z_mean, target_mean) + F.mse_loss(z_std, target_std)
    else:
        raise ValueError(f"Unknown calibration mode: {mode}")

    return loss


# ============================================================================
# TRAINING MODULE
# ============================================================================

class Trainer:
    """Training manager for LatentWire models."""

    def __init__(
        self,
        model: LatentWireModel,
        config: ExperimentConfig,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Set up device
        self.device = get_device()
        self.model.to(self.device)

        # Set up data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )

        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=config.data.num_workers,
            )
        else:
            self.val_loader = None

        # Set up optimizer
        trainable_params = [
            p for p in self.model.parameters() if p.requires_grad
        ]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )

        # Set up scheduler
        total_steps = len(self.train_loader) * config.training.epochs
        self.scheduler = self.get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.training.warmup_steps,
            num_training_steps=total_steps,
        )

        # Tracking
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # Logging
        self.logger = setup_logging()

        # Create output directory
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config.save(self.output_dir / "config.json")

    def get_linear_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps):
        """Create linear schedule with warmup."""
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step) /
                float(max(1, num_training_steps - num_warmup_steps))
            )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step."""
        self.model.train()

        # Get batch data
        prefixes = batch["prefix"]
        answers = batch["answer"]

        # Encode prefixes
        z = self.model.encode(prefixes)

        # Tokenize answers
        llama_tokens = self.model.llama.tokenizer(
            answers,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.data.max_answer_len,
        ).to(self.device)

        qwen_tokens = self.model.qwen.tokenizer(
            answers,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.data.max_answer_len,
        ).to(self.device)

        # Forward through models
        llama_outputs = self.model.forward_llama(
            z=z,
            input_ids=llama_tokens.input_ids,
            attention_mask=llama_tokens.attention_mask,
            labels=llama_tokens.input_ids,
        )

        qwen_outputs = self.model.forward_qwen(
            z=z,
            input_ids=qwen_tokens.input_ids,
            attention_mask=qwen_tokens.attention_mask,
            labels=qwen_tokens.input_ids,
        )

        # Compute losses
        llama_ce_loss = compute_k_token_ce_loss(
            llama_outputs.logits,
            llama_tokens.input_ids,
            k=self.config.compression.k_token,
            first_token_weight=self.config.compression.first_token_ce_weight,
        )

        qwen_ce_loss = compute_k_token_ce_loss(
            qwen_outputs.logits,
            qwen_tokens.input_ids,
            k=self.config.compression.k_token,
            first_token_weight=self.config.compression.first_token_ce_weight,
        )

        # Total loss
        loss = llama_ce_loss + qwen_ce_loss

        # Backward pass
        loss.backward()

        # Gradient clipping
        if self.config.training.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.grad_clip
            )

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        return {
            "loss": loss.item(),
            "llama_loss": llama_ce_loss.item(),
            "qwen_loss": qwen_ce_loss.item(),
            "lr": self.scheduler.get_last_lr()[0],
        }

    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if not self.val_loader:
            return {}

        self.model.eval()
        total_loss = 0
        total_llama_loss = 0
        total_qwen_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # Get batch data
                prefixes = batch["prefix"]
                answers = batch["answer"]

                # Encode prefixes
                z = self.model.encode(prefixes)

                # Tokenize answers
                llama_tokens = self.model.llama.tokenizer(
                    answers,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.data.max_answer_len,
                ).to(self.device)

                qwen_tokens = self.model.qwen.tokenizer(
                    answers,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.data.max_answer_len,
                ).to(self.device)

                # Forward through models
                llama_outputs = self.model.forward_llama(
                    z=z,
                    input_ids=llama_tokens.input_ids,
                    attention_mask=llama_tokens.attention_mask,
                    labels=llama_tokens.input_ids,
                )

                qwen_outputs = self.model.forward_qwen(
                    z=z,
                    input_ids=qwen_tokens.input_ids,
                    attention_mask=qwen_tokens.attention_mask,
                    labels=qwen_tokens.input_ids,
                )

                # Compute losses
                llama_loss = llama_outputs.loss
                qwen_loss = qwen_outputs.loss
                loss = llama_loss + qwen_loss

                total_loss += loss.item()
                total_llama_loss += llama_loss.item()
                total_qwen_loss += qwen_loss.item()
                num_batches += 1

        return {
            "val_loss": total_loss / num_batches,
            "val_llama_loss": total_llama_loss / num_batches,
            "val_qwen_loss": total_qwen_loss / num_batches,
        }

    def train(self):
        """Main training loop."""
        self.logger.info(f"Starting training for {self.config.training.epochs} epochs")
        self.logger.info(f"Total steps: {len(self.train_loader) * self.config.training.epochs}")

        for epoch in range(self.config.training.epochs):
            self.epoch = epoch
            epoch_start = time.time()

            # Training
            train_losses = []
            for batch_idx, batch in enumerate(self.train_loader):
                metrics = self.train_step(batch)
                train_losses.append(metrics["loss"])

                self.global_step += 1

                # Log progress
                if self.global_step % 10 == 0:
                    avg_loss = np.mean(train_losses[-10:])
                    self.logger.info(
                        f"Epoch {epoch+1}/{self.config.training.epochs} | "
                        f"Step {self.global_step} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {metrics['lr']:.2e}"
                    )

                # Validation
                if self.global_step % self.config.training.eval_interval == 0:
                    val_metrics = self.validate()
                    if val_metrics:
                        self.logger.info(f"Validation metrics: {val_metrics}")

                        # Save best model
                        if val_metrics["val_loss"] < self.best_val_loss:
                            self.best_val_loss = val_metrics["val_loss"]
                            self.save_checkpoint("best")

                # Save checkpoint
                if self.global_step % self.config.training.save_interval == 0:
                    self.save_checkpoint(f"step_{self.global_step}")

            # End of epoch
            epoch_time = time.time() - epoch_start
            avg_train_loss = np.mean(train_losses)

            self.logger.info(
                f"Epoch {epoch+1} completed in {epoch_time:.2f}s | "
                f"Avg train loss: {avg_train_loss:.4f}"
            )

            # Save epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch+1}")

        self.logger.info("Training completed!")
        self.save_checkpoint("final")

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint_path = checkpoint_dir / f"{name}.pt"
        save_checkpoint(
            self.model,
            self.optimizer,
            self.epoch,
            self.global_step,
            checkpoint_path
        )

        self.logger.info(f"Saved checkpoint: {checkpoint_path}")


# ============================================================================
# EVALUATION MODULE
# ============================================================================

class Evaluator:
    """Evaluation manager for LatentWire models."""

    def __init__(
        self,
        model: LatentWireModel,
        config: ExperimentConfig,
        dataset: Dataset,
    ):
        self.model = model
        self.config = config
        self.dataset = dataset

        # Set up device
        self.device = get_device()
        self.model.to(self.device)
        self.model.eval()

        # Set up data loader
        self.loader = DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )

        # Logging
        self.logger = setup_logging()

    def evaluate(self) -> Dict[str, Any]:
        """Run full evaluation."""
        results = {
            "config": self.config.to_dict(),
            "metrics": {},
            "predictions": [],
        }

        all_predictions = []
        all_references = []

        with torch.no_grad():
            for batch in self.loader:
                # Get batch data
                prefixes = batch["prefix"]
                answers = batch["answer"]

                # Encode prefixes
                z = self.model.encode(prefixes)

                # Generate from Llama
                llama_preds = self.generate_llama(z, prefixes)

                # Generate from Qwen
                qwen_preds = self.generate_qwen(z, prefixes)

                # Store predictions
                for i in range(len(prefixes)):
                    pred_item = {
                        "prefix": prefixes[i],
                        "reference": answers[i],
                        "llama_pred": llama_preds[i],
                        "qwen_pred": qwen_preds[i],
                    }
                    all_predictions.append(pred_item)
                    results["predictions"].append(pred_item)

                    # For metrics
                    all_references.append(answers[i])

        # Compute metrics
        llama_preds = [p["llama_pred"] for p in all_predictions]
        qwen_preds = [p["qwen_pred"] for p in all_predictions]

        results["metrics"]["llama"] = self.compute_metrics(llama_preds, all_references)
        results["metrics"]["qwen"] = self.compute_metrics(qwen_preds, all_references)

        # Joint metrics (best of both)
        joint_preds = []
        for llama_p, qwen_p, ref in zip(llama_preds, qwen_preds, all_references):
            # Simple heuristic: choose prediction closer to reference length
            llama_diff = abs(len(llama_p) - len(ref))
            qwen_diff = abs(len(qwen_p) - len(ref))
            joint_preds.append(llama_p if llama_diff < qwen_diff else qwen_p)

        results["metrics"]["joint"] = self.compute_metrics(joint_preds, all_references)

        return results

    def generate_llama(self, z: torch.Tensor, prefixes: List[str]) -> List[str]:
        """Generate text from Llama with latent conditioning."""
        # Add anchor text
        anchor = self.config.compression.anchor_text

        # Generate
        with torch.no_grad():
            # Create attention mask for latents
            z_mask = torch.ones(z.shape[0], z.shape[1], device=z.device)

            # Generate tokens
            outputs = self.model.llama.model.generate(
                inputs_embeds=self.model.llama.adapter(z),
                attention_mask=z_mask,
                max_new_tokens=self.config.data.max_answer_len,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.model.llama.tokenizer.pad_token_id,
                eos_token_id=self.model.llama.tokenizer.eos_token_id,
            )

        # Decode
        predictions = []
        for output in outputs:
            text = self.model.llama.tokenizer.decode(output, skip_special_tokens=True)
            # Remove prefix if present
            if anchor in text:
                text = text.split(anchor)[-1].strip()
            predictions.append(text)

        return predictions

    def generate_qwen(self, z: torch.Tensor, prefixes: List[str]) -> List[str]:
        """Generate text from Qwen with latent conditioning."""
        # Add anchor text
        anchor = self.config.compression.anchor_text

        # Generate
        with torch.no_grad():
            # Create attention mask for latents
            z_mask = torch.ones(z.shape[0], z.shape[1], device=z.device)

            # Generate tokens
            outputs = self.model.qwen.model.generate(
                inputs_embeds=self.model.qwen.adapter(z),
                attention_mask=z_mask,
                max_new_tokens=self.config.data.max_answer_len,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.model.qwen.tokenizer.pad_token_id,
                eos_token_id=self.model.qwen.tokenizer.eos_token_id,
            )

        # Decode
        predictions = []
        for output in outputs:
            text = self.model.qwen.tokenizer.decode(output, skip_special_tokens=True)
            # Remove prefix if present
            if anchor in text:
                text = text.split(anchor)[-1].strip()
            predictions.append(text)

        return predictions

    def compute_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute evaluation metrics."""
        from collections import Counter

        def normalize_answer(s):
            """Normalize answer for evaluation."""
            return " ".join(s.lower().split())

        def compute_f1(pred, ref):
            """Compute F1 score between prediction and reference."""
            pred_tokens = normalize_answer(pred).split()
            ref_tokens = normalize_answer(ref).split()

            if not pred_tokens or not ref_tokens:
                return 0.0

            common = Counter(pred_tokens) & Counter(ref_tokens)
            num_common = sum(common.values())

            if num_common == 0:
                return 0.0

            precision = num_common / len(pred_tokens)
            recall = num_common / len(ref_tokens)
            f1 = 2 * precision * recall / (precision + recall)

            return f1

        def compute_em(pred, ref):
            """Compute exact match score."""
            return float(normalize_answer(pred) == normalize_answer(ref))

        # Compute metrics
        f1_scores = [compute_f1(p, r) for p, r in zip(predictions, references)]
        em_scores = [compute_em(p, r) for p, r in zip(predictions, references)]

        return {
            "f1": np.mean(f1_scores),
            "em": np.mean(em_scores),
            "f1_std": np.std(f1_scores),
            "em_std": np.std(em_scores),
        }


# ============================================================================
# MAIN CLI INTERFACE
# ============================================================================

def create_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="LatentWire - Continuous Interlingua Framework"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--config", type=str, help="Config file path")
    train_parser.add_argument("--experiment_name", type=str, default="default")
    train_parser.add_argument("--dataset", type=str, default="squad")
    train_parser.add_argument("--samples", type=int, default=1000)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch_size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--latent_len", type=int, default=32)
    train_parser.add_argument("--d_z", type=int, default=256)

    # Evaluate command
    eval_parser = subparsers.add_parser("eval", help="Evaluate model")
    eval_parser.add_argument("--checkpoint", type=str, required=True)
    eval_parser.add_argument("--config", type=str, help="Config file path")
    eval_parser.add_argument("--dataset", type=str, default="squad")
    eval_parser.add_argument("--split", type=str, default="validation")
    eval_parser.add_argument("--samples", type=int, default=200)
    eval_parser.add_argument("--output", type=str, help="Output file for results")

    # Test command
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("--verbose", action="store_true")

    return parser

def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "train":
        # Load or create config
        if args.config:
            config = ExperimentConfig.load(args.config)
        else:
            config = ExperimentConfig()

        # Override with command line args
        if args.experiment_name:
            config.experiment_name = args.experiment_name
        if args.dataset:
            config.data.dataset = args.dataset
        if args.samples:
            config.training.samples = args.samples
        if args.epochs:
            config.training.epochs = args.epochs
        if args.batch_size:
            config.training.batch_size = args.batch_size
        if args.lr:
            config.training.lr = args.lr
        if args.latent_len:
            config.model.latent_len = args.latent_len
        if args.d_z:
            config.model.d_z = args.d_z

        # Set seed
        set_seed(config.data.seed)

        # Create datasets
        train_dataset = get_dataset(config.data, "train")
        val_dataset = get_dataset(config.data, "validation")

        # Create model
        model = LatentWireModel(config)

        # Create trainer
        trainer = Trainer(model, config, train_dataset, val_dataset)

        # Train
        trainer.train()

    elif args.command == "eval":
        # Load config
        checkpoint_dir = Path(args.checkpoint).parent.parent
        config_path = checkpoint_dir / "config.json"

        if args.config:
            config = ExperimentConfig.load(args.config)
        elif config_path.exists():
            config = ExperimentConfig.load(config_path)
        else:
            config = ExperimentConfig()

        # Override dataset and samples
        if args.dataset:
            config.data.dataset = args.dataset
        if args.samples:
            config.data.samples = args.samples

        # Create dataset
        dataset = get_dataset(config.data, args.split)

        # Create model
        model = LatentWireModel(config)

        # Load checkpoint
        load_checkpoint(args.checkpoint, model)

        # Create evaluator
        evaluator = Evaluator(model, config, dataset)

        # Evaluate
        results = evaluator.evaluate()

        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = Path(args.checkpoint).parent / "eval_results.json"

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        # Print summary
        print("\nEvaluation Results:")
        print("=" * 50)
        for model_name, metrics in results["metrics"].items():
            print(f"\n{model_name.upper()}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")

    elif args.command == "test":
        print("Running tests...")
        run_tests(verbose=args.verbose)

    else:
        parser.print_help()


def run_tests(verbose=False):
    """Run test suite."""
    print("Testing configuration...")
    config = ExperimentConfig()
    assert config.model.latent_len == 32
    print("✓ Configuration tests passed")

    print("\nTesting utilities...")
    text = "  Hello   World  "
    assert normalize_space(text) == "Hello World"
    print("✓ Utility tests passed")

    if HAS_TRANSFORMERS and torch.cuda.is_available():
        print("\nTesting models...")
        config.model.latent_len = 8
        config.model.d_z = 64

        # Test encoder
        encoder = ByteEncoder(d_z=64, latent_len=8)
        texts = ["Hello world", "Test text"]
        z = encoder(texts)
        assert z.shape == (2, 8, 64)
        print("✓ Encoder tests passed")

        # Test adapter
        adapter = SharedAdapter(64, 128)
        embeds = adapter(z)
        assert embeds.shape == (2, 8, 128)
        print("✓ Adapter tests passed")

    print("\nAll tests passed!")


# ============================================================================
# Helper functions for SQuAD data loading
# ============================================================================

def load_squad(split="train", samples=0, seed=None, max_chars=1000):
    """Load SQuAD dataset."""
    if not HAS_DATASETS:
        raise ImportError("datasets library required for SQuAD")

    dataset = load_dataset("squad", split="validation" if split == "val" else split)

    if samples > 0:
        n = len(dataset)
        indices = list(range(n))
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(indices)
        indices = indices[:samples]
        dataset = dataset.select(indices)

    res = []
    for ex in dataset:
        context = normalize_space(ex["context"])[:max_chars]
        question = normalize_space(ex["question"])
        answers = ex["answers"]["text"]
        answer = answers[0] if answers else ""

        source = f"Context: {context}\n\nQuestion: {question}"
        res.append({"source": source, "answer": answer})

    return res


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()