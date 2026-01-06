#!/usr/bin/env python3
"""
================================================================================
MODELS.py - Models Module for LatentWire/Telepathy
================================================================================

This module contains all model-related components including encoders,
adapters, bridge models, and language model wrappers.

Author: LatentWire Team
Date: January 2025
Version: 1.0.0 (Split from consolidated)
================================================================================
"""

import os
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict

# Deep learning
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    F = None
    HAS_TORCH = False

# Transformers
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for a language model."""
    model_id: str
    model_type: str = "llama"  # llama, qwen, or other
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float16"
    max_length: int = 2048
    use_cache: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CompressionConfig:
    """Configuration for compression/interlingua settings."""
    latent_dim: int = 256
    latent_len: int = 32
    encoder_type: str = "byte"  # byte, char, token
    decoder_type: str = "linear"  # linear, mlp, transformer

    # Loss weights
    reconstruction_weight: float = 1.0
    kl_weight: float = 0.01
    first_token_ce_weight: float = 0.5
    k_token_ce_weight: float = 0.3

    # Advanced settings
    use_vae: bool = False
    use_quantization: bool = False
    quantization_bits: int = 8

    # Calibration
    calibration_method: str = "embed_rms"
    anchor_text: str = "Answer: "
    append_bos: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExperimentConfig:
    """Minimal experiment configuration for model initialization."""
    models: List[ModelConfig] = field(default_factory=list)
    compression: CompressionConfig = field(default_factory=CompressionConfig)


# ============================================================================
# BYTE ENCODER
# ============================================================================

class ByteEncoder(nn.Module):
    """Byte-level encoder for text."""

    def __init__(self, d_model: int = 256, max_len: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Byte embedding (256 possible values)
        self.byte_embed = nn.Embedding(256, d_model)

        # Positional encoding
        self.pos_embed = nn.Embedding(max_len, d_model)

        # Transformer encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=6,
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, text: Union[str, List[str]]) -> torch.Tensor:
        """Encode text to latent representation."""
        if isinstance(text, str):
            text = [text]

        # Convert text to bytes
        byte_sequences = []
        for t in text:
            bytes_data = t.encode('utf-8')
            byte_sequence = torch.tensor([b for b in bytes_data], dtype=torch.long)
            if len(byte_sequence) > self.max_len:
                byte_sequence = byte_sequence[:self.max_len]
            byte_sequences.append(byte_sequence)

        # Pad sequences
        max_seq_len = max(len(seq) for seq in byte_sequences)
        padded_sequences = torch.zeros(len(byte_sequences), max_seq_len, dtype=torch.long)
        for i, seq in enumerate(byte_sequences):
            padded_sequences[i, :len(seq)] = seq

        device = next(self.parameters()).device
        padded_sequences = padded_sequences.to(device)

        # Embed bytes
        byte_embeds = self.byte_embed(padded_sequences)

        # Add positional encoding
        positions = torch.arange(max_seq_len, device=device).unsqueeze(0).expand(len(text), -1)
        pos_embeds = self.pos_embed(positions)

        x = byte_embeds + pos_embeds

        # Apply transformer
        x = self.encoder(x)

        # Output projection
        x = self.output_proj(x)

        return x


# ============================================================================
# LATENT ENCODER
# ============================================================================

class LatentEncoder(nn.Module):
    """Encoder that produces fixed-size latent representation."""

    def __init__(self, config: CompressionConfig):
        super().__init__()
        self.config = config

        # Input encoder (byte, char, or token-based)
        if config.encoder_type == "byte":
            self.input_encoder = ByteEncoder(d_model=config.latent_dim)
        else:
            # Simple MLP encoder as fallback
            self.input_encoder = nn.Sequential(
                nn.Linear(768, config.latent_dim * 2),  # Assuming 768-dim input
                nn.GELU(),
                nn.Linear(config.latent_dim * 2, config.latent_dim),
            )

        # Compression to fixed-size latent
        self.compressor = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.latent_dim,
                nhead=8,
                dim_feedforward=config.latent_dim * 4,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=3,
        )

        # Learnable queries for fixed-size output
        self.latent_queries = nn.Parameter(
            torch.randn(1, config.latent_len, config.latent_dim)
        )

        # Cross-attention for compression
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.latent_dim,
            num_heads=8,
            batch_first=True,
        )

        # Optional VAE components
        if config.use_vae:
            self.fc_mu = nn.Linear(config.latent_dim, config.latent_dim)
            self.fc_logvar = nn.Linear(config.latent_dim, config.latent_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAE reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Union[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Encode input to latent representation."""
        # Encode input
        if isinstance(x, str) or (isinstance(x, list) and isinstance(x[0], str)):
            encoded = self.input_encoder(x)
        else:
            # Assume tensor input
            if len(x.shape) == 2:
                x = x.unsqueeze(1)  # Add sequence dimension
            encoded = self.input_encoder(x)

        # Apply transformer
        encoded = self.compressor(encoded)

        # Cross-attention to compress to fixed size
        batch_size = encoded.shape[0]
        queries = self.latent_queries.expand(batch_size, -1, -1)

        latent, _ = self.cross_attention(queries, encoded, encoded)

        # VAE encoding if enabled
        output = {"latent": latent}
        if self.config.use_vae:
            mu = self.fc_mu(latent)
            logvar = self.fc_logvar(latent)
            z = self.reparameterize(mu, logvar)
            output.update({
                "z": z,
                "mu": mu,
                "logvar": logvar,
            })

        return output


# ============================================================================
# MODEL ADAPTER
# ============================================================================

class ModelAdapter(nn.Module):
    """Adapter to map latent representation to model-specific embeddings."""

    def __init__(self, latent_dim: int, model_dim: int, num_layers: int = 2):
        super().__init__()

        layers = []
        hidden_dim = (latent_dim + model_dim) // 2

        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(latent_dim, hidden_dim))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, model_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))

            if i < num_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.LayerNorm(hidden_dim))

        self.adapter = nn.Sequential(*layers)

        # Calibration parameters
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))

    def forward(self, latent: torch.Tensor, calibration_target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Map latent to model embeddings."""
        # Apply adapter
        embeddings = self.adapter(latent)

        # Apply calibration if target provided
        if calibration_target is not None:
            # Match RMS of target embeddings
            target_rms = torch.sqrt(torch.mean(calibration_target ** 2))
            embed_rms = torch.sqrt(torch.mean(embeddings ** 2))
            scale_factor = target_rms / (embed_rms + 1e-8)
            embeddings = embeddings * scale_factor
        else:
            # Use learned calibration
            embeddings = embeddings * self.scale + self.shift

        return embeddings


# ============================================================================
# BRIDGE MODEL
# ============================================================================

class BridgeModel(nn.Module):
    """Main bridge model combining encoder and adapters."""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config

        # Encoder
        self.encoder = LatentEncoder(config.compression)

        # Model-specific adapters
        self.adapters = nn.ModuleDict()
        for model_config in config.models:
            # Get model dimension
            if "llama" in model_config.model_id.lower():
                model_dim = 4096  # Llama-7B/8B dimension
            elif "qwen" in model_config.model_id.lower():
                model_dim = 3584  # Qwen 7B dimension
            else:
                model_dim = 768  # Default

            adapter_name = model_config.model_id.replace("/", "_")
            self.adapters[adapter_name] = ModelAdapter(
                config.compression.latent_dim,
                model_dim
            )

    def forward(self, x: Union[str, torch.Tensor], target_model: str) -> torch.Tensor:
        """Forward pass through encoder and adapter."""
        # Encode to latent
        encoded = self.encoder(x)
        latent = encoded["latent"] if isinstance(encoded, dict) else encoded

        # Apply model-specific adapter
        adapter_name = target_model.replace("/", "_")
        if adapter_name not in self.adapters:
            raise ValueError(f"No adapter for model: {target_model}")

        embeddings = self.adapters[adapter_name](latent)

        return embeddings


# ============================================================================
# LANGUAGE MODEL WRAPPER
# ============================================================================

class LMWrapper(nn.Module):
    """Wrapper for language models with bridge integration."""

    def __init__(self, model_config: ModelConfig, bridge_model: Optional[BridgeModel] = None):
        super().__init__()
        self.config = model_config
        self.bridge = bridge_model

        # Load pretrained model
        if HAS_TRANSFORMERS:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_config.model_id,
                torch_dtype=getattr(torch, model_config.dtype),
                device_map="auto" if model_config.device == "cuda" else None,
                trust_remote_code=model_config.trust_remote_code,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_config.model_id,
                trust_remote_code=model_config.trust_remote_code,
            )

            # Ensure pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            # Dummy model for testing without transformers
            self.model = nn.Linear(768, 50000)  # vocab size
            self.tokenizer = None

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        prefix_text: Optional[str] = None,
        use_bridge: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with optional bridge encoding."""

        if use_bridge and self.bridge is not None and prefix_text is not None:
            # Encode prefix with bridge
            bridge_embeddings = self.bridge(prefix_text, self.config.model_id)

            # Get answer tokens
            answer_text = kwargs.get("answer_text", "")
            answer_tokens = self.tokenizer(answer_text, return_tensors="pt")

            # Combine bridge embeddings with answer embeddings
            answer_embeds = self.model.get_input_embeddings()(answer_tokens["input_ids"])

            inputs_embeds = torch.cat([bridge_embeddings, answer_embeds], dim=1)

            # Forward through model
            outputs = self.model(inputs_embeds=inputs_embeds, **kwargs)
        else:
            # Standard forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )

        return outputs


# ============================================================================
# ELASTIC GPU CONFIGURATION (from Section 16)
# ============================================================================

class ElasticGPUConfig:
    """Elastic GPU configuration that adapts to available hardware."""

    def __init__(self, base_batch_size=64, model_size_gb=14.0, target_util=0.75):
        self.base_batch_size = base_batch_size
        self.model_size_gb = model_size_gb
        self.target_util = target_util
        self.gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.gpu_specs = self._detect_gpu_specs()
        self.config = self._configure_for_gpus()

    def _detect_gpu_specs(self):
        if self.gpu_count == 0:
            return None

        specs = []
        for i in range(self.gpu_count):
            props = torch.cuda.get_device_properties(i)
            specs.append({
                'id': i,
                'name': props.name,
                'memory_gb': props.total_memory / 1e9,
                'compute_capability': (props.major, props.minor),
                'is_h100': 'H100' in props.name or 'h100' in props.name.lower(),
                'is_a100': 'A100' in props.name or 'a100' in props.name.lower(),
            })
        return specs

    def _configure_for_gpus(self):
        if self.gpu_count == 0:
            return {
                'batch_size': 1,
                'effective_batch_size': 1,
                'grad_accum_steps': 1,
                'device': 'cpu',
                'strategy': 'single_device',
            }

        total_memory = sum(g['memory_gb'] for g in self.gpu_specs)
        min_gpu_memory = min(g['memory_gb'] for g in self.gpu_specs)

        is_h100_cluster = all(g.get('is_h100', False) for g in self.gpu_specs)

        available_per_gpu = min_gpu_memory * self.target_util
        activation_gb_per_item = 0.75

        configs = {
            1: self._config_single_gpu(available_per_gpu, activation_gb_per_item),
            2: self._config_dual_gpu(available_per_gpu, activation_gb_per_item, is_h100_cluster),
            4: self._config_four_gpu(available_per_gpu, activation_gb_per_item, is_h100_cluster),
        }

        return configs.get(self.gpu_count, self._config_single_gpu(available_per_gpu, activation_gb_per_item))

    def _config_single_gpu(self, available_gb, activation_gb_per_item):
        usable_for_activations = max(1, available_gb - self.model_size_gb - 4)
        batch_size = min(self.base_batch_size, int(usable_for_activations / activation_gb_per_item))
        batch_size = max(1, batch_size)

        target_effective = self.base_batch_size
        grad_accum = max(1, target_effective // batch_size)

        return {
            'batch_size': batch_size,
            'effective_batch_size': batch_size * grad_accum,
            'grad_accum_steps': grad_accum,
            'device': 'cuda:0',
            'strategy': 'single_gpu',
        }

    def _config_dual_gpu(self, available_gb, activation_gb_per_item, is_h100):
        if is_h100:
            batch_per_gpu = self.base_batch_size
            return {
                'batch_size': batch_per_gpu * 2,
                'effective_batch_size': batch_per_gpu * 2,
                'grad_accum_steps': 1,
                'device': 'cuda',
                'strategy': 'ddp',
            }
        else:
            batch_size = min(self.base_batch_size, int((available_gb - self.model_size_gb) / activation_gb_per_item))
            return {
                'batch_size': batch_size,
                'effective_batch_size': batch_size,
                'grad_accum_steps': 1,
                'device': 'cuda',
                'strategy': 'model_split',
            }

    def _config_four_gpu(self, available_gb, activation_gb_per_item, is_h100):
        if is_h100:
            batch_per_gpu = self.base_batch_size
            is_torchrun = 'WORLD_SIZE' in os.environ and int(os.environ.get('WORLD_SIZE', 1)) > 1

            if is_torchrun:
                return {
                    'batch_size': batch_per_gpu,
                    'effective_batch_size': batch_per_gpu * 4,
                    'grad_accum_steps': 1,
                    'device': 'cuda',
                    'strategy': 'ddp_torchrun_4gpu',
                }
            else:
                return {
                    'batch_size': batch_per_gpu * 4,
                    'effective_batch_size': batch_per_gpu * 4,
                    'grad_accum_steps': 1,
                    'device': 'cuda',
                    'strategy': 'ddp_4gpu',
                }
        else:
            batch_per_gpu = min(self.base_batch_size, int((available_gb - self.model_size_gb/2) / activation_gb_per_item))
            return {
                'batch_size': batch_per_gpu * 2,
                'effective_batch_size': batch_per_gpu * 2,
                'grad_accum_steps': 1,
                'device': 'cuda',
                'strategy': 'hybrid_4gpu',
            }