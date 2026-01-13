#!/usr/bin/env python3
"""
Production Metrics System for Telepathy Paper (MLSys).

Comprehensive benchmarking system measuring:
1. Throughput (samples/second) at different batch sizes [1, 4, 8, 16, 32, 64, 128]
2. Latency breakdown (p50, p95, p99) for each component
3. Memory usage and scaling
4. Bridge vs text generation speed comparison (validate 22x claim)
5. Quantization impact (fp32, fp16, int8, int4)

Uses real models with proper CUDA timing via torch.cuda.Event.

Usage:
    # Full benchmark suite
    python telepathy/production_metrics.py --output_dir runs/production_metrics

    # Quick validation (smaller batch sizes, fewer iterations)
    python telepathy/production_metrics.py --quick --output_dir runs/quick_metrics

    # Specific tests
    python telepathy/production_metrics.py --test throughput --batch_sizes 1 4 8 16
    python telepathy/production_metrics.py --test quantization
    python telepathy/production_metrics.py --test memory
"""

import argparse
import gc
import json
import os
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Try to import bitsandbytes for int8/int4 quantization
try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False
    print("Warning: bitsandbytes not available, int8/int4 quantization tests will be skipped")


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class LatencyBreakdown:
    """Component-wise latency breakdown in milliseconds."""
    encode_ms: float = 0.0
    bridge_ms: float = 0.0
    decode_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class LatencyStats:
    """Latency statistics with percentiles."""
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    samples_per_second: float
    num_samples: int

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class MemoryStats:
    """GPU memory statistics in MB."""
    allocated_mb: float
    reserved_mb: float
    peak_allocated_mb: float
    peak_reserved_mb: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class ThroughputResult:
    """Result for throughput benchmark at a specific batch size."""
    batch_size: int
    seq_length: int
    latency: LatencyStats
    breakdown: LatencyBreakdown
    memory: MemoryStats
    method: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "seq_length": self.seq_length,
            "latency": self.latency.to_dict(),
            "breakdown": self.breakdown.to_dict(),
            "memory": self.memory.to_dict(),
            "method": self.method
        }


@dataclass
class QuantizationResult:
    """Result for quantization benchmark."""
    precision: str
    model_size_mb: float
    latency: LatencyStats
    memory: MemoryStats
    throughput_vs_fp32: float  # Relative to fp32 baseline

    def to_dict(self) -> Dict[str, Any]:
        return {
            "precision": self.precision,
            "model_size_mb": self.model_size_mb,
            "latency": self.latency.to_dict(),
            "memory": self.memory.to_dict(),
            "throughput_vs_fp32": self.throughput_vs_fp32
        }


# =============================================================================
# Perceiver Bridge Architecture (from latent_bridge.py)
# =============================================================================

class PerceiverResampler(nn.Module):
    """
    Perceiver-style cross-attention resampler.
    Compresses variable-length input to fixed-length latent sequence.
    """
    def __init__(self, src_dim: int, tgt_dim: int, num_latents: int = 64,
                 heads: int = 8, depth: int = 4):
        super().__init__()
        self.num_latents = num_latents
        self.tgt_dim = tgt_dim

        # Learnable latent queries
        self.latents = nn.Parameter(torch.randn(num_latents, tgt_dim) * 0.02)

        # Project source to target dimension
        self.input_proj = nn.Linear(src_dim, tgt_dim) if src_dim != tgt_dim else nn.Identity()

        # Transformer layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(tgt_dim, heads, batch_first=True),
                "ln1": nn.LayerNorm(tgt_dim),
                "self_attn": nn.MultiheadAttention(tgt_dim, heads, batch_first=True),
                "ln2": nn.LayerNorm(tgt_dim),
                "ffn": nn.Sequential(
                    nn.Linear(tgt_dim, 4 * tgt_dim),
                    nn.GELU(),
                    nn.Linear(4 * tgt_dim, tgt_dim)
                ),
                "ln3": nn.LayerNorm(tgt_dim)
            }) for _ in range(depth)
        ])

    def forward(self, src_hidden: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src_hidden: [B, T, src_dim] - Source hidden states
            src_mask: [B, T] - attention mask (1=valid, 0=pad)
        Returns:
            [B, num_latents, tgt_dim] - Compressed representation
        """
        B = src_hidden.shape[0]

        # Project source to target dimension
        keys = self.input_proj(src_hidden)

        # Expand latent queries for batch
        x = self.latents.unsqueeze(0).expand(B, -1, -1).to(keys.dtype)

        # Key padding mask for attention (True = ignore)
        key_padding_mask = ~src_mask.bool() if src_mask is not None else None

        for layer in self.layers:
            # Cross-attention to source
            x_norm = layer["ln1"](x)
            attn_out, _ = layer["cross_attn"](
                query=x_norm, key=keys, value=keys,
                key_padding_mask=key_padding_mask,
                need_weights=False
            )
            x = x + attn_out

            # Self-attention among latents
            x_norm = layer["ln2"](x)
            attn_out, _ = layer["self_attn"](
                query=x_norm, key=x_norm, value=x_norm,
                need_weights=False
            )
            x = x + attn_out

            # FFN
            x = x + layer["ffn"](layer["ln3"](x))

        return x


class TelepathyBridge(nn.Module):
    """
    Production Telepathy Bridge for benchmarking.

    Transforms hidden states from sender LLM to soft tokens for receiver LLM.
    Uses Perceiver architecture for efficient compression.
    """
    def __init__(
        self,
        sender_dim: int = 4096,
        receiver_dim: int = 4096,
        num_soft_tokens: int = 16,
        heads: int = 8,
        depth: int = 4,
        target_rms: float = 0.03
    ):
        super().__init__()
        self.sender_dim = sender_dim
        self.receiver_dim = receiver_dim
        self.num_soft_tokens = num_soft_tokens

        # Perceiver resampler
        self.resampler = PerceiverResampler(
            src_dim=sender_dim,
            tgt_dim=receiver_dim,
            num_latents=num_soft_tokens,
            heads=heads,
            depth=depth
        )

        # Output scale to match receiver embedding magnitude
        self.output_scale = nn.Parameter(torch.tensor(target_rms))

    def forward(self, src_hidden: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Transform sender hidden states to soft tokens for receiver.

        Args:
            src_hidden: [B, T, sender_dim] - Sender hidden states
            src_mask: [B, T] - Attention mask

        Returns:
            [B, num_soft_tokens, receiver_dim] - Soft tokens for receiver
        """
        # Compress through Perceiver
        compressed = self.resampler(src_hidden, src_mask)  # [B, K, tgt_dim]

        # RMS normalization and scaling
        rms = torch.sqrt((compressed ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        out = (compressed / rms) * self.output_scale

        return out


# =============================================================================
# GPU Timing Utilities
# =============================================================================

class CUDATimer:
    """
    Accurate GPU timing using CUDA events.

    Usage:
        timer = CUDATimer()
        timer.start()
        # ... GPU operations ...
        elapsed_ms = timer.stop()
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        """Record start time."""
        torch.cuda.synchronize(self.device)
        self.start_event.record(torch.cuda.current_stream(self.device))

    def stop(self) -> float:
        """Record end time and return elapsed milliseconds."""
        self.end_event.record(torch.cuda.current_stream(self.device))
        torch.cuda.synchronize(self.device)
        return self.start_event.elapsed_time(self.end_event)


def reset_memory_stats(device: torch.device):
    """Reset GPU memory statistics for accurate measurement."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)


def get_memory_stats(device: torch.device) -> MemoryStats:
    """Get current GPU memory statistics."""
    torch.cuda.synchronize(device)
    return MemoryStats(
        allocated_mb=torch.cuda.memory_allocated(device) / 1024 / 1024,
        reserved_mb=torch.cuda.memory_reserved(device) / 1024 / 1024,
        peak_allocated_mb=torch.cuda.max_memory_allocated(device) / 1024 / 1024,
        peak_reserved_mb=torch.cuda.max_memory_reserved(device) / 1024 / 1024
    )


def compute_latency_stats(timings_ms: List[float]) -> LatencyStats:
    """Compute latency statistics from timing measurements."""
    arr = np.array(timings_ms)
    total_time_s = arr.sum() / 1000

    return LatencyStats(
        mean_ms=float(np.mean(arr)),
        std_ms=float(np.std(arr)),
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        samples_per_second=len(arr) / total_time_s if total_time_s > 0 else 0,
        num_samples=len(arr)
    )


# =============================================================================
# Model Loading
# =============================================================================

def load_llama(device: torch.device, dtype: torch.dtype = torch.bfloat16,
               quantization: Optional[str] = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load Llama model with optional quantization."""
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    print(f"Loading Llama from {model_id} (dtype={dtype}, quant={quantization})...")

    load_kwargs = {
        "torch_dtype": dtype,
        "device_map": device if quantization is None else "auto",
    }

    if quantization == "int8" and HAS_BITSANDBYTES:
        load_kwargs["load_in_8bit"] = True
        del load_kwargs["torch_dtype"]
    elif quantization == "int4" and HAS_BITSANDBYTES:
        load_kwargs["load_in_4bit"] = True
        load_kwargs["bnb_4bit_compute_dtype"] = dtype
        del load_kwargs["torch_dtype"]

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_mistral(device: torch.device, dtype: torch.dtype = torch.bfloat16,
                 quantization: Optional[str] = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load Mistral model with optional quantization."""
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    print(f"Loading Mistral from {model_id} (dtype={dtype}, quant={quantization})...")

    load_kwargs = {
        "torch_dtype": dtype,
        "device_map": device if quantization is None else "auto",
    }

    if quantization == "int8" and HAS_BITSANDBYTES:
        load_kwargs["load_in_8bit"] = True
        del load_kwargs["torch_dtype"]
    elif quantization == "int4" and HAS_BITSANDBYTES:
        load_kwargs["load_in_4bit"] = True
        load_kwargs["bnb_4bit_compute_dtype"] = dtype
        del load_kwargs["torch_dtype"]

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024


# =============================================================================
# Throughput Benchmark
# =============================================================================

class ThroughputBenchmark:
    """
    Benchmark throughput at various batch sizes and sequence lengths.

    Measures:
    - Samples per second
    - Component-wise latency breakdown
    - Memory usage
    """

    def __init__(
        self,
        device: torch.device,
        num_warmup: int = 10,
        num_iterations: int = 50,
        verbose: bool = True
    ):
        self.device = device
        self.num_warmup = num_warmup
        self.num_iterations = num_iterations
        self.verbose = verbose

        # Models will be loaded on demand
        self.llama = None
        self.llama_tok = None
        self.mistral = None
        self.mistral_tok = None
        self.bridge = None

    def _ensure_models_loaded(self, dtype: torch.dtype = torch.bfloat16):
        """Load models if not already loaded."""
        if self.llama is None:
            self.llama, self.llama_tok = load_llama(self.device, dtype)
            self.llama.eval()

        if self.mistral is None:
            self.mistral, self.mistral_tok = load_mistral(self.device, dtype)
            self.mistral.eval()

        if self.bridge is None:
            self.bridge = TelepathyBridge(
                sender_dim=4096,
                receiver_dim=4096,
                num_soft_tokens=16,
                heads=8,
                depth=4
            ).to(self.device).to(dtype)
            self.bridge.eval()

    def _create_synthetic_batch(
        self,
        batch_size: int,
        seq_length: int,
        tokenizer: AutoTokenizer
    ) -> Dict[str, torch.Tensor]:
        """Create synthetic batch for benchmarking."""
        # Create varied input texts
        texts = [
            f"This is test sentence number {i} for benchmarking. " * (seq_length // 20)
            for i in range(batch_size)
        ]

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=seq_length
        )

        return {k: v.to(self.device) for k, v in inputs.items()}

    def benchmark_bridge(
        self,
        batch_size: int,
        seq_length: int,
        dtype: torch.dtype = torch.bfloat16
    ) -> ThroughputResult:
        """
        Benchmark bridge inference with component-wise timing.

        Pipeline:
        1. Llama encode (forward pass, extract hidden states)
        2. Bridge transform (Perceiver compression)
        3. Mistral decode (forward pass with soft tokens)
        """
        self._ensure_models_loaded(dtype)

        if self.verbose:
            print(f"\n  Benchmarking Bridge: batch={batch_size}, seq={seq_length}")

        # Create batch
        inputs = self._create_synthetic_batch(batch_size, seq_length, self.llama_tok)

        timer = CUDATimer(self.device)
        encode_times = []
        bridge_times = []
        decode_times = []
        total_times = []

        # Warmup
        if self.verbose:
            print(f"    Warmup ({self.num_warmup} iterations)...")
        for _ in range(self.num_warmup):
            with torch.no_grad():
                # Encode
                out = self.llama(**inputs, output_hidden_states=True)
                hidden = out.hidden_states[-1]

                # Bridge
                soft_tokens = self.bridge(hidden, inputs["attention_mask"])

                # Decode
                _ = self.mistral(inputs_embeds=soft_tokens, use_cache=False)

        # Reset memory for measurement
        reset_memory_stats(self.device)

        # Timed runs
        if self.verbose:
            print(f"    Measuring ({self.num_iterations} iterations)...")

        for i in range(self.num_iterations):
            # Total time
            timer.start()
            total_start = time.perf_counter()

            with torch.no_grad():
                # Encode phase
                timer_encode = CUDATimer(self.device)
                timer_encode.start()
                out = self.llama(**inputs, output_hidden_states=True)
                hidden = out.hidden_states[-1]
                encode_ms = timer_encode.stop()

                # Bridge phase
                timer_bridge = CUDATimer(self.device)
                timer_bridge.start()
                soft_tokens = self.bridge(hidden, inputs["attention_mask"])
                bridge_ms = timer_bridge.stop()

                # Decode phase
                timer_decode = CUDATimer(self.device)
                timer_decode.start()
                _ = self.mistral(inputs_embeds=soft_tokens, use_cache=False)
                decode_ms = timer_decode.stop()

            total_ms = timer.stop()

            encode_times.append(encode_ms)
            bridge_times.append(bridge_ms)
            decode_times.append(decode_ms)
            total_times.append(total_ms)

        # Get memory stats
        memory = get_memory_stats(self.device)

        # Compute statistics
        latency = compute_latency_stats(total_times)
        # Adjust samples per second for batch size
        latency = LatencyStats(
            mean_ms=latency.mean_ms,
            std_ms=latency.std_ms,
            min_ms=latency.min_ms,
            max_ms=latency.max_ms,
            p50_ms=latency.p50_ms,
            p95_ms=latency.p95_ms,
            p99_ms=latency.p99_ms,
            samples_per_second=batch_size * 1000 / latency.mean_ms,
            num_samples=latency.num_samples
        )

        breakdown = LatencyBreakdown(
            encode_ms=float(np.mean(encode_times)),
            bridge_ms=float(np.mean(bridge_times)),
            decode_ms=float(np.mean(decode_times)),
            total_ms=latency.mean_ms
        )

        if self.verbose:
            print(f"    Results: {latency.samples_per_second:.1f} samples/s, "
                  f"latency={latency.mean_ms:.1f}ms (p99={latency.p99_ms:.1f}ms)")
            print(f"    Breakdown: encode={breakdown.encode_ms:.1f}ms, "
                  f"bridge={breakdown.bridge_ms:.1f}ms, decode={breakdown.decode_ms:.1f}ms")

        return ThroughputResult(
            batch_size=batch_size,
            seq_length=seq_length,
            latency=latency,
            breakdown=breakdown,
            memory=memory,
            method="bridge"
        )

    def benchmark_text_generation(
        self,
        batch_size: int,
        seq_length: int,
        max_new_tokens: int = 50,
        dtype: torch.dtype = torch.bfloat16
    ) -> ThroughputResult:
        """
        Benchmark text generation (autoregressive) for comparison.

        This represents the text-relay baseline where:
        1. Llama generates text summary
        2. Mistral processes that text
        """
        self._ensure_models_loaded(dtype)

        if self.verbose:
            print(f"\n  Benchmarking Text Generation: batch={batch_size}, seq={seq_length}")

        # Create batch
        inputs = self._create_synthetic_batch(batch_size, seq_length, self.llama_tok)

        timer = CUDATimer(self.device)
        generation_times = []

        # Warmup
        if self.verbose:
            print(f"    Warmup ({self.num_warmup} iterations)...")
        for _ in range(self.num_warmup):
            with torch.no_grad():
                _ = self.llama.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.llama_tok.eos_token_id
                )

        # Reset memory for measurement
        reset_memory_stats(self.device)

        # Timed runs
        if self.verbose:
            print(f"    Measuring ({self.num_iterations} iterations)...")

        for i in range(self.num_iterations):
            timer.start()

            with torch.no_grad():
                _ = self.llama.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.llama_tok.eos_token_id
                )

            elapsed_ms = timer.stop()
            generation_times.append(elapsed_ms)

        # Get memory stats
        memory = get_memory_stats(self.device)

        # Compute statistics
        latency = compute_latency_stats(generation_times)
        latency = LatencyStats(
            mean_ms=latency.mean_ms,
            std_ms=latency.std_ms,
            min_ms=latency.min_ms,
            max_ms=latency.max_ms,
            p50_ms=latency.p50_ms,
            p95_ms=latency.p95_ms,
            p99_ms=latency.p99_ms,
            samples_per_second=batch_size * 1000 / latency.mean_ms,
            num_samples=latency.num_samples
        )

        breakdown = LatencyBreakdown(
            encode_ms=0,
            bridge_ms=0,
            decode_ms=latency.mean_ms,
            total_ms=latency.mean_ms
        )

        if self.verbose:
            print(f"    Results: {latency.samples_per_second:.1f} samples/s, "
                  f"latency={latency.mean_ms:.1f}ms (p99={latency.p99_ms:.1f}ms)")

        return ThroughputResult(
            batch_size=batch_size,
            seq_length=seq_length,
            latency=latency,
            breakdown=breakdown,
            memory=memory,
            method="text_generation"
        )

    def run_batch_size_sweep(
        self,
        batch_sizes: List[int] = [1, 4, 8, 16, 32, 64, 128],
        seq_length: int = 256
    ) -> Dict[str, List[ThroughputResult]]:
        """Run throughput benchmark across batch sizes."""
        results = {
            "bridge": [],
            "text_generation": []
        }

        print("\n" + "=" * 70)
        print("THROUGHPUT BENCHMARK: Batch Size Sweep")
        print("=" * 70)

        for batch_size in batch_sizes:
            try:
                # Bridge benchmark
                bridge_result = self.benchmark_bridge(batch_size, seq_length)
                results["bridge"].append(bridge_result)

                # Text generation benchmark
                text_result = self.benchmark_text_generation(batch_size, seq_length)
                results["text_generation"].append(text_result)

                # Calculate speedup
                speedup = text_result.latency.mean_ms / bridge_result.latency.mean_ms
                print(f"  Speedup at batch={batch_size}: {speedup:.1f}x")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  OOM at batch_size={batch_size}, stopping sweep")
                    break
                raise

        return results

    def run_sequence_length_sweep(
        self,
        seq_lengths: List[int] = [128, 256, 512, 1024],
        batch_size: int = 8
    ) -> Dict[str, List[ThroughputResult]]:
        """Run throughput benchmark across sequence lengths."""
        results = {
            "bridge": [],
            "text_generation": []
        }

        print("\n" + "=" * 70)
        print("THROUGHPUT BENCHMARK: Sequence Length Sweep")
        print("=" * 70)

        for seq_length in seq_lengths:
            try:
                # Bridge benchmark
                bridge_result = self.benchmark_bridge(batch_size, seq_length)
                results["bridge"].append(bridge_result)

                # Text generation benchmark
                text_result = self.benchmark_text_generation(batch_size, seq_length)
                results["text_generation"].append(text_result)

                # Calculate speedup
                speedup = text_result.latency.mean_ms / bridge_result.latency.mean_ms
                print(f"  Speedup at seq_len={seq_length}: {speedup:.1f}x")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  OOM at seq_length={seq_length}, stopping sweep")
                    break
                raise

        return results

    def cleanup(self):
        """Clean up loaded models."""
        del self.llama, self.mistral, self.bridge
        self.llama = self.mistral = self.bridge = None
        gc.collect()
        torch.cuda.empty_cache()


# =============================================================================
# Quantization Benchmark
# =============================================================================

class QuantizationBenchmark:
    """
    Benchmark impact of quantization on latency and throughput.

    Tests: fp32, fp16/bf16, int8, int4
    """

    def __init__(
        self,
        device: torch.device,
        num_warmup: int = 10,
        num_iterations: int = 50,
        verbose: bool = True
    ):
        self.device = device
        self.num_warmup = num_warmup
        self.num_iterations = num_iterations
        self.verbose = verbose

    def benchmark_precision(
        self,
        precision: str,
        batch_size: int = 8,
        seq_length: int = 256
    ) -> QuantizationResult:
        """Benchmark a specific precision level."""
        if self.verbose:
            print(f"\n  Benchmarking precision: {precision}")

        # Determine dtype and quantization
        if precision == "fp32":
            dtype = torch.float32
            quant = None
        elif precision == "fp16":
            dtype = torch.float16
            quant = None
        elif precision == "bf16":
            dtype = torch.bfloat16
            quant = None
        elif precision == "int8":
            dtype = torch.bfloat16
            quant = "int8"
        elif precision == "int4":
            dtype = torch.bfloat16
            quant = "int4"
        else:
            raise ValueError(f"Unknown precision: {precision}")

        # Check if quantization is available
        if quant and not HAS_BITSANDBYTES:
            print(f"    Skipping {precision}: bitsandbytes not available")
            return None

        # Load model
        model, tokenizer = load_llama(self.device, dtype, quant)
        model.eval()

        model_size = get_model_size_mb(model)

        # Create batch
        texts = [f"Test input {i} " * 20 for i in range(batch_size)]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=seq_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        timer = CUDATimer(self.device)
        timings = []

        # Warmup
        if self.verbose:
            print(f"    Warmup ({self.num_warmup} iterations)...")
        for _ in range(self.num_warmup):
            with torch.no_grad():
                _ = model(**inputs, output_hidden_states=True)

        # Reset memory
        reset_memory_stats(self.device)

        # Timed runs
        if self.verbose:
            print(f"    Measuring ({self.num_iterations} iterations)...")

        for _ in range(self.num_iterations):
            timer.start()
            with torch.no_grad():
                _ = model(**inputs, output_hidden_states=True)
            elapsed_ms = timer.stop()
            timings.append(elapsed_ms)

        # Get memory stats
        memory = get_memory_stats(self.device)

        # Compute statistics
        latency = compute_latency_stats(timings)
        latency = LatencyStats(
            mean_ms=latency.mean_ms,
            std_ms=latency.std_ms,
            min_ms=latency.min_ms,
            max_ms=latency.max_ms,
            p50_ms=latency.p50_ms,
            p95_ms=latency.p95_ms,
            p99_ms=latency.p99_ms,
            samples_per_second=batch_size * 1000 / latency.mean_ms,
            num_samples=latency.num_samples
        )

        if self.verbose:
            print(f"    Results: {latency.samples_per_second:.1f} samples/s, "
                  f"model_size={model_size:.0f}MB, memory={memory.peak_allocated_mb:.0f}MB")

        # Cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()

        return QuantizationResult(
            precision=precision,
            model_size_mb=model_size,
            latency=latency,
            memory=memory,
            throughput_vs_fp32=1.0  # Will be computed later
        )

    def run_all_precisions(
        self,
        batch_size: int = 8,
        seq_length: int = 256
    ) -> List[QuantizationResult]:
        """Benchmark all precision levels."""
        print("\n" + "=" * 70)
        print("QUANTIZATION BENCHMARK")
        print("=" * 70)

        precisions = ["fp32", "bf16", "fp16"]
        if HAS_BITSANDBYTES:
            precisions.extend(["int8", "int4"])

        results = []
        fp32_throughput = None

        for precision in precisions:
            result = self.benchmark_precision(precision, batch_size, seq_length)
            if result:
                results.append(result)
                if precision == "fp32":
                    fp32_throughput = result.latency.samples_per_second

        # Compute relative throughput
        if fp32_throughput:
            for result in results:
                result.throughput_vs_fp32 = result.latency.samples_per_second / fp32_throughput

        return results


# =============================================================================
# Memory Benchmark
# =============================================================================

class MemoryBenchmark:
    """
    Benchmark memory usage and scaling.
    """

    def __init__(self, device: torch.device, verbose: bool = True):
        self.device = device
        self.verbose = verbose

    def measure_model_memory(self) -> Dict[str, Any]:
        """Measure memory usage of each component."""
        print("\n" + "=" * 70)
        print("MEMORY BENCHMARK")
        print("=" * 70)

        results = {}

        # Baseline
        reset_memory_stats(self.device)
        baseline = get_memory_stats(self.device)

        # Llama
        print("\n  Loading Llama...")
        llama, _ = load_llama(self.device)
        llama_memory = get_memory_stats(self.device)
        llama_size = get_model_size_mb(llama)
        results["llama"] = {
            "model_size_mb": llama_size,
            "gpu_allocated_mb": llama_memory.allocated_mb - baseline.allocated_mb,
            "gpu_reserved_mb": llama_memory.reserved_mb - baseline.reserved_mb
        }
        print(f"    Llama: {llama_size:.0f}MB model, {results['llama']['gpu_allocated_mb']:.0f}MB GPU")

        # Mistral
        print("  Loading Mistral...")
        mistral, _ = load_mistral(self.device)
        both_memory = get_memory_stats(self.device)
        mistral_size = get_model_size_mb(mistral)
        results["mistral"] = {
            "model_size_mb": mistral_size,
            "gpu_allocated_mb": both_memory.allocated_mb - llama_memory.allocated_mb,
            "gpu_reserved_mb": both_memory.reserved_mb - llama_memory.reserved_mb
        }
        print(f"    Mistral: {mistral_size:.0f}MB model, {results['mistral']['gpu_allocated_mb']:.0f}MB GPU")

        # Bridge
        print("  Creating Bridge...")
        bridge = TelepathyBridge(
            sender_dim=4096,
            receiver_dim=4096,
            num_soft_tokens=16,
            heads=8,
            depth=4
        ).to(self.device).to(torch.bfloat16)
        bridge_memory = get_memory_stats(self.device)
        bridge_size = get_model_size_mb(bridge)
        results["bridge"] = {
            "model_size_mb": bridge_size,
            "gpu_allocated_mb": bridge_memory.allocated_mb - both_memory.allocated_mb,
            "gpu_reserved_mb": bridge_memory.reserved_mb - both_memory.reserved_mb
        }
        print(f"    Bridge: {bridge_size:.1f}MB model, {results['bridge']['gpu_allocated_mb']:.1f}MB GPU")

        # Total system
        results["total_system"] = {
            "gpu_allocated_mb": bridge_memory.allocated_mb,
            "gpu_reserved_mb": bridge_memory.reserved_mb,
            "total_model_size_mb": llama_size + mistral_size + bridge_size
        }
        print(f"\n  Total System: {results['total_system']['gpu_allocated_mb']:.0f}MB GPU allocated")

        # Cleanup
        del llama, mistral, bridge
        gc.collect()
        torch.cuda.empty_cache()

        return results

    def measure_activation_memory(
        self,
        batch_sizes: List[int] = [1, 4, 8, 16, 32]
    ) -> Dict[int, MemoryStats]:
        """Measure activation memory at different batch sizes."""
        print("\n  Measuring activation memory scaling...")

        llama, llama_tok = load_llama(self.device)
        llama.eval()

        results = {}

        for batch_size in batch_sizes:
            try:
                # Create batch
                texts = [f"Test {i} " * 50 for i in range(batch_size)]
                inputs = llama_tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                reset_memory_stats(self.device)

                with torch.no_grad():
                    _ = llama(**inputs, output_hidden_states=True)

                memory = get_memory_stats(self.device)
                results[batch_size] = memory

                if self.verbose:
                    print(f"    batch={batch_size}: peak={memory.peak_allocated_mb:.0f}MB")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"    batch={batch_size}: OOM")
                    break
                raise

        del llama
        gc.collect()
        torch.cuda.empty_cache()

        return results


# =============================================================================
# Paper-Ready Visualization and Tables
# =============================================================================

def create_latency_table(throughput_results: Dict[str, List[ThroughputResult]]) -> str:
    """Create LaTeX table for latency results."""
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Latency breakdown by batch size (ms)}",
        "\\label{tab:latency}",
        "\\begin{tabular}{l|r|rrr|rr}",
        "\\toprule",
        "Batch & Method & Encode & Bridge & Decode & p50 & p99 \\\\",
        "\\midrule"
    ]

    bridge_results = throughput_results.get("bridge", [])
    text_results = throughput_results.get("text_generation", [])

    for i, bridge_r in enumerate(bridge_results):
        bs = bridge_r.batch_size
        lines.append(
            f"{bs} & Bridge & {bridge_r.breakdown.encode_ms:.1f} & "
            f"{bridge_r.breakdown.bridge_ms:.1f} & {bridge_r.breakdown.decode_ms:.1f} & "
            f"{bridge_r.latency.p50_ms:.1f} & {bridge_r.latency.p99_ms:.1f} \\\\"
        )
        if i < len(text_results):
            text_r = text_results[i]
            lines.append(
                f" & Text-Gen & - & - & {text_r.breakdown.decode_ms:.1f} & "
                f"{text_r.latency.p50_ms:.1f} & {text_r.latency.p99_ms:.1f} \\\\"
            )
        if i < len(bridge_results) - 1:
            lines.append("\\midrule")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])

    return "\n".join(lines)


def create_throughput_table(throughput_results: Dict[str, List[ThroughputResult]]) -> str:
    """Create LaTeX table for throughput results."""
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Throughput comparison (samples/second)}",
        "\\label{tab:throughput}",
        "\\begin{tabular}{r|rr|r}",
        "\\toprule",
        "Batch Size & Bridge & Text-Gen & Speedup \\\\",
        "\\midrule"
    ]

    bridge_results = throughput_results.get("bridge", [])
    text_results = throughput_results.get("text_generation", [])

    for i, bridge_r in enumerate(bridge_results):
        bs = bridge_r.batch_size
        bridge_tp = bridge_r.latency.samples_per_second

        if i < len(text_results):
            text_tp = text_results[i].latency.samples_per_second
            speedup = text_results[i].latency.mean_ms / bridge_r.latency.mean_ms
            lines.append(f"{bs} & {bridge_tp:.1f} & {text_tp:.1f} & {speedup:.1f}$\\times$ \\\\")
        else:
            lines.append(f"{bs} & {bridge_tp:.1f} & - & - \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])

    return "\n".join(lines)


def create_quantization_table(quant_results: List[QuantizationResult]) -> str:
    """Create LaTeX table for quantization results."""
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Impact of quantization on performance}",
        "\\label{tab:quantization}",
        "\\begin{tabular}{l|r|r|r|r}",
        "\\toprule",
        "Precision & Model (MB) & Memory (MB) & Throughput & Relative \\\\",
        "\\midrule"
    ]

    for result in quant_results:
        lines.append(
            f"{result.precision} & {result.model_size_mb:.0f} & "
            f"{result.memory.peak_allocated_mb:.0f} & {result.latency.samples_per_second:.1f} & "
            f"{result.throughput_vs_fp32:.2f}$\\times$ \\\\"
        )

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])

    return "\n".join(lines)


def create_plots(results: Dict[str, Any], output_dir: Path):
    """Create matplotlib plots for paper figures."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    # Plot 1: Throughput vs Batch Size
    if "throughput_batch_sweep" in results:
        fig, ax = plt.subplots(figsize=(8, 5))

        bridge_data = results["throughput_batch_sweep"]["bridge"]
        text_data = results["throughput_batch_sweep"]["text_generation"]

        batch_sizes = [r["batch_size"] for r in bridge_data]
        bridge_tp = [r["latency"]["samples_per_second"] for r in bridge_data]
        text_tp = [r["latency"]["samples_per_second"] for r in text_data]

        ax.plot(batch_sizes, bridge_tp, 'o-', label='Bridge', linewidth=2, markersize=8)
        ax.plot(batch_sizes[:len(text_tp)], text_tp, 's-', label='Text Generation', linewidth=2, markersize=8)

        ax.set_xlabel('Batch Size', fontsize=12)
        ax.set_ylabel('Throughput (samples/sec)', fontsize=12)
        ax.set_title('Throughput Scaling with Batch Size', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(output_dir / 'throughput_vs_batch.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'throughput_vs_batch.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Plot 2: Latency Breakdown
    if "throughput_batch_sweep" in results:
        fig, ax = plt.subplots(figsize=(10, 5))

        bridge_data = results["throughput_batch_sweep"]["bridge"]
        batch_sizes = [r["batch_size"] for r in bridge_data]

        encode = [r["breakdown"]["encode_ms"] for r in bridge_data]
        bridge = [r["breakdown"]["bridge_ms"] for r in bridge_data]
        decode = [r["breakdown"]["decode_ms"] for r in bridge_data]

        x = np.arange(len(batch_sizes))
        width = 0.25

        ax.bar(x - width, encode, width, label='Encode (Llama)', color='#2ecc71')
        ax.bar(x, bridge, width, label='Bridge', color='#3498db')
        ax.bar(x + width, decode, width, label='Decode (Mistral)', color='#e74c3c')

        ax.set_xlabel('Batch Size', fontsize=12)
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.set_title('Latency Breakdown by Component', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(batch_sizes)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_dir / 'latency_breakdown.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'latency_breakdown.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Plot 3: Speedup vs Text Generation
    if "throughput_batch_sweep" in results:
        fig, ax = plt.subplots(figsize=(8, 5))

        bridge_data = results["throughput_batch_sweep"]["bridge"]
        text_data = results["throughput_batch_sweep"]["text_generation"]

        batch_sizes = [r["batch_size"] for r in bridge_data]
        speedups = []

        for i, bridge_r in enumerate(bridge_data):
            if i < len(text_data):
                speedup = text_data[i]["latency"]["mean_ms"] / bridge_r["latency"]["mean_ms"]
                speedups.append(speedup)

        ax.bar(range(len(speedups)), speedups, color='#9b59b6', alpha=0.8)
        ax.axhline(y=22, color='red', linestyle='--', label='Claimed 22x speedup')

        ax.set_xlabel('Batch Size', fontsize=12)
        ax.set_ylabel('Speedup vs Text Generation', fontsize=12)
        ax.set_title('Bridge Speedup over Text Generation', fontsize=14)
        ax.set_xticks(range(len(speedups)))
        ax.set_xticklabels(batch_sizes[:len(speedups)])
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_dir / 'speedup_comparison.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'speedup_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  Plots saved to {output_dir}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Production Metrics System for Telepathy Paper"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/production_metrics",
        help="Directory to save results"
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["all", "throughput", "quantization", "memory"],
        default="all",
        help="Which benchmark to run"
    )
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16, 32, 64, 128],
        help="Batch sizes for throughput benchmark"
    )
    parser.add_argument(
        "--seq_lengths",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024],
        help="Sequence lengths for throughput benchmark"
    )
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=10,
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=50,
        help="Number of measurement iterations"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with fewer iterations and smaller batch sizes"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device to use"
    )
    args = parser.parse_args()

    # Quick mode adjustments
    if args.quick:
        args.batch_sizes = [1, 4, 8, 16]
        args.seq_lengths = [128, 256]
        args.num_warmup = 3
        args.num_iterations = 10

    # Setup
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TELEPATHY PRODUCTION METRICS SYSTEM")
    print("=" * 70)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"Output directory: {output_dir}")
    print(f"Test mode: {args.test}")
    print(f"Warmup iterations: {args.num_warmup}")
    print(f"Measurement iterations: {args.num_iterations}")
    print("=" * 70)

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if torch.cuda.is_available() else "N/A",
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "pytorch_version": torch.__version__,
            "num_warmup": args.num_warmup,
            "num_iterations": args.num_iterations,
            "batch_sizes": args.batch_sizes,
            "seq_lengths": args.seq_lengths
        }
    }

    # Run benchmarks
    if args.test in ["all", "throughput"]:
        benchmark = ThroughputBenchmark(
            device=device,
            num_warmup=args.num_warmup,
            num_iterations=args.num_iterations
        )

        # Batch size sweep
        batch_results = benchmark.run_batch_size_sweep(
            batch_sizes=args.batch_sizes,
            seq_length=256
        )
        results["throughput_batch_sweep"] = {
            k: [r.to_dict() for r in v] for k, v in batch_results.items()
        }

        # Sequence length sweep
        seq_results = benchmark.run_sequence_length_sweep(
            seq_lengths=args.seq_lengths,
            batch_size=8
        )
        results["throughput_seq_sweep"] = {
            k: [r.to_dict() for r in v] for k, v in seq_results.items()
        }

        benchmark.cleanup()

        # Calculate headline speedup number
        if batch_results["bridge"] and batch_results["text_generation"]:
            # Average speedup across batch sizes
            speedups = []
            for i, bridge_r in enumerate(batch_results["bridge"]):
                if i < len(batch_results["text_generation"]):
                    text_r = batch_results["text_generation"][i]
                    speedup = text_r.latency.mean_ms / bridge_r.latency.mean_ms
                    speedups.append(speedup)

            results["headline_speedup"] = {
                "min": min(speedups),
                "max": max(speedups),
                "mean": np.mean(speedups),
                "speedups_by_batch": speedups
            }
            print(f"\n  HEADLINE: Bridge is {np.mean(speedups):.1f}x faster than text generation "
                  f"(range: {min(speedups):.1f}x - {max(speedups):.1f}x)")

    if args.test in ["all", "quantization"]:
        quant_benchmark = QuantizationBenchmark(
            device=device,
            num_warmup=args.num_warmup,
            num_iterations=args.num_iterations
        )

        quant_results = quant_benchmark.run_all_precisions(batch_size=8, seq_length=256)
        results["quantization"] = [r.to_dict() for r in quant_results if r]

    if args.test in ["all", "memory"]:
        mem_benchmark = MemoryBenchmark(device=device)

        model_memory = mem_benchmark.measure_model_memory()
        results["memory_models"] = model_memory

        activation_memory = mem_benchmark.measure_activation_memory(
            batch_sizes=[1, 4, 8, 16, 32]
        )
        results["memory_activations"] = {
            str(k): v.to_dict() for k, v in activation_memory.items()
        }

    # Save results
    results_file = output_dir / "production_metrics.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Generate LaTeX tables
    if "throughput_batch_sweep" in results:
        # Convert back to ThroughputResult objects for table generation
        bridge_results = [
            ThroughputResult(
                batch_size=r["batch_size"],
                seq_length=r["seq_length"],
                latency=LatencyStats(**r["latency"]),
                breakdown=LatencyBreakdown(**r["breakdown"]),
                memory=MemoryStats(**r["memory"]),
                method=r["method"]
            )
            for r in results["throughput_batch_sweep"]["bridge"]
        ]
        text_results = [
            ThroughputResult(
                batch_size=r["batch_size"],
                seq_length=r["seq_length"],
                latency=LatencyStats(**r["latency"]),
                breakdown=LatencyBreakdown(**r["breakdown"]),
                memory=MemoryStats(**r["memory"]),
                method=r["method"]
            )
            for r in results["throughput_batch_sweep"]["text_generation"]
        ]

        throughput_dict = {"bridge": bridge_results, "text_generation": text_results}

        latency_table = create_latency_table(throughput_dict)
        throughput_table = create_throughput_table(throughput_dict)

        with open(output_dir / "latency_table.tex", "w") as f:
            f.write(latency_table)
        with open(output_dir / "throughput_table.tex", "w") as f:
            f.write(throughput_table)
        print(f"LaTeX tables saved to {output_dir}")

    if "quantization" in results:
        quant_results = [
            QuantizationResult(
                precision=r["precision"],
                model_size_mb=r["model_size_mb"],
                latency=LatencyStats(**r["latency"]),
                memory=MemoryStats(**r["memory"]),
                throughput_vs_fp32=r["throughput_vs_fp32"]
            )
            for r in results["quantization"]
        ]
        quant_table = create_quantization_table(quant_results)
        with open(output_dir / "quantization_table.tex", "w") as f:
            f.write(quant_table)

    # Generate plots
    create_plots(results, output_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"Results: {results_file}")
    print(f"Tables: {output_dir}/*.tex")
    print(f"Plots: {output_dir}/*.pdf, {output_dir}/*.png")

    if "headline_speedup" in results:
        print(f"\nHEADLINE RESULT:")
        print(f"  Bridge achieves {results['headline_speedup']['mean']:.1f}x speedup over text generation")
        print(f"  Range: {results['headline_speedup']['min']:.1f}x - {results['headline_speedup']['max']:.1f}x")


if __name__ == "__main__":
    main()
