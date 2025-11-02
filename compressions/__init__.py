"""Compression experiments for prompt/context compression.

This package contains experiments for compressing long prompts/contexts into
shorter representations while maintaining task performance.

Architectures:
- Cross-Attention: Learnable query-based compression (Perceiver-like)
- Convolutional: Strided conv-based compression
- Weighted Pooling: Window-based weighted pooling
- Gist: Learnable gist tokens (Mu et al., NeurIPS 2023)

Usage:
    From compressions import run_experiments
    python compressions/run_experiments.py
"""

from .config import CompressionConfig
from .models import (
    CrossAttentionCompressor,
    ConvolutionalCompressor,
    WeightedPoolingCompressor,
    GistCompressor,
    create_compressor
)
from .dataset import SQuADCompressionDataset

__all__ = [
    'CompressionConfig',
    'CrossAttentionCompressor',
    'ConvolutionalCompressor',
    'WeightedPoolingCompressor',
    'GistCompressor',
    'create_compressor',
    'SQuADCompressionDataset',
]
