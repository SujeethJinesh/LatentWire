"""Configuration for compression experiments."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class CompressionConfig:
    """Configuration for compression experiments."""
    # Model
    model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # Compression
    target_length: int = 64  # M: number of compressed tokens
    architecture: str = "cross_attention"  # cross_attention, conv, pooling, gist

    # Training
    batch_size: int = 8
    gradient_accumulation: int = 4
    learning_rate: float = 1e-4
    lora_lr: float = 5e-5
    epochs: int = 10
    warmup_ratio: float = 0.05

    # Loss weights
    loss_weights: Dict[str, float] = None

    # Data
    num_train_samples: int = 10000
    num_eval_samples: int = 1000
    max_input_length: int = 512

    # System
    device: str = "cuda"
    gpu_id: int = 0
    use_bf16: bool = True
    seed: int = 42
    output_dir: str = "runs/compression_ablations"

    def __post_init__(self):
        if self.loss_weights is None:
            # Default: Balanced between teacher-forcing and generation
            self.loss_weights = {
                'teacher_forcing': 0.5,
                'generation': 0.3,
                'contrastive': 0.2
            }
