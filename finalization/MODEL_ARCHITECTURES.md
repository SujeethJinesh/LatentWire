# LatentWire Model Architectures

This document describes the key model architectures in the LatentWire system for continuous interlingua-based prompt compression.

## Core Architecture Files

- **models.py**: Main model definitions (encoders, adapters, wrappers)
- **losses.py**: Loss computation modules (K-token CE, KD losses)
- **loss_bundles.py**: Loss configuration bundles
- **config.py**: Model hyperparameter configurations
- **core_utils.py**: Utility functions for model operations
- **features/**: Additional architectural components
  - **latent_adapters.py**: Adapter architectures for latent space
  - **deep_prefix.py**: Deep prefix tuning implementation
  - **coproc.py**: Co-processing features

## Primary Model Components

### 1. Encoder (`ByteLevelEncoder`, `EncoderFromEmbedLayer`)
- **Purpose**: Compresses input text into continuous latent representations
- **Input**: Raw text or token embeddings
- **Output**: Latent tensor Z ∈ R^{M × d_z} where M is the number of soft tokens
- **Key Features**:
  - Byte-level encoding support
  - Transformer-based architecture
  - Configurable latent dimensions (M, d_z)

### 2. Adapter (`Adapter`, `LearnedAdapter`)
- **Purpose**: Maps shared latent space to model-specific embedding spaces
- **Architecture**: Linear projection layers with optional normalization
- **Key Features**:
  - Model-specific parameters while keeping base LLMs frozen
  - Preserves statistical properties of embeddings
  - Lightweight (typically <1% of LLM parameters)

### 3. LMWrapper
- **Purpose**: Wraps frozen LLMs (Llama, Qwen) with adapters
- **Key Features**:
  - Manages inputs_embeds injection
  - Handles BOS/PAD token policies
  - Supports both training and inference modes
  - Implements calibration (RMS scaling)

### 4. Loss Modules
- **K-token Cross-Entropy**: Supervises first K tokens during training
- **Knowledge Distillation**: Distills teacher distributions from text-prompted models
- **Prefix KD**: Specialized KD for prefix-based conditioning

## Key Architectural Design Choices

### Frozen LLM Paradigm
- Base language models (Llama-3.1-8B, Qwen2.5-7B) remain completely frozen
- Only encoder and adapters are trained (~10M parameters vs 7-8B frozen)
- Enables cross-model transfer without catastrophic forgetting

### Continuous Latent Representation
- Soft tokens (continuous vectors) instead of discrete tokens
- Allows gradient-based optimization
- Supports arbitrary compression ratios (not limited to vocabulary)

### Model-Agnostic Interlingua
- Single encoder produces representations consumable by multiple LLMs
- No model-specific tokenization in latent space
- Enables heterogeneous model ensembles

### Calibration Mechanisms
- **embed_rms**: Scales latents to match embedding RMS statistics
- **Per-example**: Avoids batch-level amplitude drift
- **Anchor text**: Provides consistent starting point for generation

## Configuration Parameters

Key hyperparameters from config.py:
```python
LATENT_LEN = 32        # Number of soft tokens (M)
D_Z = 256              # Latent dimension per token
ENCODER_TYPE = "byte"  # Byte-level or embedding-based
K = 4                  # K-token supervision window
FIRST_TOKEN_CE = 0.5   # Weight for first-token loss
KD_TAU = 1.0          # KD temperature
```

## Training Flow

1. **Encode**: Text → Encoder → Latent Z
2. **Adapt**: Z → Adapter → Model-specific embeddings
3. **Generate**: Embeddings → Frozen LLM → Output tokens
4. **Loss**: K-token CE + KD losses guide learning

## Inference Flow

1. **Compress**: Input text → Encoder → Latent representation
2. **Transmit**: Send compressed latent (4-8× smaller than text)
3. **Decompress**: Latent → Adapter → LLM → Generated text

## Feature Extensions

### LoRA Integration (Optional)
- Supports LoRA adapters on frozen LLMs
- Target modules: attention (q,k,v,o) and MLP (gate,up,down)
- Can limit to first N layers for efficiency

### Prefix Tuning (Optional)
- Virtual token prepending
- Learnable prefix embeddings
- Alternative to adapter-based approach

### Deep Prefix Features
- Multi-layer prefix representations
- Hierarchical latent injection
- Enhanced expressiveness

## Memory Requirements

Typical memory footprint for training:
- Encoder: ~10M parameters
- Adapters: ~2M parameters per LLM
- Total trainable: ~14M parameters
- Frozen LLMs: 7-8B parameters each (no gradients)

## Performance Targets

- **Compression**: 4-8× reduction in prompt size
- **Quality**: F1 score within 10-20% of full text baseline
- **First-token accuracy**: 12-20% for cold-start generation
- **Throughput**: Process >1000 samples/minute on H100