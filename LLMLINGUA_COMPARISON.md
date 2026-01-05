# LLMLingua vs LatentWire: Comprehensive Comparison

This document provides a detailed comparison between LLMLingua prompt compression and LatentWire's learned interlingua compression.

## Table of Contents

1. [How LLMLingua Works](#how-llmlingua-works)
2. [Installation and Usage](#installation-and-usage)
3. [Compression Ratios](#compression-ratios)
4. [Fair Comparison Guidelines](#fair-comparison-guidelines)
5. [Known Limitations](#known-limitations)
6. [When to Use Each Method](#when-to-use-each-method)
7. [Code Examples](#code-examples)

---

## How LLMLingua Works

### Core Algorithm

LLMLingua uses a **coarse-to-fine prompt compression** approach based on perplexity (PPL) from a small language model:

1. **Budget Controller**: Dynamically allocates different compression ratios to various prompt components (instruction, demonstrations, question) to maintain semantic integrity under high compression ratios.

2. **Iterative Token-level Compression**:
   - First eliminates certain sentences at a coarse level
   - Then individually compresses remaining tokens
   - Uses iterative refinement to preserve coherence between tokens

3. **Alignment** (optional): Fine-tunes the small compression model to capture distribution patterns from target LLMs via instruction tuning.

### LLMLingua vs LLMLingua-2

| Feature | LLMLingua (v1) | LLMLingua-2 |
|---------|----------------|-------------|
| **Architecture** | Causal LM (GPT-2, LLaMA) | BERT-based encoder |
| **Context** | Unidirectional (left-to-right) | Bidirectional |
| **Speed** | Baseline | 3-6x faster |
| **Compression metric** | Perplexity-based | Data distillation from GPT-4 |
| **Task focus** | General | Task-agnostic |

**Key insight**: LLMLingua-2 addresses the limitation that perplexity from a causal LM only leverages unidirectional context, which may fail to capture all essential information needed for compression.

### LongLLMLingua (Question-Aware)

An extension that performs **question-aware compression**:
- Compresses context based on what question will be asked
- Better preserves relevant information for specific queries
- **Limitation**: Requires re-compression for each new question (cannot cache compressed context)

---

## Installation and Usage

### Installation

```bash
pip install llmlingua

# For quantized models (optional, requires GPU):
pip install optimum auto-gptq
```

### Basic Usage

```python
from llmlingua import PromptCompressor

# Initialize compressor
llm_lingua = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True,  # Use LLMLingua-2 (recommended)
)

# Compress to target token count
compressed = llm_lingua.compress_prompt(
    prompt="[Your long prompt here...]",
    target_token=32,  # Target compressed length
    question="What is the answer?",  # Optional: question-aware compression
    force_tokens=['\n', '?', '!', '.'],  # Preserve important tokens
)

print(f"Original: {compressed['origin_tokens']} tokens")
print(f"Compressed: {compressed['compressed_tokens']} tokens")
print(f"Ratio: {compressed['ratio']:.2f}x")
print(f"Compressed prompt: {compressed['compressed_prompt']}")
```

### Model Options

```python
# LLMLingua-2 (fastest, bidirectional, recommended)
llm_lingua = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True,
)

# LLMLingua-2 smaller variant
llm_lingua = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    use_llmlingua2=True,
)

# Original LLMLingua with GPT-2 (smallest, CPU-friendly)
llm_lingua = PromptCompressor(
    model_name="gpt2",
    use_llmlingua2=False,
)

# Original LLMLingua with LLaMA-2-7B (higher quality, needs GPU)
llm_lingua = PromptCompressor(
    model_name="NousResearch/Llama-2-7b-hf",
    use_llmlingua2=False,
)

# Quantized model (for GPUs with limited memory)
llm_lingua = PromptCompressor(
    model_name="TheBloke/Llama-2-7b-Chat-GPTQ",
    model_config={"revision": "main"},
)
```

### Compression Control

```python
# Compress to specific token count
compressed = llm_lingua.compress_prompt(
    prompt=prompt,
    target_token=32,  # Target 32 tokens
)

# Compress to target ratio
compressed = llm_lingua.compress_prompt(
    prompt=prompt,
    rate=0.5,  # Keep 50% of tokens (0.5 = less compression, 0.1 = more compression)
)

# Question-aware compression
compressed = llm_lingua.compress_prompt(
    prompt=context,
    target_token=32,
    question=question,  # Guides compression based on question
    force_tokens=['\n', '?', '!', '.'],  # Preserve structural tokens
)

# Advanced settings
compressed = llm_lingua.compress_prompt(
    prompt=prompt,
    target_token=32,
    instruction="Answer the following question.",  # System instruction
    context_budget="+100",  # Add extra context budget
    dynamic_context_compression_ratio=0.3,  # Dynamic adjustment
    reorder_context="sort",  # Reorder for relevance
)
```

---

## Compression Ratios

### Reported Performance

| Dataset | Compression | Performance Change | Notes |
|---------|-------------|-------------------|-------|
| GSM8K (math) | 20x | -1.5 points | Best performance |
| BBH (reasoning) | 20x | Minimal loss | Strong on reasoning |
| ShareGPT (chat) | 2x | Minimal gain | Modest on conversation |
| Arxiv (summarization) | 2x | Minimal gain | Modest on summarization |
| MuSicQue (multi-hop QA) | 4x | +17.1% | With LongLLMLingua |

**Key finding**: LLMLingua excels on **reasoning tasks** (GSM8K, BBH) with dramatic improvements at high compression (up to 20x). Performance on **conversation/summarization tasks** is more modest, with random selection sometimes competitive at 2x compression.

### Comparison to Baselines

On GSM8K at 20x compression:
- **LLMLingua**: 33.10 points better than Selective-Context baseline
- **LLMLingua**: Significantly better than random sentence selection
- **LLMLingua-2**: Outperforms LLMLingua on out-of-domain data

**Important**: Performance varies significantly by task type. Always benchmark on your specific use case.

---

## Fair Comparison Guidelines

### 1. Compression Budget

**Goal**: Both methods should compress to the same target.

- **LatentWire**: M=32 soft latent vectors (32 × 256 = 8,192 values)
- **LLMLingua**: ~32 text tokens after pruning

Set `target_token=32` in LLMLingua to match LatentWire's M=32.

### 2. Wire Cost Measurement

**Critical**: Compare actual bytes transmitted, not just token counts.

#### LLMLingua Wire Cost
```python
compressed_text = compressed["compressed_prompt"]
wire_bytes = len(compressed_text.encode("utf-8"))
```

#### LatentWire Wire Cost
```python
# For M=32 tokens, d_z=256 dimension
M, d_z = 32, 256

# fp16 quantization
wire_bytes = M * d_z * 2  # 16,384 bytes

# int8 quantization (1 byte per value + scale overhead)
wire_bytes = M * d_z * 1 + M * 4  # 8,320 bytes

# int6 quantization (0.75 bytes per value + scale overhead)
wire_bytes = (M * d_z * 6 + 7) // 8 + M * 4  # 6,272 bytes

# int4 quantization (0.5 bytes per value + scale overhead)
wire_bytes = M * d_z // 2 + M * 4  # 4,224 bytes
```

**For fair comparison**: LatentWire should use int6 or int4 quantization to achieve similar wire cost to LLMLingua's compressed text.

### 3. Generation Quality Evaluation

**Important**: LLMLingua only compresses prompts. It does NOT generate answers. For fair comparison, both methods must use the **same target LLM**.

```python
# LLMLingua pipeline
compressed_prompt = llm_lingua.compress_prompt(prompt, target_token=32)
answer = target_llm.generate(compressed_prompt["compressed_prompt"])

# LatentWire pipeline
latent = encoder(prompt)  # M=32 soft vectors
answer = target_llm.generate(inputs_embeds=adapter(latent))

# Compare EM/F1 scores on same evaluation set
```

### 4. Recommended Baselines

For a comprehensive comparison, include:

1. **Text baseline**: Full prompt (upper bound)
2. **Token-budget baseline**: Truncate to M tokens (fairness check)
3. **Random token selection**: Simple baseline
4. **Selective-Context**: Phrase-level self-information baseline
5. **LLMLingua (question-agnostic)**: Compress without question
6. **LLMLingua (question-aware)**: Compress with question guidance
7. **LLMLingua-2**: Bidirectional compression

### 5. Metrics to Report

| Category | Metric | LLMLingua | LatentWire |
|----------|--------|-----------|------------|
| **Compression** | Compression ratio | original / compressed tokens | original / M |
| | Wire bytes | UTF-8 bytes | Quantized latent bytes |
| | Compression time | ms/example | Training time + inference |
| **Quality** | EM (exact match) | % exact matches | % exact matches |
| | F1 | Token overlap | Token overlap |
| | First-token accuracy | N/A (generates text tokens) | % correct first token |
| | NLL on gold answer | Perplexity | Perplexity |
| **Efficiency** | Throughput | examples/sec | examples/sec |
| | Latency | ms/example | ms/example |
| | Memory usage | Compression model size | Encoder + adapter size |

---

## Known Limitations

### LLMLingua Limitations

1. **Unidirectional context (v1 only)**
   - Causal LM only uses left context
   - May miss information requiring bidirectional understanding
   - **Mitigation**: Use LLMLingua-2 (bidirectional BERT encoder)

2. **Question-agnostic compression (basic mode)**
   - Compresses without knowing the question
   - May remove crucial information for specific queries
   - **Mitigation**: Use question-aware compression (LongLLMLingua)

3. **Task-specific performance**
   - Excels on reasoning (GSM8K: 20x compression, minimal loss)
   - Modest gains on conversation/summarization
   - Random selection sometimes competitive at 2x compression
   - **Mitigation**: Benchmark on your specific task

4. **No model adaptation**
   - Cannot fine-tune target LLM to understand compressed prompts
   - Compression is text-to-text (preserves tokenization overhead)
   - **LatentWire advantage**: Trains adapters for better alignment

5. **Compression plateau**
   - Performance degrades significantly beyond ~20x compression
   - **LatentWire advantage**: Targets extreme compression (>20x) with learned representations

6. **Re-compression overhead (question-aware mode)**
   - Must re-compress context for each new question
   - Cannot cache compressed context for multiple queries
   - **LatentWire advantage**: Compress once, answer many questions

7. **Text-based representation**
   - Compressed output is still discrete text tokens
   - Cannot leverage continuous representations
   - **LatentWire advantage**: Soft latent vectors for richer conditioning

### LatentWire Limitations

1. **Training required**
   - Needs training data, compute, and time
   - Cannot work with black-box API LLMs
   - **LLMLingua advantage**: Works immediately with any LLM

2. **Model-specific**
   - Trained for specific target models (Llama, Qwen)
   - Cannot generalize to arbitrary LLMs without retraining
   - **LLMLingua advantage**: Model-agnostic compression

3. **Non-interpretable compression**
   - Latent vectors are not human-readable
   - Cannot inspect what information was retained
   - **LLMLingua advantage**: Compressed text is interpretable

4. **Complexity**
   - More complex system (encoder, adapters, calibration, etc.)
   - Harder to debug and maintain
   - **LLMLingua advantage**: Simpler perplexity-based method

---

## When to Use Each Method

### Use LLMLingua When:

1. **Black-box LLM APIs** (GPT-4, Claude, etc.)
   - Cannot fine-tune the target model
   - Need to work with API-only models

2. **No training budget**
   - No compute resources for training
   - No training data available
   - Need immediate solution

3. **Interpretability required**
   - Need to inspect compressed prompts
   - Debugging or auditing compression
   - Human review of compressed content

4. **Single-question-per-context**
   - Each context used for one question only
   - Cannot amortize compression cost

5. **Reasoning tasks**
   - Math problems (GSM8K)
   - Logical reasoning (BBH)
   - Chain-of-thought scenarios

6. **Moderate compression** (2-20x)
   - Not targeting extreme compression
   - Acceptable performance at moderate ratios

### Use LatentWire When:

1. **Own models**
   - Can fine-tune target LLMs
   - Have control over model architecture
   - Can deploy custom inference

2. **Multiple questions per context**
   - Amortize compression cost across queries
   - Cache compressed representations
   - Many questions on same document

3. **Extreme compression needed** (>20x)
   - Need aggressive compression
   - Willing to trade quality for efficiency
   - Learned representations outperform pruning

4. **Cross-model conditioning**
   - Need to condition multiple heterogeneous models
   - Shared representation across models (Llama + Qwen)
   - Multi-model ensemble scenarios

5. **Continuous representations**
   - Leverage soft embeddings
   - Differentiable compression
   - Gradient-based optimization

6. **Training budget available**
   - Have compute resources (GPUs)
   - Have training data
   - Can invest in training time

### Hybrid Approach

Consider using **both** methods:

1. **LLMLingua for initial compression**: Reduce 10,000 tokens → 200 tokens
2. **LatentWire for final compression**: 200 tokens → 32 latent vectors
3. **Benefit**: Combine strengths of both approaches

---

## Code Examples

### Example 1: Basic Compression

```python
from llmlingua import PromptCompressor

# Initialize
compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True,
)

# Compress
prompt = "Long prompt with lots of context..."
compressed = compressor.compress_prompt(
    prompt=prompt,
    target_token=32,
)

print(f"Compressed from {compressed['origin_tokens']} to {compressed['compressed_tokens']} tokens")
print(f"Compression ratio: {compressed['ratio']:.2f}x")
```

### Example 2: Question-Aware Compression for QA

```python
# Question-aware compression for better QA performance
context = "The Eiffel Tower was built in 1889 for the World's Fair..."
question = "When was the Eiffel Tower built?"

compressed = compressor.compress_prompt(
    prompt=context,
    target_token=32,
    question=question,  # Guides compression
    force_tokens=['\n', '?', '!', '.', ','],  # Preserve structure
)

# Feed compressed context to LLM
full_prompt = f"{compressed['compressed_prompt']}\n\nQuestion: {question}\nAnswer:"
answer = llm.generate(full_prompt)
```

### Example 3: Batch Compression with Progress

```python
from latentwire.data import load_examples
from tqdm import tqdm

# Load dataset
examples = load_examples("squad", split="validation", samples=200)

# Compress all examples
results = []
for ex in tqdm(examples):
    compressed = compressor.compress_prompt(
        prompt=ex["context"],
        target_token=32,
        question=ex["question"],
    )

    results.append({
        "original_tokens": compressed["origin_tokens"],
        "compressed_tokens": compressed["compressed_tokens"],
        "ratio": compressed["ratio"],
        "compressed_text": compressed["compressed_prompt"],
    })

# Compute statistics
import statistics
avg_ratio = statistics.mean(r["ratio"] for r in results)
print(f"Average compression ratio: {avg_ratio:.2f}x")
```

### Example 4: Wire Cost Comparison

```python
import json

# LLMLingua wire cost
compressed = compressor.compress_prompt(prompt, target_token=32)
llmlingua_bytes = len(compressed["compressed_prompt"].encode("utf-8"))

# LatentWire wire cost (M=32, d_z=256, int8 quantization)
M, d_z = 32, 256
latentwire_bytes = M * d_z * 1 + M * 4  # int8 + scale overhead

# Compare
original_bytes = len(prompt.encode("utf-8"))

print(f"Original: {original_bytes} bytes")
print(f"LLMLingua: {llmlingua_bytes} bytes ({original_bytes/llmlingua_bytes:.2f}x compression)")
print(f"LatentWire (int8): {latentwire_bytes} bytes ({original_bytes/latentwire_bytes:.2f}x compression)")

# Save comparison
results = {
    "original_bytes": original_bytes,
    "llmlingua_bytes": llmlingua_bytes,
    "latentwire_int8_bytes": latentwire_bytes,
    "llmlingua_compression_ratio": original_bytes / llmlingua_bytes,
    "latentwire_compression_ratio": original_bytes / latentwire_bytes,
}

with open("wire_cost_comparison.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Example 5: Run LatentWire Baseline Experiments

```bash
# Install LLMLingua
pip install llmlingua

# Run baseline experiments at different compression budgets
bash scripts/run_llmlingua_baseline.sh

# Or run manually with custom settings
python latentwire/llmlingua_baseline.py \
    --dataset squad \
    --samples 200 \
    --target_tokens 32 \
    --use_llmlingua2 \
    --question_aware \
    --output_dir runs/llmlingua_m32

# Analyze results
python latentwire/analyze_llmlingua_results.py \
    --results_dir runs/llmlingua_baseline \
    --d_z 256 \
    --output runs/llmlingua_analysis.json
```

### Example 6: Compare Multiple Compression Methods

```python
import time

prompt = "Long context for compression..."
question = "What is the answer?"

methods = {
    "llmlingua2_qaware": {
        "model": "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        "use_llmlingua2": True,
        "question_aware": True,
    },
    "llmlingua2_qagnostic": {
        "model": "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        "use_llmlingua2": True,
        "question_aware": False,
    },
    "llmlingua1": {
        "model": "gpt2",
        "use_llmlingua2": False,
        "question_aware": True,
    },
}

results = {}

for name, config in methods.items():
    compressor = PromptCompressor(
        model_name=config["model"],
        use_llmlingua2=config["use_llmlingua2"],
    )

    start = time.time()

    if config["question_aware"]:
        compressed = compressor.compress_prompt(
            prompt=prompt,
            target_token=32,
            question=question,
        )
    else:
        compressed = compressor.compress_prompt(
            prompt=prompt,
            target_token=32,
        )

    elapsed = time.time() - start

    results[name] = {
        "ratio": compressed["ratio"],
        "compressed_tokens": compressed["compressed_tokens"],
        "time_ms": elapsed * 1000,
    }

# Print comparison
for name, res in results.items():
    print(f"{name}:")
    print(f"  Ratio: {res['ratio']:.2f}x")
    print(f"  Tokens: {res['compressed_tokens']}")
    print(f"  Time: {res['time_ms']:.1f}ms")
    print()
```

---

## Running Experiments

### Quick Start

```bash
# 1. Install LLMLingua
pip install llmlingua

# 2. Run baseline compression experiments
bash scripts/run_llmlingua_baseline.sh

# 3. Analyze results
python latentwire/analyze_llmlingua_results.py \
    --results_dir runs/llmlingua_baseline
```

### Custom Experiments

```bash
# Experiment 1: LLMLingua-2 at M=32 (match LatentWire)
python latentwire/llmlingua_baseline.py \
    --dataset squad \
    --samples 200 \
    --target_tokens 32 \
    --use_llmlingua2 \
    --question_aware \
    --output_dir runs/llmlingua_m32

# Experiment 2: Compare multiple compression budgets
for M in 32 48 64 96 128; do
    python latentwire/llmlingua_baseline.py \
        --dataset squad \
        --samples 200 \
        --target_tokens $M \
        --output_dir runs/llmlingua_m${M}
done

# Experiment 3: Ablation - question-aware vs agnostic
python latentwire/llmlingua_baseline.py \
    --dataset squad \
    --samples 200 \
    --target_tokens 32 \
    --question_aware \
    --output_dir runs/llmlingua_qaware

python latentwire/llmlingua_baseline.py \
    --dataset squad \
    --samples 200 \
    --target_tokens 32 \
    --no-question_aware \
    --output_dir runs/llmlingua_qagnostic

# Experiment 4: LLMLingua-2 vs original
python latentwire/llmlingua_baseline.py \
    --dataset squad \
    --samples 200 \
    --target_tokens 32 \
    --use_llmlingua2 \
    --output_dir runs/llmlingua2

python latentwire/llmlingua_baseline.py \
    --dataset squad \
    --samples 200 \
    --target_tokens 32 \
    --no-use_llmlingua2 \
    --compressor_model gpt2 \
    --output_dir runs/llmlingua1
```

---

## References

### Papers

- **LLMLingua**: [Compressing Prompts for Accelerated Inference of Large Language Models](https://arxiv.org/abs/2310.05736) (EMNLP 2023)
- **LLMLingua-2**: [Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression](https://arxiv.org/abs/2403.12968) (ACL 2024)
- **LongLLMLingua**: [Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression](https://arxiv.org/abs/2310.06839)

### Resources

- **Official Website**: https://llmlingua.com/
- **GitHub**: https://github.com/microsoft/LLMLingua
- **Microsoft Research Blog**: https://www.microsoft.com/en-us/research/blog/llmlingua-innovating-llm-efficiency-with-prompt-compression/

### Related Work

- **Selective-Context**: Phrase-level self-information baseline
- **GIST**: Soft prompt tuning for compression
- **AutoCompressor**: Learned compression via fine-tuning
- **500xCompressor**: Extreme compression via summarization

---

## Summary

**LLMLingua** is a strong baseline for prompt compression that:
- Works with any LLM (including APIs)
- Requires no training
- Achieves up to 20x compression with minimal loss on reasoning tasks
- Produces interpretable compressed text

**LatentWire** is a learned compression method that:
- Requires training on specific models
- Can achieve extreme compression (>20x)
- Supports cross-model conditioning
- Produces continuous latent representations

**For fair comparison**:
1. Match compression budgets (M=32 tokens)
2. Compare wire costs (bytes, not tokens)
3. Use same target LLM for generation
4. Report EM/F1 on same evaluation set
5. Include multiple baselines (random, truncation, Selective-Context)

Both methods have complementary strengths and can even be combined for hybrid compression pipelines.
