"""
LLMLingua Baseline for LatentWire Comparison

This module provides a fair comparison between LLMLingua prompt compression
and LatentWire's learned interlingua compression on QA tasks.

LLMLingua is a state-of-the-art prompt compression method that uses a small
language model (e.g., GPT-2, LLaMA-7B) to identify and remove non-essential
tokens based on perplexity. It achieves up to 20x compression with minimal
performance loss on many tasks.

Key features:
- Token-level compression via perplexity-based pruning
- Budget controller for maintaining semantic integrity
- Question-aware compression (LongLLMLingua variant)
- No model fine-tuning required (works with black-box LLMs)

Installation:
    pip install llmlingua

References:
- LLMLingua: https://arxiv.org/abs/2310.05736
- LLMLingua-2: https://arxiv.org/abs/2403.12968
- GitHub: https://github.com/microsoft/LLMLingua
"""

import os
import json
import time
import argparse
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    from llmlingua import PromptCompressor
    LLMLINGUA_AVAILABLE = True
except ImportError:
    LLMLINGUA_AVAILABLE = False
    print("WARNING: llmlingua not installed. Install with: pip install llmlingua")

from latentwire.data import load_examples
from latentwire.core_utils import em, f1, _normalize


# ==============================================================================
# LLMLingua Compression
# ==============================================================================

class LLMLinguaCompressor:
    """
    Wrapper for LLMLingua prompt compression with fair comparison settings.

    Supports:
    - LLMLingua (original): Perplexity-based compression with causal LM
    - LLMLingua-2: BERT-based encoder for bidirectional context
    - LongLLMLingua: Question-aware compression for long contexts
    """

    def __init__(
        self,
        model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        use_llmlingua2: bool = True,
        device: str = "cpu",
        model_config: Optional[Dict] = None,
    ):
        """
        Initialize LLMLingua compressor.

        Args:
            model_name: Model for compression. Options:
                - "microsoft/llmlingua-2-xlm-roberta-large-meetingbank" (LLMLingua-2)
                - "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
                - "NousResearch/Llama-2-7b-hf" (original LLMLingua)
                - "microsoft/phi-2" (compact alternative)
                - "gpt2" (smallest, fastest)
            use_llmlingua2: Use LLMLingua-2 architecture (faster, bidirectional)
            device: Device for compression model
            model_config: Additional model configuration
        """
        if not LLMLINGUA_AVAILABLE:
            raise ImportError(
                "llmlingua not installed. Install with: pip install llmlingua"
            )

        self.model_name = model_name
        self.use_llmlingua2 = use_llmlingua2

        print(f"Loading LLMLingua compressor: {model_name}")
        print(f"  Architecture: {'LLMLingua-2' if use_llmlingua2 else 'LLMLingua'}")
        print(f"  Device: {device}")

        config = model_config or {}
        if "device_map" not in config:
            config["device_map"] = device

        self.compressor = PromptCompressor(
            model_name=model_name,
            use_llmlingua2=use_llmlingua2,
            model_config=config,
        )

        print("LLMLingua compressor loaded successfully")

    def compress_to_target_tokens(
        self,
        prompt: str,
        target_tokens: int,
        question: Optional[str] = None,
        instruction: str = "",
        force_tokens: Optional[List[str]] = None,
        chunk_end_tokens: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compress prompt to target number of tokens.

        Args:
            prompt: Text to compress
            target_tokens: Target number of tokens after compression
            question: Question for question-aware compression (optional)
            instruction: System instruction (optional)
            force_tokens: Tokens to always preserve (e.g., ['\n', '?', '!'])
            chunk_end_tokens: Tokens marking chunk boundaries

        Returns:
            Dict with:
                - compressed_prompt: Compressed text
                - origin_tokens: Original token count
                - compressed_tokens: Compressed token count
                - ratio: Compression ratio (original / compressed)
                - saving: Estimated token cost savings
                - compression_time: Time taken for compression
        """
        start_time = time.time()

        kwargs = {
            "target_token": target_tokens,
        }

        if question:
            kwargs["question"] = question

        if instruction:
            kwargs["instruction"] = instruction

        if force_tokens:
            kwargs["force_tokens"] = force_tokens

        if chunk_end_tokens:
            kwargs["chunk_end_tokens"] = chunk_end_tokens

        # LLMLingua-2 uses 'rate' instead of 'target_token'
        # But we'll use target_token for consistency
        result = self.compressor.compress_prompt(prompt, **kwargs)

        compression_time = time.time() - start_time

        return {
            "compressed_prompt": result["compressed_prompt"],
            "origin_tokens": result["origin_tokens"],
            "compressed_tokens": result["compressed_tokens"],
            "ratio": result["ratio"],
            "saving": result.get("saving", "N/A"),
            "compression_time": compression_time,
        }

    def compress_to_target_ratio(
        self,
        prompt: str,
        target_ratio: float,
        question: Optional[str] = None,
        instruction: str = "",
        force_tokens: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compress prompt to target compression ratio.

        Args:
            prompt: Text to compress
            target_ratio: Target compression ratio (e.g., 0.5 = 50% compression)
                         Lower values = more compression
            question: Question for question-aware compression
            instruction: System instruction
            force_tokens: Tokens to always preserve

        Returns:
            Same as compress_to_target_tokens()
        """
        start_time = time.time()

        kwargs = {
            "rate": target_ratio,
        }

        if question:
            kwargs["question"] = question

        if instruction:
            kwargs["instruction"] = instruction

        if force_tokens:
            kwargs["force_tokens"] = force_tokens

        result = self.compressor.compress_prompt(prompt, **kwargs)

        compression_time = time.time() - start_time

        return {
            "compressed_prompt": result["compressed_prompt"],
            "origin_tokens": result["origin_tokens"],
            "compressed_tokens": result["compressed_tokens"],
            "ratio": result["ratio"],
            "saving": result.get("saving", "N/A"),
            "compression_time": compression_time,
        }


# ==============================================================================
# Evaluation Against LatentWire
# ==============================================================================

def evaluate_llmlingua_on_qa(
    dataset: str = "squad",
    split: str = "validation",
    samples: int = 200,
    target_tokens: int = 32,
    use_llmlingua2: bool = True,
    compressor_model: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    output_dir: str = "runs/llmlingua_baseline",
    question_aware: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate LLMLingua compression on QA task for fair comparison with LatentWire.

    This evaluates COMPRESSION QUALITY ONLY - it does NOT evaluate generation quality
    because LLMLingua produces text that must be fed to an LLM separately.

    For fair comparison with LatentWire:
    1. Both methods compress prompts to same target token budget
    2. LLMLingua produces compressed text tokens
    3. LatentWire produces soft latent vectors
    4. Both should then feed their representations to the SAME LLM

    This function measures:
    - Compression ratio achieved
    - Token reduction statistics
    - Compression time/throughput

    Args:
        dataset: Dataset name (squad, hotpotqa)
        split: Dataset split
        samples: Number of examples to evaluate
        target_tokens: Target compressed token count (M in LatentWire notation)
        use_llmlingua2: Use LLMLingua-2 (faster, bidirectional)
        compressor_model: Model for compression
        output_dir: Where to save results
        question_aware: Use question-aware compression
        seed: Random seed

    Returns:
        Dictionary with compression statistics and per-example results
    """
    print("=" * 80)
    print("LLMLingua Baseline Evaluation")
    print("=" * 80)
    print(f"Dataset: {dataset} ({split})")
    print(f"Samples: {samples}")
    print(f"Target tokens: {target_tokens}")
    print(f"Compressor: {compressor_model}")
    print(f"LLMLingua-2: {use_llmlingua2}")
    print(f"Question-aware: {question_aware}")
    print("=" * 80)
    print()

    # Load data
    print("Loading dataset...")
    examples = load_examples(dataset, split=split, samples=samples, seed=seed)
    print(f"Loaded {len(examples)} examples")
    print()

    # Initialize compressor
    print("Initializing LLMLingua compressor...")
    compressor = LLMLinguaCompressor(
        model_name=compressor_model,
        use_llmlingua2=use_llmlingua2,
        device="cpu",  # Use CPU for compression (small model)
    )
    print()

    # Compress each example
    results = []
    total_compression_time = 0.0

    print("Compressing prompts...")
    for i, ex in enumerate(examples):
        context = ex["context"]
        question = ex["question"]
        gold_answer = ex["answer"]

        # Build prompt (same format as LatentWire eval)
        prompt = f"{context}\n\nQuestion: {question}"

        # Compress
        if question_aware:
            compressed = compressor.compress_to_target_tokens(
                prompt=context,  # Compress only context
                target_tokens=target_tokens,
                question=question,  # Question guides compression
                force_tokens=['\n', '?', '!', '.'],  # Preserve structure
            )
        else:
            compressed = compressor.compress_to_target_tokens(
                prompt=prompt,
                target_tokens=target_tokens,
                force_tokens=['\n', '?', '!', '.'],
            )

        total_compression_time += compressed["compression_time"]

        # Store results
        result = {
            "example_id": i,
            "question": question,
            "gold_answer": gold_answer,
            "original_prompt": prompt,
            "compressed_prompt": compressed["compressed_prompt"],
            "origin_tokens": compressed["origin_tokens"],
            "compressed_tokens": compressed["compressed_tokens"],
            "compression_ratio": compressed["ratio"],
            "compression_time": compressed["compression_time"],
        }
        results.append(result)

        # Progress
        if (i + 1) % 50 == 0:
            avg_time = total_compression_time / (i + 1)
            print(f"  Processed {i + 1}/{len(examples)} "
                  f"(avg time: {avg_time*1000:.1f}ms, "
                  f"avg ratio: {sum(r['compression_ratio'] for r in results)/len(results):.2f}x)")

    print()

    # Compute statistics
    avg_compression_ratio = sum(r["compression_ratio"] for r in results) / len(results)
    avg_origin_tokens = sum(r["origin_tokens"] for r in results) / len(results)
    avg_compressed_tokens = sum(r["compressed_tokens"] for r in results) / len(results)
    avg_compression_time = total_compression_time / len(results)
    throughput = len(results) / total_compression_time

    summary = {
        "dataset": dataset,
        "split": split,
        "samples": len(examples),
        "target_tokens": target_tokens,
        "compressor_model": compressor_model,
        "use_llmlingua2": use_llmlingua2,
        "question_aware": question_aware,
        "avg_compression_ratio": avg_compression_ratio,
        "avg_origin_tokens": avg_origin_tokens,
        "avg_compressed_tokens": avg_compressed_tokens,
        "avg_compression_time_ms": avg_compression_time * 1000,
        "throughput_examples_per_sec": throughput,
        "total_compression_time_sec": total_compression_time,
    }

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    results_file = Path(output_dir) / "llmlingua_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "summary": summary,
            "examples": results,
        }, f, indent=2)

    print("=" * 80)
    print("LLMLingua Compression Results")
    print("=" * 80)
    print(f"Average compression ratio: {avg_compression_ratio:.2f}x")
    print(f"Average original tokens: {avg_origin_tokens:.1f}")
    print(f"Average compressed tokens: {avg_compressed_tokens:.1f}")
    print(f"Average compression time: {avg_compression_time*1000:.1f}ms")
    print(f"Throughput: {throughput:.2f} examples/sec")
    print()
    print(f"Results saved to: {results_file}")
    print("=" * 80)

    return summary


# ==============================================================================
# Fair Comparison Notes
# ==============================================================================

"""
FAIR COMPARISON BETWEEN LLMLINGUA AND LATENTWIRE:

1. COMPRESSION BUDGET:
   - Both methods should compress to same target (e.g., 32 tokens for LatentWire = M)
   - LLMLingua: Produces ~32 text tokens after pruning
   - LatentWire: Produces 32 soft latent vectors

2. WIRE COST:
   - LLMLingua: Count UTF-8 bytes of compressed text
   - LatentWire: Count bytes of quantized latents (fp16/int8/int6/int4)
   - Both should achieve similar compression ratio for fairness

3. GENERATION QUALITY:
   - Both compressed representations should be fed to SAME target LLM
   - LLMLingua: Compressed text → tokenize → LLM
   - LatentWire: Latents → adapter → LLM
   - Compare EM/F1 on same evaluation set

4. KNOWN LIMITATIONS OF LLMLINGUA:

   a) Unidirectional context (LLMLingua v1):
      - Uses causal LM perplexity → only left context
      - May miss important info that requires right context
      - Fixed in LLMLingua-2 with bidirectional BERT encoder

   b) Question-agnostic (basic mode):
      - Compresses context without knowing what question will be asked
      - May remove info crucial for answering specific questions
      - Fixed with question-aware compression (LongLLMLingua)

   c) Task-specific performance:
      - Excels on reasoning tasks (GSM8K, BBH) with 20x compression
      - More modest gains on conversation/summarization (ShareGPT, Arxiv)
      - Random selection sometimes competitive at 2x compression

   d) No model fine-tuning:
      - Works with black-box LLMs (strength and limitation)
      - Cannot adapt LLM to compressed representation
      - LatentWire trains adapters for better alignment

   e) Compression plateau:
      - Performance degrades significantly beyond ~20x compression
      - LatentWire targets 4-8x with learned representations

   f) Re-compression overhead:
      - Question-aware mode requires re-compression for each question
      - Cannot cache compressed context for multiple questions
      - LatentWire compresses once, answers many questions

5. STRENGTHS OF LLMLINGUA:

   a) No training required:
      - Works immediately with any LLM
      - No need for checkpoints, gradients, or optimization

   b) Interpretable compression:
      - Compressed prompt is still readable text
      - Can manually inspect what was removed

   c) Faster inference (sometimes):
      - Small compression model (BERT-base) is fast
      - LLMLingua-2 is 3-6x faster than original

   d) Strong reasoning performance:
      - Up to 20 points better than baselines on GSM8K at 20x compression
      - Maintains chain-of-thought better than phrase-level methods

   e) Flexible compression control:
      - Easy to target specific token budgets or ratios
      - Can preserve specific tokens/patterns via force_tokens

6. WHEN TO PREFER EACH METHOD:

   LLMLingua:
   - Black-box LLM APIs (GPT-4, Claude)
   - No training budget or data
   - Interpretability required
   - Single-question-per-context scenarios

   LatentWire:
   - Multiple questions per context (amortize compression)
   - Own models that can be fine-tuned
   - Extreme compression needed (>20x)
   - Cross-model conditioning (Llama + Qwen)

7. RECOMMENDED BASELINES FOR COMPARISON:

   a) Text baseline: Full prompt (upper bound)
   b) Token-budget baseline: Truncate to M tokens (fairness check)
   c) LLMLingua (question-agnostic): Compress without question
   d) LLMLingua (question-aware): Compress with question
   e) LLMLingua-2: Bidirectional compression
   f) Random token selection: Simple baseline
   g) Selective-Context: Phrase-level self-information baseline

8. METRICS TO REPORT:

   Compression:
   - Compression ratio (original tokens / compressed tokens)
   - Wire bytes (UTF-8 for text, quantized bytes for latents)
   - Compression time

   Quality:
   - EM (exact match)
   - F1 (token overlap)
   - NLL on gold answer (conditioning quality)
   - First-token accuracy (for generation)

   Efficiency:
   - Throughput (examples/sec)
   - Latency (ms/example)
   - GPU memory usage

EXAMPLE USAGE:

```python
# Run LLMLingua baseline at same budget as LatentWire (M=32)
python latentwire/llmlingua_baseline.py \\
    --dataset squad \\
    --samples 200 \\
    --target_tokens 32 \\
    --use_llmlingua2 \\
    --question_aware \\
    --output_dir runs/llmlingua_m32

# Compare multiple compression budgets
for M in 32 48 64 96 128; do
    python latentwire/llmlingua_baseline.py \\
        --dataset squad \\
        --samples 200 \\
        --target_tokens $M \\
        --output_dir runs/llmlingua_m${M}
done

# Ablation: Question-aware vs agnostic
python latentwire/llmlingua_baseline.py \\
    --dataset squad \\
    --samples 200 \\
    --target_tokens 32 \\
    --question_aware \\
    --output_dir runs/llmlingua_qaware

python latentwire/llmlingua_baseline.py \\
    --dataset squad \\
    --samples 200 \\
    --target_tokens 32 \\
    --no-question_aware \\
    --output_dir runs/llmlingua_qagnostic

# Compare LLMLingua-2 (bidirectional) vs original (unidirectional)
python latentwire/llmlingua_baseline.py \\
    --dataset squad \\
    --samples 200 \\
    --target_tokens 32 \\
    --use_llmlingua2 \\
    --output_dir runs/llmlingua2

python latentwire/llmlingua_baseline.py \\
    --dataset squad \\
    --samples 200 \\
    --target_tokens 32 \\
    --no-use_llmlingua2 \\
    --compressor_model NousResearch/Llama-2-7b-hf \\
    --output_dir runs/llmlingua1
```

To evaluate generation quality, the compressed prompts must be fed to an LLM:

```python
# This requires integration with latentwire/eval.py to use same LLM
# See run_llmlingua_generation_eval.sh for full pipeline
```
"""


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LLMLingua baseline for LatentWire comparison"
    )

    # Dataset args
    parser.add_argument("--dataset", type=str, default="squad",
                        help="Dataset: squad, hotpotqa")
    parser.add_argument("--split", type=str, default="validation",
                        help="Dataset split")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of examples to evaluate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Compression args
    parser.add_argument("--target_tokens", type=int, default=32,
                        help="Target compressed token count (M in LatentWire)")
    parser.add_argument("--target_ratio", type=float, default=None,
                        help="Target compression ratio (alternative to target_tokens)")
    parser.add_argument("--question_aware", action="store_true", default=True,
                        help="Use question-aware compression")
    parser.add_argument("--no-question_aware", action="store_false", dest="question_aware",
                        help="Disable question-aware compression")

    # Model args
    parser.add_argument("--use_llmlingua2", action="store_true", default=True,
                        help="Use LLMLingua-2 (bidirectional BERT)")
    parser.add_argument("--no-use_llmlingua2", action="store_false", dest="use_llmlingua2",
                        help="Use original LLMLingua (unidirectional causal LM)")
    parser.add_argument("--compressor_model", type=str,
                        default="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                        help="Model for compression")

    # Output args
    parser.add_argument("--output_dir", type=str, default="runs/llmlingua_baseline",
                        help="Output directory")

    args = parser.parse_args()

    # Run evaluation
    summary = evaluate_llmlingua_on_qa(
        dataset=args.dataset,
        split=args.split,
        samples=args.samples,
        target_tokens=args.target_tokens,
        use_llmlingua2=args.use_llmlingua2,
        compressor_model=args.compressor_model,
        output_dir=args.output_dir,
        question_aware=args.question_aware,
        seed=args.seed,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
