#!/usr/bin/env python3
"""
Learned Adapter Ablation: Training cross-model alignment with generation loss.

Tests whether LEARNED adapters (trained with next-token prediction) outperform
training-free Procrustes alignment for cross-model hidden state transfer.

Adapter Types:
1. Linear: Full 4096x4096 projection (16.8M params)
2. Affine: Full projection + bias (16.8M params)
3. LoRA-8: Low-rank adapter (65k params) - efficient baseline

Literature Foundation:
- arXiv:2505.20142: "Affine mappings... cheap way to transfer features"
- arXiv:2506.06609: "50% cheaper training with learned alignment"
- arXiv:2304.01933: "Adapter-based PEFT achieves comparable performance"
- arXiv:2502.02013: "Intermediate layers outperform final by 16%"

GPU Allocation: GPUs 1, 2, 3 (GPU 0 reserved for Procrustes experiment)
"""

import os
import sys
import torch
import torch.nn as nn
import math
import json
import time
import shutil
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import numpy as np
from collections import defaultdict

# GPU Allocation - MUST use GPUs 1-3 only
GPU_MAPPING = {
    "linear": 1,   # Linear adapter on GPU 1
    "affine": 2,   # Affine adapter on GPU 2
    "lora": 3,     # LoRA adapter on GPU 3
}

# Configuration
LLAMA_MODEL = "meta-llama/Llama-3.1-8B"  # Base model (not instruct)
MISTRAL_MODEL = "mistralai/Mistral-7B-v0.3"  # Base model (not instruct)
LAYER_IDX = 16  # Middle layer (best per literature arXiv:2502.02013)
OUTPUT_DIR = Path("runs/learned_adapters")

# Dataset splits (non-overlapping)
DATASET_SPLITS = {
    "linear": {"start": 0, "end": 1000},
    "affine": {"start": 1000, "end": 2000},
    "lora": {"start": 2000, "end": 3000},
}
EVAL_SPLIT = {"start": 0, "end": 500}  # From validation set
GENERATION_SPLIT = {"start": 500, "end": 550}  # 50 diverse prompts

# Training hyperparameters (validated against literature)
LEARNING_RATE = 1e-4  # Standard for adapter training
EPOCHS = 3  # Avoid overfitting (literature: multi-epoch can harm static datasets)
BATCH_SIZE = 32  # Per GPU - optimized for H100 80GB VRAM (~30GB models + 384MB batch)
GRAD_ACCUM_STEPS = 1  # No accumulation needed with larger batch
MAX_SEQ_LEN = 512
LORA_RANK = 8  # Standard efficient rank
LORA_ALPHA = 16  # 2*rank (best practice from literature)

# Evaluation
TEST_PROMPTS = [
    "The capital of France is",
    "To solve this problem, we need to",
    "The future of artificial intelligence is",
    "In the year 2050,",
    "The main difference between cats and dogs is",
]


# ============================================================================
# Adapter Architectures
# ============================================================================

class LinearAdapter(nn.Module):
    """Full linear projection (16.8M params)"""

    def __init__(self, hidden_dim=4096):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Initialize with Kaiming (will be overwritten with Procrustes if available)
        nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))

    def forward(self, x):
        return self.proj(x)


class AffineAdapter(nn.Module):
    """Full affine projection with bias (16.8M params)"""

    def __init__(self, hidden_dim=4096):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        # Initialize weight with Kaiming, bias with zeros
        nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return self.proj(x)


class LoRAAdapter(nn.Module):
    """Low-rank adapter (65k params) - efficient baseline"""

    def __init__(self, hidden_dim=4096, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # Standard LoRA scaling

        self.lora_A = nn.Linear(hidden_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, hidden_dim, bias=False)

        # Standard LoRA initialization (Hu et al., 2021)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # Residual connection + scaled LoRA
        return x + self.scaling * self.lora_B(self.lora_A(x))


# ============================================================================
# Dataset
# ============================================================================

class AlignmentDataset(Dataset):
    """Dataset for adapter training with paired source-target texts"""

    def __init__(self, texts, tokenizer_a, tokenizer_b, max_length=512):
        self.texts = texts
        self.tokenizer_a = tokenizer_a
        self.tokenizer_b = tokenizer_b
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize for both models
        inputs_a = self.tokenizer_a(text, truncation=True, max_length=self.max_length,
                                    return_tensors="pt")
        inputs_b = self.tokenizer_b(text, truncation=True, max_length=self.max_length,
                                    return_tensors="pt")

        return {
            "input_ids_a": inputs_a["input_ids"][0],
            "attention_mask_a": inputs_a["attention_mask"][0],
            "input_ids_b": inputs_b["input_ids"][0],
            "attention_mask_b": inputs_b["attention_mask"][0],
        }


def collate_fn(batch):
    """Custom collate with padding"""
    # Find max lengths
    max_len_a = max(item["input_ids_a"].shape[0] for item in batch)
    max_len_b = max(item["input_ids_b"].shape[0] for item in batch)

    # Pad sequences
    input_ids_a = []
    attention_mask_a = []
    input_ids_b = []
    attention_mask_b = []

    for item in batch:
        # Pad A
        pad_len_a = max_len_a - item["input_ids_a"].shape[0]
        input_ids_a.append(torch.cat([
            item["input_ids_a"],
            torch.full((pad_len_a,), -100, dtype=torch.long)  # Use -100 to ignore in loss
        ]))
        attention_mask_a.append(torch.cat([
            item["attention_mask_a"],
            torch.zeros(pad_len_a, dtype=torch.long)
        ]))

        # Pad B
        pad_len_b = max_len_b - item["input_ids_b"].shape[0]
        input_ids_b.append(torch.cat([
            item["input_ids_b"],
            torch.full((pad_len_b,), -100, dtype=torch.long)  # Use -100 to ignore in loss
        ]))
        attention_mask_b.append(torch.cat([
            item["attention_mask_b"],
            torch.zeros(pad_len_b, dtype=torch.long)
        ]))

    return {
        "input_ids_a": torch.stack(input_ids_a),
        "attention_mask_a": torch.stack(attention_mask_a),
        "input_ids_b": torch.stack(input_ids_b),
        "attention_mask_b": torch.stack(attention_mask_b),
    }


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_adapter(adapter_type, model_a, tokenizer_a, model_b, tokenizer_b,
                  train_texts, device, log_file):
    """Train adapter with generation loss"""

    print(f"\n{'='*80}", file=log_file)
    print(f"Training {adapter_type.upper()} Adapter on GPU {device}", file=log_file)
    print(f"{'='*80}", file=log_file)

    # Create adapter
    if adapter_type == "linear":
        adapter = LinearAdapter().to(device)
    elif adapter_type == "affine":
        adapter = AffineAdapter().to(device)
    elif adapter_type == "lora":
        adapter = LoRAAdapter(rank=LORA_RANK, alpha=LORA_ALPHA).to(device)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")

    # Count parameters
    num_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"Adapter parameters: {num_params:,}", file=log_file)

    # Setup optimizer
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=LEARNING_RATE)

    # Create dataset
    dataset = AlignmentDataset(train_texts, tokenizer_a, tokenizer_b, MAX_SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=0)

    # Training loop
    adapter.train()
    training_metrics = []
    global_step = 0

    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_loss = 0
        epoch_steps = 0

        print(f"\nEpoch {epoch+1}/{EPOCHS}", file=log_file)
        print("-" * 40, file=log_file)

        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)

            # Extract source representations (frozen Model A)
            with torch.no_grad():
                outputs_a = model_a(
                    input_ids=input_ids_a,
                    attention_mask=attention_mask_a,
                    output_hidden_states=True
                )
                source_repr = outputs_a.hidden_states[LAYER_IDX]  # [B, T, D]

            # Align representations
            aligned_repr = adapter(source_repr)

            # Compute generation loss with Model B
            outputs_b = model_b(
                inputs_embeds=aligned_repr,
                attention_mask=attention_mask_b,
                labels=input_ids_b
            )

            loss = outputs_b.loss / GRAD_ACCUM_STEPS
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                # Gradient clipping for training stability
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * GRAD_ACCUM_STEPS
            epoch_steps += 1

            # Log progress
            if (batch_idx + 1) % 10 == 0:
                avg_loss = epoch_loss / epoch_steps
                print(f"  Step {batch_idx+1}/{len(dataloader)}: Loss = {avg_loss:.4f}",
                      file=log_file)
                log_file.flush()

        # Epoch metrics
        avg_epoch_loss = epoch_loss / epoch_steps
        training_metrics.append({
            "epoch": epoch + 1,
            "loss": avg_epoch_loss,
            "time_elapsed": time.time() - start_time
        })

        print(f"  Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}", file=log_file)
        log_file.flush()

    total_time = time.time() - start_time
    print(f"\nTraining Complete! Total time: {total_time:.2f}s", file=log_file)

    return adapter, training_metrics


def evaluate_perplexity(adapter, model_a, tokenizer_a, model_b, tokenizer_b,
                        eval_texts, device, log_file):
    """Evaluate perplexity on held-out set"""

    print(f"\nEvaluating Perplexity on {len(eval_texts)} examples...", file=log_file)

    adapter.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for text in eval_texts:
            # Tokenize
            inputs_a = tokenizer_a(text, truncation=True, max_length=MAX_SEQ_LEN,
                                  return_tensors="pt").to(device)
            inputs_b = tokenizer_b(text, truncation=True, max_length=MAX_SEQ_LEN,
                                  return_tensors="pt").to(device)

            # Extract + align
            outputs_a = model_a(**inputs_a, output_hidden_states=True)
            source_repr = outputs_a.hidden_states[LAYER_IDX]
            aligned_repr = adapter(source_repr)

            # Compute loss
            outputs_b = model_b(
                inputs_embeds=aligned_repr,
                labels=inputs_b["input_ids"]
            )

            total_loss += outputs_b.loss.item() * inputs_b["input_ids"].shape[1]
            total_tokens += inputs_b["input_ids"].shape[1]

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    print(f"  Perplexity: {perplexity:.2f}", file=log_file)

    return perplexity


def generate_with_adapter(adapter, model_a, tokenizer_a, model_b, tokenizer_b,
                         prompt, device, max_new_tokens=50):
    """Generate text using adapter-aligned representations"""

    adapter.eval()

    # Ensure eos_token_id is set (fallback for base models)
    eos_token_id = tokenizer_b.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer_b.convert_tokens_to_ids(tokenizer_b.eos_token)

    # Tokenize prompt with Model A
    inputs_a = tokenizer_a(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        # Extract source representation
        outputs_a = model_a(**inputs_a, output_hidden_states=True)
        source_repr = outputs_a.hidden_states[LAYER_IDX]

        # Align
        aligned_repr = adapter(source_repr)

        # Create position_ids for RoPE
        seq_len = aligned_repr.shape[1]
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)

        # Generate with Model B
        generated_ids = []
        past_key_values = None

        for step in range(max_new_tokens):
            if past_key_values is None:
                # First step: process prefix
                outputs_b = model_b.model(
                    inputs_embeds=aligned_repr,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=True,
                    output_hidden_states=True
                )
            else:
                # Subsequent steps: new token
                next_pos = position_ids[0, -1] + 1
                outputs_b = model_b.model(
                    inputs_embeds=next_embed,
                    position_ids=torch.tensor([[next_pos]], device=device),
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True
                )

            # Get next token
            logits = model_b.lm_head(outputs_b.hidden_states[-1])
            next_token_id = torch.argmax(logits[0, -1, :]).item()

            # Update state
            past_key_values = outputs_b.past_key_values

            if next_token_id == eos_token_id:
                break

            generated_ids.append(next_token_id)

            # Get embedding for next token
            next_embed = model_b.model.embed_tokens(
                torch.tensor([[next_token_id]], device=device))

            # Update positions
            next_pos = position_ids[0, -1] + 1
            position_ids = torch.cat([
                position_ids,
                torch.tensor([[next_pos]], device=device)
            ], dim=1)

        generated_text = tokenizer_b.decode(generated_ids, skip_special_tokens=True)
        return prompt + generated_text


def evaluate_generation(adapter, model_a, tokenizer_a, model_b, tokenizer_b,
                       test_prompts, device, log_file):
    """Evaluate generation quality on test prompts"""

    print(f"\nEvaluating Generation Quality on {len(test_prompts)} prompts...", file=log_file)
    print("-" * 80, file=log_file)

    generations = []

    for i, prompt in enumerate(test_prompts, 1):
        output = generate_with_adapter(
            adapter, model_a, tokenizer_a, model_b, tokenizer_b,
            prompt, device, max_new_tokens=50
        )

        generations.append({
            "prompt": prompt,
            "generation": output
        })

        print(f"[{i}/{len(test_prompts)}] {prompt}", file=log_file)
        print(f"  → {output}", file=log_file)
        log_file.flush()

    return generations


# ============================================================================
# Main Experiment Runner
# ============================================================================

def run_single_adapter_experiment(adapter_type):
    """Run experiment for single adapter type on assigned GPU"""

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    # Get GPU assignment
    gpu_id = GPU_MAPPING[adapter_type]
    # IMPORTANT: CUDA_VISIBLE_DEVICES is set by bash script, so always use cuda:0
    # The bash script restricts each process to see only 1 GPU, indexed as cuda:0
    device = "cuda:0"

    # Setup output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = OUTPUT_DIR / f"{adapter_type}_gpu{gpu_id}_{timestamp}.log"
    results_path = OUTPUT_DIR / f"{adapter_type}_results_{timestamp}.json"

    with open(log_path, 'w', buffering=1) as log_file:
        print("="*80, file=log_file)
        print(f"LEARNED ADAPTER EXPERIMENT: {adapter_type.upper()}", file=log_file)
        print("="*80, file=log_file)
        print(f"GPU: {gpu_id} ({device})", file=log_file)
        print(f"Layer: {LAYER_IDX} (middle layer)", file=log_file)
        print(f"Timestamp: {timestamp}", file=log_file)
        print(f"Log file: {log_path}", file=log_file)
        print(f"Results file: {results_path}", file=log_file)
        print("="*80, file=log_file)

        # Load models
        print(f"\nLoading models on GPU {gpu_id}...", file=log_file)

        llama_model = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL,
            torch_dtype=torch.float16,
            device_map=device
        ).eval()
        llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL, use_fast=False)
        # Set pad_token for base models (use eos_token as pad_token)
        if llama_tokenizer.pad_token is None:
            llama_tokenizer.pad_token = llama_tokenizer.eos_token
        print("  ✓ Llama 3.1 8B loaded", file=log_file)

        mistral_model = AutoModelForCausalLM.from_pretrained(
            MISTRAL_MODEL,
            torch_dtype=torch.float16,
            device_map=device
        ).eval()
        mistral_tokenizer = AutoTokenizer.from_pretrained(MISTRAL_MODEL, use_fast=False)
        # Set pad_token for base models (use eos_token as pad_token)
        if mistral_tokenizer.pad_token is None:
            mistral_tokenizer.pad_token = mistral_tokenizer.eos_token
        print("  ✓ Mistral 7B loaded", file=log_file)

        # Load datasets
        print("\nLoading datasets...", file=log_file)

        # Delete corrupted cache to force fresh download
        # The cache structure is: {HF_HOME}/datasets/rajpurkar___squad/...
        # Also handle race condition when multiple processes run in parallel
        import os

        # Get the actual HF cache root directory
        hf_cache_root = os.environ.get('HF_HOME') or os.environ.get('HF_DATASETS_CACHE')
        if not hf_cache_root:
            # Default location on Linux systems
            hf_cache_root = os.path.expanduser('~/.cache/huggingface')

        # The actual cache directory includes the 'datasets' subdirectory
        cache_dir = Path(hf_cache_root) / 'datasets' / 'rajpurkar___squad'

        print(f"  Checking cache at: {cache_dir}", file=log_file)
        print(f"  Cache exists: {cache_dir.exists()}", file=log_file)

        try:
            if cache_dir.exists():
                print(f"  Removing corrupted cache at {cache_dir}", file=log_file)
                shutil.rmtree(cache_dir)
                print(f"  ✓ Cache removed successfully", file=log_file)
            else:
                print(f"  Cache directory does not exist - proceeding with fresh download", file=log_file)
        except (FileNotFoundError, OSError) as e:
            # Another process may have already deleted it - that's fine
            print(f"  Cache already removed (likely by parallel process): {e}", file=log_file)

        # Load dataset without trust_remote_code to avoid metadata issues
        squad = load_dataset("squad", split="train")
        squad_val = load_dataset("squad", split="validation")

        # Extract training texts for this adapter
        split = DATASET_SPLITS[adapter_type]
        train_texts = [
            f"{squad[i]['context'][:200]} Question: {squad[i]['question']}"
            for i in range(split["start"], split["end"])
        ]
        print(f"  Training: {len(train_texts)} texts (indices {split['start']}-{split['end']})",
              file=log_file)

        # Extract evaluation texts
        eval_texts = [
            f"{squad_val[i]['context'][:200]} Question: {squad_val[i]['question']}"
            for i in range(EVAL_SPLIT["start"], EVAL_SPLIT["end"])
        ]
        print(f"  Evaluation: {len(eval_texts)} texts", file=log_file)

        # Train adapter
        adapter, training_metrics = train_adapter(
            adapter_type, llama_model, llama_tokenizer,
            mistral_model, mistral_tokenizer,
            train_texts, device, log_file
        )

        # Evaluate perplexity
        perplexity = evaluate_perplexity(
            adapter, llama_model, llama_tokenizer,
            mistral_model, mistral_tokenizer,
            eval_texts, device, log_file
        )

        # Evaluate generation
        generations = evaluate_generation(
            adapter, llama_model, llama_tokenizer,
            mistral_model, mistral_tokenizer,
            TEST_PROMPTS, device, log_file
        )

        # Save results
        results = {
            "metadata": {
                "adapter_type": adapter_type,
                "gpu": gpu_id,
                "device": device,
                "layer": LAYER_IDX,
                "timestamp": timestamp,
                "source_model": LLAMA_MODEL,
                "target_model": MISTRAL_MODEL,
                "training_samples": len(train_texts),
                "eval_samples": len(eval_texts),
                "hyperparameters": {
                    "learning_rate": LEARNING_RATE,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "grad_accum": GRAD_ACCUM_STEPS,
                    "max_seq_len": MAX_SEQ_LEN,
                    "lora_rank": LORA_RANK if adapter_type == "lora" else None,
                    "lora_alpha": LORA_ALPHA if adapter_type == "lora" else None,
                }
            },
            "training_metrics": training_metrics,
            "evaluation": {
                "perplexity": perplexity,
                "generations": generations
            }
        }

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print("\n" + "="*80, file=log_file)
        print("EXPERIMENT COMPLETE", file=log_file)
        print("="*80, file=log_file)
        print(f"Results saved to: {results_path}", file=log_file)
        print(f"Final Perplexity: {perplexity:.2f}", file=log_file)

        return results


if __name__ == "__main__":
    import traceback

    # Get adapter type from command line argument
    if len(sys.argv) != 2 or sys.argv[1] not in ["linear", "affine", "lora"]:
        print("Usage: python learned_adapter_ablation.py [linear|affine|lora]")
        print("  linear: Full linear projection (GPU 1)")
        print("  affine: Full affine projection (GPU 2)")
        print("  lora:   LoRA rank-8 adapter (GPU 3)")
        sys.exit(1)

    adapter_type = sys.argv[1]
    gpu_id = GPU_MAPPING[adapter_type]

    # Set up log file FIRST, before any output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = OUTPUT_DIR / f"{adapter_type}_gpu{gpu_id}_{timestamp}.log"

    # Redirect stdout and stderr to log file (also keep console output)
    class TeeLogger:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    log_file_handle = open(log_path, 'w', buffering=1)
    sys.stdout = TeeLogger(sys.stdout, log_file_handle)
    sys.stderr = TeeLogger(sys.stderr, log_file_handle)

    print("=" * 80)
    print(f"LEARNED ADAPTER ABLATION - {adapter_type.upper()} - STARTING")
    print("=" * 80)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA current device: {torch.cuda.current_device()}")
    print(f"Script: {__file__}")
    print(f"Adapter type: {adapter_type}")
    print(f"GPU assigned: {gpu_id}")
    print(f"Log file: {log_path}")
    print("=" * 80)
    print()

    try:
        results = run_single_adapter_experiment(adapter_type)
        print()
        print("=" * 80)
        print(f"{adapter_type.upper()} ADAPTER EXPERIMENT COMPLETED SUCCESSFULLY")
        print("=" * 80)
        log_file_handle.close()
        sys.exit(0)
    except Exception as e:
        print()
        print("=" * 80)
        print(f"{adapter_type.upper()} ADAPTER EXPERIMENT FAILED WITH ERROR")
        print("=" * 80)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print()
        print("Full traceback:")
        print("-" * 80)
        traceback.print_exc()
        print("-" * 80)
        print()
        print("=" * 80)
        log_file_handle.close()
        sys.exit(1)
