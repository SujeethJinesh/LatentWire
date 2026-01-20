#!/usr/bin/env python
# telepathy/train_telepathy.py
"""
Unified Telepathy Training Script

Trains the latent bridge on classification tasks.
Supports multiple datasets: SST-2 (2-class), AG News (4-class), TREC (6-class), Banking77 (77-class)

Usage:
    python telepathy/train_telepathy.py --dataset sst2 --steps 2000 --soft_tokens 8
    python telepathy/train_telepathy.py --dataset agnews --steps 3000 --source_layer 31
    python telepathy/train_telepathy.py --dataset trec --steps 2000
    python telepathy/train_telepathy.py --dataset banking77 --steps 5000

Key Features:
- Continuous soft tokens (VQ collapsed in experiments, continuous is the way)
- Batch diversity loss to prevent mode collapse
- RMS normalization to match embedding statistics
- Supports DDP for multi-GPU training
"""
import os
import math
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse
import json
from datetime import datetime

from latentwire import LatentBridge

# Import experimental bridge types
try:
    from telepathy.cross_model_experiments import (
        # NOVEL bridges from diverse fields (neuroscience, chemistry, etc.)
        PredictiveCodingBridge,
        OptimalTransportBridge,
        ContrastiveInfoNCEBridge,
        SparseKWTABridge,
        ResidualCodingBridge,
        LockAndKeyBridge,
        MoEBridge,  # Mixture-of-Experts for cross-model transfer (Mixtral/DeepSeek)
        # MATH bridges (from math opus subagents)
        SpectralCCABridge,
        FlowMatchingBridge,
        # Legacy bridges (not novel, kept for reference)
        RidgeRegressionBridge,
        VIBLatentBridge,
        MultiLayerExtractor,
        MultiLayerBridge,
        CurriculumTrainer,
        GumbelSigmoidGate,
        CLASSIFICATION_CURRICULUM,
    )
    EXPERIMENTAL_BRIDGES_AVAILABLE = True
except ImportError:
    EXPERIMENTAL_BRIDGES_AVAILABLE = False


# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================

DATASET_CONFIGS = {
    "sst2": {
        "load_args": ("glue", "sst2"),
        "text_field": "sentence",
        "label_field": "label",
        "labels": ["negative", "positive"],
        "num_classes": 2,
        "train_split": "train",
        "eval_split": "validation",
        "max_length": 128,
        "prompt_template": "Review: {text}\nSentiment (positive or negative):",
        "primer": "Sentiment:",
        "random_baseline": 50.0,
    },
    "agnews": {
        "load_args": ("ag_news",),
        "text_field": "text",
        "label_field": "label",
        "labels": ["world", "sports", "business", "science"],
        "num_classes": 4,
        "train_split": "train",
        "eval_split": "test",
        "max_length": 256,
        "prompt_template": "Article: {text}\nTopic (world, sports, business, or science):",
        "primer": "Topic:",
        "random_baseline": 25.0,
        "label_synonyms": {
            "science": ["science", "technology", "tech", "sci/tech", "scitech"],
        },
    },
    "trec": {
        "load_args": ("CogComp/trec",),
        "text_field": "text",
        "label_field": "coarse_label",
        "labels": ["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"],
        "num_classes": 6,
        "train_split": "train",
        "eval_split": "test",
        "max_length": 128,
        "prompt_template": "Question: {text}\nCategory (ABBR, ENTY, DESC, HUM, LOC, or NUM):",
        "primer": "Question Type:",
        "random_baseline": 16.7,
    },
    "banking77": {
        "load_args": ("banking77",),
        "text_field": "text",
        "label_field": "label",
        "labels": None,  # Will be populated from dataset
        "num_classes": 77,
        "train_split": "train",
        "eval_split": "test",
        "max_length": 128,
        "prompt_template": "Query: {text}\nIntent:",
        "primer": "Intent:",
        "random_baseline": 1.3,
    },
    # =========================================================================
    # REASONING BENCHMARKS
    # =========================================================================
    "arc_easy": {
        "load_args": ("allenai/ai2_arc", "ARC-Easy"),
        "text_field": "question",  # Will be formatted with choices via text_formatter
        "label_field": "answerKey",
        "labels": ["A", "B", "C", "D", "E"],  # Some questions have 5 choices
        "num_classes": 5,
        "train_split": "train",
        "eval_split": "test",
        "max_length": 256,
        "prompt_template": "Question: {text}\nAnswer (A, B, C, D, or E):",
        "primer": "Answer:",
        "random_baseline": 20.0,  # 5-way classification
        "is_reasoning": True,
        "text_formatter": "arc",  # Special formatter to include answer choices
    },
    "winogrande": {
        "load_args": ("allenai/winogrande", "winogrande_xl"),
        "text_field": "sentence",
        "label_field": "answer",
        "labels": ["1", "2"],
        "num_classes": 2,
        "train_split": "train",
        "eval_split": "validation",
        "max_length": 128,
        "prompt_template": "Sentence: {text}\nWhich option fits? (1 or 2):",
        "primer": "Option:",
        "random_baseline": 50.0,
        "is_reasoning": True,
    },
    "hellaswag": {
        "load_args": ("Rowan/hellaswag",),
        "text_field": "ctx",
        "label_field": "label",
        "labels": ["0", "1", "2", "3"],
        "num_classes": 4,
        "train_split": "train",
        "eval_split": "validation",
        "max_length": 256,
        "prompt_template": "Context: {text}\nMost likely continuation (0, 1, 2, or 3):",
        "primer": "Continuation:",
        "random_baseline": 25.0,
        "is_reasoning": True,
    },
    "boolq": {
        "load_args": ("google/boolq",),
        "text_field": "question",
        "label_field": "answer",
        "labels": ["False", "True"],
        "num_classes": 2,
        "train_split": "train",
        "eval_split": "validation",
        "max_length": 256,
        "prompt_template": "Question: {text}\nAnswer (True or False):",
        "primer": "Answer:",
        "random_baseline": 50.0,
        "is_reasoning": True,
    },
    # =========================================================================
    # MATH REASONING (Generative)
    # =========================================================================
    "gsm8k": {
        "load_args": ("openai/gsm8k", "main"),
        "text_field": "question",
        "label_field": "answer",
        "labels": None,  # GSM8K is generative - extract numeric answer from "#### N" format
        "num_classes": None,  # Not a classification task
        "train_split": "train",
        "eval_split": "test",
        "max_length": 512,  # Math problems need longer context
        "prompt_template": "Math Problem: {text}\nSolution:",
        "primer": "The answer is",
        "random_baseline": 0.0,  # Generative task - random baseline not applicable
        "is_reasoning": True,
        "is_generative": True,  # Flag for generative evaluation
    },
}


def format_text_for_item(item, config):
    """
    Format text for an item, handling special formatters.

    For ARC datasets, includes the answer choices in the formatted text:
    "What is the capital of France?\nA) Paris\nB) London\nC) Berlin\nD) Madrid"
    """
    text_field = config["text_field"]
    text = item[text_field]

    # Check if we need a special formatter
    formatter = config.get("text_formatter")

    if formatter == "arc":
        # ARC datasets have 'choices' with 'text' (list) and 'label' (list like ["A", "B", "C", "D"])
        if "choices" in item:
            choices = item["choices"]
            choice_texts = choices.get("text", [])
            choice_labels = choices.get("label", [])

            # Format choices as "A) choice1\nB) choice2\n..."
            formatted_choices = []
            for label, choice_text in zip(choice_labels, choice_texts):
                formatted_choices.append(f"{label}) {choice_text}")

            # Combine question with choices
            text = f"{text}\n" + "\n".join(formatted_choices)

    return text


def custom_collate_fn(batch):
    """
    Custom collate function that handles variable-length nested data.

    HuggingFace datasets with nested dicts (like ARC's 'choices' field)
    can have variable-length lists that the default collate can't stack.
    This function keeps them as lists instead of trying to tensorize.
    """
    if not batch:
        return {}

    # Get all keys from first item
    keys = batch[0].keys()
    collated = {}

    for key in keys:
        values = [item[key] for item in batch]

        # Check if values are dicts (nested structure like 'choices')
        if isinstance(values[0], dict):
            # Recursively collate nested dicts
            nested_keys = values[0].keys()
            collated[key] = {
                nk: [v[nk] for v in values] for nk in nested_keys
            }
        elif isinstance(values[0], (int, float, bool)):
            # Numeric types - keep as list (will be converted to tensor later if needed)
            collated[key] = values
        else:
            # Strings and other types - keep as list
            collated[key] = values

    return collated


def format_texts_for_batch(batch, config):
    """
    Format texts for a batch of items, handling special formatters.

    For standard datasets, returns batch[text_field] directly.
    For ARC datasets, formats each item to include answer choices.
    """
    text_field = config["text_field"]
    formatter = config.get("text_formatter")

    if formatter == "arc":
        # ARC datasets have 'choices' which is a dict with lists
        # batch["choices"] = {"text": [[choices1], [choices2], ...], "label": [[A,B,C,D], ...]}
        questions = batch[text_field]
        choices_data = batch.get("choices", {})
        choice_texts_list = choices_data.get("text", [])
        choice_labels_list = choices_data.get("label", [])

        formatted_texts = []
        for i, question in enumerate(questions):
            if i < len(choice_texts_list) and i < len(choice_labels_list):
                choice_texts = choice_texts_list[i]
                choice_labels = choice_labels_list[i]

                # Format choices as "A) choice1\nB) choice2\n..."
                formatted_choices = []
                for lbl, choice_text in zip(choice_labels, choice_texts):
                    formatted_choices.append(f"{lbl}) {choice_text}")

                # Combine question with choices
                text = f"{question}\n" + "\n".join(formatted_choices)
            else:
                text = question
            formatted_texts.append(text)

        return formatted_texts
    else:
        # Standard datasets - just return the text field as-is
        return batch[text_field]


def normalize_label_index(label_val, labels):
    """
    Convert a label value to the corresponding label text.

    Handles the following cases:
    - Integer labels (standard classification): labels[label_val]
    - String labels like "1", "2" (WinoGrande): find matching string in labels
    - String labels like "0", "1", "2", "3" (HellaSwag): find matching string in labels
    - Boolean labels True/False (BoolQ): map to "True"/"False" strings
    - Letter labels like "A", "B", "C", "D" (ARC): find matching string in labels

    Args:
        label_val: The raw label value from the dataset (int, str, or bool)
        labels: List of label strings from config

    Returns:
        str: The label text
    """
    if labels is None:
        return str(label_val)

    # Handle boolean labels (BoolQ)
    if isinstance(label_val, bool):
        return "True" if label_val else "False"

    # Handle string labels (WinoGrande, HellaSwag, ARC)
    if isinstance(label_val, str):
        # If the string is in labels, return it directly
        if label_val in labels:
            return label_val
        # Otherwise, try to convert to int and index
        try:
            idx = int(label_val)
            if 0 <= idx < len(labels):
                return labels[idx]
        except ValueError:
            pass
        # Fallback: return the string as-is
        return label_val

    # Handle integer labels (standard case)
    if isinstance(label_val, int):
        if 0 <= label_val < len(labels):
            return labels[label_val]
        return str(label_val)

    # Fallback for any other type
    return str(label_val)


def get_nearest_neighbors(latent_vector, embedding_matrix, tokenizer, k=5):
    """Find k nearest vocabulary tokens to a latent vector."""
    latent_vector = latent_vector.float()
    embedding_matrix = embedding_matrix.float()

    latent_norm = F.normalize(latent_vector.unsqueeze(0), p=2, dim=-1)
    emb_norm = F.normalize(embedding_matrix, p=2, dim=-1)
    similarity = torch.matmul(latent_norm, emb_norm.t())

    scores, indices = torch.topk(similarity, k)

    neighbors = []
    for score, idx in zip(scores[0], indices[0]):
        token_str = tokenizer.decode([idx.item()]).replace('\n', '\\n').replace('\t', '\\t')
        if token_str.strip() == '':
            token_str = repr(tokenizer.decode([idx.item()]))
        neighbors.append((token_str, score.item()))
    return neighbors


def analyze_latent_interpretability(bridge, src_model, tgt_model, src_tok, tgt_tok, device, args, eval_ds, config):
    """Analyze what the soft tokens 'mean' by finding nearest vocabulary neighbors."""
    print("\n" + "=" * 70)
    print("LATENT INTERPRETABILITY ANALYSIS")
    print("=" * 70)
    print("What vocabulary tokens are closest to each soft token?")

    bridge_module = bridge.module if hasattr(bridge, 'module') else bridge
    bridge_module.eval()
    mistral_embeddings = tgt_model.get_input_embeddings().weight.detach()

    # Sample one from each class (up to 4)
    samples = []
    seen_labels = set()
    labels = config["labels"]

    for i in range(min(200, len(eval_ds))):
        item = eval_ds[i]
        label_idx = item[config["label_field"]]
        label = normalize_label_index(label_idx, labels)
        if label not in seen_labels:
            text = format_text_for_item(item, config)
            samples.append((text, label))
            seen_labels.add(label)
        if len(samples) >= 4:
            break

    for text, label in samples:
        print(f"\n--- Label: {label} ---")
        print(f"    Input: \"{text[:50]}...\"")

        src_input = config["prompt_template"].format(text=text[:256])
        src_enc = src_tok(src_input, return_tensors="pt", truncation=True, max_length=config["max_length"]).to(device)

        with torch.no_grad():
            src_out = src_model(**src_enc, output_hidden_states=True)
            src_h = src_out.hidden_states[args.source_layer]
            if args.bf16:
                src_h = src_h.bfloat16()
            latents, _, _, _ = bridge_module(src_h, src_enc.attention_mask)

        latents = latents[0]  # Remove batch dim

        for i in range(min(args.soft_tokens, latents.shape[0])):
            neighbors = get_nearest_neighbors(latents[i], mistral_embeddings, tgt_tok, k=5)
            neighbor_str = ", ".join([f"'{tok}'({score:.2f})" for tok, score in neighbors])
            print(f"  Token {i+1}: {neighbor_str}")

    # Geometry analysis
    print("\n--- Latent Geometry (last sample) ---")
    latents_norm = F.normalize(latents.float(), dim=-1)
    sim_matrix = torch.matmul(latents_norm, latents_norm.t())
    num_tokens = latents.shape[0]
    off_diag = sim_matrix[~torch.eye(num_tokens, dtype=torch.bool, device=device)]
    print(f"  Mean pairwise similarity: {off_diag.mean().item():.3f}")
    print(f"  Token RMS range: {latents.float().pow(2).mean(dim=-1).sqrt().min().item():.4f} - {latents.float().pow(2).mean(dim=-1).sqrt().max().item():.4f}")


def extract_numeric_answer(text):
    """
    Extract numeric answer from text.

    Handles various formats:
    - GSM8K ground truth format: "...#### 42"
    - Generated text: "The answer is 42", "42", "= 42", etc.

    Returns the extracted number as a string, or None if not found.
    """
    import re

    # First, try GSM8K format: "#### <number>"
    gsm8k_match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
    if gsm8k_match:
        # Remove commas from numbers like "1,234"
        return gsm8k_match.group(1).replace(',', '')

    # Try common answer patterns in generated text
    # "The answer is X", "answer: X", "= X", etc.
    patterns = [
        r'(?:the\s+)?answer\s*(?:is|:)\s*(-?\d+(?:,\d+)*(?:\.\d+)?)',
        r'=\s*(-?\d+(?:,\d+)*(?:\.\d+)?)\s*$',
        r'(-?\d+(?:,\d+)*(?:\.\d+)?)\s*$',  # Number at end of string
    ]

    for pattern in patterns:
        match = re.search(pattern, text.lower().strip())
        if match:
            return match.group(1).replace(',', '')

    # Fallback: find any number in the text (last one is usually the answer)
    numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].replace(',', '')

    return None


def check_label_match(label, output, config):
    """Check if label matches output, with permissive matching for synonyms."""

    # For generative tasks (like GSM8K), compare numeric answers
    if config.get("is_generative", False):
        gt_number = extract_numeric_answer(str(label))
        pred_number = extract_numeric_answer(output)

        if gt_number is None or pred_number is None:
            return False

        # Compare as floats to handle "42" vs "42.0" etc.
        try:
            return float(gt_number) == float(pred_number)
        except ValueError:
            return gt_number == pred_number

    # Standard classification matching
    output_lower = output.lower()

    # Check synonyms if defined
    if "label_synonyms" in config and label in config["label_synonyms"]:
        return any(syn in output_lower for syn in config["label_synonyms"][label])

    return label.lower() in output_lower


def setup_ddp():
    if "RANK" in os.environ:
        torch.distributed.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True
    return False


def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint in the output directory for auto-resume."""
    import glob
    import re

    # Look for checkpoint_step*.pt files
    pattern = os.path.join(output_dir, "checkpoint_step*.pt")
    checkpoints = glob.glob(pattern)

    if not checkpoints:
        return None

    # Extract step numbers and find the latest
    step_pattern = re.compile(r'checkpoint_step(\d+)\.pt')
    latest_step = -1
    latest_ckpt = None

    for ckpt in checkpoints:
        match = step_pattern.search(os.path.basename(ckpt))
        if match:
            step = int(match.group(1))
            if step > latest_step:
                latest_step = step
                latest_ckpt = ckpt

    return latest_ckpt


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Telepathy Training")

    # Dataset selection
    parser.add_argument("--dataset", type=str, default="sst2",
                       choices=["sst2", "agnews", "trec", "banking77",
                                "arc_easy", "winogrande", "hellaswag", "boolq", "gsm8k"],
                       help="Dataset to train on")

    # Model configuration
    parser.add_argument("--source_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--target_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--source_layer", type=int, default=31,
                       help="Which layer to extract hidden states from (31=final)")

    # Bridge architecture
    parser.add_argument("--soft_tokens", type=int, default=8,
                       help="Number of soft tokens (information bottleneck)")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)

    # Training configuration
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--diversity_weight", type=float, default=0.1,
                       help="Weight for batch diversity loss (prevents collapse)")

    # Output configuration
    parser.add_argument("--output_dir", default=None,
                       help="Output directory (default: runs/{dataset})")
    parser.add_argument("--save_path", default=None,
                       help="Checkpoint path (default: bridge_{dataset}.pt)")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=200)

    # Other settings
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--use_fsq", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index (for single GPU)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume training from")
    parser.add_argument("--auto_resume", action="store_true", default=True,
                       help="Automatically resume from latest checkpoint if available")

    # =============================================================================
    # EXPERIMENTAL FEATURES (from cross_model_experiments.py)
    # =============================================================================

    # Bridge type selection
    parser.add_argument("--bridge_type", type=str, default="standard",
                       choices=["standard", "ridge", "vib", "multi_layer",
                               # NOVEL bridge types (diverse fields)
                               "predictive_coding", "optimal_transport", "infonce",
                               "sparse_kwta", "residual_coding", "lock_and_key", "moe",
                               # MATH bridge types
                               "spectral_cca", "flow_matching",
                               # HAIL MARY bridge types (literature-informed)
                               "cross_modal_distillation", "mine", "mixture_of_depths",
                               "thalamic_relay", "domain_adversarial", "successive_refinement"],
                       help="Bridge architecture: standard, ridge, vib, multi_layer, "
                            "NOVEL types: predictive_coding, optimal_transport, infonce, "
                            "sparse_kwta, residual_coding, lock_and_key, moe, "
                            "MATH types: spectral_cca, flow_matching, "
                            "HAIL MARY types: cross_modal_distillation, mine, mixture_of_depths, "
                            "thalamic_relay, domain_adversarial, successive_refinement")

    # Ridge Regression (LatentMAS)
    parser.add_argument("--lambda_reg", type=float, default=1e-4,
                       help="Regularization for ridge regression alignment")
    parser.add_argument("--eval_only", action="store_true",
                       help="Skip training, only evaluate (for ridge regression)")

    # Multi-layer extraction
    parser.add_argument("--extract_layers", nargs="+", type=int, default=None,
                       help="Layers to extract from (e.g., 16 24 31)")
    parser.add_argument("--learn_layer_weights", action="store_true",
                       help="Learn weights for layer combination")

    # Multi-layer injection (Cache2Cache)
    parser.add_argument("--inject_layers", nargs="+", type=int, default=None,
                       help="Layers to inject soft tokens (e.g., 0 8 16 24)")
    parser.add_argument("--use_gumbel_gates", action="store_true",
                       help="Use Gumbel-sigmoid gates for layer selection")
    parser.add_argument("--temp_start", type=float, default=2.0,
                       help="Starting temperature for Gumbel annealing")
    parser.add_argument("--temp_end", type=float, default=0.1,
                       help="Ending temperature for Gumbel annealing")

    # Variational Information Bottleneck
    parser.add_argument("--use_vib", action="store_true",
                       help="Use VIB for stochastic soft tokens")
    parser.add_argument("--vib_beta", type=float, default=0.001,
                       help="KL divergence weight for VIB")
    parser.add_argument("--vib_beta_anneal", action="store_true",
                       help="Anneal VIB beta from 0 to target")

    # Curriculum Training (COCONUT)
    parser.add_argument("--use_curriculum", action="store_true",
                       help="Use curriculum training (text -> latent)")
    parser.add_argument("--curriculum_stages", type=int, default=5,
                       help="Number of curriculum stages")

    # =============================================================================
    # NOVEL BRIDGE PARAMETERS
    # =============================================================================

    # Optimal Transport (Sinkhorn)
    parser.add_argument("--ot_epsilon", type=float, default=0.1,
                       help="Entropy regularization for Sinkhorn OT (lower = sharper)")
    parser.add_argument("--ot_iters", type=int, default=20,
                       help="Number of Sinkhorn iterations")

    # Contrastive InfoNCE
    parser.add_argument("--infonce_temp", type=float, default=0.07,
                       help="Temperature for InfoNCE contrastive loss")
    parser.add_argument("--infonce_weight", type=float, default=0.1,
                       help="Weight for InfoNCE auxiliary loss")

    # Sparse k-WTA
    parser.add_argument("--sparsity_k", type=int, default=128,
                       help="Number of active dimensions in k-WTA (lower = sparser)")

    # Residual Coding
    parser.add_argument("--num_refinement_steps", type=int, default=2,
                       help="Number of progressive refinement steps")

    # Lock-and-Key Binding
    parser.add_argument("--num_receptors", type=int, default=32,
                       help="Number of receptor sites for binding")
    parser.add_argument("--binding_sparsity", type=float, default=0.1,
                       help="Fraction of receptors each key can bind to")

    # Spectral CCA (Math subagent)
    parser.add_argument("--cca_dim", type=int, default=256,
                       help="Dimensionality of CCA shared subspace")

    # Flow Matching (Math subagent)
    parser.add_argument("--num_flow_steps", type=int, default=4,
                       help="Number of ODE integration steps for flow matching")
    parser.add_argument("--flow_loss_weight", type=float, default=0.1,
                       help="Weight for flow matching auxiliary loss")

    # Mixture-of-Experts (MoE) Bridge
    parser.add_argument("--moe_num_experts", type=int, default=8,
                       help="Number of expert FFNs (default 8, like Mixtral)")
    parser.add_argument("--moe_top_k", type=int, default=2,
                       help="Number of experts to route to (default 2, like Mixtral)")
    parser.add_argument("--moe_use_shared_expert", action="store_true", default=True,
                       help="Include a shared expert (DeepSeek-style)")
    parser.add_argument("--moe_no_shared_expert", action="store_false", dest="moe_use_shared_expert",
                       help="Disable shared expert")
    parser.add_argument("--moe_aux_loss_weight", type=float, default=0.01,
                       help="Weight for load balancing auxiliary loss")
    parser.add_argument("--moe_aux_loss_free", action="store_true", default=False,
                       help="Use learnable bias instead of aux loss (DeepSeek-V3)")

    args = parser.parse_args()

    # Set defaults based on dataset
    if args.output_dir is None:
        args.output_dir = f"runs/{args.dataset}"
    if args.save_path is None:
        args.save_path = f"bridge_{args.dataset}.pt"

    return args


def quick_eval(bridge, src_model, tgt_model, src_tok, tgt_tok, eval_ds, device, args, config, step):
    """Quick evaluation during training."""
    bridge_module = bridge.module if hasattr(bridge, 'module') else bridge
    bridge_module.eval()

    correct = 0
    total = 0
    labels = config["labels"]
    n_eval = min(100, len(eval_ds))

    # Per-class tracking
    class_correct = {l: 0 for l in labels} if labels else {}
    class_total = {l: 0 for l in labels} if labels else {}

    print(f"\n{'='*60}")
    print(f"QUICK EVAL @ Step {step}")
    print(f"{'='*60}")

    for i in range(n_eval):
        item = eval_ds[i]
        text = format_text_for_item(item, config)
        label_idx = item[config["label_field"]]
        label = normalize_label_index(label_idx, labels)

        src_input = config["prompt_template"].format(text=text[:256])

        with torch.no_grad():
            src_enc = src_tok(src_input, return_tensors="pt", truncation=True,
                            max_length=config["max_length"]).to(device)
            src_out = src_model(**src_enc, output_hidden_states=True)
            src_h = src_out.hidden_states[args.source_layer]
            if args.bf16:
                src_h = src_h.bfloat16()
            soft_tokens, _, _, _ = bridge_module(src_h, src_enc.attention_mask)

            # Target: [Primer] + [Soft Tokens] -> Generate
            primer = config["primer"]
            primer_enc = tgt_tok(primer, return_tensors="pt", add_special_tokens=False).to(device)
            primer_embeds = tgt_model.get_input_embeddings()(primer_enc.input_ids)
            if args.bf16:
                primer_embeds = primer_embeds.bfloat16()

            combined_embeds = torch.cat([primer_embeds, soft_tokens], dim=1)
            attn_mask = torch.ones(combined_embeds.shape[:2], device=device, dtype=torch.long)

            # Use more tokens for generative tasks like GSM8K
            max_tokens = 30 if config.get("is_generative", False) else 10

            out_ids = tgt_model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=attn_mask,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tgt_tok.eos_token_id,
            )
            output = tgt_tok.decode(out_ids[0], skip_special_tokens=True).strip().lower()

        # Check if correct
        is_correct = check_label_match(label, output, config)
        if is_correct:
            correct += 1
            if labels and label in class_correct:
                class_correct[label] += 1
        total += 1
        if labels and label in class_total:
            class_total[label] += 1

        # Print first few samples
        if i < 4:
            print(f"[{i}] GT: {label:15} | Pred: {output[:30]}")

    accuracy = 100 * correct / total
    print(f"\nAccuracy: {correct}/{total} ({accuracy:.1f}%) [Random={config['random_baseline']:.1f}%]")

    # Per-class breakdown for multi-class
    if labels and config["num_classes"] > 2:
        for label in labels[:6]:  # Show first 6 classes
            if class_total.get(label, 0) > 0:
                class_acc = 100 * class_correct[label] / class_total[label]
                print(f"  {label}: {class_correct[label]}/{class_total[label]} ({class_acc:.1f}%)")

    print(f"{'='*60}\n")

    bridge_module.train()
    return {"accuracy": accuracy, "correct": correct, "total": total}


def train_step(batch, src_tok, tgt_tok, src_model, bridge, tgt_model, device, args, config):
    """Single training step."""
    labels = config["labels"]
    inputs = format_texts_for_batch(batch, config)
    is_generative = config.get("is_generative", False)

    # For generative tasks like GSM8K, extract just the final numeric answer
    # This prevents training on the full reasoning chain (which would be truncated anyway)
    if is_generative:
        label_texts = [extract_numeric_answer(str(l)) or str(l) for l in batch[config["label_field"]]]
    else:
        label_texts = [normalize_label_index(l, labels) for l in batch[config["label_field"]]]

    B = len(inputs)

    # 1. Source (Llama reads text)
    src_texts = [config["prompt_template"].format(text=t[:256]) for t in inputs]

    with torch.no_grad():
        src_enc = src_tok(
            src_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=config["max_length"]
        ).to(device)
        src_out = src_model(**src_enc, output_hidden_states=True)
        src_h = src_out.hidden_states[args.source_layer]
        if args.bf16:
            src_h = src_h.bfloat16()
        src_mask = src_enc.attention_mask

    # 2. Bridge (continuous soft tokens)
    bridge_module = bridge.module if hasattr(bridge, 'module') else bridge
    soft_tokens, aux_loss, diversity, z_variance = bridge_module(src_h, src_mask)

    # 3. Batch diversity loss (prevent mode collapse)
    batch_div_loss = torch.tensor(0.0, device=device)
    if B > 1:
        flat_tokens = soft_tokens.reshape(B, -1).float()
        flat_norm = F.normalize(flat_tokens, dim=1)
        sim_matrix = torch.mm(flat_norm, flat_norm.t())
        mask = ~torch.eye(B, dtype=torch.bool, device=device)
        off_diag_sim = sim_matrix[mask].mean()
        batch_div_loss = off_diag_sim

    # 4. Target (Mistral predicts label)
    primer_text = config["primer"]
    with torch.no_grad():
        primer_enc = tgt_tok(
            [primer_text] * B, return_tensors="pt", add_special_tokens=False
        ).to(device)
        primer_embeds = tgt_model.get_input_embeddings()(primer_enc.input_ids)
        if args.bf16:
            primer_embeds = primer_embeds.bfloat16()

        tgt_texts = [f" {l}{tgt_tok.eos_token}" for l in label_texts]
        # Use longer max_length for generative tasks (GSM8K answers can be longer numbers)
        # For classification, labels are short (16 is plenty)
        tgt_max_length = 32 if is_generative else 16
        tgt_enc = tgt_tok(
            tgt_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=tgt_max_length, add_special_tokens=False
        ).to(device)
        answer_embeds = tgt_model.get_input_embeddings()(tgt_enc.input_ids)
        if args.bf16:
            answer_embeds = answer_embeds.bfloat16()

    # Concatenate: [Primer] + [Soft Tokens] + [Answer]
    inputs_embeds = torch.cat([primer_embeds, soft_tokens, answer_embeds], dim=1)

    K = soft_tokens.shape[1]
    P_len = primer_embeds.shape[1]

    # Labels: Mask primer and soft tokens, predict answer
    ignore_prefix = torch.full((B, P_len + K), -100, dtype=torch.long, device=device)
    answer_labels = tgt_enc.input_ids.clone()
    answer_labels[tgt_enc.attention_mask == 0] = -100
    labels_tensor = torch.cat([ignore_prefix, answer_labels], dim=1)

    # Attention mask
    soft_mask = torch.ones(B, K, dtype=torch.long, device=device)
    full_mask = torch.cat([primer_enc.attention_mask, soft_mask, tgt_enc.attention_mask], dim=1)

    # Forward through Mistral
    outputs = tgt_model(
        inputs_embeds=inputs_embeds,
        attention_mask=full_mask,
        labels=labels_tensor
    )
    loss_lm = outputs.loss

    # Total loss: LM + diversity penalty + auxiliary loss (MoE load balancing, etc.)
    aux_loss_val = aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
    total_loss = loss_lm + args.diversity_weight * batch_div_loss + aux_loss

    return total_loss, {
        "total": total_loss.item(),
        "lm": loss_lm.item(),
        "div": batch_div_loss.item(),
        "aux": aux_loss_val,  # MoE load balancing, VIB KL, etc.
        "z_var": z_variance.item() if isinstance(z_variance, torch.Tensor) else z_variance
    }


def main():
    setup_ddp()
    args = parse_args()
    config = DATASET_CONFIGS[args.dataset]

    # Set seeds for reproducibility (all random sources)
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        # CUDNN determinism for exact reproducibility
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if torch.distributed.is_initialized():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Create output directory
    if local_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    # Track training progress for JSON output
    training_log = []

    if local_rank == 0:
        print("=" * 60)
        print(f"TELEPATHY TRAINING: {args.dataset.upper()}")
        print("=" * 60)
        print("")
        print(f"Dataset: {args.dataset} ({config['num_classes']} classes)")
        print(f"Random baseline: {config['random_baseline']:.1f}%")
        print(f"Source layer: {args.source_layer}")
        print(f"Soft tokens: {args.soft_tokens}")
        print(f"Steps: {args.steps}")
        print(f"Diversity weight: {args.diversity_weight}")
        print("=" * 60)

    # Load models
    src_model = AutoModelForCausalLM.from_pretrained(
        args.source_model, torch_dtype=torch.bfloat16,
        device_map={"": local_rank} if torch.distributed.is_initialized() else device
    ).eval()
    tgt_model = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=torch.bfloat16,
        device_map={"": local_rank} if torch.distributed.is_initialized() else device
    ).eval()

    src_tok = AutoTokenizer.from_pretrained(args.source_model)
    src_tok.pad_token = src_tok.eos_token
    tgt_tok = AutoTokenizer.from_pretrained(args.target_model)
    tgt_tok.pad_token = tgt_tok.eos_token

    # Compute target RMS
    with torch.no_grad():
        tgt_embeds = tgt_model.get_input_embeddings().weight.float()
        target_rms = tgt_embeds.pow(2).mean(dim=1).sqrt().median().item()
        if local_rank == 0:
            print(f"Target embedding RMS: {target_rms:.4f}")

    # Initialize bridge based on bridge_type
    if args.bridge_type == "ridge":
        if not EXPERIMENTAL_BRIDGES_AVAILABLE:
            raise ImportError("Experimental bridges not available. Check telepathy/cross_model_experiments.py")
        if local_rank == 0:
            print(f"Using Ridge Regression Bridge (lambda={args.lambda_reg})")
        bridge = RidgeRegressionBridge(
            src_model, tgt_model,
            lambda_reg=args.lambda_reg,
            pooling='last'
        )
        # Ridge is training-free, so we only evaluate
        args.eval_only = True

    elif args.bridge_type == "vib" or args.use_vib:
        if not EXPERIMENTAL_BRIDGES_AVAILABLE:
            raise ImportError("Experimental bridges not available. Check telepathy/cross_model_experiments.py")
        if local_rank == 0:
            print(f"Using VIB Bridge (beta={args.vib_beta})")
        bridge = VIBLatentBridge(
            src_dim=src_model.config.hidden_size,
            tgt_dim=tgt_model.config.hidden_size,
            num_latents=args.soft_tokens,
            depth=args.depth,
            heads=args.heads,
            target_rms=target_rms,
        )

    elif args.bridge_type == "multi_layer" or args.extract_layers:
        if not EXPERIMENTAL_BRIDGES_AVAILABLE:
            raise ImportError("Experimental bridges not available. Check telepathy/cross_model_experiments.py")
        extract_layers = args.extract_layers or [16, 24, 31]
        if local_rank == 0:
            print(f"Using Multi-Layer Bridge (layers={extract_layers})")
        bridge = MultiLayerBridge(
            src_dim=src_model.config.hidden_size,
            tgt_dim=tgt_model.config.hidden_size,
            extract_layers=extract_layers,
            num_latents=args.soft_tokens,
            depth=args.depth,
            heads=args.heads,
            learn_layer_weights=args.learn_layer_weights,
            target_rms=target_rms,
        )

    # =========================================================================
    # NOVEL BRIDGE TYPES
    # =========================================================================

    elif args.bridge_type == "predictive_coding":
        if not EXPERIMENTAL_BRIDGES_AVAILABLE:
            raise ImportError("Experimental bridges not available")
        if local_rank == 0:
            print(f"Using Predictive Coding Bridge (NOVEL)")
        bridge = PredictiveCodingBridge(
            args,
            src_dim=src_model.config.hidden_size,
            tgt_dim=tgt_model.config.hidden_size,
            target_rms=target_rms,
        )

    elif args.bridge_type == "optimal_transport":
        if not EXPERIMENTAL_BRIDGES_AVAILABLE:
            raise ImportError("Experimental bridges not available")
        if local_rank == 0:
            print(f"Using Optimal Transport Bridge (NOVEL) - epsilon={args.ot_epsilon}")
        bridge = OptimalTransportBridge(
            args,
            src_dim=src_model.config.hidden_size,
            tgt_dim=tgt_model.config.hidden_size,
            target_rms=target_rms,
            epsilon=args.ot_epsilon,
            n_iters=args.ot_iters,
        )

    elif args.bridge_type == "infonce":
        if not EXPERIMENTAL_BRIDGES_AVAILABLE:
            raise ImportError("Experimental bridges not available")
        if local_rank == 0:
            print(f"Using Contrastive InfoNCE Bridge (NOVEL) - temp={args.infonce_temp}")
        bridge = ContrastiveInfoNCEBridge(
            args,
            src_dim=src_model.config.hidden_size,
            tgt_dim=tgt_model.config.hidden_size,
            target_rms=target_rms,
            temperature=args.infonce_temp,
        )

    elif args.bridge_type == "sparse_kwta":
        if not EXPERIMENTAL_BRIDGES_AVAILABLE:
            raise ImportError("Experimental bridges not available")
        if local_rank == 0:
            print(f"Using Sparse k-WTA Bridge (NOVEL) - k={args.sparsity_k}")
        bridge = SparseKWTABridge(
            args,
            src_dim=src_model.config.hidden_size,
            tgt_dim=tgt_model.config.hidden_size,
            target_rms=target_rms,
            sparsity_k=args.sparsity_k,
        )

    elif args.bridge_type == "residual_coding":
        if not EXPERIMENTAL_BRIDGES_AVAILABLE:
            raise ImportError("Experimental bridges not available")
        if local_rank == 0:
            print(f"Using Residual Coding Bridge (NOVEL) - steps={args.num_refinement_steps}")
        bridge = ResidualCodingBridge(
            args,
            src_dim=src_model.config.hidden_size,
            tgt_dim=tgt_model.config.hidden_size,
            target_rms=target_rms,
            num_refinement_steps=args.num_refinement_steps,
        )

    elif args.bridge_type == "lock_and_key":
        if not EXPERIMENTAL_BRIDGES_AVAILABLE:
            raise ImportError("Experimental bridges not available")
        if local_rank == 0:
            print(f"Using Lock-and-Key Bridge (NOVEL) - receptors={args.num_receptors}")
        bridge = LockAndKeyBridge(
            args,
            src_dim=src_model.config.hidden_size,
            tgt_dim=tgt_model.config.hidden_size,
            target_rms=target_rms,
            num_receptors=args.num_receptors,
            binding_sparsity=args.binding_sparsity,
        )

    # =========================================================================
    # MATH BRIDGE TYPES (from math opus subagents)
    # =========================================================================

    elif args.bridge_type == "spectral_cca":
        if not EXPERIMENTAL_BRIDGES_AVAILABLE:
            raise ImportError("Experimental bridges not available")
        if local_rank == 0:
            print(f"Using Spectral CCA Bridge (MATH) - cca_dim={args.cca_dim}")
        bridge = SpectralCCABridge(
            args,
            src_dim=src_model.config.hidden_size,
            tgt_dim=tgt_model.config.hidden_size,
            target_rms=target_rms,
            cca_dim=args.cca_dim,
        )

    elif args.bridge_type == "flow_matching":
        if not EXPERIMENTAL_BRIDGES_AVAILABLE:
            raise ImportError("Experimental bridges not available")
        if local_rank == 0:
            print(f"Using Flow Matching Bridge (MATH) - steps={args.num_flow_steps}")
        bridge = FlowMatchingBridge(
            args,
            src_dim=src_model.config.hidden_size,
            tgt_dim=tgt_model.config.hidden_size,
            target_rms=target_rms,
            num_flow_steps=args.num_flow_steps,
        )

    elif args.bridge_type == "moe":
        if not EXPERIMENTAL_BRIDGES_AVAILABLE:
            raise ImportError("Experimental bridges not available")
        if local_rank == 0:
            print(f"Using MoE Bridge - {args.moe_num_experts} experts, top-{args.moe_top_k}, "
                  f"shared_expert={args.moe_use_shared_expert}, aux_loss_free={args.moe_aux_loss_free}")
        bridge = MoEBridge(
            args,
            src_dim=src_model.config.hidden_size,
            tgt_dim=tgt_model.config.hidden_size,
            target_rms=target_rms,
            num_experts=args.moe_num_experts,
            top_k=args.moe_top_k,
            use_shared_expert=args.moe_use_shared_expert,
            aux_loss_weight=args.moe_aux_loss_weight,
            use_aux_loss_free=args.moe_aux_loss_free,
        )

    # =========================================================================
    # HAIL MARY BRIDGE TYPES (Literature-Informed Novel Experiments)
    # =========================================================================

    elif args.bridge_type == "cross_modal_distillation":
        if not EXPERIMENTAL_BRIDGES_AVAILABLE:
            raise ImportError("Experimental bridges not available")
        if local_rank == 0:
            print(f"Using Cross-Modal Distillation Bridge - KL divergence to sender logits")
        from telepathy.cross_model_experiments import CrossModalDistillationBridge
        bridge = CrossModalDistillationBridge(
            args,
            src_dim=src_model.config.hidden_size,
            tgt_dim=tgt_model.config.hidden_size,
            target_rms=target_rms,
            temperature=2.0,
            kd_weight=0.5,
        )

    elif args.bridge_type == "mine":
        if not EXPERIMENTAL_BRIDGES_AVAILABLE:
            raise ImportError("Experimental bridges not available")
        if local_rank == 0:
            print(f"Using MINE Bridge - Mutual Information Neural Estimation")
        from telepathy.cross_model_experiments import MINEBridge
        bridge = MINEBridge(
            args,
            src_dim=src_model.config.hidden_size,
            tgt_dim=tgt_model.config.hidden_size,
            target_rms=target_rms,
            mine_weight=0.1,
        )

    elif args.bridge_type == "mixture_of_depths":
        if not EXPERIMENTAL_BRIDGES_AVAILABLE:
            raise ImportError("Experimental bridges not available")
        if local_rank == 0:
            print(f"Using Mixture-of-Depths Bridge - Adaptive compute via early exit")
        from telepathy.cross_model_experiments import MixtureOfDepthsBridge
        bridge = MixtureOfDepthsBridge(
            args,
            src_dim=src_model.config.hidden_size,
            tgt_dim=tgt_model.config.hidden_size,
            target_rms=target_rms,
            capacity_factor=0.5,
        )

    elif args.bridge_type == "thalamic_relay":
        if not EXPERIMENTAL_BRIDGES_AVAILABLE:
            raise ImportError("Experimental bridges not available")
        if local_rank == 0:
            print(f"Using Thalamic Relay Bridge - Inhibitory gating")
        from telepathy.cross_model_experiments import ThalamicRelayBridge
        bridge = ThalamicRelayBridge(
            args,
            src_dim=src_model.config.hidden_size,
            tgt_dim=tgt_model.config.hidden_size,
            target_rms=target_rms,
            gate_temperature=1.0,
        )

    elif args.bridge_type == "domain_adversarial":
        if not EXPERIMENTAL_BRIDGES_AVAILABLE:
            raise ImportError("Experimental bridges not available")
        if local_rank == 0:
            print(f"Using Domain Adversarial Bridge - Gradient reversal alignment")
        from telepathy.cross_model_experiments import DomainAdversarialBridge
        bridge = DomainAdversarialBridge(
            args,
            src_dim=src_model.config.hidden_size,
            tgt_dim=tgt_model.config.hidden_size,
            target_rms=target_rms,
            adv_weight=0.1,
        )

    elif args.bridge_type == "successive_refinement":
        if not EXPERIMENTAL_BRIDGES_AVAILABLE:
            raise ImportError("Experimental bridges not available")
        if local_rank == 0:
            print(f"Using Successive Refinement Bridge - Progressive token generation")
        from telepathy.cross_model_experiments import SuccessiveRefinementBridge
        bridge = SuccessiveRefinementBridge(
            args,
            src_dim=src_model.config.hidden_size,
            tgt_dim=tgt_model.config.hidden_size,
            target_rms=target_rms,
            max_tokens=16,
        )

    else:
        # Standard LatentBridge
        bridge = LatentBridge(
            args,
            src_dim=src_model.config.hidden_size,
            tgt_dim=tgt_model.config.hidden_size,
            target_rms=target_rms
        )

    if args.bf16:
        bridge = bridge.bfloat16()
    bridge.train()
    bridge.to(device)

    if torch.distributed.is_initialized():
        bridge = DDP(bridge, device_ids=[local_rank])

    # Optimizer
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=0.01)

    # LR Scheduler with linear warmup + cosine decay
    def lr_lambda(current_step):
        if current_step < args.warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, args.warmup_steps))
        # Cosine decay after warmup
        progress = float(current_step - args.warmup_steps) / float(max(1, args.steps - args.warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # =========================================================================
    # CHECKPOINT LOADING (for resume after preemption)
    # =========================================================================
    start_step = 0
    best_accuracy = 0.0

    # Determine checkpoint to resume from
    resume_path = args.resume
    if resume_path is None and args.auto_resume:
        # Auto-detect latest checkpoint
        resume_path = find_latest_checkpoint(args.output_dir)
        if resume_path and local_rank == 0:
            print(f"Auto-resume: Found checkpoint {resume_path}")

    if resume_path and os.path.exists(resume_path):
        if local_rank == 0:
            print(f"Loading checkpoint from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)

        # Handle both old (state_dict only) and new (full checkpoint) formats
        if isinstance(checkpoint, dict) and "bridge_state_dict" in checkpoint:
            # New format with full training state
            bridge_module = bridge.module if torch.distributed.is_initialized() else bridge
            bridge_module.load_state_dict(checkpoint["bridge_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_step = checkpoint["step"]
            best_accuracy = checkpoint.get("best_accuracy", 0.0)
            if local_rank == 0:
                print(f"  Resumed from step {start_step}, best_accuracy={best_accuracy:.1f}%")
        else:
            # Old format (just bridge state_dict) - can't resume optimizer/scheduler
            bridge_module = bridge.module if torch.distributed.is_initialized() else bridge
            bridge_module.load_state_dict(checkpoint)
            if local_rank == 0:
                print(f"  Loaded bridge weights (old format, starting training from step 0)")

    # Load dataset
    if len(config["load_args"]) == 2:
        train_ds = load_dataset(config["load_args"][0], config["load_args"][1],
                               split=config["train_split"], trust_remote_code=True)
        eval_ds = load_dataset(config["load_args"][0], config["load_args"][1],
                              split=config["eval_split"], trust_remote_code=True)
    else:
        train_ds = load_dataset(config["load_args"][0],
                               split=config["train_split"], trust_remote_code=True)
        eval_ds = load_dataset(config["load_args"][0],
                              split=config["eval_split"], trust_remote_code=True)

    # Get labels from dataset if not predefined (e.g., banking77)
    # Skip for generative tasks like GSM8K where labels should stay None
    if config["labels"] is None and not config.get("is_generative", False):
        try:
            config["labels"] = train_ds.features[config["label_field"]].names
        except AttributeError:
            # Dataset doesn't have ClassLabel with .names (e.g., string labels)
            # Try to extract unique values instead
            unique_labels = sorted(set(str(item[config["label_field"]]) for item in train_ds))
            config["labels"] = unique_labels
            print(f"Warning: Extracted {len(unique_labels)} unique labels from dataset")

    if torch.distributed.is_initialized():
        train_ds = train_ds.shard(world_size, local_rank)

    # Seeded generator for reproducible shuffling
    dl_generator = torch.Generator()
    dl_generator.manual_seed(args.seed)
    dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, generator=dl_generator, collate_fn=custom_collate_fn)

    if local_rank == 0:
        print(f"\nTraining on {len(train_ds)} samples")
        print(f"Validation: {len(eval_ds)} samples")
        if args.eval_only:
            print("EVAL ONLY MODE - skipping training\n")
        else:
            print("Starting training...\n")

    # Initialize curriculum trainer if enabled
    curriculum_trainer = None
    if args.use_curriculum and EXPERIMENTAL_BRIDGES_AVAILABLE:
        curriculum_trainer = CurriculumTrainer(
            stages=CLASSIFICATION_CURRICULUM,
            total_steps=args.steps
        )
        if local_rank == 0:
            print(f"Curriculum training enabled with {len(CLASSIFICATION_CURRICULUM)} stages")

    # Skip training for eval_only mode (e.g., ridge regression)
    if args.eval_only:
        if local_rank == 0:
            print("Skipping training loop (eval_only=True)")
    else:
        # Use start_step for resume support
        remaining_steps = args.steps - start_step
        if remaining_steps <= 0:
            if local_rank == 0:
                print(f"Training already complete (start_step={start_step} >= steps={args.steps})")
        else:
            if local_rank == 0 and start_step > 0:
                print(f"Resuming training from step {start_step} ({remaining_steps} steps remaining)")

        progress = tqdm(range(start_step, args.steps), disable=(local_rank != 0),
                       desc=f"{args.dataset.upper()}", ncols=100)
        iter_dl = iter(dl)
        running = {"total": 0, "lm": 0, "div": 0, "aux": 0, "z_var": 0, "grad_norm": 0}
        grad_accum = args.grad_accum

        # VIB beta annealing
        vib_beta = args.vib_beta if (args.use_vib or args.bridge_type == "vib") else 0.0

        # Learning rate (will be updated by scheduler each step)
        current_lr = scheduler.get_last_lr()[0]

        # NaN/Inf detection counters
        nan_inf_count = 0
        max_nan_inf_allowed = 10  # Stop training if too many NaN/Inf losses

        # Helper function to save emergency checkpoint
        def save_emergency_checkpoint(step, error_msg):
            """Save emergency checkpoint when training fails."""
            if local_rank != 0:
                return
            bridge_to_save = bridge.module if torch.distributed.is_initialized() else bridge
            emergency_path = os.path.join(args.output_dir, f"emergency_checkpoint_step{step}.pt")
            torch.save({
                "bridge_state_dict": bridge_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "step": step,
                "best_accuracy": best_accuracy,
                "config": vars(args),
                "error_message": error_msg,
            }, emergency_path)
            print(f"\n[EMERGENCY] Checkpoint saved to {emergency_path}")
            print(f"[EMERGENCY] Error: {error_msg}")

        try:
          for step in progress:
            optimizer.zero_grad()
            accum_loss_dict = {"total": 0, "lm": 0, "div": 0, "aux": 0, "z_var": 0}

            for _ in range(grad_accum):
                try:
                    batch = next(iter_dl)
                except StopIteration:
                    iter_dl = iter(dl)
                    batch = next(iter_dl)

                loss, loss_dict = train_step(
                    batch, src_tok, tgt_tok, src_model, bridge, tgt_model, device, args, config
                )

                # NaN/Inf detection
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_inf_count += 1
                    if local_rank == 0:
                        print(f"\n[WARNING] NaN/Inf loss detected at step {step+1} (count: {nan_inf_count}/{max_nan_inf_allowed})")
                        print(f"  Loss values: total={loss_dict['total']:.4f}, lm={loss_dict['lm']:.4f}, "
                              f"aux={loss_dict['aux']:.4f}, div={loss_dict['div']:.4f}")
                        # Log batch info for debugging
                        batch_texts = format_texts_for_batch(batch, config)
                        print(f"  Batch size: {len(batch_texts)}, first text length: {len(batch_texts[0]) if batch_texts else 0}")

                    if nan_inf_count >= max_nan_inf_allowed:
                        error_msg = f"Too many NaN/Inf losses ({nan_inf_count}). Training stopped."
                        save_emergency_checkpoint(step, error_msg)
                        raise RuntimeError(error_msg)

                    # Skip this gradient accumulation step
                    continue

                # Add VIB KL loss if applicable
                if (args.use_vib or args.bridge_type == "vib") and vib_beta > 0:
                    # loss_dict should contain kl_loss from VIBLatentBridge
                    kl_loss = loss_dict.get("kl", 0.0)
                    if args.vib_beta_anneal:
                        # Anneal beta from 0 to target over first half of training
                        anneal_factor = min(1.0, step / (args.steps * 0.5))
                        effective_beta = vib_beta * anneal_factor
                    else:
                        effective_beta = vib_beta
                    loss = loss + effective_beta * kl_loss

                scaled_loss = loss / grad_accum
                scaled_loss.backward()

                for k in accum_loss_dict:
                    accum_loss_dict[k] += loss_dict[k] / grad_accum

            # Track gradient norm before clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
            grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            optimizer.step()
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            # Update running stats
            for k in accum_loss_dict:
                running[k] += accum_loss_dict[k]
            running["grad_norm"] += grad_norm_val

            # Progress bar with key metrics
            progress.set_postfix({
                "lm": f"{accum_loss_dict['lm']:.2f}",
                "aux": f"{accum_loss_dict['aux']:.3f}",
                "gn": f"{grad_norm_val:.2f}"
            })

            # Periodic logging (every 50 steps)
            if local_rank == 0 and (step + 1) % 50 == 0:
                avg = {k: v / 50 for k, v in running.items()}
                print(f"\n[Step {step+1}/{args.steps}] lr={current_lr:.2e}")
                print(f"  LM Loss: {avg['lm']:.3f} | Aux Loss: {avg['aux']:.4f} | Div Loss: {avg['div']:.4f}")
                print(f"  Z Variance: {avg['z_var']:.4f} | Grad Norm: {avg['grad_norm']:.3f}")

                # GPU memory usage logging
                if torch.cuda.is_available():
                    gpu_mem_allocated = torch.cuda.memory_allocated(device) / (1024**3)  # GB
                    gpu_mem_reserved = torch.cuda.memory_reserved(device) / (1024**3)  # GB
                    gpu_mem_max = torch.cuda.max_memory_allocated(device) / (1024**3)  # GB
                    print(f"  GPU Memory: {gpu_mem_allocated:.2f}GB allocated, {gpu_mem_reserved:.2f}GB reserved, {gpu_mem_max:.2f}GB peak")

                # Log to training_log for JSON output
                log_entry = {
                    "step": step + 1,
                    "type": "metrics",
                    "lm_loss": avg['lm'],
                    "aux_loss": avg['aux'],
                    "div_loss": avg['div'],
                    "z_var": avg['z_var'],
                    "grad_norm": avg['grad_norm'],
                    "lr": current_lr
                }
                # Add GPU memory to log if available
                if torch.cuda.is_available():
                    log_entry["gpu_mem_allocated_gb"] = gpu_mem_allocated
                    log_entry["gpu_mem_reserved_gb"] = gpu_mem_reserved
                    log_entry["gpu_mem_peak_gb"] = gpu_mem_max
                training_log.append(log_entry)

                running = {k: 0 for k in running}

            # Quick eval
            if local_rank == 0 and (step + 1) % args.eval_every == 0:
                eval_result = quick_eval(bridge, src_model, tgt_model, src_tok, tgt_tok,
                                        eval_ds, device, args, config, step + 1)
                training_log.append({"step": step + 1, "type": "eval", **eval_result})

                # Track best checkpoint
                current_accuracy = eval_result["accuracy"]
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    bridge_to_save = bridge.module if torch.distributed.is_initialized() else bridge
                    best_ckpt_path = os.path.join(args.output_dir, "best_checkpoint.pt")
                    torch.save({
                        "bridge_state_dict": bridge_to_save.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "step": step + 1,
                        "best_accuracy": best_accuracy,
                        "config": vars(args),
                    }, best_ckpt_path)
                    print(f"  New best accuracy: {best_accuracy:.1f}% - saved to {best_ckpt_path}")

            # Save checkpoint (full state for resume)
            if local_rank == 0 and (step + 1) % args.save_every == 0:
                bridge_to_save = bridge.module if torch.distributed.is_initialized() else bridge
                ckpt_path = os.path.join(args.output_dir, f"checkpoint_step{step+1}.pt")
                torch.save({
                    "bridge_state_dict": bridge_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "step": step + 1,
                    "best_accuracy": best_accuracy,
                    "config": vars(args),
                }, ckpt_path)
                print(f"  Checkpoint saved: {ckpt_path}")

        except Exception as e:
            # Emergency checkpoint on any unexpected error
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}"
            if local_rank == 0:
                print(f"\n[ERROR] Training interrupted by exception at step {step+1}")
                print(f"[ERROR] {error_msg}")
                traceback.print_exc()
            save_emergency_checkpoint(step, error_msg)
            raise  # Re-raise the exception after saving checkpoint

    # Final save and evaluation
    if local_rank == 0:
        bridge_to_save = bridge.module if torch.distributed.is_initialized() else bridge
        final_path = os.path.join(args.output_dir, args.save_path)
        torch.save({
            "bridge_state_dict": bridge_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": args.steps,
            "best_accuracy": best_accuracy,
            "config": vars(args),
        }, final_path)

        print("\n" + "=" * 60)
        print(f"{args.dataset.upper()} Training Complete!")
        print(f"Final checkpoint: {final_path}")
        print(f"Best accuracy during training: {best_accuracy:.1f}%")
        print("=" * 60)

        # Final evaluation
        print("\n" + "=" * 60)
        print("FINAL EVALUATION (200 samples)")
        print("=" * 60)

        bridge_module = bridge.module if hasattr(bridge, 'module') else bridge
        bridge_module.eval()

        correct = 0
        total = 0
        labels = config["labels"]
        n_eval = min(200, len(eval_ds))

        for i in range(n_eval):
            item = eval_ds[i]
            text = format_text_for_item(item, config)
            label_idx = item[config["label_field"]]
            label = normalize_label_index(label_idx, labels)

            src_input = config["prompt_template"].format(text=text[:256])

            with torch.no_grad():
                src_enc = src_tok(src_input, return_tensors="pt", truncation=True,
                                max_length=config["max_length"]).to(device)
                src_out = src_model(**src_enc, output_hidden_states=True)
                src_h = src_out.hidden_states[args.source_layer]
                if args.bf16:
                    src_h = src_h.bfloat16()
                soft_tokens, _, _, _ = bridge_module(src_h, src_enc.attention_mask)

                primer = config["primer"]
                primer_enc = tgt_tok(primer, return_tensors="pt", add_special_tokens=False).to(device)
                primer_embeds = tgt_model.get_input_embeddings()(primer_enc.input_ids)
                if args.bf16:
                    primer_embeds = primer_embeds.bfloat16()

                combined_embeds = torch.cat([primer_embeds, soft_tokens], dim=1)
                attn_mask = torch.ones(combined_embeds.shape[:2], device=device, dtype=torch.long)

                # Use more tokens for generative tasks like GSM8K
                max_tokens = 30 if config.get("is_generative", False) else 10

                out_ids = tgt_model.generate(
                    inputs_embeds=combined_embeds,
                    attention_mask=attn_mask,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tgt_tok.eos_token_id,
                )
                output = tgt_tok.decode(out_ids[0], skip_special_tokens=True).strip().lower()

            if check_label_match(label, output, config):
                correct += 1
            total += 1

        final_accuracy = 100 * correct / total
        print(f"Final Accuracy: {final_accuracy:.1f}% ({correct}/{total})")
        final_results = {"accuracy": final_accuracy, "correct": correct, "total": total}

        # Save JSON results with full config and library versions
        import transformers
        results = {
            "experiment": args.dataset,
            "timestamp": datetime.now().isoformat(),
            "config": vars(args),  # Full config for reproducibility
            "library_versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda": torch.version.cuda if torch.cuda.is_available() else None,
                "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            },
            "num_classes": config["num_classes"],
            "random_baseline": config["random_baseline"],
            "final_results": final_results,
            "training_log": training_log
        }

        json_path = os.path.join(args.output_dir, f"{args.dataset}_seed{args.seed}_results.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {json_path}")

        # Latent Interpretability Analysis
        analyze_latent_interpretability(bridge, src_model, tgt_model, src_tok, tgt_tok,
                                       device, args, eval_ds, config)


if __name__ == "__main__":
    main()
