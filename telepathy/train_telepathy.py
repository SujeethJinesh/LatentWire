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
        "text_field": "question",
        "label_field": "answerKey",
        "labels": ["A", "B", "C", "D"],
        "num_classes": 4,
        "train_split": "train",
        "eval_split": "test",
        "max_length": 256,
        "prompt_template": "Question: {text}\nAnswer (A, B, C, or D):",
        "primer": "Answer:",
        "random_baseline": 25.0,
        "is_reasoning": True,
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
}


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
        label = labels[label_idx] if labels else str(label_idx)
        if label not in seen_labels:
            text = item[config["text_field"]]
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


def check_label_match(label, output, config):
    """Check if label matches output, with permissive matching for synonyms."""
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


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Telepathy Training")

    # Dataset selection
    parser.add_argument("--dataset", type=str, default="sst2",
                       choices=["sst2", "agnews", "trec", "banking77",
                                "arc_easy", "winogrande", "hellaswag", "boolq"],
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
                               "spectral_cca", "flow_matching"],
                       help="Bridge architecture: standard, ridge, vib, multi_layer, "
                            "NOVEL types: predictive_coding, optimal_transport, infonce, "
                            "sparse_kwta, residual_coding, lock_and_key, moe, "
                            "MATH types: spectral_cca, flow_matching")

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
        text = item[config["text_field"]]
        label_idx = item[config["label_field"]]
        label = labels[label_idx] if labels else str(label_idx)

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

            out_ids = tgt_model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=attn_mask,
                max_new_tokens=10,
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
    inputs = batch[config["text_field"]]
    label_texts = [labels[l] for l in batch[config["label_field"]]] if labels else [str(l) for l in batch[config["label_field"]]]

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
        tgt_enc = tgt_tok(
            tgt_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=16, add_special_tokens=False
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

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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
    if config["labels"] is None:
        config["labels"] = train_ds.features[config["label_field"]].names

    if torch.distributed.is_initialized():
        train_ds = train_ds.shard(world_size, local_rank)

    dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

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
        progress = tqdm(range(args.steps), disable=(local_rank != 0),
                       desc=f"{args.dataset.upper()}", ncols=100)
        iter_dl = iter(dl)
        running = {"total": 0, "lm": 0, "div": 0, "aux": 0, "z_var": 0, "grad_norm": 0}
        grad_accum = args.grad_accum

        # VIB beta annealing
        vib_beta = args.vib_beta if (args.use_vib or args.bridge_type == "vib") else 0.0

        # Learning rate for logging (no scheduler currently, but track it)
        current_lr = args.lr

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

                # Log to training_log for JSON output
                training_log.append({
                    "step": step + 1,
                    "type": "metrics",
                    "lm_loss": avg['lm'],
                    "aux_loss": avg['aux'],
                    "div_loss": avg['div'],
                    "z_var": avg['z_var'],
                    "grad_norm": avg['grad_norm'],
                    "lr": current_lr
                })

                running = {k: 0 for k in running}

            # Quick eval
            if local_rank == 0 and (step + 1) % args.eval_every == 0:
                eval_result = quick_eval(bridge, src_model, tgt_model, src_tok, tgt_tok,
                                        eval_ds, device, args, config, step + 1)
                training_log.append({"step": step + 1, "type": "eval", **eval_result})

            # Save checkpoint
            if local_rank == 0 and (step + 1) % args.save_every == 0:
                bridge_to_save = bridge.module if torch.distributed.is_initialized() else bridge
                ckpt_path = os.path.join(args.output_dir, f"checkpoint_step{step+1}.pt")
                torch.save(bridge_to_save.state_dict(), ckpt_path)
                print(f"  Checkpoint saved: {ckpt_path}")

    # Final save and evaluation
    if local_rank == 0:
        bridge_to_save = bridge.module if torch.distributed.is_initialized() else bridge
        final_path = os.path.join(args.output_dir, args.save_path)
        torch.save(bridge_to_save.state_dict(), final_path)

        print("\n" + "=" * 60)
        print(f"{args.dataset.upper()} Training Complete!")
        print(f"Checkpoint: {final_path}")
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
            text = item[config["text_field"]]
            label_idx = item[config["label_field"]]
            label = labels[label_idx] if labels else str(label_idx)

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

                out_ids = tgt_model.generate(
                    inputs_embeds=combined_embeds,
                    attention_mask=attn_mask,
                    max_new_tokens=10,
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

        # Save JSON results
        results = {
            "experiment": args.dataset,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "output_dir": args.output_dir,
                "steps": args.steps,
                "soft_tokens": args.soft_tokens,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "eval_every": args.eval_every,
                "diversity_weight": args.diversity_weight,
                "source_layer": args.source_layer,
                "seed": args.seed,
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
