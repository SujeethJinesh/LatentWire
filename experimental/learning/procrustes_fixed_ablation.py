#!/usr/bin/env python3
"""
Procrustes Fixed Ablation: Testing cross-model transfer with proper implementation.

This script implements Procrustes alignment with all critical fixes:
1. Proper centering and normalization (scipy.spatial.procrustes standard)
2. Explicit position_ids tracking for RoPE
3. Layer ablation (embedding, early, mid, late, final layers)

Ablation Matrix:
- Layers: [0 (embedding), 8, 16, 24, 32 (final)]
- Configs: Mistral→Mistral, Llama→Mistral, Mistral→Llama
- Prompts: 5 diverse test cases

Expected outcome: Demonstrates that even with proper implementation,
training-free Procrustes cannot match learned approaches like LatentWire.
"""

import torch
import json
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Configuration
LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MISTRAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("runs/procrustes_fixed_ablation")
CALIBRATION_SIZE = 100
LAYERS_TO_TEST = [0, 8, 16, 24, 32]  # 0 = embedding, 32 = final

TEST_PROMPTS = [
    "The capital of France is",
    "To solve this problem, we need to",
    "The future of artificial intelligence is",
    "In the year 2050,",
    "The main difference between cats and dogs is",
]


class FixedProcrustesAlignment:
    """
    Procrustes alignment with proper preprocessing (scipy standard).

    Key fixes:
    1. Centers both source and target (subtract means)
    2. Normalizes to unit Frobenius norm (tr(AA^T) = 1)
    3. Computes optimal orthogonal rotation via SVD
    4. Stores normalization parameters for inference
    """

    def __init__(self, layer_idx=0):
        """
        Args:
            layer_idx: Which layer to use for alignment
                0 = embedding space (model.embed_tokens)
                >0 = hidden_states[layer_idx]
        """
        self.layer_idx = layer_idx
        self.W = None
        self.source_mean = None
        self.target_mean = None
        self.source_norm = None
        self.target_norm = None
        self.is_calibrated = False

    def _extract_representations(self, model, tokenizer, texts, device):
        """Extract representations from specified layer."""
        all_reprs = []

        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=512).to(device)

            with torch.no_grad():
                if self.layer_idx == 0:
                    # Embedding space
                    repr = model.model.embed_tokens(inputs['input_ids'])
                else:
                    # Hidden states from specified layer
                    outputs = model(**inputs, output_hidden_states=True)
                    repr = outputs.hidden_states[self.layer_idx]

                # Extract all tokens (no padding in this simplified version)
                all_reprs.append(repr[0])  # [seq_len, hidden_dim]

        # Concatenate all tokens from all texts
        return torch.cat(all_reprs, dim=0)  # [total_tokens, hidden_dim]

    def calibrate(self, model_a, tokenizer_a, model_b, tokenizer_b,
                  calibration_texts, device):
        """
        Calibrate Procrustes alignment with proper preprocessing.

        Steps (following scipy.spatial.procrustes):
        1. Extract representations from both models
        2. Center both sets (subtract column means)
        3. Normalize to unit Frobenius norm
        4. Compute optimal rotation W via SVD
        5. Store normalization parameters
        """
        print(f"  Calibrating Procrustes for layer {self.layer_idx}...")
        print(f"  Extracting representations from {len(calibration_texts)} texts...")

        # Extract representations
        source_states = self._extract_representations(
            model_a, tokenizer_a, calibration_texts, device)
        target_states = self._extract_representations(
            model_b, tokenizer_b, calibration_texts, device)

        # Handle different numbers of tokens (shouldn't happen with same texts, but safe)
        min_tokens = min(source_states.shape[0], target_states.shape[0])
        source_states = source_states[:min_tokens]
        target_states = target_states[:min_tokens]

        print(f"  Representation shape: {source_states.shape}")

        # Step 1: Center both matrices (subtract column means)
        self.source_mean = source_states.mean(dim=0, keepdim=True)
        self.target_mean = target_states.mean(dim=0, keepdim=True)

        source_centered = source_states - self.source_mean
        target_centered = target_states - self.target_mean

        print(f"  Source mean norm: {self.source_mean.norm().item():.4f}")
        print(f"  Target mean norm: {self.target_mean.norm().item():.4f}")

        # Step 2: Normalize to unit Frobenius norm (tr(AA^T) = 1)
        self.source_norm = torch.sqrt((source_centered ** 2).sum())
        self.target_norm = torch.sqrt((target_centered ** 2).sum())

        source_normalized = source_centered / self.source_norm
        target_normalized = target_centered / self.target_norm

        print(f"  Source Frobenius norm: {self.source_norm.item():.4f}")
        print(f"  Target Frobenius norm: {self.target_norm.item():.4f}")

        # Step 3: Compute optimal orthogonal rotation
        # Minimize ||W @ source - target||_F
        # Solution: W = U @ V^T where target^T @ source = U @ S @ V^T
        H = (target_normalized.T @ source_normalized).float()

        # Add small regularization for numerical stability
        eps = 1e-6
        H_reg = H + eps * torch.eye(H.shape[0], device=H.device)

        try:
            U, S, Vt = torch.linalg.svd(H_reg)
            self.W = (U @ Vt).to(source_states.dtype)
            print(f"  ✓ Rotation matrix W: {self.W.shape}")
            print(f"  Singular values (top 5): {S[:5].tolist()}")
        except Exception as e:
            print(f"  Warning: SVD failed ({e}), using identity")
            self.W = torch.eye(H.shape[0], device=device, dtype=source_states.dtype)

        self.is_calibrated = True
        print(f"  ✓ Calibration complete")

    def align(self, source_repr):
        """
        Apply Procrustes alignment to source representations.

        Steps (inverse of calibration preprocessing):
        1. Center: subtract source_mean
        2. Normalize: divide by source_norm
        3. Rotate: apply W
        4. Denormalize: multiply by target_norm
        5. Uncenter: add target_mean
        """
        if not self.is_calibrated:
            raise ValueError("Must calibrate before aligning")

        # Preserve original shape
        original_shape = source_repr.shape

        # Flatten to [total_tokens, hidden_dim]
        flat = source_repr.reshape(-1, source_repr.shape[-1])

        # Apply transformation
        centered = flat - self.source_mean
        normalized = centered / self.source_norm
        rotated = (self.W @ normalized.T).T
        scaled = rotated * self.target_norm
        aligned = scaled + self.target_mean

        # Restore shape
        return aligned.reshape(original_shape)


def generate_cross_model_fixed(model_a, tokenizer_a, model_b, tokenizer_b,
                                prompt, alignment_method, max_new_tokens=50):
    """
    Generate text using Model B conditioned on Model A's representations.

    Critical fix: Explicit position_ids tracking for RoPE.

    Args:
        model_a: Source model
        tokenizer_a: Source tokenizer
        model_b: Target model (will generate text)
        tokenizer_b: Target tokenizer
        prompt: Input text
        alignment_method: FixedProcrustesAlignment instance
        max_new_tokens: Maximum tokens to generate

    Returns:
        Generated text (prompt + continuation)
    """
    # Tokenize with Model A
    inputs_a = tokenizer_a(prompt, return_tensors="pt").to(model_a.device)

    with torch.no_grad():
        # Extract source representation from specified layer
        if alignment_method.layer_idx == 0:
            # Embedding space
            source_repr = model_a.model.embed_tokens(inputs_a['input_ids'])
        else:
            # Hidden states from specified layer
            outputs_a = model_a(**inputs_a, output_hidden_states=True)
            source_repr = outputs_a.hidden_states[alignment_method.layer_idx]

        # Align to target model's representation space
        aligned_repr = alignment_method.align(source_repr)

        # CRITICAL FIX: Create position_ids for the prefix
        seq_len = aligned_repr.shape[1]
        position_ids = torch.arange(0, seq_len, dtype=torch.long,
                                   device=model_b.device).unsqueeze(0)

        # Generate tokens autoregressively
        generated_ids = []
        past_key_values = None
        current_embeds = aligned_repr

        for step in range(max_new_tokens):
            if past_key_values is None:
                # First step: process entire prefix with explicit position_ids
                outputs_b = model_b.model(
                    inputs_embeds=current_embeds,
                    position_ids=position_ids,  # ✅ CRITICAL: Explicit positions for RoPE
                    past_key_values=None,
                    use_cache=True,
                    output_hidden_states=True
                )
            else:
                # Subsequent steps: process only new token
                # CRITICAL: Update position_ids for new token
                next_pos = position_ids[0, -1] + 1
                outputs_b = model_b.model(
                    inputs_embeds=next_embed,
                    position_ids=torch.tensor([[next_pos]], device=model_b.device),
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True
                )

            # Get logits and select next token (greedy decoding)
            logits = model_b.lm_head(outputs_b.hidden_states[-1])
            next_token_id = torch.argmax(logits[0, -1, :]).item()

            # Update KV cache
            past_key_values = outputs_b.past_key_values

            # Check for EOS
            if next_token_id == tokenizer_b.eos_token_id:
                break

            generated_ids.append(next_token_id)

            # Get embedding for next token
            next_embed = model_b.model.embed_tokens(
                torch.tensor([[next_token_id]], device=model_b.device))

            # Update position_ids tracking (calculate next position)
            next_pos = position_ids[0, -1] + 1
            position_ids = torch.cat([
                position_ids,
                torch.tensor([[next_pos]], device=model_b.device)
            ], dim=1)

        # Decode generated tokens
        generated_text = tokenizer_b.decode(generated_ids, skip_special_tokens=True)
        return prompt + generated_text


def load_calibration_texts(num_samples=100):
    """Load calibration texts from SQuAD dataset."""
    print(f"Loading {num_samples} calibration texts from SQuAD...")
    dataset = load_dataset("rajpurkar/squad", split="train", trust_remote_code=True)

    # Extract diverse questions
    texts = []
    for i in range(min(num_samples, len(dataset))):
        question = dataset[i]['question']
        context = dataset[i]['context'][:200]  # Truncate context
        texts.append(f"{context} Question: {question}")

    print(f"  ✓ Loaded {len(texts)} texts")
    return texts


def run_layer_ablation():
    """
    Run complete layer ablation experiment.

    For each layer in [0, 8, 16, 24, 32]:
        1. Calibrate Procrustes alignment
        2. Test on 3 configurations:
            - Mistral→Mistral (sanity check)
            - Llama→Mistral (cross-model)
            - Mistral→Llama (cross-model)
        3. Generate outputs for 5 test prompts
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("="*80)
    print("PROCRUSTES FIXED ABLATION - Layer Selection Study")
    print("="*80)
    print(f"Layers to test: {LAYERS_TO_TEST}")
    print(f"Test prompts: {len(TEST_PROMPTS)}")
    print(f"Device: {DEVICE}")
    print()

    # Load models
    print("Loading models...")
    print(f"  Llama: {LLAMA_MODEL}")
    llama_model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map=DEVICE
    )
    llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL)
    print(f"  ✓ Llama loaded")

    print(f"  Mistral: {MISTRAL_MODEL}")
    mistral_model = AutoModelForCausalLM.from_pretrained(
        MISTRAL_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map=DEVICE
    )
    mistral_tokenizer = AutoTokenizer.from_pretrained(MISTRAL_MODEL)
    print(f"  ✓ Mistral loaded")
    print()

    # Load calibration texts
    calibration_texts = load_calibration_texts(CALIBRATION_SIZE)
    print()

    # Results storage
    results = {
        "metadata": {
            "timestamp": timestamp,
            "llama_model": LLAMA_MODEL,
            "mistral_model": MISTRAL_MODEL,
            "calibration_size": CALIBRATION_SIZE,
            "layers_tested": LAYERS_TO_TEST,
        },
        "results": {}
    }

    # Run ablation for each layer
    for layer_idx in LAYERS_TO_TEST:
        print("="*80)
        print(f"LAYER {layer_idx}: {'Embedding Space' if layer_idx == 0 else f'Hidden States Layer {layer_idx}'}")
        print("="*80)

        layer_results = {}

        # Configuration 1: Mistral → Mistral (sanity check)
        print("\n[1/3] Mistral → Mistral (sanity check)")
        print("-" * 40)

        alignment_mm = FixedProcrustesAlignment(layer_idx=layer_idx)
        alignment_mm.calibrate(
            mistral_model, mistral_tokenizer,
            mistral_model, mistral_tokenizer,
            calibration_texts, DEVICE
        )

        mm_outputs = {}
        for i, prompt in enumerate(TEST_PROMPTS, 1):
            print(f"  [{i}/{len(TEST_PROMPTS)}] Generating: {prompt[:50]}...")
            output = generate_cross_model_fixed(
                mistral_model, mistral_tokenizer,
                mistral_model, mistral_tokenizer,
                prompt, alignment_mm, max_new_tokens=50
            )
            mm_outputs[f"prompt_{i}"] = output
            print(f"       → {output}")

        layer_results["mistral_to_mistral"] = mm_outputs

        # Configuration 2: Llama → Mistral
        print("\n[2/3] Llama → Mistral (cross-model)")
        print("-" * 40)

        alignment_lm = FixedProcrustesAlignment(layer_idx=layer_idx)
        alignment_lm.calibrate(
            llama_model, llama_tokenizer,
            mistral_model, mistral_tokenizer,
            calibration_texts, DEVICE
        )

        lm_outputs = {}
        for i, prompt in enumerate(TEST_PROMPTS, 1):
            print(f"  [{i}/{len(TEST_PROMPTS)}] Generating: {prompt[:50]}...")
            output = generate_cross_model_fixed(
                llama_model, llama_tokenizer,
                mistral_model, mistral_tokenizer,
                prompt, alignment_lm, max_new_tokens=50
            )
            lm_outputs[f"prompt_{i}"] = output
            print(f"       → {output}")

        layer_results["llama_to_mistral"] = lm_outputs

        # Configuration 3: Mistral → Llama
        print("\n[3/3] Mistral → Llama (cross-model)")
        print("-" * 40)

        alignment_ml = FixedProcrustesAlignment(layer_idx=layer_idx)
        alignment_ml.calibrate(
            mistral_model, mistral_tokenizer,
            llama_model, llama_tokenizer,
            calibration_texts, DEVICE
        )

        ml_outputs = {}
        for i, prompt in enumerate(TEST_PROMPTS, 1):
            print(f"  [{i}/{len(TEST_PROMPTS)}] Generating: {prompt[:50]}...")
            output = generate_cross_model_fixed(
                mistral_model, mistral_tokenizer,
                llama_model, llama_tokenizer,
                prompt, alignment_ml, max_new_tokens=50
            )
            ml_outputs[f"prompt_{i}"] = output
            print(f"       → {output}")

        layer_results["mistral_to_llama"] = ml_outputs

        results["results"][f"layer_{layer_idx}"] = layer_results
        print()

    # Save results
    results_file = OUTPUT_DIR / f"procrustes_layer_ablation_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"Results saved to: {results_file}")
    print()

    # Print summary
    print("SUMMARY:")
    print("-" * 80)
    for layer_idx in LAYERS_TO_TEST:
        layer_name = "Embedding" if layer_idx == 0 else f"Layer {layer_idx}"
        print(f"{layer_name}:")

        layer_data = results["results"][f"layer_{layer_idx}"]

        # Check for common failure patterns
        for config_name, config_data in layer_data.items():
            failures = []
            for prompt_key, output in config_data.items():
                if len(output) < 50:  # Too short (likely immediate EOS)
                    failures.append("immediate_EOS")
                elif "�" in output or len(set(output.split())) < 3:  # Gibberish
                    failures.append("gibberish")

            if failures:
                print(f"  {config_name}: {len(failures)}/{len(TEST_PROMPTS)} failures ({', '.join(set(failures))})")
            else:
                print(f"  {config_name}: All prompts generated successfully")
        print()

    return results


if __name__ == "__main__":
    run_layer_ablation()
