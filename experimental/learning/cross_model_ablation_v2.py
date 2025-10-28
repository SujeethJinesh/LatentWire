"""
Cross-Model Hidden State Transfer Ablation v2

Enhanced experiment testing multiple alignment methods and layer-wise transfer.

New features:
1. Procrustes alignment (SVD-based, no training)
2. Layer-wise transfer (early/middle/late layers)
3. Evaluation metrics (cosine similarity, perplexity, generation time)
4. Fixed KV cache and position handling
5. Comprehensive comparison table
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from typing import Optional, Tuple, Dict
import numpy as np

def procrustes_alignment(source_states: torch.Tensor, target_states: torch.Tensor) -> torch.Tensor:
    """
    Compute Procrustes alignment matrix using SVD.

    Given source and target hidden states, finds optimal rotation W such that:
    W @ source ≈ target

    Formula: W = V @ U.T where U, S, V = SVD(target.T @ source)

    Args:
        source_states: (N, hidden_dim) or (batch, seq_len, hidden_dim)
        target_states: (N, hidden_dim) or (batch, seq_len, hidden_dim)

    Returns:
        W: (hidden_dim, hidden_dim) alignment matrix
    """
    # Flatten to (N, hidden_dim) where N = batch * seq_len (if needed)
    if source_states.ndim == 3:
        source_flat = source_states.reshape(-1, source_states.shape[-1])  # (N, D)
        target_flat = target_states.reshape(-1, target_states.shape[-1])  # (N, D)
    else:
        source_flat = source_states
        target_flat = target_states

    # Compute covariance matrix: H = target.T @ source
    # Convert to float32 for SVD (float16 not supported)
    # Add small regularization for numerical stability
    H = (target_flat.T @ source_flat).float()  # (D, D)

    # Add regularization (ridge regression-style)
    eps = 1e-4
    H_reg = H + eps * torch.eye(H.shape[0], device=H.device)

    # SVD: H = U @ S @ V.T
    try:
        U, S, Vt = torch.linalg.svd(H_reg)
    except:
        # If SVD still fails, return identity matrix
        print("  Warning: SVD failed, using identity matrix")
        return torch.eye(H.shape[0], device=source_states.device, dtype=source_states.dtype)

    # Optimal rotation: W = U @ V.T
    W = U @ Vt

    # Convert back to original dtype
    W = W.to(source_states.dtype)

    return W


def compute_perplexity(model, tokenizer, text: str, device: str) -> float:
    """
    Compute perplexity of generated text under the model.

    Lower perplexity = better quality (model is more confident)
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss).item()

    return perplexity


def cosine_similarity_metric(hidden1: torch.Tensor, hidden2: torch.Tensor) -> float:
    """Compute average cosine similarity between two hidden state sequences."""
    # Flatten to (N, D)
    h1 = hidden1.reshape(-1, hidden1.shape[-1])
    h2 = hidden2.reshape(-1, hidden2.shape[-1])

    # Compute cosine similarity for each position
    cos_sim = F.cosine_similarity(h1, h2, dim=-1)

    return cos_sim.mean().item()


def generate_with_hidden_states(
    model,
    tokenizer,
    initial_hidden_states: torch.Tensor,
    max_new_tokens: int,
    device: str
) -> Tuple[str, float]:
    """
    Generate text starting from given hidden states with proper KV cache.

    Returns:
        (generated_text, generation_time_seconds)
    """
    generated_ids = []
    past_key_values = None
    current_hidden = initial_hidden_states

    start_time = time.time()

    with torch.no_grad():
        for step in range(max_new_tokens):
            # Forward pass with KV cache
            if past_key_values is None:
                # First step: process all initial hidden states
                outputs = model.model(
                    inputs_embeds=current_hidden,
                    past_key_values=None,
                    use_cache=True,
                    return_dict=True
                )
            else:
                # Subsequent steps: only process new token embedding
                outputs = model.model(
                    inputs_embeds=current_hidden[:, -1:, :],  # Only last token
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )

            # Get logits and select next token (greedy)
            logits = model.lm_head(outputs.last_hidden_state)
            next_token_id = torch.argmax(logits[0, -1, :]).item()

            # Store KV cache for next iteration
            past_key_values = outputs.past_key_values

            # Check for EOS
            if next_token_id == tokenizer.eos_token_id:
                break

            generated_ids.append(next_token_id)

            # Get embedding for next token and concatenate
            next_embedding = model.model.embed_tokens(torch.tensor([[next_token_id]]).to(device))
            current_hidden = torch.cat([current_hidden, next_embedding], dim=1)

    gen_time = time.time() - start_time
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generated_text, gen_time


def extract_layer_hidden_states(
    model,
    tokenizer,
    text: str,
    layer_idx: int,
    device: str
) -> torch.Tensor:
    """
    Extract hidden states from a specific layer of the model.

    Args:
        layer_idx: Layer index to extract from (0 = embedding, -1 = last layer)
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        # outputs.hidden_states is tuple of (num_layers + 1) tensors
        # Index 0 is embeddings, 1 is after first layer, etc.
        hidden_states = outputs.hidden_states[layer_idx]

    return hidden_states


def run_experiment(
    model_a, tokenizer_a,
    model_b, tokenizer_b,
    prompt: str,
    max_new_tokens: int,
    device: str,
    alignment_method: str = "none",
    source_layer: Optional[int] = None,
    alignment_matrix: Optional[torch.Tensor] = None
) -> Dict:
    """
    Run a single cross-model transfer experiment.

    Args:
        alignment_method: "none", "linear", "procrustes"
        source_layer: Layer index to extract from source model (None = final layer)
        alignment_matrix: Pre-computed alignment matrix (for procrustes)

    Returns:
        Dictionary with results
    """
    start_time = time.time()

    # Tokenize prompt with source model
    inputs_a = tokenizer_a(prompt, return_tensors="pt").to(device)

    # Extract hidden states from source model
    with torch.no_grad():
        outputs_a = model_a(**inputs_a, output_hidden_states=True, return_dict=True)

        if source_layer is not None:
            # Extract from specific layer
            hidden_states_a = outputs_a.hidden_states[source_layer]
        else:
            # Use final layer
            hidden_states_a = outputs_a.hidden_states[-1]

    # Apply alignment if specified
    if alignment_method == "none":
        hidden_states_b = hidden_states_a
    elif alignment_method == "procrustes":
        if alignment_matrix is None:
            raise ValueError("Procrustes alignment requires pre-computed matrix")
        # Apply: W @ source
        hidden_states_b = (alignment_matrix @ hidden_states_a.reshape(-1, hidden_states_a.shape[-1]).T).T
        hidden_states_b = hidden_states_b.reshape(hidden_states_a.shape)
    elif alignment_method == "linear":
        # Random linear projection (untrained baseline)
        dim_a = hidden_states_a.shape[-1]
        dim_b = model_b.config.hidden_size
        if dim_a != dim_b:
            W = torch.randn(dim_b, dim_a).to(device) * 0.01
            hidden_states_b = (W @ hidden_states_a.reshape(-1, dim_a).T).T
            hidden_states_b = hidden_states_b.reshape(hidden_states_a.shape[0], hidden_states_a.shape[1], dim_b)
        else:
            hidden_states_b = hidden_states_a

    # Generate with target model
    generated_text, gen_time = generate_with_hidden_states(
        model_b, tokenizer_b, hidden_states_b, max_new_tokens, device
    )

    full_text = prompt + generated_text

    # Compute metrics
    try:
        perplexity = compute_perplexity(model_b, tokenizer_b, full_text, device)
    except:
        perplexity = float('inf')

    # Compute cosine similarity with native hidden states
    with torch.no_grad():
        inputs_b_native = tokenizer_b(prompt, return_tensors="pt").to(device)
        outputs_b_native = model_b(**inputs_b_native, output_hidden_states=True, return_dict=True)
        hidden_native = outputs_b_native.hidden_states[-1]  # Last layer

        # Compare prompt portion only (since generation differs)
        min_len = min(hidden_states_b.shape[1], hidden_native.shape[1])
        cos_sim = cosine_similarity_metric(
            hidden_states_b[:, :min_len, :],
            hidden_native[:, :min_len, :]
        )

    total_time = time.time() - start_time

    return {
        "output": full_text,
        "generated_only": generated_text,
        "perplexity": perplexity,
        "cosine_sim": cos_sim,
        "gen_time": gen_time,
        "total_time": total_time,
        "num_tokens": len(generated_text.split())
    }


def compute_procrustes_matrix(
    model_a, tokenizer_a,
    model_b, tokenizer_b,
    calibration_texts: list,
    device: str
) -> torch.Tensor:
    """
    Compute Procrustes alignment matrix using calibration data.

    Uses multiple texts to get better alignment estimate.
    """
    all_source = []
    all_target = []

    for text in calibration_texts:
        # Get hidden states from both models
        inputs_a = tokenizer_a(text, return_tensors="pt").to(device)
        inputs_b = tokenizer_b(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs_a = model_a(**inputs_a, output_hidden_states=True, return_dict=True)
            outputs_b = model_b(**inputs_b, output_hidden_states=True, return_dict=True)

            # Get last hidden state from hidden_states tuple
            hidden_a = outputs_a.hidden_states[-1]  # Last layer
            hidden_b = outputs_b.hidden_states[-1]  # Last layer

            # Handle different sequence lengths (tokenizers differ)
            min_len = min(hidden_a.shape[1], hidden_b.shape[1])
            all_source.append(hidden_a[:, :min_len, :])
            all_target.append(hidden_b[:, :min_len, :])

    # Concatenate all calibration data along batch AND sequence dimensions
    # This flattens to (total_tokens, hidden_dim) format
    source_states = torch.cat([s.reshape(-1, s.shape[-1]) for s in all_source], dim=0)
    target_states = torch.cat([t.reshape(-1, t.shape[-1]) for t in all_target], dim=0)

    # Compute alignment
    W = procrustes_alignment(source_states, target_states)

    return W


def main():
    print("="*80)
    print("Cross-Model Hidden State Transfer Ablation v2")
    print("="*80)

    # Configuration
    prompt = "The future of artificial intelligence is"
    max_new_tokens = 50

    llama_model_id = "meta-llama/Llama-3.1-8B"
    mistral_model_id = "mistralai/Mistral-7B-v0.1"

    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
        print("Using device: cuda")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using device: mps")
    else:
        device = "cpu"
        print("Using device: cpu")

    print(f"\nLoading models...")

    # Load models
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_id)
    llama_model = AutoModelForCausalLM.from_pretrained(
        llama_model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)

    mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_id)
    mistral_model = AutoModelForCausalLM.from_pretrained(
        mistral_model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)

    print("✓ Models loaded")

    # Calibration texts for Procrustes alignment
    calibration_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
    ]

    print(f"\nComputing Procrustes alignment matrix...")
    procrustes_matrix = compute_procrustes_matrix(
        llama_model, llama_tokenizer,
        mistral_model, mistral_tokenizer,
        calibration_texts,
        device
    )
    print(f"✓ Procrustes matrix computed: {procrustes_matrix.shape}")

    # Run all experiments
    results = []

    print(f"\n{'='*80}")
    print("Running Experiments")
    print(f"{'='*80}\n")

    # Experiment 1: Llama baseline
    print("[1/10] Llama 3.1 8B alone (baseline)...")
    llama_inputs = llama_tokenizer(prompt, return_tensors="pt").to(device)
    start = time.time()
    with torch.no_grad():
        llama_outputs = llama_model.generate(
            **llama_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=llama_tokenizer.eos_token_id
        )
    llama_time = time.time() - start
    llama_text = llama_tokenizer.decode(llama_outputs[0], skip_special_tokens=True)
    llama_ppl = compute_perplexity(llama_model, llama_tokenizer, llama_text, device)

    results.append({
        "experiment": "Llama alone",
        "method": "baseline",
        "output": llama_text,
        "perplexity": llama_ppl,
        "cosine_sim": 1.0,
        "gen_time": llama_time
    })
    print(f"   Output: {llama_text[:80]}...")

    # Experiment 2: Mistral baseline
    print("[2/10] Mistral 7B alone (baseline)...")
    mistral_inputs = mistral_tokenizer(prompt, return_tensors="pt").to(device)
    start = time.time()
    with torch.no_grad():
        mistral_outputs = mistral_model.generate(
            **mistral_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=mistral_tokenizer.eos_token_id
        )
    mistral_time = time.time() - start
    mistral_text = mistral_tokenizer.decode(mistral_outputs[0], skip_special_tokens=True)
    mistral_ppl = compute_perplexity(mistral_model, mistral_tokenizer, mistral_text, device)

    results.append({
        "experiment": "Mistral alone",
        "method": "baseline",
        "output": mistral_text,
        "perplexity": mistral_ppl,
        "cosine_sim": 1.0,
        "gen_time": mistral_time
    })
    print(f"   Output: {mistral_text[:80]}...")

    # Experiment 3: Llama → Llama (sanity check)
    print("[3/10] Llama → Llama (no alignment)...")
    result = run_experiment(
        llama_model, llama_tokenizer,
        llama_model, llama_tokenizer,
        prompt, max_new_tokens, device,
        alignment_method="none"
    )
    results.append({
        "experiment": "Llama → Llama",
        "method": "no alignment",
        **result
    })
    print(f"   Output: {result['output'][:80]}...")

    # Experiment 4: Mistral → Mistral (sanity check)
    print("[4/10] Mistral → Mistral (no alignment)...")
    result = run_experiment(
        mistral_model, mistral_tokenizer,
        mistral_model, mistral_tokenizer,
        prompt, max_new_tokens, device,
        alignment_method="none"
    )
    results.append({
        "experiment": "Mistral → Mistral",
        "method": "no alignment",
        **result
    })
    print(f"   Output: {result['output'][:80]}...")

    # Experiment 5: Llama → Mistral (no alignment)
    print("[5/10] Llama → Mistral (no alignment)...")
    result = run_experiment(
        llama_model, llama_tokenizer,
        mistral_model, mistral_tokenizer,
        prompt, max_new_tokens, device,
        alignment_method="none"
    )
    results.append({
        "experiment": "Llama → Mistral",
        "method": "no alignment",
        **result
    })
    print(f"   Output: {result['output'][:80]}...")

    # Experiment 6: Llama → Mistral (Procrustes)
    print("[6/10] Llama → Mistral (Procrustes alignment)...")
    result = run_experiment(
        llama_model, llama_tokenizer,
        mistral_model, mistral_tokenizer,
        prompt, max_new_tokens, device,
        alignment_method="procrustes",
        alignment_matrix=procrustes_matrix
    )
    results.append({
        "experiment": "Llama → Mistral",
        "method": "Procrustes",
        **result
    })
    print(f"   Output: {result['output'][:80]}...")

    # Experiment 7: Layer 8 transfer (early)
    print("[7/10] Llama layer 8 → Mistral (no alignment)...")
    result = run_experiment(
        llama_model, llama_tokenizer,
        mistral_model, mistral_tokenizer,
        prompt, max_new_tokens, device,
        alignment_method="none",
        source_layer=8
    )
    results.append({
        "experiment": "Llama L8 → Mistral",
        "method": "layer 8 (early)",
        **result
    })
    print(f"   Output: {result['output'][:80]}...")

    # Experiment 8: Layer 16 transfer (middle)
    print("[8/10] Llama layer 16 → Mistral (no alignment)...")
    result = run_experiment(
        llama_model, llama_tokenizer,
        mistral_model, mistral_tokenizer,
        prompt, max_new_tokens, device,
        alignment_method="none",
        source_layer=16
    )
    results.append({
        "experiment": "Llama L16 → Mistral",
        "method": "layer 16 (middle)",
        **result
    })
    print(f"   Output: {result['output'][:80]}...")

    # Experiment 9: Layer 24 transfer (late)
    print("[9/10] Llama layer 24 → Mistral (no alignment)...")
    result = run_experiment(
        llama_model, llama_tokenizer,
        mistral_model, mistral_tokenizer,
        prompt, max_new_tokens, device,
        alignment_method="none",
        source_layer=24
    )
    results.append({
        "experiment": "Llama L24 → Mistral",
        "method": "layer 24 (late)",
        **result
    })
    print(f"   Output: {result['output'][:80]}...")

    # Experiment 10: Mistral → Llama (Procrustes reverse)
    print("[10/10] Mistral → Llama (Procrustes alignment)...")
    # Compute reverse Procrustes matrix
    procrustes_matrix_rev = compute_procrustes_matrix(
        mistral_model, mistral_tokenizer,
        llama_model, llama_tokenizer,
        calibration_texts,
        device
    )
    result = run_experiment(
        mistral_model, mistral_tokenizer,
        llama_model, llama_tokenizer,
        prompt, max_new_tokens, device,
        alignment_method="procrustes",
        alignment_matrix=procrustes_matrix_rev
    )
    results.append({
        "experiment": "Mistral → Llama",
        "method": "Procrustes",
        **result
    })
    print(f"   Output: {result['output'][:80]}...")

    # Print results table
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}\n")

    print(f"{'Experiment':<25} {'Method':<20} {'PPL':<8} {'CosSim':<8} {'Time(s)':<8}")
    print("-" * 80)

    for r in results:
        exp = r['experiment']
        method = r['method']
        ppl = r['perplexity']
        cos = r['cosine_sim']
        t = r['gen_time']

        ppl_str = f"{ppl:.2f}" if ppl < 1000 else "inf"
        print(f"{exp:<25} {method:<20} {ppl_str:<8} {cos:<8.3f} {t:<8.2f}")

    print("\n" + "="*80)
    print("OUTPUT COMPARISON")
    print("="*80 + "\n")

    for r in results:
        print(f"{'='*80}")
        print(f"{r['experiment']} ({r['method']})")
        print(f"{'='*80}")
        print(r['output'])
        print(f"\nMetrics: PPL={r['perplexity']:.2f}, CosSim={r['cosine_sim']:.3f}, Time={r['gen_time']:.2f}s")
        print()


if __name__ == "__main__":
    main()
