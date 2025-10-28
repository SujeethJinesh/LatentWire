"""
Cross-Model Hidden State Transfer Experiment

This script tests whether hidden states from one LLM can be used to condition
another LLM for text generation. This is relevant for LatentWire's goal of
creating a universal interlingua that works across models.

Models tested:
- Llama 3.1 8B (hidden_size=4096)
- Mistral 7B (hidden_size=4096)

Both models have matching hidden dimensions, allowing direct transfer without
an alignment layer. This lets us test pure architectural compatibility.

Ablations:
1. Llama alone (baseline)
2. Llama → Llama (sanity check, should match baseline)
3. Llama → Mistral (cross-model transfer)
4. Mistral → Mistral (sanity check)
5. Mistral alone (baseline)
6. Mistral → Llama (reverse cross-model transfer)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def cross_model_experiment():
    """
    Test passing hidden states between different models.
    Both Llama 3.1 8B and Mistral 7B have hidden_size=4096, so no alignment needed!
    """

    # Load models
    print("Loading models...")
    llama_model_id = "meta-llama/Llama-3.1-8B"
    mistral_model_id = "mistralai/Mistral-7B-v0.1"

    # Detect device - use MPS on Mac, CUDA on Linux/Windows
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_id)
    llama_model = AutoModelForCausalLM.from_pretrained(
        llama_model_id,
        torch_dtype=torch.float16,  # MPS compatible (not bfloat16)
        low_cpu_mem_usage=True
    ).to(device)

    mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_id)
    mistral_model = AutoModelForCausalLM.from_pretrained(
        mistral_model_id,
        torch_dtype=torch.float16,  # MPS compatible (not bfloat16)
        low_cpu_mem_usage=True
    ).to(device)
    
    # Test prompt
    prompt = "The future of artificial intelligence is"
    max_new_tokens = 50
    
    # Ablation 1: Llama 3.1 8B alone (baseline)
    print("\n=== Llama 3.1 8B Alone ===")
    llama_only = generate_baseline(llama_model, llama_tokenizer, prompt, max_new_tokens)
    print(llama_only)
    
    # Ablation 2: Llama → Llama (sanity check)
    print("\n=== Llama 3.1 8B → Llama 3.1 8B ===")
    llama_to_llama = generate_cross_model(
        llama_model, llama_tokenizer,
        llama_model, llama_tokenizer,
        prompt, max_new_tokens
    )
    print(llama_to_llama)
    
    # Ablation 3: Llama → Mistral (cross-model, matching dimensions!)
    print("\n=== Llama 3.1 8B → Mistral 7B ===")
    llama_to_mistral = generate_cross_model(
        llama_model, llama_tokenizer,
        mistral_model, mistral_tokenizer,
        prompt, max_new_tokens
    )
    print(llama_to_mistral)

    # Ablation 4: Mistral → Mistral (sanity check)
    print("\n=== Mistral 7B → Mistral 7B ===")
    mistral_to_mistral = generate_cross_model(
        mistral_model, mistral_tokenizer,
        mistral_model, mistral_tokenizer,
        prompt, max_new_tokens
    )
    print(mistral_to_mistral)

    # Ablation 5: Mistral alone (baseline)
    print("\n=== Mistral 7B Alone ===")
    mistral_only = generate_baseline(mistral_model, mistral_tokenizer, prompt, max_new_tokens)
    print(mistral_only)

    # Ablation 6: Mistral → Llama (reverse direction)
    print("\n=== Mistral 7B → Llama 3.1 8B ===")
    mistral_to_llama = generate_cross_model(
        mistral_model, mistral_tokenizer,
        llama_model, llama_tokenizer,
        prompt, max_new_tokens
    )
    print(mistral_to_llama)


def generate_baseline(model, tokenizer, prompt, max_new_tokens):
    """Standard generation"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for reproducibility
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_cross_model(
    model_a, tokenizer_a,
    model_b, tokenizer_b,
    prompt, max_new_tokens
):
    """
    Generate using Model A's hidden states as input to Model B.
    Since Llama 3.1 8B and Mistral 7B both have hidden_size=4096, no alignment needed!
    """
    # Tokenize with Model A's tokenizer
    inputs_a = tokenizer_a(prompt, return_tensors="pt").to(model_a.device)

    # Get hidden states from Model A (no generation)
    with torch.no_grad():
        outputs_a = model_a.model(**inputs_a, output_hidden_states=True)
        hidden_states_a = outputs_a.last_hidden_state  # (1, seq_len, 4096)

    print(f"Model A hidden states shape: {hidden_states_a.shape}")

    # Verify dimension compatibility (should match!)
    model_a_dim = hidden_states_a.shape[-1]
    model_b_dim = model_b.config.hidden_size
    print(f"Model A dim: {model_a_dim}, Model B dim: {model_b_dim}")

    if model_a_dim != model_b_dim:
        raise ValueError(f"Dimension mismatch! {model_a_dim} != {model_b_dim}. Use matching models.")

    # Transfer hidden states directly (same device already)
    hidden_states_b = hidden_states_a
    
    # Generate with Model B starting from Model A's hidden states
    with torch.no_grad():
        # Start generation from hidden states
        generated_ids = []
        current_hidden = hidden_states_b
        
        for _ in range(max_new_tokens):
            # Forward through Model B's layers (skip embeddings)
            outputs_b = model_b.model(
                inputs_embeds=current_hidden,
                output_hidden_states=True
            )
            
            # Get logits
            logits = model_b.lm_head(outputs_b.last_hidden_state)
            
            # Get next token (greedy)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            generated_ids.append(next_token.item())
            
            # Check for EOS
            if next_token.item() == tokenizer_b.eos_token_id:
                break
            
            # Get embedding for next token
            next_embedding = model_b.model.embed_tokens(next_token.unsqueeze(0))
            
            # Concatenate for next iteration
            current_hidden = torch.cat([current_hidden, next_embedding], dim=1)
    
    # Decode
    generated_text = tokenizer_b.decode(generated_ids, skip_special_tokens=True)
    return prompt + generated_text


# Run experiments
if __name__ == "__main__":
    cross_model_experiment()
