"""
Cross-Model Hidden State Transfer Experiment

Tests whether hidden states from one LLM can condition another LLM.
Relevant for LatentWire's cross-model interlingua goal.

Models: Llama 3.1 8B (hidden_size=4096) ↔ Mistral 7B (hidden_size=4096)

Ablations:
1. Llama alone (baseline)
2. Llama → Llama (sanity: same model, should work)
3. Llama → Mistral (cross-model)
4. Mistral → Mistral (sanity)
5. Mistral alone (baseline)
6. Mistral → Llama (reverse)
7. Llama → Mistral with Procrustes alignment (geometric)
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def cross_model_experiment():
    """
    Test passing hidden states between different models.
    Both Llama 3.1 8B and Mistral 7B have hidden_size=4096, so no alignment needed!
    """

    # Load models
    print("Loading models...")
    llama_model_id = "meta-llama/Llama-3.1-8B"
    mistral_model_id = "mistralai/Mistral-7B-v0.1"

    # Detect device and choose loading strategy
    if torch.cuda.is_available():
        # HPC/CUDA: Load directly to GPU with device_map (most efficient)
        print("Using device: cuda")
        device = "cuda"
        load_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto"  # Automatically loads to GPU
        }
    elif torch.backends.mps.is_available():
        # MacBook: Load to CPU then move to MPS
        print("Using device: mps")
        device = "mps"
        load_kwargs = {
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True
        }
    else:
        print("⚠️ Using device: cpu (will be very slow!)")
        device = "cpu"
        load_kwargs = {
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True
        }

    llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_id)
    print(f"Loading {llama_model_id}...")
    llama_model = AutoModelForCausalLM.from_pretrained(llama_model_id, **load_kwargs)
    if device != "cuda":  # device_map="auto" already places on GPU
        llama_model = llama_model.to(device)

    mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_id)
    print(f"Loading {mistral_model_id}...")
    mistral_model = AutoModelForCausalLM.from_pretrained(mistral_model_id, **load_kwargs)
    if device != "cuda":
        mistral_model = mistral_model.to(device)

    print("✓ Models loaded successfully!")
    print(f"Llama device: {next(llama_model.parameters()).device}")
    print(f"Mistral device: {next(mistral_model.parameters()).device}")
    
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
    """Standard generation with MPS compatibility"""
    # Set pad_token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # MPS has issues with model.generate(), so we use manual generation
    device = model.device
    if str(device).startswith("mps"):
        # Manual generation for MPS
        input_ids = inputs.input_ids
        generated_ids = input_ids[0].tolist()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                curr_input = torch.tensor([generated_ids]).to(device)
                outputs = model(curr_input)
                next_token_logits = outputs.logits[0, -1, :]
                next_token_id = torch.argmax(next_token_logits).item()

                if next_token_id == tokenizer.eos_token_id:
                    break

                generated_ids.append(next_token_id)

        return tokenizer.decode(generated_ids, skip_special_tokens=True)
    else:
        # Use built-in generate for CUDA/CPU
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy for reproducibility
                pad_token_id=tokenizer.pad_token_id
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
        hidden_states_a = outputs_a.hidden_states[-1]  # Last layer (1, seq_len, 4096)

    print(f"Model A hidden states shape: {hidden_states_a.shape}")

    # Verify dimension compatibility (should match!)
    model_a_dim = hidden_states_a.shape[-1]
    model_b_dim = model_b.config.hidden_size
    print(f"Model A dim: {model_a_dim}, Model B dim: {model_b_dim}")

    if model_a_dim != model_b_dim:
        raise ValueError(f"Dimension mismatch! {model_a_dim} != {model_b_dim}. Use matching models.")

    # Transfer hidden states directly (same device already)
    hidden_states_b = hidden_states_a
    
    # Generate with Model B from Model A's hidden states (with KV cache)
    generated_ids = []
    past_key_values = None
    current_hidden = hidden_states_b

    with torch.no_grad():
        for step in range(max_new_tokens):
            # Forward pass with KV cache
            if past_key_values is None:
                # First step: process all initial hidden states
                outputs_b = model_b.model(
                    inputs_embeds=current_hidden,
                    past_key_values=None,
                    use_cache=True
                )
            else:
                # Subsequent steps: only process new token embedding
                outputs_b = model_b.model(
                    inputs_embeds=current_hidden[:, -1:, :],
                    past_key_values=past_key_values,
                    use_cache=True
                )

            # Get logits and select next token (greedy)
            logits = model_b.lm_head(outputs_b.hidden_states[-1])
            next_token_id = torch.argmax(logits[0, -1, :]).item()

            # Store KV cache
            past_key_values = outputs_b.past_key_values

            # Check for EOS
            if next_token_id == tokenizer_b.eos_token_id:
                break

            generated_ids.append(next_token_id)

            # Get embedding for next token and concatenate
            next_embedding = model_b.model.embed_tokens(torch.tensor([[next_token_id]]).to(model_b.device))
            current_hidden = torch.cat([current_hidden, next_embedding], dim=1)

    # Decode
    generated_text = tokenizer_b.decode(generated_ids, skip_special_tokens=True)
    return prompt + generated_text


# Run experiments
if __name__ == "__main__":
    cross_model_experiment()
