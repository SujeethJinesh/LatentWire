from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

# Downloading a model from huggingface
model_id = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load model directly to MPS (Mac GPU) - no offloading needed with 64GB RAM
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # MPS supports float16 but not bfloat16
    low_cpu_mem_usage=True
).to(device)

# Sample text with better context
text = "Once upon a time, there was a curious cat who loved to explore. One day,"

# Tokenize the text
token_ids = tokenizer.encode(text, add_special_tokens=True)
print(f"Input text: {text}")
print("Token IDs:", token_ids)
print("Decoded tokens:", [tokenizer.decode([id]) for id in token_ids])

# Now we need to pass these token ids into the embedding layer of the model
# That is what we can feed into the model
input_tensor = torch.tensor([token_ids]).to(model.device)
embeddings = model.model.embed_tokens(input_tensor)

print(f"\nInput shape: {input_tensor.shape}")  # (1, 7)
print(f"Embedding shape: {embeddings.shape}")  # (1, 7, 4096)
print(f"Each token is now {embeddings.shape[-1]} dimensions")
print(f"Embeddings tensor:\n{embeddings}")

embedding_weights = model.model.embed_tokens.weight
print(f"\nEmbedding weights shape: {embedding_weights.shape}")  # (128256, 4096)
print(f"Total number of parameters in embedding layer: {embedding_weights.numel()}")
print(f"Sample embedding vector for first token:\n{embedding_weights[0]}")

# Forward pass through the model
print("\n" + "="*50)
print("Running full forward pass...")
print("="*50)
with torch.no_grad():
    outputs = model(input_tensor)
    logits = outputs.logits
    print(f"Output logits shape: {logits.shape}")  # (1, 7, 128256)
    print(f"Last token logits shape: {logits[0, -1].shape}")  # (128256,)

    # Get the predicted next token
    next_token_id = torch.argmax(logits[0, -1]).item()
    next_token = tokenizer.decode([next_token_id])
    print(f"\nPredicted next token ID: {next_token_id}")
    print(f"Predicted next token: '{next_token}'")

# Autoregressive generation with sampling (more interesting than greedy!)
print("\n" + "="*50)
print("Autoregressive text generation with sampling...")
print("="*50)

max_new_tokens = 50
temperature = 0.8  # Lower = more conservative, Higher = more creative
top_p = 0.9  # Nucleus sampling - only sample from top 90% probability mass

generated_ids = token_ids.copy()
past_key_values = None  # KV cache - starts empty

start_time = time.time()
step_times = []

with torch.no_grad():
    for i in range(max_new_tokens):
        step_start = time.time()
        # With KV caching, we only pass the last token (not the whole sequence!)
        if past_key_values is None:
            # First iteration: pass the entire prompt
            input_tensor = torch.tensor([generated_ids]).to(device)
        else:
            # Subsequent iterations: only pass the new token
            input_tensor = torch.tensor([[generated_ids[-1]]]).to(device)

        # Get model predictions with KV cache
        outputs = model(input_tensor, past_key_values=past_key_values, use_cache=True)
        logits = outputs.logits[0, -1]  # Get logits for last token
        past_key_values = outputs.past_key_values  # Store KV cache for next iteration

        # Apply temperature scaling
        logits = logits / temperature

        # Convert to probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # Apply top-p (nucleus) sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[0] = False

        # Zero out probabilities for removed tokens
        sorted_probs[sorted_indices_to_remove] = 0.0

        # Renormalize
        sorted_probs = sorted_probs / sorted_probs.sum()

        # Sample from the filtered distribution
        next_token_idx = torch.multinomial(sorted_probs, num_samples=1)
        next_token_id = sorted_indices[next_token_idx].item()

        # Stop if we hit end-of-sequence token
        if next_token_id == tokenizer.eos_token_id:
            print(f"\n[Hit EOS token at step {i}]")
            break

        # Append to sequence
        generated_ids.append(next_token_id)

        # Print the token
        token_text = tokenizer.decode([next_token_id])
        step_time = time.time() - step_start
        step_times.append(step_time)
        print(f"Step {i+1}: Token {next_token_id} = '{token_text}' ({step_time*1000:.1f}ms)")

# Decode the full generated sequence
total_time = time.time() - start_time
generated_text = tokenizer.decode(generated_ids)

print("\n" + "="*50)
print("Full generated text:")
print("="*50)
print(generated_text)

print("\n" + "="*50)
print("Performance with KV caching:")
print("="*50)
print(f"Total time: {total_time:.2f}s")
print(f"Tokens generated: {len(step_times)}")
print(f"Average time per token: {sum(step_times)/len(step_times)*1000:.1f}ms")
print(f"First token (prefill): {step_times[0]*1000:.1f}ms")
if len(step_times) > 1:
    print(f"Avg decode token (with KV cache): {sum(step_times[1:])/len(step_times[1:])*1000:.1f}ms")
    print(f"Speedup: {step_times[0]/sum(step_times[1:])*len(step_times[1:]):.1f}x faster per token")
print(f"Tokens/second: {len(step_times)/total_time:.1f}")

# Note: KV caching stores key/value tensors from previous tokens so we don't
# recompute attention for the entire sequence each time. Without it, generation
# would get slower and slower as the sequence grows!