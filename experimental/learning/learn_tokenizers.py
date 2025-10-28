from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

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

# Sample text
text = "the cat sat on a mat"

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