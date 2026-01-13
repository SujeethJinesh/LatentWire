#!/usr/bin/env python
# telepathy/inspect_latents.py
"""
Latent Inspector: What do the 8 soft tokens "mean"?

This script finds the nearest vocabulary tokens (in Mistral's embedding space)
to each soft token produced by the bridge. It reveals what "concepts" the
bridge is transmitting.

Example:
    python telepathy/inspect_latents.py \
        --checkpoint runs/agnews_*/bridge_agnews.pt \
        --text "NASA launched a new satellite to study Mars."

Expected output for Science article:
    Token 1: space, NASA, satellite (Topic cluster)
    Token 2: research, study, science (Domain cluster)
    ...
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from latent_bridge import LatentBridge
import argparse


class Args:
    """Args object for LatentBridge interface."""
    def __init__(self, soft_tokens=8, heads=8, depth=2, use_fsq=False, stats_path=None):
        self.soft_tokens = soft_tokens
        self.heads = heads
        self.depth = depth
        self.use_fsq = use_fsq
        self.stats_path = stats_path


def get_nearest_neighbors(latent_vector, embedding_matrix, tokenizer, k=5):
    """
    Finds the k tokens in the vocabulary whose embeddings are
    closest (cosine similarity) to the given latent vector.
    """
    # Ensure float32 for numerical stability
    latent_vector = latent_vector.float()
    embedding_matrix = embedding_matrix.float()

    # Normalize latent vector
    latent_norm = F.normalize(latent_vector.unsqueeze(0), p=2, dim=-1)

    # Normalize embedding matrix (for cosine similarity)
    emb_norm = F.normalize(embedding_matrix, p=2, dim=-1)

    # Cosine similarity = dot product of normalized vectors
    similarity = torch.matmul(latent_norm, emb_norm.t())

    # Get top k
    scores, indices = torch.topk(similarity, k)

    neighbors = []
    for score, idx in zip(scores[0], indices[0]):
        token_str = tokenizer.decode([idx.item()]).replace('\n', '\\n').replace('\t', '\\t')
        # Clean up for display
        if token_str.strip() == '':
            token_str = repr(tokenizer.decode([idx.item()]))
        neighbors.append((token_str, score.item()))

    return neighbors


def analyze_latent_geometry(latents):
    """Analyze the geometric properties of the latent tokens."""
    latents = latents.float()

    # Pairwise similarities between tokens
    latents_norm = F.normalize(latents, dim=-1)
    sim_matrix = torch.matmul(latents_norm, latents_norm.t())

    # RMS of each token
    rms = torch.sqrt((latents ** 2).mean(dim=-1))

    # Variance across tokens
    token_var = latents.var(dim=0).mean()

    return {
        "similarity_matrix": sim_matrix,
        "rms_per_token": rms,
        "cross_token_variance": token_var.item()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="The stock market crashed today causing widespread panic among investors.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to bridge checkpoint")
    parser.add_argument("--soft_tokens", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=5, help="Number of nearest neighbors per token")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("LATENT INSPECTOR: What do the soft tokens mean?")
    print("=" * 70)
    print(f"\nInput: \"{args.text}\"")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Soft tokens: {args.soft_tokens}")

    # 1. Load Models
    print("\nLoading models...")
    src_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    if src_tok.pad_token is None:
        src_tok.pad_token = src_tok.eos_token
    src_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16
    ).to(device)
    src_model.eval()

    tgt_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    if tgt_tok.pad_token is None:
        tgt_tok.pad_token = tgt_tok.eos_token
    tgt_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        torch_dtype=torch.bfloat16
    ).to(device)
    tgt_model.eval()

    # 2. Load Bridge with correct interface
    print("Loading bridge...")
    bridge_args = Args(soft_tokens=args.soft_tokens, heads=8, depth=2, use_fsq=False)
    bridge = LatentBridge(
        bridge_args,
        src_dim=4096,
        tgt_dim=4096,
        target_rms=0.03
    ).to(device).to(torch.bfloat16)

    # Load checkpoint
    state_dict = torch.load(args.checkpoint, map_location=device)
    bridge.load_state_dict(state_dict)
    bridge.eval()
    print(f"  Loaded from {args.checkpoint}")

    # 3. Get Mistral's Embedding Matrix (our "dictionary")
    mistral_embeddings = tgt_model.get_input_embeddings().weight.detach()
    print(f"  Mistral vocabulary: {mistral_embeddings.shape[0]} tokens")

    # 4. Run Inference
    print("\nRunning inference...")
    inputs = src_tok(args.text, return_tensors="pt").to(device)

    with torch.no_grad():
        # Get Llama hidden states (Layer 31)
        src_out = src_model(**inputs, output_hidden_states=True)
        src_hidden = src_out.hidden_states[31]
        src_mask = inputs.attention_mask

        # Get bridge output
        latents, aux_loss, diversity, z_var = bridge(src_hidden, src_mask)

    latents = latents[0]  # Remove batch dim: [8, 4096]

    # 5. Analyze Geometry
    print("\n" + "=" * 70)
    print("LATENT GEOMETRY ANALYSIS")
    print("=" * 70)

    geometry = analyze_latent_geometry(latents)

    print(f"\nRMS per token (target ~0.03):")
    for i, rms in enumerate(geometry["rms_per_token"]):
        bar = "█" * int(rms * 100)
        print(f"  Token {i+1}: {rms:.4f} {bar}")

    print(f"\nPairwise similarity matrix (1.0 = identical, 0.0 = orthogonal):")
    sim = geometry["similarity_matrix"]
    print("        " + "  ".join([f"T{i+1:02d}" for i in range(args.soft_tokens)]))
    for i in range(args.soft_tokens):
        row = "  ".join([f"{sim[i,j]:.2f}" for j in range(args.soft_tokens)])
        print(f"  T{i+1:02d}:  {row}")

    # 6. Decode Each Token
    print("\n" + "=" * 70)
    print("NEAREST VOCABULARY TOKENS")
    print("=" * 70)
    print("(What words in Mistral's vocabulary are closest to each soft token?)\n")

    for i in range(args.soft_tokens):
        neighbors = get_nearest_neighbors(latents[i], mistral_embeddings, tgt_tok, k=args.top_k)
        neighbor_str = ", ".join([f"'{tok}' ({score:.3f})" for tok, score in neighbors])
        print(f"Token {i+1}: {neighbor_str}")

    # 7. Interpretation Guide
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    # Check if tokens are diverse or collapsed
    off_diag = sim[~torch.eye(args.soft_tokens, dtype=torch.bool, device=device)]
    mean_sim = off_diag.mean().item()

    if mean_sim > 0.9:
        print("⚠️  HIGH SIMILARITY: Tokens are nearly identical (mode collapse?)")
        print("   The bridge may be transmitting the same 'vibe' in all tokens.")
    elif mean_sim > 0.7:
        print("⚡ MODERATE SIMILARITY: Tokens share common structure")
        print("   Some redundancy, but tokens may encode different aspects.")
    else:
        print("✓  LOW SIMILARITY: Tokens are diverse")
        print("   Each token likely encodes different information.")

    print(f"\nMean off-diagonal similarity: {mean_sim:.3f}")
    print(f"Cross-token variance: {geometry['cross_token_variance']:.6f}")

    # Check if nearest neighbors make semantic sense
    print("\n" + "-" * 70)
    print("WHAT TO LOOK FOR:")
    print("-" * 70)
    print("""
For Classification (AG News, SST-2):
  - Tokens should cluster around category-related words
  - E.g., "business" input → tokens near "market", "trade", "economy"
  - E.g., "sports" input → tokens near "game", "team", "player"

For Passkey (exact retrieval):
  - Tokens should include digit-like tokens (0-9, numbers)
  - If no digits appear, the bridge is "vibe only" - can't transmit data

For GSM8K (math):
  - If tokens are generic ("the", "a", "is"), bridge lost the specifics
  - This explains why classification works but math fails
""")


if __name__ == "__main__":
    main()
