import torch
import math

# Setup
batch = 1
seq_len = 7
d_model = 4096
n_heads = 32
d_k = d_model // n_heads  # 128 per head

# Projections
Q = torch.randn(batch, n_heads, seq_len, d_k)  # (1, 32, 7, 128)
K = torch.randn(batch, n_heads, seq_len, d_k)  # (1, 32, 7, 128)
V = torch.randn(batch, n_heads, seq_len, d_k)  # (1, 32, 7, 128)

# Step 1: Q @ K^T
K_transposed = K.transpose(-2, -1)  # (1, 32, 128, 7)
scores = torch.matmul(Q, K_transposed)  # (1, 32, 7, 7)
print(f"Q @ K^T shape: {scores.shape}")
# (batch, n_heads, seq_len_q, seq_len_k)
# (1, 32, 7, 7) - attention scores matrix

# Step 2: Scale
scores = scores / math.sqrt(d_k)

# Step 3: Softmax
attn_weights = torch.softmax(scores, dim=-1)  # (1, 32, 7, 7)
print(f"Attention weights shape: {attn_weights.shape}")

# Step 4: (Q @ K^T) @ V  ← This is the key!
output = torch.matmul(attn_weights, V)
# (1, 32, 7, 7) @ (1, 32, 7, 128) → (1, 32, 7, 128)
print(f"Output shape: {output.shape}")

# The math:
# attn_weights[i,j] = softmax(Q[i] · K[j])  # For each token i and j
# output[i] = Σ_j attn_weights[i,j] * V[j]  # Weighted sum of all values