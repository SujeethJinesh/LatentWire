
# Patch Notes (Multi-Depth Adapters)
- Insert LatentAdapterBlock into selected transformer layers (e.g., model.layers[i].post_attention).
- Train only adapter and latent projection params.
- Track adapter strengths (alpha) and hidden-state KD at adapter layers.
