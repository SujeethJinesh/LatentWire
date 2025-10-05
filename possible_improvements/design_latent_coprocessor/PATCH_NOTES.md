
# Patch Notes (Latent Coprocessor)
- Add `LatentCoprocessor` module.
- In train/eval loops, after building `past_key_values`, compute K,V deltas and call `augment_past_kv(...)`.
- For KD teacher, wrap model forward with `disable_adapters()` if PEFT is present to get base distributions.
- New logs: first-token acc, KD, and cache-delta norms per layer.
