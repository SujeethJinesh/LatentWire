
# Patch Notes (Teach Base to Listen)
- Apply PEFT LoRA to early attention blocks.
- Add LoraGate to disable LoRA effects on text-only batches (optional).
- Keep strong first-token/K-token CE + KD from text teacher (adapters disabled).
