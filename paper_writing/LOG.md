# GSM8K Work Log (Codex, Nov 2025)

- **2025-11-13 – Eval + Train Fixes**
  - Fixed GSM8K evaluator to extract gold answers from `sample.tgt_answer` only, added question/output dumps (JSONL), and logged both source-only (Mistral) and target-only (Llama) baselines alongside bridged accuracy.
  - Created `run_1c_validation.sh` to re-run experiment 1c after evaluation fixes.
  - Added decoder prompt modes:
    - `soft_only`: Llama sees only soft tokens + a short “end with ####” instruction.
    - `soft_plus_text`: Llama sees literal prompt/question plus soft tokens (default for meaningful runs).
  - Removed soft-token compression (default `--soft_tokens=-1` → capped at 2048 tokens) and added semantics-aware losses:
    - prompt-alignment loss (soft tokens vs. Llama’s prompt embeddings, padded to same length)
    - format penalty on teacher-forced logits (missing `####`)
    - KL regularizer re-enabled with aligned positions + InfoNCE.
  - Iterative debugging steps:
    - Hybrid textual anchor initially caused Llama to parrot the few-shot prompt; reverted to literal prompt for `soft_plus_text`.
    - Soft-only path revealed the bridge still collapses (0% accuracy), but we keep it as a diagnostic bound.
    - Fixed truncation to remove follow-up questions while preserving the ground-truth answer.
    - Resolved repeated CUDA OOMs by capping soft tokens, dropping `cache_implementation="static"`, and halving `per_device_batch` from 4 → 2 in both ablation and validation scripts.
    - Added padding to prompt-alignment tensors so shapes always match the soft-token length.
  - Current status: `soft_plus_text` configs ready for rerun with the new losses; `soft_only` runs continue to serve as a lower-bound (expected near 0%). Source/target baselines report realistic numbers again, so bridged metrics are trustworthy.

