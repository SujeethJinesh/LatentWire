# Next Steps — Finalized Execution Plan

## 0. Prep (Today)
- Sync to latest `main`, archive prior run logs, and reserve a 4× H100 slot.
- Double-check module stack (`gcc/13.1.0`, `conda/24.3.0-0`, `stockcuda/12.6.2`, `cudnn/cuda12/9.3.0.75`) before launching any job.

## 1. Phase 1 Run — All Fixes at Once (ETA: 2 hrs GPU)
Implement in `paper_writing/cross_attention.py`:
1. **KL Slice Alignment** — compare answer tokens to answer tokens by re-tokenizing baselines with answer text and shifting the bridged logits past `soft_tokens + prompt_length`.
2. **Prompt Alignment Weight** — reduce coefficient from `0.05` to `0.001`.
3. **Decode-Aware Supervision** — once per `eval_every`, run the translator+Llama stack in free-generation mode on a small mini-batch, compute GSM8K loss on decoded tokens, and backpropagate into the translator.
4. **RoPE/Tokenizer Projection** — insert a learnable projection layer right after we capture Mistral hidden states to map the 32 768-token/roto-phase space into Llama’s 128 256-token geometry before diffusion.

After code changes:
- Run `git pull`, stage edits, and launch `PYTHONPATH=. bash paper_writing/run_ablations.sh` but **only** the Phase 1 config (1b_dit_4step_64tok_soft_plus_text) with the updated flags (`PROMPT_MODES=("soft_plus_text")`, `PER_DEVICE_BATCH=2`, `eval_batch=36`). Log run ID and GPU node.
- Success criterion: bridged accuracy ≥ 70 %.

## 1.5 Conditional Ablations (only if Phase 1 < 70 %)
- **If 65–69 %:** run two jobs back-to-back (~4 hrs total):
  - **B:** KL fix only (no prompt/decoder/RoPE changes).
  - **C:** KL + prompt-weight + decode-aware (no RoPE projection).
- **If <65 %:** add **A** (KL fix only) so we complete A/B/C alongside the Phase 1 “full” run for attribution.
- Keep configs identical except for toggled features; reuse the same node if available to avoid cache thrash.

## 2. Phase 2 Experiments (post-success)
1. **Bidirectional Swap:** run Llama 3.1 as source and Mistral as target using the stabilized translator; reuse Phase 1 hyperparameters. Goal: test directionality sensitivity without adding new knobs.
2. **Hybrid Conditioning Baseline:** keep literal prompts in place and inject DiT outputs via adapters or residual addition, verifying whether textual anchoring alone closes the remaining gap.

## 3. Phase 3 (only if GPU queue opens up)
- Compression sweep (64/128/256/512 soft tokens) using the stabilized codepath.
- Attention-pooling diagnostics and DiT + cross-attention hybrids.
- Broader evals (HotpotQA, MATH) and larger targets (Llama 70B) once GSM8K is solid.

## 4. Reporting
- After every run, update `paper_writing/CODEX_REPORT.md`, `LOG.md`, and `EXPERIMENTS_SUMMARY.md` with metrics plus JSONL paths.
- If conditional ablations fire, add a short appendix summarizing the comparative gains so Claude’s publication concerns are satisfied.
