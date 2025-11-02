# Gist Tokens - Base Model Degenerate Output Results

**Date**: 2025-11-02
**Experiment**: Gist tokens reproduction with base (non-instruct) LLaMA model
**Status**: Pipeline successful, but outputs degenerate

## Summary

This preserved data shows the results of training gist tokens on `meta-llama/Meta-Llama-3.1-8B` (base model, not Instruct). While the pipeline completed successfully end-to-end (all 5 critical bugs fixed), the evaluation revealed that the base model generates degenerate repetitive outputs because it was never instruction-tuned.

## Key Findings

### Training
- **Model**: meta-llama/Meta-Llama-3.1-8B (base)
- **Configuration**: 1 gist token, 2,000 samples, 2 epochs
- **Loss progression**: 0.217 → 0.157
- **Time**: 0.98 minutes
- **Status**: ✅ Successful

### Evaluation Results

**ROUGE-L Scores**:
- Full text: 0.0517 (upper bound)
- Gist: 0.0270 (52% of full text)
- Truncated: 0.0290 (56% of full text)

**Compression**: 2.69× (33.3 tokens → 12.4 tokens)

### Critical Problem: Degenerate Outputs

All three baselines (full text, gist, truncated) generated repetitive garbage instead of following instructions:

**Example 1**:
- Instruction: "Suggest two songs that may be used for a jogging playlist."
- Full text output: `"best of the best of the best of the best..."`
- Gist output: `"www.google.com\nOutput:www.google.com\nOutput:www..."`

**Example 2**:
- Instruction: "Make a one sentence prediction of the stock market..."
- Gist output: `": The S&P500\nInput: The S&P500\nInput: The S&P500..."`
- Truncated output: `"2018/2018/2018/2018/2018/2018..."`

## Root Cause

The base model (`meta-llama/Meta-Llama-3.1-8B`) was never instruction-tuned and doesn't understand Alpaca-style prompts like:
```
Instruction: <task description>
Input: <optional input>
Output:
```

The model simply tries to continue the text pattern instead of following the instruction, resulting in repetitive/nonsensical outputs.

## Paper's Approach vs Our Approach

**Mu et al. (NeurIPS 2023)**:
- Started with base LLaMA-7B
- **Simultaneously** trained gist tokens AND performed full instruction tuning on Alpaca
- Both model parameters AND gist embeddings were trained together
- Result: Model learns instruction following + gist compression

**Our Initial Attempt**:
- Used base Llama 3.1 8B
- Only trained gist embeddings (model frozen)
- No instruction tuning
- Result: Model can't follow instructions, evaluation meaningless

## Decision: Switch to Instruct Model

After these results, we switched to `meta-llama/Meta-Llama-3.1-8B-Instruct` for validation:
- Instruct model already knows instruction following
- Can train only gist embeddings (fast, <1 min)
- Validates core compression mechanism with meaningful outputs
- See next experiment results for comparison

## Files in This Directory

- `train_20251102_110457.log` - Training log showing successful training
- `eval_20251102_110457.log` - Evaluation log
- `metrics.json` - Training metrics (loss, time, configuration)
- `eval_results.json` - ROUGE scores and compression metrics
- `sample_outputs.json` - Sample generations showing degenerate outputs
- `README.md` - This file

## Technical Notes

This experiment validated that:
1. ✅ All 5 critical bugs are fixed (pipeline runs end-to-end)
2. ✅ Training completes successfully with correct hyperparameters
3. ✅ Evaluation runs with all baselines (full, gist, truncated)
4. ❌ Base model without instruction tuning cannot be evaluated meaningfully on Alpaca

The infrastructure works correctly; the issue is solely that base models don't follow instructions.

## Next Steps

See subsequent experiments with Instruct model for meaningful gist token compression results.
