# RotAlign Re-Audit Notes

## What changed in the read

The second literature pass and harness audit changed one important conclusion:
the current `0.06` control result is encouraging, but it is **not** yet a clean
headline number because the control suite picked the best gate on the same
`GSM8K-100` slice it reported on.

So the story is now:
- the method is **alive**
- sparse, low-gate transfer still looks like the right regime
- but the next run must be a **fairness-corrected rerun**, not a broader sweep

## Highest-value additions

### Add immediately

1. **Held-out gate search**
   - Tune on `data/gsm8k_gate_search_30.jsonl`
   - Report on `data/gsm8k_eval_70.jsonl`
   - Why: avoids optimistic small-slice selection

2. **Fair text baseline sweep**
   - `text-to-text` with `plain`, `brief_analysis`, and `cot`
   - Why: current comparisons still mix method changes with prompt-template changes

3. **Matched protocol study**
   - fused KV
   - translated-only KV
   - text+KV hybrid
   - Why: separates “latent transfer helps” from “the prompt changed”

4. **One knowledge task**
   - small `MMLU-Redux` or `ARC-Challenge` slice
   - Why: tests the paper’s main claim that reasoning benefits more than knowledge

### Add next if the held-out rerun stays positive

5. **Per-head / grouped-head alignment**
   - Motivation: [KVTC](</Users/sujeethjinesh/Desktop/LatentWire/references/17_kv_cache_transform_coding_kvtc.pdf>)
   - Why: current failure mode may be head-structured, not layer-structured

6. **Steering-vector baseline**
   - Motivation: [Linear Representation Hypothesis](</Users/sujeethjinesh/Desktop/LatentWire/references/35_linear_representation_hypothesis.pdf>), [Function Vectors](</Users/sujeethjinesh/Desktop/LatentWire/references/36_function_vectors_in_large_language_models.pdf>), [RepE](</Users/sujeethjinesh/Desktop/LatentWire/references/37_representation_engineering.pdf>)
   - Why: tells us whether the useful signal is a full KV state or just a low-dimensional direction

7. **Soft token alignment for cross-tokenizer stress tests**
   - Motivation: [Relative Representations](</Users/sujeethjinesh/Desktop/LatentWire/references/13_relative_representations_enable_zero_shot_latent_space_communication.pdf>), [MoSECroT](</Users/sujeethjinesh/Desktop/LatentWire/references/33_mosecrot.pdf>), [Gromov-Wasserstein](</Users/sujeethjinesh/Desktop/LatentWire/references/34_gromov_wasserstein_alignment_of_word_embedding_spaces.pdf>)
   - Why: Qwen → Gemma should be treated as a transport problem, not just a KV problem

## What not to do yet

- Do not widen to more model pairs before the held-out control rerun
- Do not use the current `0.06` as a paper headline
- Do not assume quantization is the main story yet; sparsity may be doing more work than the codec

## Immediate next run

1. `Qwen/Qwen2.5-0.5B-Instruct -> Qwen/Qwen3-0.6B`
2. held-out gate search on `gsm8k_gate_search_30`
3. report on `gsm8k_eval_70`
4. compare:
   - baseline `plain`
   - baseline `brief_analysis`
   - baseline `cot`
   - fused KV
   - translated-only KV
   - text+KV hybrid
5. keep the strongest current branch:
   - `CKA`
   - `selection_ratio = 0.5`
   - low gates
   - quantized and no-quantized variants
