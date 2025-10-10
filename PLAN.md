# LatentWire Phase-2 Accuracy Plan

## Overview
This document outlines the implementation plan for improving LatentWire's accuracy from its current baseline to achieve meaningful compression while maintaining generation quality.

## Phase A: Foundation Fixes
### Milestone 1: K-Token Supervision
- **Item**: Implement K-token teacher-forced cross-entropy loss (`k_token_ce_from_prefix`)
  - Supervise first K tokens instead of just first token
  - Target: Improve FirstTok@1 from 5-7% to 12-20%

### Milestone 2: Knowledge Distillation
- **Item**: Add prefix-based KD (`kd_first_k_prefix_vs_text`)
  - Distill text-prompted teacher distributions into latent student
  - Use temperature-scaled soft targets for better gradient flow

### Milestone 3: Tokenization Alignment
- **Item**: Ensure exact t=0 alignment after anchor text
  - Fix BOS policy mismatches between train and eval
  - Validate alignment with `_assert_t0_alignment()`

### Milestone 4: Calibration & Normalization
- **Item**: Per-example embedding RMS calibration
  - Scale latents to match embedding statistics per example
  - Prevent amplitude drift during training

## Phase B: Architecture Improvements
### Milestone 5: Embedding Baseline Testing
- **Item**: Validate inputs_embeds mechanism with real embeddings
  - Test raw passthrough, anchor patterns, and adapter modes
  - Isolate whether issue is with inputs_embeds or learned representations

### Milestone 6: Feature Integration
- **Item**: Systematically test and integrate auxiliary features
  - LoRA adapters for parameter efficiency
  - Prefix tuning for soft prompt optimization
  - Latent adapters for cross-attention mechanisms
  - Coprocessor for KV cache manipulation

## Success Criteria
- **FirstTok@1**: 12-20% at M∈{32,48,64}
- **F1 Score**: 0.10-0.20 with ≥4× compression
- **NLL**: Within 2-3 nats of text baseline
- **Compression**: ≥4× reduction in wire bytes

## Current Status
- Phase A Milestone 1-4: In progress
- Embedding baseline implementation: Complete, testing in progress
- Feature smoke tests: All passing with correct implementations

## Next Steps
1. Run embedding baseline tests to validate inputs_embeds
2. Analyze results to determine if issue is architectural or training-related
3. Implement K-token objectives if embedding tests show promise
4. Iterate on calibration and normalization strategies