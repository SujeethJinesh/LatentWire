# 2026-05-27T21:35Z Tier 1.5 architecture-grounded methods queue

## Trigger

Run this batch only if M-PRED and M-FISH both kill, defined as median recovery
less than or equal to M11b on all measured architectures. If either method
passes, surface that result before running Tier 1.5.

## Architecture constraints

- M-ROUTE targets Nemotron-3 only because it depends on MoE routing.
- M-GATE Option B targets SwiGLU models only: Granite attention MLPs and
  DeepSeek dense Qwen2.5 MLPs. It does not apply to Nemotron MoE squared-ReLU
  layers.
- M-TOKEN-SOFT is architecture-agnostic but lower priority.

## Queue

1. M-ROUTE on Nemotron, cap 12 GPU hours.
   - Run a fresh scoop check first.
   - Run diagnostic per-expert set-leaving first.
   - Differentiate from EAQuant and MxMoE as decode-time W4A16 top-K
     channel-set retrieval conditioned on active routing, not static smoothing
     or per-expert bit-width allocation.
2. M-GATE Option B on Granite and DeepSeek, cap 10 GPU hours.
   - Run a fresh scoop check first.
   - Differentiate from Smooth-SwiGLU and Depth Registers as inference-time
     W4A16 top-K channel protection, not training-time rescaling or register
     interventions.
3. M-TOKEN-SOFT, cap 12 GPU hours, only if budget remains and prior Tier 1.5
   results are ambiguous or killed.
   - Run a fresh scoop check first.
   - Soft token routing is specifically intended to avoid the hard-switching
     failure mode observed in M2/M10.

## Priority and budget

M-PRED DeepSeek/Falcon remains higher priority than this Tier 1.5 batch. M-FISH
also remains ahead of this batch. Skip M-TOKEN-SOFT if cumulative GPU time is
above 330 hours after M-ROUTE and M-GATE.

## Expectation

Estimated chance that one Tier 1.5 method passes on one architecture is around
50%, but this is conditional on both M-PRED and M-FISH failing first.
