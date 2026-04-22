# LatentWire Community Update (2026-04-22)

## Status

We are **not ICLR-ready yet**. We are still in method-discovery mode, not paper-writing mode.

The clearest current story is:

- On the **real same-pair benchmark lane**, the only live lift is still `dynalign_module_replace_residrank16 = 0.1250` on a frozen 32-example GSM8K contract.
- On the **toy/shared-basis lane**, quotient-aware symmetry fixing + GPA + a sparse shared dictionary + interface sidecars is consistently positive, especially in low-shot settings.
- Those two stories have **not yet fused into one benchmark-ready positive method**.

This memo is meant to make that boundary explicit, summarize what we have actually tried, and ask for ideas on what we have not yet tried well.

## Goal

The goal is a positive method for **cross-model communication / reasoning transfer**:

- move useful latent/KV information from one model to another
- beat or match direct text relay at better communication cost when possible
- stay robust across tokenizer, vocabulary, and representation mismatch
- eventually survive broader benchmarks beyond a frozen smoke contract

## Contract We Are Using Right Now

The real lane is currently gated by one strict contract:

- source/target: `Qwen2.5-0.5B-Instruct -> Qwen3-0.6B`
- task: GSM8K
- slice size: `32` frozen examples
- checks: exact ID parity, no empty predictions, numeric extraction coverage, paired comparison against `target_alone`

This is intentionally narrow. We do **not** want to widen the benchmark story until a method survives this contract first.

## Real Same-Pair Benchmark Story

### Baselines and early bridge rows

| Row | Accuracy | Read |
| --- | ---: | --- |
| `target_alone` | `0.0625` | target model on its own |
| `text_to_text` | `0.0312` | weak relay control |
| `c2c_generate` | `0.1250` | current same-pair communication smoke bar |
| `rotalign_kv` | `0.0625` | tied target, but only `28/32` numeric coverage |
| `dynalign_module_replace` | `0.0938` | first real same-pair lift above target |
| `spanalign_module_replace` | `0.0938` | ties dynalign |
| `tokenbasis_replace` | `0.0938` | ties dynalign |
| `bytespan_module_replace` | `0.0312` | negative |
| `sae_adapter` | `0.0000` | collapsed |

Takeaway:

- `rotalign_kv` did **not** become a durable benchmark story.
- `dynalign`/`spanalign`/`tokenbasis` all raised the ceiling to `0.0938`.
- Stronger teacher-family elaborations did **not** reliably improve beyond that.

### Residual repair: the only real live lift

| Row | Accuracy | Read |
| --- | ---: | --- |
| `dynalign_module_replace_residrank16` | `0.1250` | only live real same-pair lift |
| `tokenbasis_replace_residrank16` | `0.0625` | matched control fails |
| `dynalign_resid16_adaptive` | `0.1250` | adaptive canonicalization preserves, but does not improve |
| `dynalign_value_routed_module_replace_residrank16` | `0.1250` | first routed variant that preserves the lift |

Key facts:

- `dynalign_module_replace_residrank16` reaches the current `C2C` smoke row on this contract.
- It has full numeric extraction coverage `32/32`.
- It wins on `2/32` examples over `target_alone`, with `0/32` losses.
- The matched `tokenbasis + rank16 residual` control drops back to `0.0625`, so the effect is **basis-specific**, not a generic residual trick.

### What failed or saturated after that

| Row | Accuracy | Read |
| --- | ---: | --- |
| `dynalign_resid16_fitted_rotation` | `0.0000` | fixed gauge wrapper collapsed |
| `dynalign_resid16_shared_basis` | `0.0000` | fixed shared-basis wrapper collapsed |
| `dynalign_preserve_module_replace_residrank16` | `0.0625` | raw-basis preserve-core failed |
| `dynalign_eigenspace_module_replace_residrank16` | `0.0312` | naive dominant-eigenspace repair failed |
| `dynalign_saliency_module_replace_residrank16` | `0.0312` | simple saliency weighting failed |
| `dynalign_saliency_preserve_module_replace_residrank16` | `0.0625` | saliency-preserve plus tail failed |
| `dynalign_routed_module_replace_residrank16` | `0.0625` | one-gate routed repair failed |
| `dynalign_value_bank_module_replace_residrank16` | `0.0938` | richer bank route fell back to old ceiling |
| `dynalign_value_routed_bank_module_replace_residrank16` | `0.0938` | top-2 routed bank also fell back |
| `dynalign_value_verifier_sidecar_module_replace_residrank16` | `0.0938` | verifier-gated sidecar valid but non-additive |

Takeaway:

- Most geometry-only or one-gate repair ideas are now **negative controls**.
- The story is **not** “residual repair generically solves cross-model communication.”
- The story is “the right output-aware basis matters, and most nearby tweaks do not preserve the live lift.”

## Toy / Shared-Basis Story

This lane is stronger than the real benchmark lane in terms of architectural signal, but it is still toy-backed.

### What worked

| Toy row | Result | Read |
| --- | --- | --- |
| `quotient_match_after_fix` | best non-oracle at `1` shot/class | quotient/gauge fixing is real |
| `quotient_gpa_sparse_dictionary` | `0.0568` MSE at `1` shot, `0.0576` at `2` shots | best low-shot compositional lane |
| `quotient_gpa_sparse_dictionary_byte_sidecar_remap` | `0.0392` / `0.0394` MSE | byte sidecar improves interface mismatch |
| `quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_remap` | `0.0360` / `0.0362` MSE | best shared-basis interface variant |
| `preserve_topk_uniform_tail` | `0.9896` accuracy, `0.0284` MSE | dominant-anchor preservation is real |

Takeaway:

- The best toy story is:
  - quotient-aware symmetry fixing
  - plus GPA canonicalization
  - plus a sparse shared dictionary
  - plus a tokenizer-agnostic interface sidecar
  - with sequence alignment helping further

### What saturated on the toy side

- Direct family-specific fitting retakes the lead once paired data becomes abundant.
- Current repair gates are still inert in the sparse-dictionary held-out-family toy.
- Naive codebook tails underperform the simpler `preserve_topk_uniform_tail` codec baseline.
- Mixed-bit allocation looks promising as an efficiency component, but not yet as a standalone method story.

## What We Think Is Actually True Right Now

### Real lane

- Output-aware alignment matters more than plain latent geometry.
- Residual repair can help, but only on the right basis.
- Fixed gauge wrappers and simple weighting/routing tweaks are not the missing ingredient.

### Toy lane

- Symmetry, gauge, and interface mismatch have to be handled together.
- Shared-basis communication is much better in the low-shot regime once those are handled jointly.
- That still has not crossed into a real benchmark-positive method.

## What We Have Probably Saturated

We do **not** think the next win is likely to come from:

- more `rotalign`-style fixed geometry wrappers
- stronger teacher-family variants on the old `0.0938` dynalign ceiling
- one-gate dense routed repair
- naive eigenspace-only or saliency-only residual weighting
- single verifier-gated sidecars
- naive codebook tails without preserved anchors

## Strongest Ideas We Have Not Yet Tried Well

### 1. Anchor-preserving codebook tail

The strongest codec-side open question is whether we should:

- preserve dominant anchors or trusted subspaces
- quantize or codebook only the tail
- add a small learned repair on top of the tail rather than choosing pure codebook or pure dense residual

Relevant directions:

- [Preserve-Then-Quantize](https://arxiv.org/abs/2602.02001)
- [QERA](https://arxiv.org/abs/2410.06040)
- [ResQ](https://arxiv.org/abs/2412.14363)
- [SERQ](https://arxiv.org/abs/2603.08185)
- [AWQ](https://arxiv.org/abs/2306.00978)
- [AQLM](https://arxiv.org/abs/2401.06118)
- [TurboQuant](https://arxiv.org/abs/2504.19874)

### 2. Stronger multi-expert / value-side repair

The first value-routed branch is live, but richer bank variants are not.

Open question:

- is the missing ingredient better gating, better experts, better initialization, or a different repair location?

Relevant directions:

- [ResMoE](https://arxiv.org/abs/2503.06881)
- [S'MoRE](https://arxiv.org/abs/2504.06426)
- [Attractor Patch Networks](https://arxiv.org/abs/2602.06993)
- [ERMoE](https://arxiv.org/abs/2511.10971)
- [DirMoE](https://openreview.net/forum?id=a15cDnzr6r)

### 3. Better verifier-gated repair

The first verifier sidecar was valid but not additive.

Open question:

- what signal actually predicts that a repair will help here?

Candidate triggers:

- disagreement
- entropy gap
- calibration drop
- retrieval mismatch
- latent consistency checks

Relevant directions:

- [Goedel-Prover-V2](https://arxiv.org/abs/2508.03613)
- [PAG](https://arxiv.org/abs/2506.10406)
- [FLASH](https://arxiv.org/abs/2505.12728)
- [Reflective Verification](https://arxiv.org/abs/2505.18629)

### 4. Interface redesign instead of more basis surgery

The low-shot toy signal keeps pointing back to interface mismatch.

Open question:

- should the next serious branch abandon the current transport basis and move toward:
  - byte-level universal interfaces
  - tokenizer/vocab surgery
  - query/resampler-style connectors
  - QK-fidelity or transport-based matching

Related inspirations:

- [The Vision Wormhole](https://arxiv.org/abs/2602.15382)
- [MM1](https://arxiv.org/abs/2403.09611)
- [Libra](https://arxiv.org/abs/2405.10140)
- [Latent Space Communication via K-V Cache Alignment](https://arxiv.org/abs/2601.06123)
- [KVComm](https://arxiv.org/abs/2510.03346)

## Benchmark Discipline

Our current view is that benchmark expansion should stay block-structured:

1. frozen same-pair GSM8K32 gate
2. `RULER`
3. one matched cross-family pair
4. `SCBench`
5. `LongBench v2`

We do **not** want to blur:

- same-pair communication
- cross-family transfer
- long-context communication
- multimodal grafting

into one mixed leaderboard too early.

Relevant benchmarks and comparators:

- [C2C](https://arxiv.org/abs/2510.03215)
- [KVComm](https://arxiv.org/abs/2510.03346)
- [RULER](https://arxiv.org/abs/2404.06654)
- [SCBench](https://arxiv.org/abs/2412.10319)
- [LongBench v2](https://arxiv.org/abs/2412.15204)

## Questions We Would Love Community Feedback On

### 1. What is the highest-leverage branch we have not yet tried well?

If you had to bet on one of:

- anchor-preserving codebook tails
- stronger multi-expert value-side repair
- verifier-gated repair
- interface redesign / byte-level transport

which would you bet on, and why?

### 2. What single ablation would falsify our current story fastest?

We want the fastest clean test of whether the live `dynalign + resid16` row is:

- a real structural effect
- a narrow calibration artifact
- or a brittle same-pair quirk

### 3. What telemetry would convince you a gain is structural?

Examples:

- consistent example-level win patterns
- stable coverage under frozen contracts
- byte/latency win at matched accuracy
- route/expert utilization that matches the intervention
- tail-only repair proving it is actually doing work

### 4. What should come immediately after GSM8K32 if a branch survives?

Our current default is:

- `RULER`
- then `SCBench`
- then `LongBench v2`

If you think that order is wrong, we want to know why.

## Bottom Line

If we had to compress the current state into one line:

> We have a real but narrow same-pair lift from output-aware dynalign residual repair, and a stronger toy story showing that symmetry, gauge, and interface mismatch matter together, but we still do not have a benchmark-ready positive method that unifies those two stories.

If you have ideas, the branches we most want help ranking are:

- anchor-preserving codebook tails
- stronger multi-expert value-side repair
- verifier-gated or disagreement-triggered repair
- interface redesign that reduces tokenizer/basis mismatch directly
