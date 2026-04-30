# Balanced Binary-Verifier Receiver References

- date: `2026-04-30`
- purpose: grounding for the balanced Qwen3 binary-verifier receiver gate and
  the next systems/selector comparisons.

## GPT-3 / LM Likelihood Scoring

- source: https://arxiv.org/abs/2005.14165
- blocker helped: a frozen target model must consume the compact packet without
  depending on free-form generation or parser luck.
- mechanism idea: score constrained candidate choices by model likelihood
  rather than sampling a long answer.
- next experiment change: use constrained receiver modes and report validity,
  candidate choice priors, and source-destroying controls.
- role: selector baseline / method framing.

## Calibrate Before Use

- source: https://arxiv.org/abs/2102.09690
- blocker helped: multiple-choice and verbalizer prompts can be dominated by
  label/option priors.
- mechanism idea: calibrate the target model's decision surface and include
  content-free or no-source controls.
- next experiment change: binary yes/no margins now require positive evidence;
  all no-match packets fall back to target prior instead of choosing the least
  negative candidate.
- role: mandatory control / selector calibration.

## PET / Verbalizer Classification

- source: https://arxiv.org/abs/2001.07676
- blocker helped: arbitrary candidate labels are brittle as generative outputs.
- mechanism idea: map decisions to short verbalizers or option tokens, then
  compare under prompt/verbalizer controls.
- next experiment change: keep the choice-token receiver as a diagnostic and
  avoid promoting it when option priors dominate.
- role: interface design / ablation.

## Neural Text Degeneration

- source: https://arxiv.org/abs/1904.09751
- blocker helped: raw likelihood is a poor story for open-ended generated text.
- mechanism idea: likelihood is more defensible on short constrained outputs
  than on unconstrained free-form continuations.
- next experiment change: interpret the binary verifier as a constrained
  closed-form receiver gate, not as a general generation result.
- role: guardrail / framing.

## PICARD

- source: https://arxiv.org/abs/2109.05093
- blocker helped: invalid outputs can contaminate structured-decoding claims.
- mechanism idea: constrain decoding to valid formal outputs and count invalid
  rejections separately.
- next experiment change: future receiver rows should use grammar/choice
  constraints when measuring endpoint latency, not permissive parsing.
- role: constrained-decoding baseline.

## Outlines

- source: https://github.com/outlines-dev/outlines
- blocker helped: practical structured generation must be reproducible and
  parser-independent.
- mechanism idea: compile regex/grammar constraints into token-level guided
  generation.
- next experiment change: implement a candidate-ID grammar receiver if the
  binary-verifier path is widened to endpoint/server runs.
- role: practical baseline / implementation path.

## SGLang

- source: https://arxiv.org/abs/2312.07104
- blocker helped: multi-candidate selection can become a systems artifact if
  prompts are evaluated naively.
- mechanism idea: structured LM programs and shared-prefix/cache reuse can
  reduce repeated target-side work.
- next experiment change: a GPU/server follow-up should batch or share-prefix
  the four candidate verifier calls and report TTFT/TPOT/goodput.
- role: systems baseline / future implementation.

## FlashAttention

- source: https://arxiv.org/abs/2205.14135
- blocker helped: raw packet bytes alone are not hardware traffic.
- mechanism idea: IO-aware accounting matters; memory movement and transfer
  quanta can dominate realized latency.
- next experiment change: keep reporting raw bytes, 64B line bytes, 128B burst
  bytes, batch packing, and source-state exposure.
- role: systems framing / overclaim guard.

## vLLM / PagedAttention

- source: https://arxiv.org/abs/2309.06180
- blocker helped: KV-cache serving systems are the natural comparator for any
  model-to-model communication systems claim.
- mechanism idea: account for KV storage, reuse, and page movement separately
  from tiny packet payloads.
- next experiment change: do not claim production serving wins until the packet
  receiver is measured in a native serving stack.
- role: serving baseline.

## DistServe

- source: https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin
- blocker helped: systems reviewers expect TTFT, TPOT, goodput, and SLO
  language.
- mechanism idea: split prefill and decode effects under serving SLOs.
- next experiment change: Mac CPU latency is supporting telemetry only; future
  NVIDIA runs must report TTFT/TPOT/goodput and HBM/KV counters.
- role: systems metric standard.

## KIVI

- source: https://arxiv.org/abs/2402.02750
- blocker helped: naive full-precision KV byte floors are weak baselines.
- mechanism idea: asymmetric 2-bit KV-cache quantization lowers cache movement
  for same-model/visible-cache settings.
- next experiment change: compare LatentWire packet traffic against compressed
  KV byte floors under explicit access assumptions.
- role: systems/compression baseline.

## QJL

- source: https://arxiv.org/abs/2406.03482
- blocker helped: compact vector sketches and one-bit JL transforms are close
  mathematical neighbors to low-rate latent communication.
- mechanism idea: random projection plus sign quantization can be a strong
  compression comparator.
- next experiment change: keep QJL/sign-sketch rows as ablations; do not claim
  novelty from compression alone.
- role: mathematical baseline / ablation.

## TurboQuant

- source: https://arxiv.org/abs/2504.19874
- blocker helped: modern KV/vector quantization narrows simple byte-count
  arguments.
- mechanism idea: rotation plus quantization can compress source state when KV
  exposure is allowed.
- next experiment change: use TurboQuant-style byte floors as baselines in the
  systems table; LatentWire's novelty is source-private task communication
  under destructive controls.
- role: systems/compression baseline.

## Diffusion Transformers

- source: https://arxiv.org/abs/2212.09748
- blocker helped: suggests iterative denoising receivers, but not a direct
  baseline for the current packet protocol.
- mechanism idea: predict clean latent/task state from noisy or partial latent
  tokens under a learned denoising objective.
- next experiment change: one future branch is a packet-consistency denoiser
  that treats corrupted packets as destructive controls.
- role: inspiration.

## V-JEPA

- source: https://arxiv.org/abs/2404.08471
- blocker helped: learned receiver branches need anti-collapse latent
  objectives and held-out semantic targets.
- mechanism idea: predict target-side latent representations from masked
  context rather than reconstructing pixels/tokens.
- next experiment change: future latent bridge work should train with
  source-control negatives and report effective rank/collapse diagnostics.
- role: inspiration / objective design.

## Bottom Line

The new receiver evidence is strongest as calibrated, constrained
source-private packet decoding. The closest prior work provides likelihood
classification, structured decoding, cache-serving systems, and compression
baselines; it does not remove the need for source-destroying controls. The
remaining paper risk is that the interface is still protocol-shaped, so broad
latent-transfer claims must stay out of the headline until a less explicit
receiver passes.
