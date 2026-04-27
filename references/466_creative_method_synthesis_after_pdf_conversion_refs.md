# Creative Method Synthesis After PDF Conversion

Date: 2026-04-27

## Status

The current paper is not ICLR-ready. The recent source-likelihood sketch and
post-sketch syndrome branches are killed, and the MPS execution lane is blocked
by orphaned PID `31103`. This memo converts the broader reference set into
bounded next methods rather than another open-ended search.

The reference PDFs have been extracted to `references/pdf_markdown/` with a
manifest so future literature subagents can inspect local papers directly.

## Highest-Value Branch: Candidate-Syndrome Decoder

Problem it helps with:

- The existing source signal is weak when measured as a standalone answer
  generator.
- Target-side candidate context has headroom, but old sketches let target
  priors and formatting artifacts masquerade as communication.

Mechanism/design idea:

- Treat the source sidecar as a tiny syndrome over a target candidate pool, not
  as a decoded answer or generic source likelihood.
- The target decoder enumerates its own candidate answers/traces and uses the
  source syndrome only to disambiguate among them.
- The first implementation should use candidate predicates available from
  existing artifacts, then graduate to learned source predicates once MPS is
  unblocked.

Primary sources and role:

- Slepian-Wolf code design via source-channel correspondence
  (`https://arxiv.org/abs/cs/0607021`): theory support for syndrome-style
  compression with decoder side information.
- Distributed arithmetic coding for asymmetric Slepian-Wolf
  (`https://arxiv.org/abs/0712.0271`): finite-block inspiration for short,
  soft candidate-disambiguation codes.
- Universal Wyner-Ziv coding
  (`https://arxiv.org/abs/1302.0050`): theory support for treating same-family
  and cross-family side information as different channels.
- Neural Distributed Source Coding
  (`https://arxiv.org/abs/2106.02797`,
  `https://github.com/UTAustin-ITML/neural-dsc`): learned baseline/inspiration
  for conditional codebooks.
- DeepJSCC-WZ
  (`https://arxiv.org/abs/2310.04311`,
  `https://github.com/ipc-lab/deepjscc-wz`): inspiration for injecting decoder
  side information at multiple stages rather than only at a prefix.
- Distributed indirect source coding with decoder side information
  (`https://arxiv.org/abs/2405.13483`): theory support for optimizing answer
  latent recovery rather than source-state reconstruction.
- Semantic rate-distortion-perception
  (`https://arxiv.org/abs/2312.05437`): ablation support for explicit
  zero-byte, target-only, slots-only, and shuffled-source controls.
- NeuroLogic A*esque decoding
  (`https://aclanthology.org/2022.naacl-main.57/`): decoding inspiration for
  treating source bits as constraints over target beams/candidates.

Does it change the next experiment?

Yes. The next CPU-feasible code change was a candidate-syndrome decoder
prototype over existing SVAMP70 candidate artifacts, with random-syndrome,
shuffled-source, zero-source, target-only, and slots-only controls from the
first gate.

CPU artifact result:

- Command: `./venv_arm64/bin/python scripts/analyze_candidate_syndrome_decoder.py --live-target-set results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/source_contrastive_target_set.json --holdout-target-set results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_contrastive_target_set.json --output-dir results/candidate_syndrome_decoder_20260427 --controls zero_source shuffled_source random_syndrome target_only slots_only --run-date 2026-04-27`
- Status: `candidate_syndrome_decoder_fails_smoke`.
- Live: matched clean source-necessary `1`, matched target-self harms `17`,
  control clean union `0`.
- Holdout: matched clean source-necessary `4`, matched target-self harms `14`,
  control clean union `0`.
- Decision: do not promote the numeric hash-syndrome artifact probe. The family
  needs learned source predicates or a stronger source surface before another
  run is worth doing.

Classification:

- Theory support, inspiration, and ablation design.

## Second Branch: Zero-Init Gated Query Bottleneck

Problem it helps with:

- Many historical bridges recovered isolated IDs by damaging target-correct
  examples, so target-self preservation is now a core blocker.

Mechanism/design idea:

- Compress source activations/traces into fixed-K query tokens.
- Inject them into the target with a zero- or near-zero-initialized gate so the
  target begins exactly as target-alone and learns measured source influence.
- Compare simple linear/MLP projection, Q-Former-style learned queries, and a
  Perceiver/Flamingo-style resampler before adding complexity.

Primary sources and role:

- BLIP-2 / Q-Former (`https://arxiv.org/abs/2301.12597`): inspiration and
  baseline for learned query extraction from a frozen source encoder.
- Flamingo (`https://arxiv.org/abs/2204.14198`): inspiration for Perceiver
  resampling plus gated cross-attention.
- InstructBLIP (`https://arxiv.org/abs/2305.06500`): inspiration for
  task-conditioned source extraction.
- LLaVA (`https://arxiv.org/abs/2304.08485`) and LLaVA-1.5
  (`https://arxiv.org/abs/2310.03744`): projector and MLP connector baselines.
- LLaMA-Adapter (`https://arxiv.org/abs/2303.16199`): zero-init/gated adapter
  inspiration for preserving the pretrained target.
- Perceiver and Perceiver IO (`https://arxiv.org/abs/2103.03206`,
  `https://arxiv.org/abs/2107.14795`): fixed latent bottleneck inspiration.
- MM1 (`https://arxiv.org/abs/2403.09611`): ablation warning that connector
  architecture alone is not the contribution.

Does it change the next experiment?

Yes, but only after MPS clears or a tiny CPU feature-only prototype is possible.
The next learned connector should start from zero-gated target preservation and
must include linear/MLP, fixed-query, and resampler ablations under the same
source-destroying controls.

Classification:

- Inspiration, baseline, and ablation design.

## Third Branch: Anchor-Relative Sparse Difference Atoms

Problem it helps with:

- RotAlign/DynAlign positives look like mechanism clues but are seed-unstable.
- Direct latent coordinate matching is brittle under gauge, head permutation,
  and local geometry mismatch.

Mechanism/design idea:

- Canonicalize by quotient/head matching before fitting residuals.
- Represent source and target examples relative to anchors and local
  neighborhoods instead of raw coordinates.
- Decompose the bridge into shared atoms plus source-difference atoms, then
  control by zeroing only the source-difference lane.

Primary sources and role:

- Relative Representations (`https://arxiv.org/abs/2209.15430`): anchor-relative
  invariance inspiration.
- Bricks to Bridges / Product of Invariances
  (`https://openreview.net/forum?id=vngVydDWft`): inspiration for combining
  multiple invariant views instead of one global rotation.
- Latent Space Translation via Semantic Alignment
  (`https://openreview.net/forum?id=pBa70rGHlr&noteId=9MWnfMIOv7`): Procrustes
  baseline/control.
- Gromov-Wasserstein embedding alignment
  (`https://aclanthology.org/D18-1214/`): relational alignment inspiration.
- Git Re-Basin (`https://openreview.net/forum?id=CQsmMYmlP5T`): symmetry and
  permutation-gauge support.
- Universal Sparse Autoencoders (`https://openreview.net/forum?id=UoaxRN88oR`):
  shared cross-model dictionary baseline/inspiration.
- Anthropic Sparse Crosscoders
  (`https://transformer-circuits.pub/2024/crosscoders/index.html`):
  interpretability support for shared vs differential atoms.
- Aristotelian representation critique (`https://arxiv.org/abs/2602.14486`):
  caution against overclaiming global similarity.

Does it change the next experiment?

Yes. Do not run another global RotAlign tweak. If this branch is selected,
the first gate should be local/anchor-relative and sparse-difference by design,
with seed agreement and source-difference zeroing as promotion checks.

Classification:

- Inspiration, interpretability support, and ablation design.

## Systems Branch: Protected-Tail Quantized Residual

Problem it helps with:

- A positive method must eventually show systems value, not only accuracy.
- The current project lacks a credible matched-byte compression story for
  latent transport.

Mechanism/design idea:

- Preserve a small anchor/top-saliency core exactly.
- Encode only the residual tail with rotation/QJL/product-style codes.
- Use query-aware KV/key selection as a matched-byte baseline rather than a
  separate headline.

Primary sources and role:

- TurboQuant (`https://arxiv.org/abs/2504.19874`): inspiration for
  rotation/polar residual quantization plus 1-bit residual correction.
- SpinQuant (`https://arxiv.org/abs/2405.16406`): inspiration for learned
  rotations that reduce outlier-driven quantization loss.
- Q-Filters (`https://arxiv.org/abs/2503.02812`): systems baseline for
  QK-geometry-based selection without materializing full attention maps.
- Expected Attention (`https://arxiv.org/abs/2510.00636`): future-query prior
  baseline for cache/transport selection.
- KQ-SVD (`https://arxiv.org/abs/2512.05916`): attention-inner-product
  preservation baseline for low-rank transport.

Does it change the next experiment?

Not as the first branch. It becomes a matched-byte systems ablation after the
candidate-syndrome or gated-query branch recovers real source-necessary wins.

Classification:

- Systems baseline and compression inspiration.

## Explicit Non-Goals After This Sweep

- Do not run another shallow source-text likelihood sketch and call it
  communication.
- Do not run a generic Perceiver memory without target-only, slots-only, and
  source-destroyed controls.
- Do not spend another cycle on global RotAlign unless quotient/local controls
  are built into the gate.
- Do not scale weak SVAMP surfaces that lack source-only headroom.

## Next Exact Gate

The CPU-first hash-syndrome decoder has already failed. Do not rerun it without
new source predicates or a stronger source surface. The next exact gate is to
clear the MPS blocker, then run the stronger-source scout:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If PID `31103` is absent:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file data/svamp_eval_70.jsonl \
  --results-dir results/qwen25math7b_qwen3_svamp70_surface_scout_20260427 \
  --translator checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt \
  --source-model Qwen/Qwen2.5-Math-7B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --methods target source t2t \
  --limit 70 \
  --device mps \
  --max-new-tokens 64 \
  --source-reasoning-mode brief_analysis \
  --use-chat-template \
  --no-enable-thinking \
  --continue-on-error
```

If that stronger surface has sufficient source-only headroom, use it to test
learned source predicates for candidate-syndrome decoding or the zero-init gated
query bottleneck. If not, move directly to zero-init gated query bottlenecks on
the strongest frozen surface available.
