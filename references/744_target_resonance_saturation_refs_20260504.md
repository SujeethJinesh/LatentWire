# Target-Resonance Saturation And Novelty Boundaries

Date: 2026-05-04

## Why This Memo Exists

The latest HellaSwag target-self-resonance runs separate a useful capacity fact
from a publishable communication method:

- Per-row optimized soft prefixes can make the target reproduce its own full-prompt
  behavior on tiny slices.
- Held-out learned encoders, query resamplers, source-hidden residual decoders,
  source-codebook repair, and one-step consistency refinement do not beat
  slots-only, zero-source, wrong-source, candidate-roll, or source-choice controls.

This means the target-resonance framing is still conceptually useful, but the
current reusable receiver family is saturated. The next method must create a new
information path or diagnostic, not another shallow source-to-prefix variant.

## Primary Related Work To Cite

- Cache-to-Cache (C2C): Fu et al. propose projecting and fusing source/target
  KV caches directly for model-to-model semantic transfer.
  Primary source: https://arxiv.org/abs/2510.03215
  LatentWire distinction: our target story should emphasize low-rate packets,
  source privacy, auditability, destructive controls, and utility per byte rather
  than dense high-bandwidth KV fusion.

- Q-KVComm: Kriuk and Ng transmit compressed KV-cache representations with
  adaptive layer-wise quantization and heterogeneous calibration.
  Primary source: https://arxiv.org/abs/2512.17914
  LatentWire distinction: this is a compressed dense-representation protocol;
  our novelty requires sparse/auditable packet fields and strict source-private
  controls, not only lower-bit cache transfer.

- TurboQuant: Zandieh et al. give an online vector quantization method with
  near-optimal distortion rates and KV-cache quantization results.
  Primary source: https://arxiv.org/abs/2504.19874
  LatentWire distinction: TurboQuant is a quantization baseline and systems
  inspiration for low-bit coefficients; it does not by itself solve source-causal
  model-to-model communication under destructive controls.

- Prefix-Tuning: Li and Liang optimize continuous prefixes while keeping the LM
  frozen.
  Primary source: https://aclanthology.org/2021.acl-long.353/
  Novelty risk: any target-native soft-prefix receiver must be framed as a
  communication protocol with source-private packet controls, not just prompt
  tuning.

- Prompt Tuning: Lester et al. show that learned soft prompts can steer frozen
  language models.
  Primary source: https://arxiv.org/abs/2104.08691
  Novelty risk: optimized/eval-row prefixes are only capacity probes.

- Gist Tokens: Mu, Li, and Goodman train models to compress prompts into special
  gist tokens.
  Primary source: https://arxiv.org/abs/2304.08467
  LatentWire distinction: gist tokens are prompt compression; our claim must be
  source-private cross-model communication with byte/accounting and destructive
  controls.

- AutoCompressors: Chevalier et al. adapt language models to compress long
  contexts into summary vectors.
  Primary source: https://arxiv.org/abs/2305.14788
  Novelty risk: soft or latent context compression is prior art unless the
  packet protocol, source privacy, and cross-model controls are central.

- Activation Addition / Contrastive Activation Addition: Rimsky et al. steer LM
  behavior through activation vectors.
  Primary source: https://arxiv.org/abs/2312.06681
  Novelty risk: target-side residual steering must be evaluated against
  target-derived and shuffled/wrong-source controls.

- Representation Engineering: Zou et al. describe control through representation
  directions.
  Primary source: https://arxiv.org/abs/2310.01405
  LatentWire distinction: we need packetized source-to-target transfer, not only
  target-local control vectors.

- Coconut / Continuous Thought: Hao et al. reason in latent space.
  Primary source: https://arxiv.org/abs/2412.06769
  Novelty risk: do not overclaim latent reasoning unless the evaluation directly
  shows communication/reasoning gains beyond text and source-choice baselines.

## Current Decision Boundary

Promote:

- Target self-resonance oracle soft prefixes only as a capacity/headroom probe.
- Conditional PQ shared-schema packet and HellaSwag fixed-byte packet rows for
  narrow COLM_v2 evidence.

Rule out for the current implementation family:

- Chunk encoder, oracle-distill encoder, and query-resampler target-native
  prefixes as held-out reusable receivers.
- Source-hidden residual, source-oracle distill, source-codebook candidate
  repair, and one-step consistency refinement as source-conditioned target-native
  receivers.
- Deterministic public-zscore and public-SVD conditioning as held-out-family
  conditional-PQ rescues.

Next gate:

- Backport the refreshed triage into COLM_v2 tables/figures.
- Before another decoder, run a complementarity-frontier diagnostic: isolate rows
  where the target is wrong and source top1/top2 could help, then test whether a
  source-private packet field offers separable source-causal signal beyond
  source-choice/wrong-row/candidate-roll controls.
