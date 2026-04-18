# References

This folder contains the papers requested for `latent_bridge`, numbered to match the list you provided.

- Files `01_` through `41_` correspond to items 1 through 41 in your reading list.
- Most entries are saved as PDFs.
- `38_scaling_monosemanticity.html` is saved as HTML because the source link is a web article on Transformer Circuits rather than a direct PDF.
- `download_manifest.json` records the source URL, output path, and download status for each item.

The search-only items were resolved to concrete paper links before download, so the folder should be complete.

## Math Grounding Addendum

Files `42_` through `75_` were added after the initial live validation runs. They cover representation similarity, CCA/CKA diagnostics, structured random/Fourier/Hadamard transforms, product quantization, incoherence processing, randomized linear algebra, control-theoretic gating, source coding, predictive-coding style residual transmission, model stitching, and representation-alignment theory. These are intended to ground the next RotAlign-KV ablations:

- `42_` through `44_`: CKA/SVCCA/PWCCA layer-pairing and representation-similarity diagnostics.
- `45_` through `48_`: random orthogonal, Hadamard/Fourier, and butterfly transform families.
- `49_` through `51_`, `53_`, `54_`: rotation/scale/product/incoherence quantization methods.
- `52_`: randomized low-rank methods for reduced-rank and subspace alignment ablations.
- `55_` through `57_`: Kalman, adaptive-computation, and James-Stein style shrinkage/gating references.
- `58_` through `60_`: xKV, MoH, and causal head gating for head-aware transmission.
- `61_` through `66_`: saliency-guided cache compression, rotation repair, and low-rank KV projection papers.
- `67_` through `71_`: source coding, adaptive routing, event-triggered control, rate-distortion, and predictive-coding controls.
- `72_` through `75_`: model stitching, representation-alignment theory, Fourier-domain compression, and multiresolution attention references.

Files `76_` through `85_` extend the KV-specific compression and selection literature around the new `k_only` branch:

- `76_`: layer-wise asymmetric KV quantization.
- `77_` through `82_`: retrieval-head, head-level, key-token, heavy-hitter, and benchmark papers for selective KV retention.
- `83_`: task-aware adaptive KV budgeting.
- `84_` and `85_`: redundancy-aware reasoning compression and key/value-asymmetric quantization arguments.

Files `86_` through `89_` extend the current overnight branch around K/V asymmetry,
attention-fidelity preservation, and token-level KV selection:

- `86_`: stronger evidence that keys and values should be handled asymmetrically.
- `87_` and `88_`: attention-fidelity and lossless-periodic KV compression ideas.
- `89_`: dynamic token-level KV selection for selective key transport.

Files `90_` through `93_` extend the same branch toward query-aware sparsity,
attention-space preservation, and query-centric fusion:

- `90_`: query-aware sparsity for selective key-position retention.
- `91_`: attention-score-weighted KV merging as a reliability-weighted fusion baseline.
- `92_`: low-rank attention-space compression for attention-preservation objectives.
- `93_`: query-centric cache fusion as a direct source-conditioned fusion reference.

Files `94_` through `99_` deepen the current selective-routing branch around
query-aware eviction, task-aware head differentiation, and retrieval-head
interpretability:

- `94_`: self-attention-guided KV eviction as a query-aware token-retention baseline.
- `95_`: task-aware semantic differentiation of attention heads for task-conditioned selector ablations.
- `96_`: query-agnostic KV compression with context reconstruction as a stronger blind-selector baseline.
- `97_` and `98_`: retrieval-head mechanistic papers linking head subsets to long-context factuality and reasoning.
- `99_`: an L2-norm KV compression baseline that grounds simple norm-based sparse selectors.

Files `100_` through `103_` extend the same branch toward future-query priors,
QK-geometry preservation, variable per-head budgets, and offline head
reordering:

- `100_`: future-query expected-attention priors for stronger fixed-selector baselines.
- `101_`: QK-geometry filters for attention-logit-preserving sparse selection.
- `102_`: variable per-head compression rates instead of one flat transport budget.
- `103_`: offline-calibrated head reordering for stronger retrieval-head-only transport.

Files `104_` through `106_` extend the next likely branch around dynamic
budgeting, principal-key concentration, and mechanistic head localization:

- `104_`: dynamic KV budgets for task-adaptive selector and head-budget ablations.
- `105_`: principal-key attention as a direct sparse-key concentration reference.
- `106_`: scalable component localization for causal or attribution-based head selection.

Files `107_` through `113_` deepen the next head-routing branch around
head-wise offloading, fine-grained retrieval, asymmetric K/V importance, and
OpenReview-only head-budget papers that are currently stored as link notes:

- `107_`: head-wise offloading as a direct head-specific transport systems baseline.
- `108_`: fine-grained KV retrieval for stronger query-aware token and head routing.
- `109_`: asymmetric K/V quantization, reinforcing the current `K-only` paper direction.
- `110_` through `113_`: link notes for hierarchical sharing, head-specific retention,
  adaptive budgeting, and head-level key pruning where direct shell PDF download
  returned HTML landing pages on this machine.

Files `114_` through `119_` extend the current paper-tightening loop around
calibration-derived head priors, attention-causality cautions, and stronger
dynamic budgeting alternatives:

- `114_`: layer-wise dynamic budget allocation as a direct next-step comparator for fixed head priors.
- `115_`: classic evidence that attention-head importance is sharply nonuniform.
- `116_`: a cautionary reference against overclaiming live attention as explanation or mechanism.
- `117_`: adaptive probabilistic memory retention as a stronger query-blind retention baseline.
- `118_`: heterogeneous group-attention experts for dynamic token-wise KV optimization.
- `119_`: evolutionary KV compression as a broader search-based budgeting reference.

Files `120_` and `121_` extend the next head-budget and transfer loop around
retrieval-vs-streaming head structure and sink-token priors:

- `120_`: retrieval heads vs streaming heads as a direct head-budgeting baseline.
- `121_`: attention sinks and streaming retention as a stronger blind prior / sink-token control.

`75_transformers_with_multiresolution_attention_heads.md` is a canonical-link note rather than a PDF because OpenReview blocked direct shell download from this machine on 2026-04-16.

See `math_grounding_manifest.json` for source URLs and the reason each paper was added.
