# Source-Private Literature Sprint References

Date: `2026-04-27`

## Blocker Helped

The blocker is no longer a lack of connector ideas. The blocker is a missing
strict-small benchmark where a source-private, rate-capped message improves the
target only when the matched source evidence is present. These sources support a
one-month path: evidence-packet/candidate-syndrome first, learned latent
connectors second.

## Sources And Experiment Implications

### JEPA And Anti-Collapse

- I-JEPA, "Self-Supervised Learning from Images with a Joint-Embedding
  Predictive Architecture", https://arxiv.org/abs/2301.08243
  - Helps: avoids reconstructing high-entropy observations.
  - Mechanism: predict frozen latent targets from context.
  - Experiment change: use JEPA only after source-private residual IDs exist;
    predict detached target candidate/process latents.
  - Role: inspiration and latent-objective framing.
- V-JEPA, "Revisiting Feature Prediction for Learning Visual Representations
  from Video", https://arxiv.org/abs/2404.08471 and
  https://ai.meta.com/vjepa/
  - Helps: masked latent prediction without token/pixel reconstruction.
  - Mechanism: fill in missing latent target features.
  - Experiment change: add masked target-state prediction as a second-stage
    connector objective.
  - Role: inspiration and ablation.
- LeJEPA, "Provable and Scalable Self-Supervised Learning Without the
  Heuristics", https://arxiv.org/abs/2511.08544
  - Helps: representation collapse risk.
  - Mechanism: JEPA-style prediction with isotropic/variance constraints.
  - Experiment change: require effective rank, variance floor, covariance
    off-diagonal, query entropy, and dual-view agreement telemetry.
  - Role: theory support and anti-collapse diagnostics.
- VICReg, https://arxiv.org/abs/2105.04906, and Barlow Twins,
  https://proceedings.mlr.press/v139/zbontar21a.html
  - Helps: noncontrastive objectives can collapse or become redundant.
  - Mechanism: variance, covariance, and redundancy-reduction losses.
  - Experiment change: collapse telemetry is mandatory before any learned
    connector claim.
  - Role: ablation and diagnostics.

### VLM Connectors And Target Preservation

- Perceiver IO, https://arxiv.org/abs/2107.14795
  - Helps: variable-size source state needs a fixed-rate interface.
  - Mechanism: learned queries compress input into a latent workspace.
  - Experiment change: query bottlenecks should be fixed-rate and byte-counted.
  - Role: connector design.
- Flamingo, https://arxiv.org/abs/2204.14198
  - Helps: bridge frozen encoders and decoders.
  - Mechanism: Perceiver Resampler plus gated cross-attention.
  - Experiment change: learned connectors should use target-preserving gates.
  - Role: connector design and baseline shape.
- BLIP-2 / Q-Former, https://arxiv.org/abs/2301.12597
  - Helps: avoid direct raw latent alignment.
  - Mechanism: query transformer bridges frozen visual encoder to frozen LLM.
  - Experiment change: source-state compression should be query-based before
    projection to target space.
  - Role: connector design.
- LLaVA, https://arxiv.org/abs/2304.08485, and improved LLaVA baselines,
  https://arxiv.org/abs/2310.03744
  - Helps: simple projectors and prompt formatting can be strong baselines.
  - Mechanism: MLP projector plus instruction-tuned target.
  - Experiment change: include an MLP/projector baseline and target prompt
    wrappers before claiming a connector gain.
  - Role: baseline discipline.
- LLaMA-Adapter, https://arxiv.org/abs/2303.16199 and
  https://openreview.net/forum?id=d4UiXAHN2W
  - Helps: target-self preservation.
  - Mechanism: zero-initialized attention/gating for new cues.
  - Experiment change: learned source adapters must be zero-init and exactly
    target-identical at step zero.
  - Role: target-preserving mechanism.

### Compression, KV, And Rate Accounting

- KVComm local memo, `references/pdf_markdown/02_kvcomm.md`
  - Helps: establishes KV/cache transfer as a relevant baseline.
  - Mechanism: selective KV sharing by matched layer selection.
  - Experiment change: cache rows need matched, zero, shuffled, target-only
    controls plus communicated-byte accounting.
  - Role: baseline and systems comparator.
- KVTC local memo, `references/pdf_markdown/17_kv_cache_transform_coding_kvtc.md`
  - Helps: compression realism.
  - Mechanism: decorrelation, adaptive quantization, entropy coding.
  - Experiment change: report simulated low-bit/KV bytes rather than raw tensor
    bytes only.
  - Role: systems baseline.
- KIVI, `references/pdf_markdown/64_kivi_a_tuning_free_asymmetric_2bit_quantization_for_kv_cache.md`
  - Helps: strong low-bit KV baseline.
  - Mechanism: asymmetric 2-bit key/value quantization.
  - Experiment change: include low-bit simulated cache budgets if cache transfer
    becomes live.
  - Role: baseline.

### Information Theory And Geometry

- Slepian-Wolf coding, https://www.itsoc.org/publications/papers/noiseless-coding-of-correlated-information-sources
  - Helps: source can send residual bits when decoder has side information.
  - Mechanism: syndrome/bin index decoded with target-side context.
  - Experiment change: use fixed-byte syndrome packets over target candidate
    pools.
  - Role: theory support.
- Wyner-Ziv coding,
  https://www.mit.edu/~6.454/www_fall_2001/kusuma/wynerziv.pdf
  - Helps: target has side information unavailable to source encoder.
  - Mechanism: lossy coding with decoder side information.
  - Experiment change: report accuracy/log-loss versus transmitted bytes.
  - Role: theory support.
- Distributed indirect source coding with decoder side information,
  https://arxiv.org/abs/2405.13483
  - Helps: source observes evidence for a task latent, not the final target
    representation.
  - Mechanism: task-distortion objective under decoder-side information.
  - Experiment change: evaluate candidate correctness and margins, not source
    reconstruction.
  - Role: closest formal framing.
- Relative representations, https://arxiv.org/abs/2209.15430 and
  https://openreview.net/forum?id=SrC-nwieGJ
  - Helps: cross-model coordinate gauges differ.
  - Mechanism: encode representations relative to anchors.
  - Experiment change: anchor-relative candidate syndromes are preferable to
    raw hidden-state transfer.
  - Role: geometry design.
- SVCCA, https://papers.nips.cc/paper/7188-svcca-singular-vector-canonical-correlation-analysis-for-deep-learning-dynamics-and-interpretability,
  and CKA, https://proceedings.mlr.press/v97/kornblith19a
  - Helps: layer/anchor choice should be audited, not assumed.
  - Mechanism: representation similarity diagnostics.
  - Experiment change: use CKA/SVCCA for diagnostics only, not as communication
    evidence.
  - Role: diagnostics.

### Multi-Agent, Tool, And Memory Systems

- ReAct, https://arxiv.org/abs/2210.03629
  - Helps: tool observations are private source state.
  - Mechanism: interleaved reasoning/actions/observations.
  - Experiment change: private tool traces are a strong second benchmark.
  - Role: task design.
- Toolformer, https://arxiv.org/abs/2302.04761
  - Helps: APIs/tools supply non-parametric evidence.
  - Mechanism: tool outputs become intermediate evidence.
  - Experiment change: source-only tool access is a clean source-private setup.
  - Role: task design.
- AutoGen, https://arxiv.org/abs/2308.08155
  - Helps: multi-agent systems require explicit handoff contracts.
  - Mechanism: agents with tools and conversation workflows.
  - Experiment change: handoff messages need byte/latency telemetry.
  - Role: systems framing.
- Chain-of-Agents, https://arxiv.org/abs/2406.02818
  - Helps: long-context work can be decomposed into bounded messages.
  - Mechanism: agents process chunks and pass compact summaries.
  - Experiment change: include private evidence packets before learned
    connectors.
  - Role: multi-agent framing.

## Next Experiment Consequence

The first runnable method should be a source-private evidence packet /
candidate-syndrome gate. Learned JEPA or Q-Former adapters are not the first
move; they are second-stage methods after the source-private benchmark exposes
residual IDs that target prompt-wrapper/no-source controls cannot solve.
