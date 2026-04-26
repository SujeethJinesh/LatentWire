# Target-Conditioned Side-Information Query Bottleneck References

- date: `2026-04-26`
- blocker: learned Perceiver/query-innovation connectors show positive clean
  margins, but the wins are explained by shuffled-source, target-only, or
  slots-only controls
- next experiment implication: the next connector should make target queries
  attend into source states while treating target cache/candidate information
  as decoder side information, not as an unconstrained memory that can solve
  the task alone

## Sources And Use

1. [Perceiver IO: A General Architecture for Structured Inputs and Outputs](https://openreview.net/forum?id=fILj7WpI-g)
   - problem helped: variable-length source states need a fixed-rate interface
   - mechanism: learned latent arrays plus cross-attention read/write queries
   - experiment change: make the bottleneck explicitly target-query-conditioned
     rather than a generic source resampler
   - class: inspiration / baseline

2. [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)
   - problem helped: injecting heterogeneous frozen-source information without
     rewriting the receiver
   - mechanism: Perceiver Resampler plus gated cross-attention into a frozen LM
   - experiment change: add gate telemetry and gate-free/always-on ablations
     if a source-conditioned prefix reaches the pre-generation gate
   - class: baseline / inspiration

3. [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://proceedings.mlr.press/v202/li23q.html)
   - problem helped: query bottlenecks must select task-relevant source
     information for a frozen decoder
   - mechanism: Q-Former query tokens cross-attend into a frozen encoder before
     handoff to a frozen language model
   - experiment change: train matched-vs-source-destroyed target-query
     bottlenecks and sweep query count as a rate-distortion curve
   - class: baseline / ablation

4. [Wyner-Ziv coding with decoder side information](https://doi.org/10.1109/TIT.1976.1055508)
   - problem helped: target cache/candidate pools already contain side
     information, so the source message should encode only conditional
     innovation
   - mechanism: lossy source coding with side information at the decoder
   - experiment change: report bytes against target-only and slots-only
     side-information controls at the same budget
   - class: theory

5. [Noiseless coding of correlated information sources](https://www.itsoc.org/publications/papers/noiseless-coding-of-correlated-information-sources)
   - problem helped: separates redundant target-inferable content from real
     source communication
   - mechanism: distributed coding can approach conditional entropy when
     sources are jointly decoded
   - experiment change: score clean C2C/source-only residual IDs separately
     from source/text-explained IDs
   - class: theory / ablation

6. [Relative Representations Enable Zero-Shot Latent Space Communication](https://openreview.net/forum?id=SrC-nwieGJ)
   - problem helped: cross-model hidden spaces can differ by rotations,
     scaling, and gauge choices
   - mechanism: represent examples by similarities to anchors
   - experiment change: use relative/anchor features as an ablation for
     target-query/source-key alignment, not as the main claim
   - class: ablation / inspiration

7. [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://aclanthology.org/2021.acl-long.353/)
   - problem helped: continuous soft-prefixes are the minimal frozen-target
     message interface
   - mechanism: learn prefix vectors while keeping the language model frozen
   - experiment change: add target-only learned-prefix and slots-only prefix
     controls at the same token/byte budget
   - class: baseline

8. [SPoT: Better Frozen Model Adaptation through Soft Prompt Transfer](https://aclanthology.org/2022.acl-long.346/)
   - problem helped: distinguishes instance-level source communication from
     generic target adaptation
   - mechanism: transfer learned prompts between tasks
   - experiment change: include projected-soft-prompt controls trained without
     source instance content
   - class: baseline / ablation

## Next Experiment Rule

Do not rerun another generic Perceiver memory variant. The next learned
connector must include, from the first gate:

- matched target-query-to-source bottleneck
- zero-source and shuffled-source controls
- target-only learned-prefix control
- slots-only learned-prefix control
- projected-soft-prompt control without source instance content
- matched byte/query budgets across controls
