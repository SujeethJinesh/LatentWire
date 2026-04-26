# Target-Loss Resampler Branch References

Focused consolidation for the next target-side LM-loss resampler branch after
tiny prefix emitters failed source controls.

## Sources And Use

1. [Perceiver IO: A General Architecture for Structured Inputs and Outputs](https://openreview.net/forum?id=fILj7WpI-g)
   - problem helped: variable source token states need a fixed-rate latent
     interface
   - mechanism: learned latent arrays read structured inputs with
     cross-attention and write task-specific outputs
   - experiment change: use a query/resampler bottleneck with query-count
     sweeps instead of a global source summary or two-token prefix
   - role: inspiration / architecture baseline

2. [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)
   - problem helped: frozen decoders need gated access to external source
     features
   - mechanism: Perceiver Resampler plus gated cross-attention into a frozen LM
   - experiment change: compare always-on, gated, and source-destroyed resampler
     variants at matched query budgets
   - role: inspiration / ablation

3. [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://proceedings.mlr.press/v202/li23q.html)
   - problem helped: learned queries must extract decoder-useful information
     from a frozen source encoder
   - mechanism: Q-Former query tokens bridge frozen encoders and language models
   - experiment change: train target-side next-token loss, not only cache MSE or
     answer-token readout margins
   - role: inspiration / baseline

4. [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://aclanthology.org/2021.acl-long.353/)
   - problem helped: continuous prefixes are the minimal frozen-target interface
   - mechanism: optimize soft prompts while keeping the LM frozen
   - experiment change: keep target-only and slots-only prefix controls as hard
     baselines; the 2026-04-26 target-CE gate shows this interface is too weak
     as the main method
   - role: baseline / pruning support

5. [Wyner-Ziv coding with decoder side information](https://doi.org/10.1109/TIT.1976.1055508)
   - problem helped: target cache/candidates already contain side information,
     so the source message should encode conditional innovation
   - mechanism: lossy coding with decoder side information
   - experiment change: report matched bytes/query count against target-only,
     shuffled-source, and C2C-fuser controls
   - role: theory support

## Next Experiment Rule

The next branch should not emit a tiny generic prefix. It should expose a
larger but rate-controlled source-token memory to target-side LM loss, with
source-destroying controls, target-only memory, slots-only memory, matched
C2C-fuser, and generation-time scoring in the first gate.
