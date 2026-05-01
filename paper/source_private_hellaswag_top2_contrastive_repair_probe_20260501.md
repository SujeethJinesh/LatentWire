# HellaSwag Top-2 Contrastive Repair Probe

- pass gate: `False`
- selected view: `hidden_score_contrast`
- selected eval accuracy: `0.449`
- source-label copy eval accuracy: `0.462`
- trained label-copy eval accuracy: `0.459`
- selected minus best label-copy: `-0.013`
- selected minus best zero-hidden control: `0.007`
- wrong-example packet accuracy: `0.389`
- source top-2 oracle accuracy: `0.716`

## Interpretation

This probe asks whether a tiny switch packet can use contrastive source evidence to repair
the source model's top HellaSwag choice. A pass would require beating label-copy controls;
otherwise the branch remains diagnostic and HellaSwag should stay out of the headline result.
