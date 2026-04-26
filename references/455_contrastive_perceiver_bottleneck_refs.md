# Contrastive Perceiver Bottleneck References

Date: `2026-04-26`

## Why This Memo Exists

After C2C mechanism-summary syndrome probes failed, the next branch tested a
receiver-conditioned Perceiver/query connector with answer-teacher supervision
and source-control contrast. This memo records the primary-source motivation
and how the sources should influence the next branch.

## Primary Sources

- [Cache-to-Cache: Direct Semantic Communication Between Large Language Models](https://arxiv.org/abs/2510.03215)
  Problem: C2C is the strongest direct comparator and source of the current
  residual headroom. Mechanism: learned KV projection/fusion with layer-level
  gating. Experiment impact: scalar summaries are too weak; any future C2C
  distillation should target token/layer residual behavior. Role: baseline.
- [BLIP-2 / Q-Former](https://arxiv.org/abs/2301.12597)
  Problem: compact communication between frozen heterogeneous systems.
  Mechanism: trainable query transformer extracts target-consumable information
  from frozen encoder states. Experiment impact: keep learned-query bottlenecks
  but require source controls before promotion. Role: inspiration.
- [Perceiver IO](https://arxiv.org/abs/2107.14795)
  Problem: variable source/target memories need a fixed-rate interface.
  Mechanism: learned latent/query array cross-attends to arbitrary inputs and
  emits structured outputs. Experiment impact: query-count/rate curves should
  be measured if this branch revives. Role: architecture.
- [Conditional Contrastive Learning](https://arxiv.org/abs/2106.02866)
  Problem: separate source-specific signal from nuisance/control effects.
  Mechanism: positives and negatives are conditioned on nuisance variables.
  Experiment impact: matched-source objectives must include zero/shuffle and
  target-only/slots-only controls, not just margin maximization. Role:
  objective ablation.
- [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)
  Problem: preserve predictive information rather than formatting artifacts.
  Mechanism: InfoNCE-style predictive contrast. Experiment impact: use as an
  auxiliary constraint around a stronger connector, not as the whole method.
  Role: theory/objective.
- [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)
  Problem: sparse clean residual IDs make instance-only contrast brittle.
  Mechanism: group positives by class/phenotype. Experiment impact: only useful
  if source-only examples can be grouped on a larger surface. Role: ablation.
- [Deep Variational Information Bottleneck](https://arxiv.org/abs/1612.00410)
  Problem: avoid target-cache leakage while preserving source innovation.
  Mechanism: rate-limited predictive representation. Experiment impact:
  future connectors should sweep bottleneck size/rate/noise and report bytes.
  Role: theory.
- [Slepian-Wolf / Wyner-Ziv Side-Information Coding](https://www.mit.edu/~6.454/www_fall_2001/kusuma/summary.pdf)
  Problem: the target already has decoder-side information; full source-state
  transfer is inefficient and leaky. Mechanism: encode source information
  conditional on decoder side information. Experiment impact: optimize
  source-minus-target innovation and evaluate target-only controls. Role:
  theory support.

## Practical Read

The failed SVAMP32 Perceiver answer-teacher checkpoint shows that a stronger
connector plus contrastive regularization is still insufficient if target-only
or shuffled-source controls beat matched source on the same clean IDs. The next
experiment should either use a larger/headroom-rich source-only surface or add
a source-necessity objective that explicitly penalizes target-only recovery
before answer-teacher supervision dominates.
