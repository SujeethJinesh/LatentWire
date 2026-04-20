# Causal Head Gating

- Title: Causal Head Gating
- Date: 2025-05-19
- Link: https://arxiv.org/abs/2505.13737

Why this is in the backlog:

- Direct reference for lightweight, query-conditioned head-level control.
- Useful for the current lane where fixed calibration transport looks plausible
  offline but fails under the live query.
- Most relevant adaptation here is a tiny runtime gate calibrator on top of a
  frozen transport checkpoint, or a learned query-conditioned gate inside the
  translator.
