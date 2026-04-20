# MoRA

- Title: `MoRA: On-the-fly Molecule-aware Low-Rank Adaptation Framework for LLM-based Multi-Modal Molecular Assistant`
- Date: 2025-10-14
- Link: https://arxiv.org/abs/2510.12245
- Why it matters here:
  - strongest recent pattern for generating instance-specific low-rank adapters on top of a frozen backbone
  - useful template if the next bridge should be query-conditioned and materially more expressive than the current tiny static residuals

Most transplantable mechanism:
- generate a low-rank adapter from prompt-local context instead of fitting one fixed bridge for all prompts

Immediate use in our setting:
- use it as the literature anchor for a dynamically generated bridge module that sits on top of grouped transport
