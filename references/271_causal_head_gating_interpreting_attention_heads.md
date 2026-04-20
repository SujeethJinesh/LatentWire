# Causal Head Gating

- Title: `Causal Head Gating: A Framework for Interpreting Roles of Attention Heads in Transformers`
- Date: 2025-05-19
- Link: https://arxiv.org/abs/2505.13737
- Why it matters here:
  - best recent fit for turning our current blocker story into component-level evidence rather than only aggregate accuracy tables
  - useful if the next reviewer-facing artifact should localize which layers/heads are actually causing wins and losses in the sparse transport family

Most transplantable mechanism:
- treat head contributions as causal interventions and summarize outcome flips under selective gating or removal

Immediate use in our setting:
- anchor a head-localization / paired-flip artifact built from existing evaluation traces and per-head selection logs
