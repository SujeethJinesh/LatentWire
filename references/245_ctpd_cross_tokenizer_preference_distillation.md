## CTPD: Cross Tokenizer Preference Distillation

- Title: `CTPD: Cross Tokenizer Preference Distillation`
- Link: https://arxiv.org/abs/2601.11865
- Why it matters here:
  - useful if stronger-teacher bridge fitting eventually needs a teacher target higher than local interactions and closer to end-task preference or generation quality
  - relevant for longer-term heterogeneous pairs where aligned token spans and tokenizer mismatch become a bigger blocker

Most transplantable mechanism:
- distill a higher-level teacher signal that survives tokenizer mismatch instead of assuming token-level latent correspondence is enough

Immediate use in our setting:
- keep it as a reserve reference if prediction-level bridge losses still plateau and we need a stronger output-side teacher objective
