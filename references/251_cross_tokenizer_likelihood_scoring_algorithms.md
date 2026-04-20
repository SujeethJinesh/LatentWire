## Cross-Tokenizer Likelihood Scoring Algorithms for Language Model Distillation

- Title: `Cross-Tokenizer Likelihood Scoring Algorithms for Language Model Distillation`
- Date: 2025-12-16
- Link: https://arxiv.org/abs/2512.14954
- Why it matters here:
  - strongest recent reference for moving the bridge teacher fully into prediction space when teacher and student token spaces are not identical
  - directly relevant if the next positive-method branch uses exact or approximate cross-tokenizer likelihood supervision instead of local structural losses

Most transplantable mechanism:
- compute a teacher-side likelihood target in the student vocabulary rather than assuming shared-token next-token supervision is the only viable teacher

Immediate use in our setting:
- use it as the strongest literature anchor for a likelihood-style bridge teacher that sits above prompt-local attention or affinity distillation
