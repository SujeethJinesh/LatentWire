# HellaSwag Hidden-Innovation Repair Probe

- pass gate: `True`
- selected view: `score_hidden_residual`
- selected ridge: `1000.0`
- eval accuracy: `0.499023`
- source-label copy accuracy: `0.461914`
- trained-label copy accuracy: `0.458984`
- delta vs best label copy: `0.037109`
- paired CI95 vs best label copy: `[0.009766, 0.061060]`
- zero-hidden control accuracy: `0.461914`
- wrong-example hidden control accuracy: `0.395508`
- candidate-roll hidden control accuracy: `0.360352`
- source top-2 oracle accuracy: `0.715820`

## Interpretation

Unlike the failed top-2 switcher, this branch treats the source hidden state as an innovation signal: for each candidate, a train-only denoiser scores whether the candidate is the answer using source scores plus the candidate's hidden residual against the source top choice. The receiver receives only a fixed-byte candidate/confidence packet, not source text, KV, raw scores, or raw hidden vectors.
