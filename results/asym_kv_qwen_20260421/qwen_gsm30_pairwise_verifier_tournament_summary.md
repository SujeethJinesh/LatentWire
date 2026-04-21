# Pairwise Verifier Tournament Summary

| Method | Accuracy | Delta vs target | Method-only | Baseline-only | Both correct | Both wrong | Fallback rate | Target selected | Seed selected | Target was left | Target was right | Avg comparisons |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pairwise_verifier_tournament | 0.0667 | +0.0000 | 2 | 2 | 0 | 26 | 0.0000 | 0.2000 | 0.8000 | 0.1667 | 0.2667 | 3.00 |

Interpretation:

This ablation converts the verifier into a bounded pairwise tournament over a shuffled candidate order,
with left/right orientation randomized per match. The raw responses, parsed winners, fallback flags,
pair order seed, and win counts are logged so the remaining failure mode can be separated into candidate
quality, pairwise order bias, and tournament aggregation error.
