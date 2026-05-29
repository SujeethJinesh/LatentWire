# Tail Trace Debug

Trace: `opencompass_AIME2025_I_4`.

| Regime | Perplexity / recovery |
|---|---:|
| BF16 | 1.036758 |
| Static top-1% | 1.037530 |
| Tight ParoQuant reference | recovery 5.940 |
| Tight ParoQuant + residual | perplexity 1.047024; recovery -12.294 |

The residual correction improves over the original loose ParoQuant catastrophe reported elsewhere, but it is much worse than the tight ParoQuant reference. That means the selected residual columns are not safe as a tail-control mechanism.
