# Cross-Field Source Innovation Controls References

Date: `2026-04-27`

## Blocker Helped

The current blocker is not candidate reachability. It is proving that a compact
message is source-derived after target-prior sampling and prompt-wrapper
controls. These references suggest compact sidecars and source-destroying
controls for future gates, but do not change the negative decision on the
SVAMP32 two-ID replay.

## Sources And Implications

### Distributed Source Coding

- Source: Slepian and Wolf, "Noiseless Coding of Correlated Information
  Sources"; Wyner and Ziv, "The Rate-Distortion Function for Source Coding with
  Side Information at the Decoder"
- URLs: https://ieeexplore.ieee.org/document/1055037 and
  https://ieeexplore.ieee.org/document/1055508
- Role: theory support, experiment framing
- Mechanism: the receiver already has side information, so the transmitter
  should send only the conditional residue.
- Experiment change: future source sidecars should encode answer-masked
  innovation relative to target/no-source candidate state, not source answers.
  Controls must include target-only side information, shuffled-source syndrome,
  and same-byte random syndrome.

### Kalman Filtering And Predictive Coding

- Source: Kalman, "A New Approach to Linear Filtering and Prediction Problems";
  Rao and Ballard, "Predictive coding in the visual cortex"
- URLs: https://www.cs.unc.edu/~welch/kalman/media/pdf/Kalman1960.pdf and
  https://www.nature.com/articles/nn0199_79
- Role: inspiration, ablation design
- Mechanism: transmit innovation or prediction error rather than the full
  state.
- Experiment change: test compact innovation sketches that subtract target
  prior features from answer-masked source process features. Destroy the signal
  with shuffled source, target-only innovation, same-norm noise, and token-order
  scrambling.

### One-Bit Sketching And Compressed Sensing

- Source: Charikar, "Similarity Estimation Techniques from Rounding
  Algorithms"; Donoho, "Compressed Sensing"
- URLs: https://www.cs.princeton.edu/courses/archive/spr05/cos598E/bib/p380-charikar.pdf
  and https://doi.org/10.1109/TIT.2006.871582
- Role: baseline, compact sidecar inspiration
- Mechanism: a small number of random projections can preserve similarity or a
  sparse signal enough for candidate selection.
- Experiment change: before a learned connector, run a 16-bit answer-masked
  process fingerprint baseline over candidate traces. Kill it if wrong-question,
  answer-only, or random same-byte fingerprints recover the same clean IDs.

## Next Experiment Consequence

The immediate next experiment does not become a learned connector. The S16
replay found that the target model with the same brief-analysis wrapper reaches
`2/2`, so the source-sampled two-ID surface is prompt-wrapper reachable. The
next valid use of these references is a new surface where target wrapper
controls remain below matched source, or a larger held-out source surface with
predeclared prompt controls.
