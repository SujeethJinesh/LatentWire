# Source-Private Communication Pivot References

Date: `2026-04-27`

## Blocker Helped

The current blocker is target-prior contamination: ordinary math surfaces let
the receiver recover apparent source gains through prompt wrappers, sampling,
or answer leakage. These references motivate a pivot to source-private
communication with explicit decoder side information and rate constraints.

## Sources And Implications

### Slepian-Wolf Distributed Source Coding

- Source: Slepian and Wolf, "Noiseless Coding of Correlated Information
  Sources", 1973.
- URL: https://www.scholarpedia.org/article/Slepian-Wolf_coding
- Role: theory support and method framing.
- Mechanism: correlated sources can be compressed separately and jointly decoded
  using side information.
- Experiment change: target candidate pool/KV state should be treated as decoder
  side information; source should transmit only residual bits.

### Wyner-Ziv Coding

- Source: Wyner and Ziv, "The Rate-Distortion Function for Source Coding with
  Side Information at the Decoder", 1976.
- URL: https://www.mit.edu/~6.454/www_fall_2001/kusuma/wynerziv.pdf
- Role: theory support.
- Mechanism: lossy source coding when only the decoder has side information.
- Experiment change: report accuracy/log-loss vs transmitted bytes, with the
  receiver's no-source candidate pool fixed.

### Shannon Rate-Distortion Theory

- Source: Shannon, "Coding Theorems for a Discrete Source with a Fidelity
  Criterion", 1959.
- URL: https://gwern.net/doc/cs/algorithm/information/1959-shannon.pdf
- Role: theory support and paper framing.
- Mechanism: compare methods by distortion at a communication rate.
- Experiment change: every method row should report bytes, latency, accuracy,
  and rate-normalized gain, not only final accuracy.

### Lamport Logical Clocks

- Source: Lamport, "Time, Clocks, and the Ordering of Events in a Distributed
  System", 1978.
- URL: https://lamport.org/pubs/time-clocks.pdf
- Role: distant inspiration.
- Mechanism: causal order can be a transferable invariant independent of raw
  state.
- Experiment change: test causal-order checksums of answer-masked source traces
  as compact sidecars.

### Universal Hashing

- Source: Carter and Wegman, "Universal Classes of Hash Functions", 1979.
- URL: https://www.cs.princeton.edu/courses/archive/fall09/cos521/Handouts/universalclasses.pdf
- Role: distant inspiration and control design.
- Mechanism: random challenge-response hashes should fail without the correct
  source signal.
- Experiment change: use per-example challenge hashes over answer-masked source
  features; wrong-seed controls must fail.

### Kalman Filtering And Predictive Coding

- Sources: Kalman, "A New Approach to Linear Filtering and Prediction
  Problems", 1960; Rao and Ballard, "Predictive coding in the visual cortex",
  1999.
- URLs: https://cs-www.cs.yale.edu/homes/yry/readings/general/Kalman1960.pdf
  and https://www.nature.com/articles/nn0199_79
- Role: inspiration for source innovation.
- Mechanism: useful communication can be residual innovation over a receiver
  prediction, not full state transfer.
- Experiment change: encode `source_features - target_prior_features`, and
  require target-prior and same-norm-noise controls to fail.

### I-JEPA, V-JEPA, LeJEPA, VICReg, Barlow Twins

- Sources: I-JEPA https://arxiv.org/abs/2301.08243, V-JEPA
  https://arxiv.org/abs/2404.08471, V-JEPA project
  https://ai.meta.com/vjepa/, LeJEPA https://arxiv.org/abs/2511.08544,
  VICReg https://arxiv.org/abs/2105.04906, Barlow Twins
  https://proceedings.mlr.press/v139/zbontar21a.html
- Role: connector design and anti-collapse diagnostics.
- Mechanism: latent prediction and variance/covariance controls can avoid
  representation collapse.
- Experiment change: use Query-JEPA only after source-private residual IDs
  exist; report effective rank, variance floor, covariance off-diagonal mass,
  query entropy, and matched-vs-control margins.

### Cache-To-Cache And Communicating Activations

- Sources: C2C https://arxiv.org/abs/2510.03215 and Communicating Activations
  https://arxiv.org/abs/2501.14082
- Role: baseline and competitor framing.
- Mechanism: direct activation/cache communication is now a relevant comparator
  for cross-model communication.
- Experiment change: include C2C/KVComm/activation rows or, at minimum, cite
  them as strong baselines for cache/latent transfer once the private-source
  task is established.

## Next Experiment Consequence

The next experiment should be a source-private strict-small benchmark, not
another ordinary math connector. The first method should be an interpretable
candidate-syndrome or private evidence packet with byte accounting and full
source-destroying controls. Learned JEPA/adapters should wait until that surface
produces stable residual IDs.
