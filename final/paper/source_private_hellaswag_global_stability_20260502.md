# HellaSwag Hidden-Innovation Global Stability

## Status

This is the strongest current positive-method evidence for the paper.

Current paper story: LatentWire uses a source-private candidate-wise
hidden-innovation packet to transfer useful information from a source model to a
receiver without exposing source text, source KV/cache state, raw hidden
vectors, or raw score vectors. On the full frozen HellaSwag validation split, a
`2B` raw / `5B` framed packet beats label-copy, score-only, zero-hidden,
wrong-hidden, and candidate-roll controls.

Remaining blocker for comfortable ICLR: strict cross-family falsification and
native NVIDIA/vLLM/SGLang systems rows are still missing. This result is enough
to promote the HellaSwag method branch, not enough to claim broad cross-model
latent communication or native serving speedups.

## Lay Explanation

For each multiple-choice example, the source model privately looks at its own
hidden-state evidence for the answer choices. Instead of sending its text,
cache, scores, or hidden vector, it sends a tiny code saying which candidate its
hidden evidence supports. This experiment asks whether that tiny code still
helps when we evaluate all frozen validation examples, and whether the result is
stable if we leave out one training sample seed at a time.

## Artifact

`results/source_private_hellaswag_hidden_innovation_global_stability_20260502/hellaswag_hidden_innovation_global_stability.json`

Supporting tables:

- `results/source_private_hellaswag_hidden_innovation_global_stability_20260502/policy_rows.csv`
- `results/source_private_hellaswag_hidden_innovation_global_stability_20260502/subbag_rows.csv`
- `results/source_private_hellaswag_hidden_innovation_global_stability_20260502/slice_policy_rows.csv`
- `results/source_private_hellaswag_hidden_innovation_global_stability_20260502/predictions.jsonl`

## Gate Definition

Primary policy: pre-existing mean-zscore aggregation over the candidate-wise
hidden-innovation model bank.

The global audit passes only if:

- full validation selected accuracy beats best label-copy by at least `+0.02`;
- paired CI95 low vs best label-copy is positive;
- selected accuracy beats score-only and zero-hidden controls by at least
  `+0.02`;
- paired CI95 low vs score-only is positive;
- wrong-example hidden and candidate-roll hidden controls do not exceed best
  label-copy;
- every leave-one-train-sample subbag passes;
- any slice-level failure is limited to the previously identified terminal-tail
  surface.

## Results

Full HellaSwag validation (`10042` rows):

| Policy | Accuracy | Delta vs best label-copy | CI95 low vs best label-copy | Subbags | Slices | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| mean-zscore | `0.526688` | `+0.045808` | `+0.039634` | `3/3` | `10/10` | yes |
| vote | `0.527584` | `+0.046704` | `+0.040278` | n/a | n/a | yes |
| hybrid vote-on-score-agreement | `0.532464` | `+0.051583` | `+0.045556` | `3/3` | `10/10` | yes |

Controls for the primary mean-zscore policy:

- best label-copy: `0.480880`
- score-only bagged control: `0.480880`
- zero-hidden control: `0.480880`
- wrong-example hidden control: `0.452699`
- candidate-roll hidden control: `0.416152`
- source-private packet: `2B` raw / `5B` framed

Terminal tail slice (`9216:10042`, `826` rows):

- mean-zscore: `0.539952`, delta vs best label-copy `+0.042373`,
  CI95 low `+0.019370`, pass
- vote: `0.536320`, delta `+0.038741`, CI95 low `+0.015738`, pass
- hybrid: `0.547215`, delta `+0.049637`, CI95 low `+0.024788`, pass

Leave-one-train-sample subbags for mean-zscore:

- leave out `1729`: `0.517726`, delta `+0.036845`, CI95 low `+0.030614`
- leave out `2027`: `0.524398`, delta `+0.043517`, CI95 low `+0.037186`
- leave out `2039`: `0.529277`, delta `+0.048397`, CI95 low `+0.041526`

## Decision

Promoted:

1. Candidate-wise hidden-innovation is now the main positive method branch.
2. HellaSwag full-validation global stability clears the current Mac-feasible
   evidence gate.
3. The hybrid vote-on-score-agreement rule becomes an optional stronger policy
   row, but the primary contribution should remain the simpler mean-zscore
   policy because it was already live before this audit.

Ruled out or demoted:

1. Top-2 trust-or-switch remains cut as a contribution.
2. The paper should not claim native systems wins yet.
3. The paper should not claim a solved universal latent basis; sparse residual
   dictionary and relative-residual gates are still future branches.

## Contribution Set

The paper can now defend three technical contributions if the next gates hold:

1. **Extreme-rate source-private hidden-innovation packet.** A `2B` raw /
   `5B` framed packet transfers useful candidate-level source evidence without
   exposing source text, KV/cache state, raw scores, or hidden vectors.
2. **Destructive-control evaluation ladder.** The method is tested against
   label-copy, score-only, zero-hidden, wrong-example hidden,
   candidate-roll hidden, top-2 switch decomposition, train-sample subbags, and
   frozen contiguous HellaSwag slices.
3. **Systems boundary for non-KV communication.** The communicated object is
   bytes-scale task evidence rather than source KV/cache state. Current evidence
   supports byte/exposure accounting; native vLLM/SGLang/C2C/KVComm/KV
   compression comparisons remain required before throughput or HBM claims.

## Reviewer-Framing Boundary

HellaSwag is a commonsense continuation benchmark, so the result supports a
commonsense multiple-choice repair claim, not general reasoning. Source:
https://arxiv.org/abs/1905.07830.

Prefix tuning and prompt tuning learn continuous task-specific vectors or soft
tokens for conditioning frozen models. LatentWire instead sends a per-example
discrete packet derived from private source hidden evidence and does not insert
soft prompt vectors into the receiver. Sources:
https://aclanthology.org/2021.acl-long.353/ and
https://arxiv.org/abs/2104.08691.

C2C and KVComm are closer inter-model communication baselines because they
communicate or fuse source-side KV/cache state. LatentWire must be framed as a
stricter rate/privacy point, not as a native-performance winner until GPU
systems rows exist. Sources: https://arxiv.org/abs/2510.03215 and
https://arxiv.org/abs/2510.03346.

TurboQuant, KIVI, KVQuant, and QJL are systems comparators for quantized
vectors or KV/cache state. They set a required systems comparison floor, but
they do not duplicate a fixed-byte source-private task packet. Sources:
https://arxiv.org/abs/2504.19874, https://arxiv.org/abs/2402.02750,
https://arxiv.org/abs/2401.18079, and https://arxiv.org/abs/2406.03482.

Sparse autoencoder universality and relative representations motivate the next
common-basis branches. They are not yet solved by the current dense packet.
Sources: https://arxiv.org/abs/2410.06981 and https://arxiv.org/abs/2209.15430.

vLLM and SGLang define the native serving systems baselines for future
TTFT/TPOT/goodput/HBM rows. Sources: https://arxiv.org/abs/2309.06180 and
https://arxiv.org/abs/2312.07104.

## Next Gate

Run the strict cross-family falsification pair before widening benchmarks:

1. same HellaSwag full-validation protocol;
2. source-private packet generated from one model family and consumed by a
   different family;
3. same destructive controls and paired uncertainty;
4. pass only if the cross-family row remains positive or cleanly explains where
   source evidence fails.

After that, build the native NVIDIA systems runbook rows for vLLM, SGLang,
C2C/KVComm, QJL/TurboQuant/KIVI/KVQuant floors, same-byte text, target-only,
and LatentWire packet variants.
