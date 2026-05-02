# ARC Candidate-Syndrome Connector Gate

Date: 2026-05-02

## Status

- Current paper readiness: COLM workshop remains plausible; ICLR full paper is
  still blocked.
- Current story: the defensible core is fixed-byte source-private packets,
  public-basis ARC/OpenBookQA packet methods, and byte/exposure systems
  accounting. The ARC Fourier/anchor-syndrome row is a positive common-basis
  packet, but it is not yet source-family robust.
- Exact gap: cached packet/score geometry still fails the strict
  TinyLlama-vs-Qwen ARC disagreement surface. ICLR still needs a true
  hidden-state/query-resampler/common-basis connector, or a stronger alternate
  source-family run.

## Command

```bash
./venv_arm64/bin/python scripts/build_source_private_arc_challenge_candidate_syndrome_connector_gate.py \
  --output-dir results/source_private_arc_challenge_candidate_syndrome_connector_gate_20260502 \
  --bootstrap-samples 500
```

Primary artifact:
`results/source_private_arc_challenge_candidate_syndrome_connector_gate_20260502/candidate_syndrome_connector_gate.json`.

## Lay Explanation

The earlier routers only chose between two complete hints: the TinyLlama hint
or the Qwen-substituted hint. This experiment gave the receiver a little more
freedom. It trained a small validation-only scorer that could pick any answer
candidate using the cached packet scores and source-score shapes. If the
problem were only that the router was too rigid, this should have recovered
some of the oracle headroom. It did not.

## Result

| View | Primary | Validation | Frozen test | Qwen | Delta | CI95 low | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tiny_packet_only_connector` | yes | `0.304` | `0.246` | `0.317` | `-0.071` | `-0.142` | `0.586` |
| `tiny_score_shape_connector` | yes | `0.344` | `0.288` | `0.317` | `-0.029` | `-0.091` | `0.586` |
| `paired_family_diagnostic_connector` | no | `0.418` | `0.316` | `0.317` | `-0.001` | `-0.027` | `0.586` |

The selected primary view is `tiny_score_shape_connector`. It improves over
the Tiny packet-only connector but remains below Qwen-substituted packets on
frozen test. The paired-family diagnostic can see both TinyLlama and Qwen
cached score features, but it only ties Qwen within noise and is not a valid
source-private primary claim.

## Decision

Rule out cached candidate-level packet/score-shape connectors for the current
ARC source-family repair. The remaining oracle headroom is real, but it is not
accessible from selected candidate index, packet receiver scores, or scalar
source-score geometry.

The next highest-value branch is a true common-basis connector that sees richer
source information under a fixed-byte or explicitly byte-accounted protocol:

1. query-resampler connector inspired by Perceiver IO / Q-Former style learned
   queries;
2. SAE or crosscoder feature dictionary that maps source activations into a
   receiver-usable feature basis;
3. stronger alternate source-family falsification on NVIDIA, keeping the same
   frozen ARC disagreement protocol.

## Positioning

This negative gate tightens the uniqueness boundary. It is not prefix tuning:
there is no learned continuous prompt injected into one frozen model. It is not
C2C/KVComm/KVCOMM: it does not transmit or fuse KV caches. It also falls short
of relative-representation or SAE-style common-basis communication because it
uses cached score geometry rather than an explicit shared representation.

Use the result as a reviewer-facing ablation: simple learned connectors over
cached packet/score features do not solve cross-family latent transfer, so the
paper must either develop a richer common-basis connector or keep COLM scoped
to fixed-byte packets plus rigorous negative gates.

Relevant primary sources for the next branch and novelty boundary include
relative representations (`https://arxiv.org/abs/2209.15430`), BLIP-2 /
Q-Former (`https://arxiv.org/abs/2301.12597`), Perceiver IO
(`https://arxiv.org/abs/2107.14795`), Flamingo
(`https://arxiv.org/abs/2204.14198`), SAE universality
(`https://arxiv.org/abs/2410.06981`), universal SAEs
(`https://arxiv.org/abs/2502.03714`), sparse crosscoders
(`https://arxiv.org/abs/2603.05805`), prefix tuning
(`https://arxiv.org/abs/2101.00190`), prompt tuning
(`https://arxiv.org/abs/2104.08691`), C2C
(`https://arxiv.org/abs/2510.03215`), KVComm
(`https://arxiv.org/abs/2510.03346`), KVCOMM
(`https://arxiv.org/abs/2510.12872`), and TurboQuant
(`https://arxiv.org/abs/2504.19874`).
