# Source-Private ARC-Challenge Fourier/Anchor-Syndrome Gate, 2026-05-02

## Status

- current paper readiness: COLM is strong; ICLR is closer but still needs
  cross-family and native systems evidence.
- current story: LatentWire now has fixed-byte source-private packets,
  OpenBookQA receiver-fusion, and a positive ARC common-basis packet using a
  public anchor-relative spectral chart.
- exact blocking gap: the source decision is still a Qwen choice cache and the
  new common-basis row still needs cross-family falsification plus native
  C2C/KVComm/TurboQuant systems baselines.

Artifact:
`results/source_private_arc_challenge_fourier_anchor_syndrome_gate_20260502/`

## What This Gate Does

Plain-language version: source and receiver agree on a public coordinate grid
made from ARC train-set anchors. The source compresses its selected candidate
into a low-frequency cosine sketch of that grid and sends a `12B` packet. The
receiver decodes the sketch against its own public candidate coordinates.

This is not a new source-model forward pass. It reuses the frozen
answer-key-forbidden ARC source-choice caches, then changes only the packet
basis/codec. The predeclared default uses `384` public anchors, a `96`
dimensional low-frequency DCT-II syndrome, `96` projection dimensions, `12B`
payloads, and five packet seeds.

## Results

ARC-Challenge test:

| Row | Result |
|---|---:|
| matched Fourier/anchor-syndrome seed pass count | 5/5 |
| matched mean / min accuracy | 0.344 / 0.343 |
| target-only accuracy | 0.265 |
| same-byte text accuracy | 0.311 |
| min lift over target | +0.078 |
| min lift over same-byte text | +0.032 |
| min CI95 low vs target | +0.038 |
| candidate derangement max | 0.216 |

ARC validation also passes `5/5` seeds, with matched mean/min `0.387/0.385`,
target `0.244`, same-byte text `0.348`, and min CI95 low `+0.067`.

## Mismatch Controls

The key evidence is not higher accuracy than the previous ARC packet; the key
evidence is that the low-frequency shared-basis packet preserves the signal
while basis mismatch destroys it.

On ARC test:

- anchor-ID shuffle: `0/5` pass, matched mean `0.253`
- anchor-value shuffle: `0/5` pass, matched mean `0.247`
- spectral-bin permutation: `0/5` pass, matched mean `0.266`
- random shared anchors: `5/5` pass, matched mean `0.344`

The random shared-anchor diagnostic weakens any semantic-anchor overclaim. The
safe claim is public shared-coordinate communication, not semantic latent
isomorphism. The anchor-ID/value and spectral-bin mismatch controls still show
that source and receiver must agree on the basis identity for the packet to
work.

## Decision

Promote this as a third technical contribution: a Fourier/anchor-syndrome
common-basis packet that preserves the frozen ARC packet result while adding
explicit basis-mismatch controls.

Follow-up rate gate:
`results/source_private_arc_challenge_fourier_anchor_syndrome_gate_20260502_budget8/`
reduces the payload from `12B` to `8B` (`11B` framed with header/CRC) and still
passes validation/test at `5/5` seeds. On ARC test it reaches matched
mean/min `0.344/0.342`, target `0.265`, same-byte text `0.300`, min CI95 low
vs target `+0.038`, and all anchor/spectral mismatch controls collapse. The
safe headline should now use the `8B` payload row while retaining the same
shared-coordinate, not semantic-anchor, claim boundary.

Stronger follow-up:
`results/source_private_arc_challenge_fourier_anchor_syndrome_gate_20260502_budget8_10seed/`
keeps the same `8B` payload / `11B` framed packet and upgrades the seed
stability gate to `10/10` matched seeds on validation and test. ARC test
matched mean/min remains `0.344/0.342`, target `0.265`, same-byte text `0.300`,
min CI95 low vs target `+0.038`, and anchor-ID/value plus spectral-bin mismatch
controls remain `0/10`. Use this as the current headline artifact.

Uncertainty follow-up:
`results/source_private_arc_challenge_fourier_anchor_syndrome_gate_20260502_budget8_10seed_b2000/`
reruns the same `8B` packet with `10` seeds and `2000` paired bootstrap
samples. The pass result is unchanged: ARC test matched mean/min `0.344/0.342`,
target `0.265`, same-byte text `0.300`, min CI95 low vs target `+0.038`, and
anchor-ID/value plus spectral-bin mismatch controls remain `0/10`. This is the
current headline artifact.

Do not claim ICLR readiness yet. Next exact gate: run a strict cross-family
falsification or train-source selector on top of this shared basis, then run
native NVIDIA systems comparisons against C2C, KVComm, TurboQuant/QJL/KV-cache
quantization, vLLM, and SGLang.
