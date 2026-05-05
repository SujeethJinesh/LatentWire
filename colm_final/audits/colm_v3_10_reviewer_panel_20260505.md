# COLM_v3 Ten-Reviewer Panel

Date: 2026-05-05

Purpose: stress-test the integrated COLM_v3 draft before external reviewer
circulation. Scores use a 1-10 workshop-style scale where 5 is borderline,
6 is weak accept, 7 is accept, and 8+ would require stronger positive-method
evidence than the current packet supports.

## Aggregate Read

- Mean score: 6.4 / 10
- Median score: 6.5 / 10
- Plausible decision: workshop weak accept if scoped as a packet protocol,
  destructive-control study, and systems accounting artifact.
- Main risk: reviewers will reject any framing that sounds like solved latent
  communication, broad cross-family transfer, or superiority over C2C.

## Reviewer Panel

| Reviewer | Lens | Score | Main concern | Action taken |
|---|---:|---:|---|---|
| R1 | method novelty | 6 | Packet mostly preserves source candidate choice; novelty over source-index is limited. | Retitled around candidate-transfer packets and added a claim-boundary paragraph. |
| R2 | evaluation/statistics | 7 | Uncertainty reporting should make fragile OBQA text separation visible. | Added compact uncertainty table with packet-target, packet-text, and packet-source lower bounds. |
| R3 | LLM collaboration impact | 7 | Practical use case is unclear if the method is only multiple choice. | Added "Where the current result is useful" paragraph and scoped away open-ended generation. |
| R4 | systems | 6 | Systems rows are mostly analytical, not native GPU evidence. | Added measured/accounted versus analytical/native-pending distinction in systems section. |
| R5 | privacy/threat model | 7 | Source privacy can be confused with source-choice secrecy. | Threat model now distinguishes receiver-visible source-state privacy from candidate-choice leakage. |
| R6 | reproducibility/workshop fit | 7 | Artifact bundle is strong, but main text lacked enough rerun detail. | Added reproducibility card and kept artifact manifest in appendix. |
| R7 | skeptical NLP/LLM | 5 | Does not beat source-index and is same-family MCQA only. | Main table and text now foreground source-index parity as the claim boundary. |
| R8 | workshop area chair | 6 | Paper is stronger as a benchmark/protocol paper than as a positive-method claim. | Framing changed from broad packet communication to candidate-transfer under destructive controls. |
| R9 | related work | 6 | Activation transfer, crosscoder/transcoder, and prompt-compression baselines need clearer placement. | Related work now adds crosscoder/transcoder and LLMLingua/LongLLMLingua comparisons. |
| R10 | reproducibility auditor | 7 | Frozen artifacts and rerun/hash status must be easy to find. | Review packet and appendix point to exact artifact manifest, hashes, scripts, and rerun status. |

## Supported Claim After Panel

LatentWire provides a source-private, fixed-byte candidate-transfer packet
protocol and destructive-control evaluation framework. On ARC-Challenge and
OpenBookQA, tiny same-family packets beat target-only and same-budget structured
text controls, but they do not beat explicit source-index communication and do
not support broad cross-family or C2C-superiority claims.

## Reviewer-Driven Changes Made

1. Changed title to emphasize source-private candidate-transfer packets.
2. Added practical use cases and non-use cases in the introduction.
3. Added an explicit claim-boundary paragraph after the main results.
4. Added a compact uncertainty table that exposes fragile comparator surfaces.
5. Strengthened the source-private threat model against privacy overclaim.
6. Strengthened measured-versus-estimated systems language.
7. Added LLMLingua and LongLLMLingua as prompt-compression baselines.
8. Kept negative source-index, cross-family, and cached-connector evidence in
   the paper rather than hiding it in appendix-only text.

## Remaining Reviewer Risks

- The paper still has no positive beyond explicit source-index transfer.
- The packet is multiple-choice and same-family; open-ended and cross-family
  results remain future work.
- Systems numbers are object-size/accounting comparisons, not native GPU
  latency, throughput, HBM, energy, or kernel wins.
- Page budget may require moving the claim audit or related-work matrix out of
  the submitted PDF depending on the final workshop format.

## Decision

Proceed to reviewer circulation after final PDF build validation and human
copyedit. Do not strengthen the main claim unless a new experiment beats
source-index or provides native systems evidence.
