# Ten-Reviewer Panel Feedback

Date: 2026-05-02

## Aggregate Read

The plausible COLM outcome is borderline-to-weak-accept if the paper is judged
as a scoped artifact/evaluation paper: reviewers who value honest scope,
reproducible artifacts, and negative results lean weak accept; reviewers who
expect broad latent communication, formal compression, or systems throughput
lean weak reject. After the explicit source-index/rate-curve audit, a realistic
score is about 6.0/10. The audit improves correctness and trust, but it does not
raise the paper to a comfortable accept because the strict positive-beyond-
source-index gate fails.

## Scores

| Reviewer | Lens | Overall | Clarity | Novelty | Evidence | Reproducibility | Recommendation |
|---|---|---:|---:|---:|---:|---:|---|
| A | Systems/serving | 5 | 7 | 5 | 4 | 6 | Borderline / weak reject |
| B | LLM evaluation | 7 | 8 | 6 | 7 | 7 | Weak accept |
| C | Representation/latent transfer | 5 | 7 | 5 | 5 | 6 | Weak reject |
| D | Reproducibility/artifact | 7 | 8 | 6 | 6 | 8 | Weak accept |
| E | Correctness/skeptical ML | 5 | 8 | 4 | 5 | 7 | Borderline / weak reject |
| F | COLM methods | 7 | 8 | 6 | 7 | 6 | Weak accept |
| G | Information theory/compression | 5 | 6 | 6 | 4 | 6 | Weak reject / borderline |
| H | Statistics/evaluation | 6 | 7 | 5 | 6 | 7 | Borderline |
| I | General ML | 6 | 7 | 5 | 6 | 6 | Borderline |
| J | Area chair | 7 | 8 | 6 | 7 | 7 | Weak accept if scoped |

Average overall score: 6.0 / 10.

## Strongest Positive Feedback

- The ARC evidence is no longer a tiny smoke result: 1172 test examples, 10
  seeds, destructive controls, and 2000 paired bootstraps are credible for a
  workshop paper.
- OpenBookQA gives a second benchmark and reduces the risk that the positive
  row is only an ARC artifact.
- The paper is unusually honest about strict cross-family failure and failed
  cached connector repairs.
- The byte/exposure accounting is useful if it remains explicitly separate
  from native serving throughput.
- The current title and abstract are scoped to "source-private packet" transfer,
  which reviewers found more defensible than "latent language."
- The explicit source-index table removes the largest correctness objection by
  showing the current method does not beat a direct selected-candidate code.

## Main Reviewer Risks

1. The packet mostly preserves the source's selected candidate and does not beat
   explicit source-index/source-rank communication. This is now disclosed, but
   it remains the largest novelty concern.
2. Cross-family evidence is negative. Phi-3 fails, and TinyLlama/cached
   connector repairs do not close the gap.
3. The systems contribution is accounting only. It should not be sold as a
   latency, throughput, HBM, or GPU result.
4. Random anchors passing weakens any semantic-anchor interpretation.
5. Reproducibility is documented in the bundle with exact commands, seeds, and
   frozen output hashes, but a reviewer may still ask for model snapshot IDs and
   a normalized rerun diff that strips timestamps/latency fields.
6. Raw source-score-vector quantization is still missing because the headline
   frozen caches store top choice rather than calibrated score vectors.

## Reviewer-Recommended Fixes Before Submission

- Keep the direct source-choice/index/source-rank rows in the main paper table.
- Keep full paired CIs for packet-vs-text, packet-vs-source-index, and
  packet-vs-best-destructive controls in the packaged audit.
- Keep the compact rate curve, but do not claim scaling because it is flat on
  the current surfaces.
- Keep "packet mostly follows source choice" in the central framing.
- Add a PDF-visible reproducibility table with exact commands, input cache
  hashes, seeds, bootstrap settings, and expected output paths; the bundle has
  these details, but the paper only summarizes them compactly.
- Keep cross-family failure in the thesis, not as an appendix-only limitation.

## Recommended COLM Framing

Submit as:

> A fixed-byte, source-private candidate-evidence packet protocol for
> same-family multiple-choice model collaboration, with strict destructive
> controls and falsification evidence showing where the protocol fails.

Avoid submitting as:

> A general cross-model latent communication language or a native systems
> acceleration method.

## Next Exact Gate

The most important experiment before camera-ready or ICLR extension is a
strict receiver-family or source-score gate. The receiver must beat target-only,
explicit source-index/source-rank, calibrated source-score quantization,
same-budget text, and destructive controls with paired confidence intervals.
