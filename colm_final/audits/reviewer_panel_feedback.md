# Ten-Reviewer Panel Feedback

Date: 2026-05-02

## Aggregate Read

The plausible COLM outcome is borderline: reviewers who value honest scope,
reproducible artifacts, and negative results lean weak accept; reviewers who
expect broad latent communication, formal compression, or systems throughput
lean weak reject. The modal recommendation is borderline / weak reject unless
the submission is framed as a narrow fixed-byte source-private packet protocol.

## Scores

| Reviewer | Lens | Overall | Clarity | Novelty | Evidence | Reproducibility | Recommendation |
|---|---|---:|---:|---:|---:|---:|---|
| A | Systems/serving | 5 | 7 | 6 | 4 | 5 | Borderline reject |
| B | LLM evaluation | 6 | 8 | 7 | 6 | 5 | Weak accept / borderline |
| C | Representation/latent transfer | 6 | 7 | 6 | 6 | 6 | Weak accept / borderline |
| D | Mechanistic interpretability/common features | 5 | 7 | 5 | 5 | 6 | Weak reject |
| E | Benchmarks/statistics | 6 | 7 | 6 | 6 | 7 | Weak accept / borderline |
| F | Information theory/compression | 5 | 6 | 7 | 5 | 6 | Weak reject / borderline |
| G | Multi-agent LLM communication | 5 | 7 | 6 | 5 | 5 | Weak reject |
| H | Workshop practicality/reproducibility | 7 | 7 | 7 | 6 | 7 | Weak accept |
| I | Skeptical general ML | 5 | 7 | 5 | 5 | 4 | Weak reject / borderline |
| J | Friendly rigorous methods | 7 | 8 | 7 | 7 | 5 | Weak accept if scoped honestly |

Average overall score: 5.7 / 10.

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
- The current title and abstract are mostly scoped to "source-private packet"
  transfer, which reviewers found more defensible than "latent language."

## Main Reviewer Risks

1. The packet appears to mostly preserve the source's selected candidate. A
   reviewer may ask why this is not just source-choice/index transfer.
2. Cross-family evidence is negative. Phi-3 fails, and TinyLlama/cached
   connector repairs do not close the gap.
3. The systems contribution is accounting only. It should not be sold as a
   latency, throughput, HBM, or GPU result.
4. Random anchors passing weakens any semantic-anchor interpretation.
5. Reproducibility still needs exact commands, model snapshot IDs, cache hashes,
   and output hashes.
6. Same-byte text is not enough as the only nonlatent compression baseline.

## Reviewer-Recommended Fixes Before Submission

- Add a direct source-choice/index baseline and source-choice text baseline.
- Report packet-only, receiver-fusion, target-only, and source-copy rows.
- Add full paired CIs for packet-vs-text and packet-vs-best-destructive
  controls, not only packet-vs-target.
- Add a compact rate curve over byte budgets and entropy-matched baselines.
- Move "packet mostly follows source choice" from caveat to explicit framing.
- Add a reviewer-facing reproducibility table with exact commands, input cache
  hashes, seeds, bootstrap settings, and expected output paths.
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
source-choice/index baseline and a strict receiver-family gate. The receiver
must beat target-only, packet-only/source-choice, same-byte text, and destructive
controls with paired confidence intervals.
