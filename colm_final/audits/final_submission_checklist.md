# Final Submission Checklist

Date: 2026-05-05

## Ready

- [x] Official COLM style files are present in `paper/`.
- [x] The compiled paper PDF is present in `paper/latentwire_colm2026.pdf`.
- [x] Figures are present in `paper/figures/`.
- [x] Evidence directories used by paper claims are copied into
  `evidence/results/`.
- [x] Reproduction scripts and targeted tests are copied into `code/`.
- [x] Full repository tests and targeted COLM tests pass.
- [x] LaTeX build completes from `colm_final/paper`.
- [x] Citation metadata for high-risk recent papers has been audited against
  primary sources.
- [x] Reviewer-risk panel is recorded.
- [x] Late HellaSwag diagnostics are packaged separately as excluded evidence,
  not silently mixed into the current PDF claim set.
- [x] Source-choice preservation is surfaced in the paper instead of hidden as
  an appendix caveat.
- [x] Explicit source-index/source-rank audit is packaged and included in the
  main PDF table.
- [x] Packet payload rate curve is packaged and included as a PDF figure.
- [x] COLM_v3 claim audit is generated and mirrored in the appendix.
- [x] Source-private threat model is integrated into the PDF.
- [x] Control-suite table is integrated into the PDF.
- [x] Related-work/baseline matrix is integrated into the PDF.
- [x] Systems table distinguishes measured/accounted packet objects from
  analytical KV/cache byte floors and native-pending rows.
- [x] Native NVIDIA runbook exists for future hard systems measurements.

## Not Yet Ready / Must Be Framed Carefully

- [ ] Human copyedit and page-budget review remain before final submission.
- [ ] No strict cross-family positive is available.
- [ ] No native systems throughput row is available.
- [ ] The strict positive-beyond-source-index gate fails; this is acceptable
  only under the current scoped COLM framing.
- [ ] Full command/hash provenance is in `audits/reproducibility_report.md`, but
  the PDF appendix only summarizes it compactly.
- [ ] Calibrated source-score-vector quantization is not available for the
  headline frozen caches.
- [ ] The integrated PDF is now 11 pages; final workshop page limits may require
  moving the claim-audit or baseline matrix out of the main submission.

## Submission Decision

The bundle is now a COLM_v3 integrated draft suitable for internal/reviewer
circulation after a human copyedit and page-budget pass. It should be pitched as
a practical packet protocol and evaluation framework with narrow positives,
failure boundaries, and byte/exposure systems accounting. It is not ready for an
ICLR full-paper claim without the next receiver-family/cross-family positive and
stronger native systems evidence.
