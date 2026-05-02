# Final Submission Checklist

Date: 2026-05-02

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

## Not Yet Ready / Must Be Framed Carefully

- [ ] No strict cross-family positive is available.
- [ ] No native systems throughput row is available.
- [ ] No direct source-choice/index baseline is included.
- [ ] Full command/hash provenance is in `audits/reproducibility_report.md`, but
  the PDF appendix only summarizes it compactly.
- [ ] Same-byte text is not a complete compression baseline suite.

## Submission Decision

The bundle is reasonable for a COLM workshop submission if the paper is pitched
as a narrow positive packet protocol with explicit failure boundaries. It is not
ready for an ICLR full-paper claim without the next receiver-family/cross-family
positive and stronger systems evidence.
