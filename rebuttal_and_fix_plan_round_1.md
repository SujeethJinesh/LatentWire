# Rebuttal and Fix Plan Round 1

Date: 2026-05-29

## Distinct Weaknesses and Actions

| Weakness | Class | Fix / response |
|---|---|---|
| Novelty versus ParoQuant | HONEST-LIMITATION + FIXABLE-IN-PROSE | Keep ParoQuant baseline-not-our-method language. Added explicit local ParoQuant-style implementation caveat. Rebuttal: novelty is decode-time channel identity drift measurement, matched-control failure taxonomy, and regime checklist, not a new rotation quantizer. |
| Protocol not validated | FIXABLE-IN-PROSE | Changed “chooses/selects” language to “provisionally suggests” and “checklist”; stated local confirmation packet is required. |
| Analytical-only systems evidence | HONEST-LIMITATION + FIXABLE-IN-TABLE | Added aggregate systems-cost row and cache/batching/device-agnostic caveat. Rebuttal: envelope is a constraint for future residual selectors after K-RES failure, not a claimed implemented system. |
| Small-N and wide CIs | HONEST-LIMITATION + FIXABLE-IN-PROSE | Strengthened small-N/no-gap caveat and clarified descriptive margins. Keep negative CIs visible. |
| No-gap estimand shifts | FIXABLE-IN-PROSE | Added statement that early gates and later positive-gap packets are not identical population quantities and should not be compared as such. |
| MATH-500 overread | FIXABLE-IN-PROSE | Added that MATH-500 supports drift generalization, not intervention recovery generalization. |
| ParoQuant-style fidelity | FIXABLE-IN-PROSE | Added local implementation caveat: reported scaled pairwise-rotation design, not official code. |
| DecDEC proxy fairness | HONEST-LIMITATION | Keep proxy terminology in Table 1 and text; do not claim full DecDEC implementation. |
| Quamba2 comparison overread | HONEST-LIMITATION | Existing surface-distinction framing remains. Do not strengthen beyond block-output top-K protection. |
| Granite-only composition overgeneralization | FIXABLE-IN-PROSE | Added “in the tested Granite composition packet.” |

## Anticipated Reviewer Responses

**No new method beyond ParoQuant.** Correct; the paper is not a new quantizer paper. The contribution is the measured long-decode failure mode, controls showing why channel-set remedies fail, and a descriptive calibration checklist. ParoQuant is the strongest prior-work baseline and helps explain why basis removal works better than chasing channel identities.

**Systems numbers are only analytical.** Correct; the paper now states this wherever the energy numbers are used. The envelope is still useful because it bounds what any future residual-correction selector must justify and shows that selector quality, not only kernel feasibility, is the bottleneck.

**Small models and small trace counts.** Correct; this remains a limitation. The defense is matched controls, visible CIs, BCa/Holm correction for the table family, cross-prompt drift replication, and restraint: protocol evaluation is descriptive, not validated.

**Just ParoQuant.** The paper does not claim ParoQuant. It uses ParoQuant-style rotation as a strong baseline to show the mechanism: basis conditioning handles a signal that original-basis channel tracking cannot reliably exploit.

**Quamba2 contradiction.** The comparison is at the measured block-output top-K surface, plus a small Granite internal-surface screen. It does not deny Quamba2 internal SSM persistence.

## Status

Round 1 fixes were applied to the paper source. They do not change numerical claims or scope.
