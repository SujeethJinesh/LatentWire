# Prose Changes

Date: 2026-05-29

Scope: main paper TeX and `sections/decision_rule.tex`. The LLM Use Disclosure section was checked and left intact.

## Section-Level Edits

| Section | Changes |
|---|---|
| Abstract | Added DeepSeek tail-risk caveat next to the `0.756` median. Standardized Nemotron ParoQuant rounding to `1.05`. |
| Introduction | Added a short diagnostic/prescriptive framing paragraph before the contribution list. This makes the absence of a new universal quantizer explicit as a finding rather than an unstated gap. |
| Related Work | Tightened the DecDEC attribution to `around 30% or below` static-channel recall. Kept the ParoQuant baseline-not-our-method boundary. |
| Results | Standardized Nemotron ParoQuant wording to `1.05`. Kept DeepSeek caveat in prose. Corrected the tight-clip worst-trace table entry to `-26.37`. |
| Systems Cost | Added one sentence linking per-projection Table 9 numbers to the aggregate Granite top-8x32 estimate, and repeated that the energy model is analytical. |
| Limitations / Appendix | Replaced unsupported `preregistered` wording with `fixed` or `planned`; no threshold is now represented as preregistered unless the text separately supports it. |
| Conclusion | Replaced vague `substantial` wording with the measured `53--67%` set-leaving range. |

## Keyword Regression

Checked terms:

- `1.047`: 0 remaining instances in active paper/claim audit.
- `[0.254, 0.923]`: 0 remaining instances.
- `preregistered`: 0 remaining instances in active paper/claim audit.
- `worth noting`, `important to note`, `notably`, `interestingly`, `in order to`, `Furthermore`, `Moreover`, `Additionally`, `plays a crucial role`: 0 remaining instances.

`significant` remains only in the technical phrase `Holm-significant`; `substantially` remains only in the technical definition of within-set rank shuffling.

## Title Options

No title change was made because the requested instruction was to evaluate options and flag the tradeoff.

- Current title: `Channel-Set Drift in Long-Reasoning W4A16: Why Static Protection Fails and When Rotation Helps`
  - Accurate for the regime map; slight risk that `When Rotation Helps` sounds like a rotation-method contribution.
- Alternative: `Channel-Set Drift in Long-Reasoning W4A16: Why Static Protection Fails and Why Rotation Helps`
  - Stronger mechanism language; slightly repetitive.
- Alternative: `Channel-Set Drift in Long-Reasoning W4A16: Why Static Protection Fails Under Long Decode`
  - Safest against method-contribution confusion; weaker at foregrounding the rotation baseline result.

## Disclosure

The `LLM Use Disclosure` section remains present and unchanged in substance.
