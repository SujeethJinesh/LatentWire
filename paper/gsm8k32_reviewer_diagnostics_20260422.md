# GSM8K32 Reviewer Diagnostics

Date: `2026-04-22`

Candidate row: `dynalign_module_replace_residrank16`

## Why This Exists

Reviewer feedback correctly pushed on three questions that the frozen GSM8K32
contract had not answered yet:

1. Is the live `dynalign + resid16` row just copying a correct source answer?
2. Is `text_to_text` failing because the source model is actively poisoning the
   target with wrong answers?
3. Is there any verifier-side headroom left on this exact 32-example slice?

This note records the answers from the new diagnostics pass.

## Main Read

- `source_alone = 0.0312` (`1/32`)
- `target_alone = 0.0625` (`2/32`)
- `text_to_text = 0.0312` (`1/32`)
- `dynalign_module_replace_residrank16 = 0.1250` (`4/32`)
- oracle verification bound `max(target_alone, dynalign_module_replace_residrank16) = 0.1250`

## Interpretation

### 1. The live row is not explained by source-answer copying on this slice

`dynalign_module_replace_residrank16` wins on exactly `2/32` examples over
`target_alone`, with `0/32` losses.

On both of those latent-only win examples, `source_alone` is still wrong.

So on this exact slice, the live row is not just forwarding a correct source
final answer through a hidden channel.

### 2. The `text_to_text` regression is consistent with source-side poisoning

`text_to_text` drops below `target_alone` on this slice.

On both target-only `text_to_text` loss examples, `source_alone` is also wrong.

So the weak text relay is plausibly poisoning the target with incorrect source
reasoning rather than merely failing to help.

### 3. The slice is already oracle-saturated for the live row

Because the live row has `2` wins and `0` losses over `target_alone`, the
oracle bound `max(target_alone, dynalign_module_replace_residrank16)` is exactly
the live row itself: `4/32 = 0.1250`.

That means:

- there is no verifier-side headroom left on this exact 32-example slice
- this slice is still useful as a reproducible falsification and regression
  check
- but it is no longer the right place to hunt for additional selector or
  verifier gains

## Operational Consequence

Use GSM8K32 for:

- reproducibility
- exact paired comparisons
- fast falsification

Do not use GSM8K32 alone for:

- claiming further verifier-side method gains
- arguing headroom for the current live row
- deciding whether another tiny same-slice improvement is structural

The next evaluation investment should be:

1. a larger frozen slice
2. a cross-family falsification pair
3. only then further same-pair benchmark expansion

## Artifact

The raw local artifact from the new script lives under:

`results/gsm8k_contract_residual_rank16_dynalign_20260421/dynalign_module_replace_residrank16_diagnostics_20260422.{json,md}`
