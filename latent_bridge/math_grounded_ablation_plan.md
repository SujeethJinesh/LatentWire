# Math-Grounded Ablation Plan

This addendum turns the post-ARC result into concrete tests. The current signal is:

- GSM8K generation: latent/KV protocols underperform target and text-to-text.
- ARC-Challenge MCQ: quantized fused KV is slightly above target on the small split, while no-quant fused KV ties target.

The working hypothesis is therefore narrower than "latent reasoning transfers": quantized, aligned KV may preserve or regularize compact knowledge/choice information, but the current protocol does not yet transfer multi-step reasoning traces.

## Highest-Priority Confirmation

1. Run a larger ARC-Challenge confirmation with `cka_half_seed1`, `fused_quant_brief`, `fused_noquant_brief`, `target`, and `text_to_text`.
2. Repeat with at least one additional seed, ideally `cka_half_seed0`.
3. Add negative controls: identity alignment, no rotation, shifted/random layer maps, and random/zero/shuffled source KV.

Pass condition: quantized fused KV should beat target by more than sampling noise and fail under the negative controls.

## Representation-Similarity Ablations

Motivation: CKA/SVCCA/PWCCA test whether the source and target layers are actually geometrically compatible.

Tests:

- Compare layer pairing by `interp`, linear CKA, RBF/kernel CKA, SVCCA, and PWCCA.
- Log per-layer pre/post alignment similarity and correlate it with downstream accuracy.
- Add a "pairing confidence" threshold: only transmit layers whose similarity exceeds a cutoff.

What it proves or disproves:

- If higher similarity predicts downstream gains, RotAlign is using real representational alignment.
- If not, the ARC gain may be noise or a quantization artifact.

## Rotation And Transform Ablations

Motivation: Fourier, Hadamard, butterfly, and random orthogonal transforms all spread outliers and make coordinates more quantization-friendly.

Tests:

- Rotation family: identity, random orthogonal, Hadamard, block-Hadamard, Fourier/DCT, butterfly.
- Rotation granularity: full hidden dimension, per-head, grouped-head, per-layer.
- Rotation learnedness: fixed random rotation vs small learned/procrustes-refined rotation.
- Rotation seed robustness: at least 5 seeds on the ARC confirmation split.

What it proves or disproves:

- If only quantized rotated KV helps, the useful effect may be incoherence/outlier smoothing rather than semantic alignment alone.
- If full-precision and quantized variants both help, alignment is likely doing more than compression regularization.

Implemented now:

- `--rotation dct` and the `dct_cka_half_seed1` control-suite spec cover the Fourier/DCT transform family.
- `shifted_layers_half_seed1` and `random_layers_half_seed1` are layer-map negative controls.
- `no_rotation_cka_half_seed1`, `hadamard_cka_half_seed1`, and seeded CKA specs cover the first robustness pass.

## Quantization Ablations

Motivation: The ARC run suggests quantization can improve the fused result rather than merely compressing it.

Tests:

- Bits: no-quant, 8, 6, 4, 3, 2.
- Quantizer: scalar Lloyd-Max, uniform, per-channel scale, product quantization, residual/bias-corrected quantization.
- Calibrate quantizer on source KV only vs translated KV vs target KV.
- Add a "quantization noise only" control: inject matched noise without discrete quantization.

What it proves or disproves:

- If moderate quantization helps but matched Gaussian noise does not, structured quantization is acting like a useful regularizer.
- If noise also helps, the result may be stochastic smoothing rather than latent communication.

Implemented now:

- `fused_quant_noise_brief` runs `--quantization-control matched_noise`.
- `bits2_`, `bits3_`, `bits6_`, and `bits8_` calibration specs bracket the current 4-bit default.
- `target_attenuation_brief` and `fused_quant_random_translated_brief` test
  whether the gain remains when target-space translated KVs are removed or
  replaced after translation. The target-attenuation control reports zero
  transmitted latent bytes because no source KV is sent.

## Linear-Algebra Structure Ablations

Motivation: KV spaces may align by head, low-rank subspace, or principal angles rather than one global orthogonal map.

Tests:

- Alignment map: global Procrustes, per-head Procrustes, grouped-head Procrustes, low-rank Procrustes, CCA/reduced-rank regression.
- Rank sweep: full rank, 75%, 50%, 25%, 10%.
- Principal-angle diagnostics between source and target K/V subspaces.
- Whitening modes: none, source-only, target-only, both, shrinkage whitening.

What it proves or disproves:

- Per-head wins would support the KVTC-style head-structured alignment story.
- Low-rank wins would suggest only a compact shared subspace is transferable.

## Statistical Robustness

Motivation: The current ARC gain is one question on 35 examples, so statistical controls are mandatory.

Tests:

- Bootstrap confidence intervals over examples.
- McNemar paired test against target and text-to-text predictions.
- Report per-example agreement: target correct/RotAlign wrong, target wrong/RotAlign correct, both correct, both wrong.
- Use fixed splits and record prompt/model hashes.

What it proves or disproves:

- A one-question gain without paired significance is only a lead.
- A consistent paired improvement under bootstrap/McNemar is publishable evidence.

Implemented now:

- `--prediction-output` writes per-example JSONL predictions.
- Evaluation summaries include paired bootstrap deltas and McNemar counts against target and text-to-text.
- `scripts/compare_prediction_files.py` compares separate prediction files on
  paired examples, which is required for real translated KV versus zero-byte
  target attenuation.

## Failure Analysis

Tests:

- Save per-example predictions for target, text-to-text, fused quantized, fused no-quant.
- Categorize examples where quantized fused KV flips target from wrong to right.
- Compare answer entropy/logit margins before and after KV fusion.
- Track whether improvements come from more confident correct answers or accidental option flips.

What it proves or disproves:

- If gains are margin-improving on semantically related questions, the mechanism is plausible.
- If gains are random option flips, the method needs redesign.
