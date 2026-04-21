# Current Bytes / Accuracy Frontier (2026-04-20)

| Split | Method | Family | Accuracy | Avg bytes | Notes |
|---|---|---|---:|---:|---|
| gsm8k_eval_70 | target-alone | control | 0.0429 | 0 | no source communication |
| gsm8k_eval_70 | text-to-text | control | 0.1000 | - | text communication baseline |
| gsm8k_eval_70 | shuffled fixed prior | selector-null | 0.0429 | 151,163.7 | query-blind prior null |
| gsm8k_eval_70 | fixed prior | selector | 0.0857 | 151,163.7 | best internal same-pair branch |
| gsm8k_eval_70 | grouped signature transport | transport | 0.0429 | 147,812.6 | best transport-only branch |
| gsm8k_eval_70 | grouped subspace + rank-4 residual | transport+correction | 0.0571 | 145,508.8 | best transport-plus-correction branch |
| gsm8k_eval_70 | bridge_ridge | bridge | 0.0429 | 295,614.9 | best bridge branch that survives held-out slices |
| gsm8k_eval_70 | QK-fidelity budget | query-conditioned selector | 0.0429 | 157,989.2 | runtime query-conditioned selector on top of best transport+correction checkpoint |
| gsm8k_eval_70 | KVComm-compatible replay | external comparator | 0.0000 | - | adjacent heterogeneous replay baseline |
| gsm8k_eval_70 | C2C | external comparator | 0.1286 | - | main external fair bar |
| gsm8k_eval_10_controlled | target-alone | control | 0.1000 | 0 | shared chat serialization + enable_thinking=false |
| gsm8k_eval_10_controlled | bridge_ridge | bridge | 0.1000 | 340,376.0 | fair controlled bridge baseline |
| gsm8k_eval_10_controlled | shared-plus-private asym adapter | modular bridge | 0.1000 | 681,668.4 | AsymLoRA-style shared bottleneck plus private K/V residual heads |
| gsm8k_eval_10_controlled | shared-plus-private dynmap adapter | modular bridge | 0.1000 | 681,668.4 | shared-plus-private bridge with context-reweighted top-k teacher |
| gsm8k_eval_10_controlled | xattn adapter | attention bridge | 0.1000 | 681,668.4 | tiny query-conditioned cross-attention bridge over live K/V-side memory signals |
| gsm8k_eval_10_controlled | xattn dynmap adapter | attention bridge | 0.1000 | 681,668.4 | xattn bridge plus context-reweighted top-k output teacher |
| gsm8k_eval_10_controlled | module adapter | attention bridge | 0.1000 | 681,668.4 | slotted attention-side transfer module with nonlinear readout and prediction distillation |
| gsm8k_eval_10_controlled | module replace | attention bridge | 0.1000 | 681,668.4 | slotted attention-side transfer module trained to predict full corrected K/V directly |
| gsm8k_eval_10_controlled | span-aligned module replace | token-remapped attention bridge | 0.1000 | 681,668.4 | direct-output slotted module fit from raw-prompt monotone span-aligned calibration pairs |
| gsm8k_eval_10_controlled | byte-span module replace | token-remapped attention bridge | 0.1000 | 681,668.4 | direct-output slotted module fit from dominant UTF-8 byte-overlap calibration pairs on a 16-prompt diagnostic slice |
| gsm8k_eval_10_controlled | dynamic-aligned context-only module replace | token-remapped attention bridge | 0.1000 | 681,668.4 | matched dynalign null with the same candidate window but prediction-overlap scoring disabled on a 16-prompt diagnostic slice |
| gsm8k_eval_10_controlled | dynamic-aligned module replace | token-remapped attention bridge | 0.1000 | 681,668.4 | direct-output slotted module fit from context-plus-output-overlap token mixtures |
| gsm8k_eval_10_controlled | dynamic-aligned DWA module replace | token-remapped attention bridge | 0.1000 | 681,668.4 | dynalign module replace plus confidence-weighted samples and dynamic prediction teacher on a 16-prompt diagnostic slice |
| gsm8k_eval_10_controlled | dynamic-aligned likelihood module replace | token-remapped attention bridge | 0.1000 | 681,668.4 | dynalign plus target next-token likelihood teacher and confidence weights on a 16-prompt diagnostic slice |
| gsm8k_eval_10_controlled | dynamic-aligned span-ALM module replace | token-remapped attention bridge | 0.1000 | 681,668.4 | dynalign plus span-window approximate-likelihood teacher and confidence weights on a 16-prompt diagnostic slice |
| gsm8k_eval_10_controlled | dynamic-aligned DWA-interaction module replace | token-remapped attention bridge | 0.1000 | 681,668.4 | dynalign plus confidence-weighted dynamic prediction teacher and prompt-local interaction distillation on a 16-prompt diagnostic slice |
| gsm8k_eval_10_controlled | dynamic-aligned preference-distilled module replace | token-remapped attention bridge | 0.1000 | 681,668.4 | dynalign plus confidence-weighted dynamic prediction teacher and pairwise preference distillation over aligned target output rows on a 16-prompt diagnostic slice |
| gsm8k_eval_10_controlled | dynamic-aligned preference-distilled attention-stratified selector | selector ablation | 0.1000 | 681,668.4 | same dynalign preference-distilled checkpoint with four-bin attention-stratified position selection to test route-collapse coverage |
| gsm8k_eval_10_controlled | readout adapter | stronger-teacher bridge | 0.0000 | 681,668.4 | prompt-local attention-readout teacher; survives GSM5 smoke but drops below the controlled target-alone floor |
| gsm8k_eval_10_controlled | dynamic-aligned interaction module replace | token-remapped attention bridge | 0.1000 | 681,668.4 | dynalign module replace plus prompt-local interaction distillation on a 16-prompt diagnostic slice |
| gsm8k_eval_10_controlled | token-basis replace | token-native attention bridge | 0.1000 | 681,668.4 | slotted attention-side module constrained to a basis distilled from target next-token output rows |
| gsm8k_eval_10_controlled | grouped rotational transport | geometry | 0.1000 | 681,668.4 | first geometry-side branch to survive the controlled slice |
| gsm8k_eval_10_controlled | grouped fitted rotation transport | geometry | 0.1000 | 681,668.4 | calibration-fit gauge-fixing follow-up |
| gsm8k_eval_10_controlled | grouped shared-basis transport | geometry | 0.1000 | 681,668.4 | shared-basis coefficient-space transport |
| gsm8k_eval_10_controlled | KVPress no-press | external comparator | 0.1000 | - | exact external KVPress harness floor |
| gsm8k_eval_10_controlled | KVPress ExpectedAttentionPress | external comparator | 0.1000 | - | exact external Expected Attention comparator |
| gsm8k_5_controlled_smoke | shared-plus-private asym adapter | modular bridge | 0.2000 | 686,026.6 | AsymLoRA-style shared-plus-private bridge survives smoke and controlled slice |
| gsm8k_5_controlled_smoke | shared-plus-private dynmap adapter | modular bridge | 0.2000 | 686,026.6 | shared-plus-private bridge with context-reweighted top-k teacher |
| gsm8k_5_controlled_smoke | xattn adapter | attention bridge | 0.2000 | 686,026.6 | tiny query-conditioned cross-attention bridge over live K/V-side memory signals |
| gsm8k_5_controlled_smoke | xattn dynmap adapter | attention bridge | 0.2000 | 686,026.6 | xattn bridge plus context-reweighted top-k output teacher |
| gsm8k_5_controlled_smoke | module adapter | attention bridge | 0.2000 | 686,026.6 | slotted attention-side transfer module with nonlinear readout and prediction distillation |
| gsm8k_5_controlled_smoke | module replace | attention bridge | 0.2000 | 686,026.6 | slotted attention-side transfer module trained to predict full corrected K/V directly |
| gsm8k_5_controlled_smoke | span-aligned module replace | token-remapped attention bridge | 0.2000 | 686,026.6 | direct-output slotted module fit from raw-prompt monotone span-aligned calibration pairs |
| gsm8k_5_controlled_smoke | byte-span module replace | token-remapped attention bridge | 0.2000 | 686,026.6 | direct-output slotted module fit from dominant UTF-8 byte-overlap calibration pairs on a 16-prompt diagnostic slice |
| gsm8k_5_controlled_smoke | dynamic-aligned context-only module replace | token-remapped attention bridge | 0.2000 | 686,026.6 | matched dynalign null with the same candidate window but prediction-overlap scoring disabled on a 16-prompt diagnostic slice |
| gsm8k_5_controlled_smoke | contextual-aligned module replace | token-remapped attention bridge | 0.0000 | 686,026.6 | direct-output slotted module fit from context-weighted source-to-target token mixtures |
| gsm8k_5_controlled_smoke | dynamic-aligned module replace | token-remapped attention bridge | 0.4000 | 686,026.6 | direct-output slotted module fit from context-plus-output-overlap token mixtures |
| gsm8k_5_controlled_smoke | dynamic-aligned DWA module replace | token-remapped attention bridge | 0.4000 | 686,026.6 | dynalign module replace plus confidence-weighted samples and dynamic prediction teacher on a 16-prompt diagnostic slice |
| gsm8k_5_controlled_smoke | dynamic-aligned likelihood module replace | token-remapped attention bridge | 0.2000 | 686,026.6 | dynalign plus target next-token likelihood teacher and confidence weights on a 16-prompt diagnostic slice |
| gsm8k_5_controlled_smoke | dynamic-aligned span-ALM module replace | token-remapped attention bridge | 0.2000 | 686,026.6 | dynalign plus span-window approximate-likelihood teacher and confidence weights on a 16-prompt diagnostic slice |
| gsm8k_5_controlled_smoke | dynamic-aligned DWA-interaction module replace | token-remapped attention bridge | 0.2000 | 686,026.6 | dynalign plus confidence-weighted dynamic prediction teacher and prompt-local interaction distillation on a 16-prompt diagnostic slice |
| gsm8k_5_controlled_smoke | dynamic-aligned preference-distilled module replace | token-remapped attention bridge | 0.4000 | 686,026.6 | dynalign plus confidence-weighted dynamic prediction teacher and pairwise preference distillation over aligned target output rows on a 16-prompt diagnostic slice |
| gsm8k_5_controlled_smoke | dynamic-aligned preference-distilled attention-stratified selector | selector ablation | 0.2000 | 686,026.6 | four-bin attention-stratified selector over the same preference-distilled checkpoint; tests whether broader prompt coverage fixes selector collapse |
| gsm8k_5_controlled_smoke | dynamic-aligned top-5 layer knockout | layer-localization ablation | 0.2000 | 563,521.8 | dynalign module replace with translated signal removed from recurrent top layer-localization signature L27,L5,L23,L22,L8 |
| gsm8k_5_controlled_smoke | dynamic-aligned offset-5 layer knockout | layer-localization ablation | 0.2000 | 563,521.8 | matched offset-layer knockout L26,L4,L21,L20,L7 for broad layer-budget sensitivity control |
| gsm8k_5_controlled_smoke | dynamic-aligned interaction module replace | token-remapped attention bridge | 0.2000 | 686,026.6 | dynalign module replace plus prompt-local interaction distillation on a 16-prompt diagnostic slice |
| gsm8k_5_controlled_smoke | token-basis replace | token-native attention bridge | 0.2000 | 686,026.6 | direct-output slotted module constrained to a target next-token output basis |
| gsm8k_5_controlled_smoke | shared-plus-private asym projector | projector bridge | 0.0000 | 686,026.6 | shared-plus-private post-transport projector combining full-rank query projector with the paired K/V interface |
| gsm8k_5_controlled_smoke | readout adapter | stronger-teacher bridge | 0.2000 | 686,026.6 | stronger prompt-local teacher survives smoke only |
| gsm8k_5_controlled_smoke | prediction-KL adapter | stronger-teacher bridge | 0.0000 | 722,107.7 | first prediction-level bridge teacher |
| gsm8k_5_controlled_smoke | prediction-KL bank | stronger-teacher bridge | 0.0000 | 722,107.7 | small modular bank follow-up to prediction-level teacher |
