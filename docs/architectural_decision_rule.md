# Architectural Decision Rule for W4A16 Long-Decode Protection

Date: 2026-05-27

## Purpose

The Stage 1 and V1/V2 packets show that neither channel-set protection nor
rotation is a universal remedy across the measured models. This note turns the
evidence into a conservative model-selection rule for W4A16 long-reasoning
deployment.

## Evidence Table

| Model | Architecture class | AIME set-leaving | MATH-500 set-leaving | FFT/KL diagnostic | No-gap fraction in M11b packet | M11b top-10 recovery | Static top-10 recovery | ParoQuant recovery | Source |
|---|---|---:|---:|---|---:|---:|---:|---:|---|
| Granite-4.0-H-Small | Hybrid Mamba-2 | 0.566 | 0.561 | Granite dense KL: square-root fit best; AR residuals 0.44-0.51 across regimes | 0.333 | 0.241 CI [-0.727, 0.436]; top-5 0.449 CI [-1.30, 1.00] | -0.052 CI [-53.4, 0.432] | 0.754 CI [0.477, 1.00] | `om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z`, `om_paroquant_granite_small_20260520T1555Z`, `om_stage1_e2_mathonly_20260526T1202Z` |
| Nemotron-3-Nano-30B-A3B | Hybrid Mamba-2 + MoE | 0.534 | not rerun | Dense KL/FFT deferred for throughput | 0.167 | 0.815 CI [0.254, 0.923] | 0.594 CI [0.317, 0.814] | V1 deferred after 2/12 traces due 5h cap | `om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z`, `om_v1_paroquant_nemotron_20260527T1125Z` |
| DeepSeek-R1-Distill-Qwen-1.5B | Transformer | 0.671 | 0.651 | square-root KL fit in all regimes; entropy 0.850; autocorr 100 tokens | 0.083 | 0.335 CI [-0.409, 0.494] | 0.377 CI [-1.14, 0.538] | not measured | `om_stage1_e1_deepseek_falcon_20260526T2335Z`, `om_v2_m11b_deepseek_20260527T1210Z` |
| Falcon-H1-0.5B | Parallel hybrid | 0.674 | 0.675 | square-root KL fit in all regimes; entropy 0.891; autocorr 100 tokens | 0.000 | 0.044 CI [-0.144, 0.203] | -0.025 CI [-0.126, 0.172] | not measured | `om_stage1_e1_deepseek_falcon_20260526T2335Z`, `om_v2_m11b_falcon_20260527T1438Z` |

## Conservative Rule

The validated rule is a gate-before-deploy policy, not a universal predictor:

1. Measure strict set-leaving at the target horizon. If the rate is below the
   static-policy tolerance for the deployment, static protection may be
   sufficient. All four measured long-reasoning models exceed 0.53, so this
   gate triggers method validation in the current paper.
2. Run a matched-budget EMA gate: M11b top-10 must achieve median recovery at
   least 0.30, CI lower above 0.10, and a median margin of at least 0.15 over
   static top-10. Nemotron passes; DeepSeek and Falcon do not.
3. When a rotation baseline is available, compare it directly rather than
   stacking it with M11b. Granite favors ParoQuant over M11b, and the direct
   ParoQuant+M11b composition is sub-additive.
4. If high drift persists but both M11b and rotation are unvalidated on the
   target model, do not deploy channel-set EMA as a positive method. Treat the
   result as diagnostic evidence and run a model-local validation packet.

## Interpretation

The rule explains the current evidence without overclaiming:

- Granite: choose rotation where available. ParoQuant recovers 0.754, while
  M11b is positive only with wide uncertainty and composition underperforms.
- Nemotron: choose budgeted EMA if rotation is unavailable or too expensive.
  M11b top-10 clears the preregistered gate and beats static top-10 by 0.220.
- DeepSeek and Falcon: do not claim M11b generalization. Both have high
  set-leaving and replicated KL/FFT diagnostics, but M11b top-10 remains
  ambiguous against the matched static control.

## Paper Integration

Add the rule as a short Discussion subsection after the composition result.
The main body should call it a conservative selection rule, while Appendix B
should carry the V1/V2 baseline-vetting rows. The Limitations section should
state that ParoQuant-on-Nemotron was throughput-deferred and that M11b did not
replicate as a broad architecture-agnostic positive method on DeepSeek/Falcon.
