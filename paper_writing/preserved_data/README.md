# Preserved Runs

## phase1_full_20251116_201212
- Config: Phase1 full stack (KL+prompt+RoPE, decode loss disabled).
- Result: Peak bridged 0.680, final 0.645 (source 0.540, target 0.770).
- Reason: Primary baseline for paper; referenced in analysis.

## ablB_20251116_234242
- Config: KL-only ablation (prompt weight 0.05, no RoPE, decode off).
- Result: Final bridged 0.625.
- Reason: Shows KL alone is insufficient; required for ablation table.

## ablC_20251117_013909
- Config: KL + prompt-weight drop (no RoPE, decode off).
- Result: Final bridged 0.655.
- Reason: Demonstrates prompt alignment is the dominant factor; near-publishable accuracy.
