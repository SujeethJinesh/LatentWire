# Candidate-Syndrome Decoder 20260427

Status: `candidate_syndrome_decoder_fails_smoke`

Command:

```bash
./venv_arm64/bin/python scripts/analyze_candidate_syndrome_decoder.py \
  --live-target-set results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/source_contrastive_target_set.json \
  --holdout-target-set results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_contrastive_target_set.json \
  --output-dir results/candidate_syndrome_decoder_20260427 \
  --controls zero_source shuffled_source random_syndrome target_only slots_only \
  --run-date 2026-04-27
```

Inputs:

- Live target set:
  `results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/source_contrastive_target_set.json`
- Holdout target set:
  `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_contrastive_target_set.json`

Outputs:

- `candidate_syndrome_decoder_probe.json`
  - sha256:
    `2ae78c4f3c31cf674f334fe5f755d6f80a8ccf3c66e777a49d4daed01c25cc81`
- `candidate_syndrome_decoder_probe.md`
  - sha256:
    `719990b7b4dff43278920cbbdbc807e4bdf2c359accc80dc81bf38bcf2f5a4f5`

Summary:

- Live: matched clean source-necessary `1`, target-self harms `17`, control
  clean union `0`.
- Holdout: matched clean source-necessary `4`, target-self harms `14`, control
  clean union `0`.

Decision:

Do not promote the numeric hash-syndrome artifact probe. Candidate-syndrome
decoding remains worth revisiting only with learned source predicates or a
stronger source surface.
