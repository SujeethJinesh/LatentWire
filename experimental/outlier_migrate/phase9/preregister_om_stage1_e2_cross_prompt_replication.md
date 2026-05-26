# Stage 1 E2 Cross-Prompt Replication Preregistration

Status: frozen before Stage 1 E2 inference.

## Question

Does the strict top-1% channel-set leaving effect measured on AIME-2025
generalize to adjacent reasoning benchmarks?

## Scope

Models:

- `ibm-granite/granite-4.0-h-small`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- `tiiuae/Falcon-H1-0.5B-Instruct`

Nemotron-3 is excluded from E2 because measured throughput would dominate the
15 GPU-hour cap. Nemotron remains covered by the AIME-2025 packet.

Prompts:

- 30 MATH-500 prompts
- 20 GPQA-Diamond prompts

If GPQA-Diamond loading is unavailable, the runner may be invoked with
`--math-only`. This is an explicit fallback and must be recorded in
`command_metadata.json`; GPQA is never silently omitted.

## Measurement

For each model and prompt, capture post-block residual output magnitudes at
decode positions 100 and 20000 under deterministic greedy decoding. The
metric is strict set-leaving: channels ranked in the top 1% at position 100
whose rank is outside the top 1% at position 20000. Aggregate by mean over
layers per trace, then mean over traces. Bootstrap uses trace resampling.

## Reference Values

AIME-2025 strict set-leaving references:

- Granite-Small: 0.566234756098
- DeepSeek-R1-Distill-Qwen-1.5B: 0.670572916667
- Falcon-H1: 0.673611111111

## Decision Rule

For each model, compare the cross-prompt strict set-leaving aggregate against
the corresponding AIME-2025 reference.

- Model replicates if absolute difference is at most 0.15.
- PASS_E2: all three measured models replicate.
- AMBIGUOUS_E2: one or two measured models replicate.
- KILL_E2: zero measured models replicate.

If fewer than three models complete because the cap is exhausted, the checker
reports `INCOMPLETE_E2_CAP_EXHAUSTED` rather than a scientific decision.

## Runtime Discipline

Cap: 15 GPU hours. The runner checks the cap before starting each model. If
the cap is exhausted, it writes the partial packet and exits without starting
another model.
