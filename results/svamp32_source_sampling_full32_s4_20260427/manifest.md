# SVAMP32 Full32 Source Sampling S4

- date: `2026-04-27`
- status: `source_reachability_pass_total_oracle_regression`
- git commit at run time: `7fa3f2b35a3dacd1bb789f7a3b2c563e5bb6d45a`
- scale rung: `smoke`
- model: `Qwen/Qwen2.5-Math-1.5B`
- prompt mode: `source_reasoning`
- source reasoning mode: `brief_analysis`
- eval file: `results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl`
- samples per ID: `4`
- sample rows: `128`
- sample SHA256: `3520951745cd92b53ff8bbe01d3b4b9d47d5985027ab3b5abc4e9d0b247fb18b`

## Metrics

- source-sampled candidate oracle: `10/32`
- target/no-source full32 S8 baseline oracle: `14/32`
- source minus target/no-source oracle: `-4`
- new source oracle IDs beyond target/no-source: `5`
- lost target/no-source oracle IDs: `9`
- C2C clean residual in source pool: `3/6`
- new C2C clean residual IDs beyond target/no-source: `2`
- C2C teacher-only IDs in source pool: `4/9`
- source-contrastive clean IDs in source pool: `1/4`
- mean unique sampled answers per ID: `3.406`
- duplicate nonempty row fraction: `0.148`

## New C2C-Clean Residual IDs

- `6e9745b37ab6fc45`
- `de1bf4d142544e5b`

## Interpretation

This is not positive communication evidence. The source-sampled pool is weaker
than the target/no-source pool in total oracle reachability (`10/32` vs
`14/32`). Its value is sharper: it exposes two C2C-clean residual gold answers
that the target/no-source full32 S8 pool did not reach. Those two IDs are now
the strict smoke surface for a source-destroyable selector or connector.

The raw rows retain the inherited generic sampler method names
`target_sample_s0` to `target_sample_s3`; the artifact metadata records
`prompt_mode=source_reasoning`, `source_reasoning_mode=brief_analysis`, and
`model=Qwen/Qwen2.5-Math-1.5B`.

## Decision

Pass only as source-surface discovery. Do not claim a method. The next gate must
show matched source selects or generates at least one of the two new clean IDs
while zero-source, shuffled-source, target-only/slots-only, random same-byte,
answer-only, and answer-masked controls miss it.

## Artifacts

- `source_samples.jsonl`
- `source_samples.json`
- `source_samples.md`
- `reachability.json`
- `reachability.md`
- `source_vs_target_reachability.json`
- `source_vs_target_reachability.md`
