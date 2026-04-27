# Source-Headroom Surface Scan Manifest

- date: `2026-04-26`
- status: `surface_scan_completed`
- min source-only threshold: `5`

## Top Read

The strongest existing source-complementary surface is
`qwen25math_qwen3_svamp70`:

- target: `21/70`
- source: `13/70`
- source-only over target: `9`
- target/source oracle: `30/70`
- exact ordered ID parity: `true`

The disjoint holdout also has source-only headroom but is weaker:

- target: `8/70`
- source: `8/70`
- source-only over target: `6`
- target/source oracle: `14/70`

## Files

- `scan.json`
  - sha256: `9611574620e91181a029e1b60165555bba8234ebbb02fcb78748d7ced52b4a6b`
- `scan.md`
  - sha256: `421f4bdf2a90c636e41da4f90f05c5aac0fa49bea5a5c21f28ceac0c64755afd`
- `scan_after_query_memory_prune.json`
  - sha256: `3ebd8aff86b732ca6b4137fb11a9306ba3b92b2e9be5a4a9d531dfefb7875b0b`
- `scan_after_query_memory_prune.md`
  - sha256: `0738cfe945fa4925983334801eedaa7ba24e7330e25c32bbbc6dd1c80b1c2f54`

## Decision

Surface discovery confirms the SVAMP70 source-sidecar surface was a reasonable
medium branch, but holdout testing of fixed guards fails. Next work should
avoid fixed length/numeric guard tuning unless a new router feature family is
introduced and evaluated with a frozen holdout gate.

After query-memory pruning, the re-scan still ranks `svamp70_live_source` and
`svamp70_holdout_source` as the only strong source-complementary surfaces. The
next branch should use these as live/holdout decision surfaces, but should not
reuse fixed decoded guards, shallow source-text routers, tiny prefix emitters,
or Perceiver target-memory checkpoints.
