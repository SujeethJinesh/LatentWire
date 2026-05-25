# Code Audit: Reproducibility

Date: 2026-05-25

Scope: `release/src/scripts/`, `release/configs/`, `release/tests/`.

## Findings

| Severity | Finding | Status |
|---|---|---|
| SUBSTANTIAL | FAST_VERIFY validates frozen claim values, not raw-model recomputation. | Explicitly documented in README, reproducing guide, architecture overview, and VERIFICATION. |
| SUBSTANTIAL | Full checkpoint download and model load verification were not exercised. | Documented as not implemented in `release/`; no misleading full-mode pass remains. |
| MINOR | `reproduce_all.sh` uses the Granite config for all fast-verify claim families. | Acceptable because frozen claim subsets include all paper models; full-mode future work should dispatch per-model configs. |

## Script Trace

Each reproduction script:

1. parses `--config`, `--output-dir`, `--dry-run`, and `--fast-verify`;
2. loads the YAML config;
3. selects a fixed claim subset;
4. writes a deterministic JSON payload under `results/` or the requested output
   directory.

`--dry-run` now produces a config-valid payload without claims. `--fast-verify`
produces frozen claim values plus tolerance checks. No script silently performs
fast verification when full mode is requested.

## Test Quality

The tests cover:

- INT4 quantization range and reconstruction sanity;
- protected mask shape and exact budget;
- registry extension with a trivial channel-0 method;
- hand-computed recovery, set-leaving, and KL examples.

These tests are basic but non-trivial and run on CPU in under five minutes.
