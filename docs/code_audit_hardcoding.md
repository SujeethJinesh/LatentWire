# Code Audit: Hardcoding And Determinism

Date: 2026-05-25

Scope: `release/src/`, `release/configs/`, `release/README.md`.

## Findings

| Severity | Finding | Status |
|---|---|---|
| SUBSTANTIAL | Paper claim values are hardcoded in `release/src/outlier_migrate/data.py`. | Intentional for FAST_VERIFY frozen-claim replay; documented as fast verification, not full reproduction. |
| SUBSTANTIAL | Full GPU runner paths, model checkpoint caches, and packet execution are not implemented in `release/`. | Documented; scripts now fail loudly outside dry-run / fast-verify. |
| MINOR | Model identifiers and bootstrap seeds live in YAML configs, not CLI flags. | Acceptable; configs are the intended reproducibility surface. |

## Determinism Check

- Bootstrap randomness uses `np.random.default_rng(seed)` in
  `metrics.bootstrap_ci`; seeds are caller-provided.
- Config files document model identifiers, trace counts, decode positions, and
  bootstrap seeds for the four model families.
- Reproduction scripts do not use random sources in FAST_VERIFY mode.
- `reproduce_all.sh` sets `PYTHONPATH` relative to the current `release/`
  directory and writes outputs under `results/`.

## Hardcoded Path Check

No hardcoded `/workspace`, `/tmp`, user account, or model-cache path is present
in `release/src` or `release/docs`. The verification log contains historical
clean-clone paths, which are expected audit evidence rather than executable
configuration.
