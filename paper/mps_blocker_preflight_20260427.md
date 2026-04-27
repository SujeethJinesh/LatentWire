# MPS Blocker Preflight

- date: `2026-04-27`
- status: `hard_blocker_confirmed_preflight_added`

## Cycle Header

1. Current ICLR readiness and distance: not ICLR-ready; no live positive method
   survives the required source-destroying controls.
2. Current paper story: the credible next story is still stronger-source
   answer-masked side information followed by erasure-aware learned syndrome or
   zero-init query bottleneck, but it cannot advance while MPS is blocked.
3. Exact blocker to submission: PID `31103` remains orphaned under PID `1` in
   `STAT=UE` and is still the MPS `scripts/calibrate.py --device mps` process.
4. Current live branch: none while this blocker persists.
5. Highest-priority gate: make the MPS blocker check executable and
   machine-readable before stopping.
6. Scale-up rung: operational hard-blocker preflight.

## What Changed

Added:

- `scripts/check_mps_blocker.py`
- `tests/test_check_mps_blocker.py`

The helper checks PID `31103` by default and can be used as a preflight before
MPS runs:

```bash
./venv_arm64/bin/python scripts/check_mps_blocker.py --json
```

To fail fast in shell scripts:

```bash
./venv_arm64/bin/python scripts/check_mps_blocker.py --require-clear
```

## Current Output

```json
{
  "blocked": true,
  "next_action": "use_cpu_only_or_clear_os_session",
  "pid": 31103,
  "present": true
}
```

The full command remains the stuck `scripts/calibrate.py ... --device mps`
process.

## Tests

```bash
./venv_arm64/bin/python -m pytest tests/test_check_mps_blocker.py -q
```

Result: `3 passed`.

```bash
./venv_arm64/bin/python -m py_compile scripts/check_mps_blocker.py
```

Result: passed.

## Decision

No method branch is promoted. The hard blocker remains OS/session-level cleanup
of PID `31103`.

Next exact command:

```bash
./venv_arm64/bin/python scripts/check_mps_blocker.py --json
```

Proceed with MPS only when it reports `"blocked": false`.
