# Code Audit: Error Handling And Defensive Code

Date: 2026-05-25

Scope: `release/src/`.

## Findings

| Severity | Finding | Status |
|---|---|---|
| CRITICAL | Full-mode invocation previously completed successfully without doing full GPU reproduction. | Fixed: full mode raises `RuntimeError` with a direct explanation. |
| MINOR | `methods.build_mask` catches `KeyError` only to add method context. | Acceptable; exception is re-raised and not swallowed. |
| MINOR | Tests use `assert`, but release library code does not rely on assertions for runtime validation. | Acceptable for pytest tests. |

## Defensive Behavior

- Missing or malformed YAML configs raise `ValueError` or file I/O exceptions.
- Invalid budgets raise `ValueError` in mask builders.
- Out-of-range protected channels raise `ValueError`.
- Zero BF16-vs-static gap raises `ValueError` in `recovery_fraction`, preventing
  accidental division by a non-recoverable trace.
- Full reproduction mode raises explicitly rather than returning frozen values.
- No broad `try/except` blocks silently swallow errors.
- No debug `print()` statements were found in release code.

## GPU/OOM Handling

`release/` does not implement GPU execution. OOM handling therefore remains a
property of archived experimental runners, not the minimal release helpers.
