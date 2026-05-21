# Extension Guide

New methods should be added as small selector modules under
`src/outlier_migrate/methods/` and registered through `default_registry`.

Each method should:

- use only explicit score tables provided by the caller;
- break ties deterministically by lower channel index;
- return a set of protected hidden-channel indices;
- include CPU tests for selection behavior and registry integration.

## Rotation And Budget Composition

Future rotation-plus-budget methods can compose two operations:

1. rotate or transform channel scores into a method-specific comparison space;
2. allocate a protected-channel budget across decode positions or layer groups.

The current `paroquant` placeholder demonstrates deterministic round-robin
budget composition across positions. Final rotation logic should be introduced
only after its release contract and expected artifacts are fixed.
