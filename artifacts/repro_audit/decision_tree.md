# Rotation-First Decision Tree

## Immediate Gate

1. Run ParoQuant Falcon smoke.
2. Run ParoQuant DeepSeek smoke.
3. Only after those results decide whether to resume Falcon channel rescue.

## Branches

### If ParoQuant passes Falcon and DeepSeek

Paper framing becomes rotation-dominant:

- Channel identity drifts during long reasoning.
- Channel-set policies are fragile under drift.
- Rotation removes the basis dependence and is robust across the measured
  four-model set.
- DriftRot attempts become optional robustness/tail improvements rather than
  the core positive result.

### If ParoQuant fails Falcon

Falcon remains the live positive-method surface:

- Promote Falcon BranchRot and/or M-SURFACE if their diagnostics show lower
  drift than post-block output.
- Run Falcon LAMBDA/HYST only after rotation is known weak or impossible.
- Use restricted KLLOOK if branch/channel smoke is ambiguous.

### If ParoQuant fails DeepSeek but passes Falcon

DeepSeek becomes the live surface:

- Consider Fisher/GATE only if cheap diagnostics show headroom.
- Keep the paper regime-aware rather than rotation-dominant.

### If DriftRot beats ParoQuant

Paper method becomes drift-aware rotation calibration:

- Static ParoQuant remains baseline.
- DriftRot must show held-out gain or tail/CI improvement.
- Do not frame as ParoQuant itself.

### If DriftRot fails but ParoQuant remains strong

Paper becomes mechanism/protocol:

- Channel drift breaks channel identity.
- Rotation is surprisingly robust.
- The contribution is the calibrated decision protocol and failure-to-design
  map.

