# OutlierMigrate Elevator Pitch

## Current Pitch

Long reasoning traces do not keep the same high-magnitude activation channels
hot. Across Granite-4-H, Nemotron-3, DeepSeek-R1-Distill, and Falcon-H1, the
top-channel set changes substantially with decode position, and the dominant
systems-relevant component is strict set-leaving rather than harmless
within-set rank shuffling. This undermines static protected-channel
quantization maps.

The strongest current method evidence is negative but mechanistically useful:
M2's position-conditional protected-set switching killed because a random-bin
control beat it by `0.6675482901676153` median recovery. That points to
boundary-discontinuous switching as a failure mode. M10-hard is running now to
test whether hard-binned scale tables show the same pattern. M11 EMA-smoothed
drift protection is preregistered next as the first continuous-update method.

## Paper Status

Draft 0 builds and is credible as a COLM workshop characterization/mechanism
paper. It is not yet an ICLR-positive-method paper because no method has
passed. The live path to a stronger paper is M11 or M17 passing; the honest
fallback is the "set-leaving cliff" negative-result framing if M11/M17/M18 all
kill.
