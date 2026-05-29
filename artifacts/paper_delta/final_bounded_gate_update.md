# Paper Delta: Final Bounded Gate Update

Add the bounded-gate outcome as a final design screen:

- K-RES top-8x32 rotated residual correction is killed as a valid implementation, not an infra failure.
- The restricted residual KLLOOK oracle was not executed because the repo only has an original-basis M-KLLOOK runner. Do not imply oracle evidence exists.
- M-SURFACE measured cheap Granite projection-input surfaces and found higher drift than post-block; SSM input/B/C remain unmeasured.
- Falcon BranchRot remains deferred because branch-local activations were not cached.

Recommended framing: mechanism/regime paper. ParoQuant is the strongest baseline and should remain clearly labeled as prior work. Our contribution is the drift measurement, failure taxonomy, regime map, and bounded negative screens that explain why basis-dependent channel protection fails.
