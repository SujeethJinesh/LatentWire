# Complete M-SURFACE Diagnostic Readout

Decision: `MEASURED_CHEAP_SURFACES_NO_PROMOTE__SSM_BC_INCOMPLETE`.

The same-run Granite diagnostic measured module-hookable internal surfaces. Neither measured internal surface clears the promotion gate:

- Post-block mean strict leaving: 0.578.
- Mamba out-projection input mean strict leaving: 0.887 (+0.311 versus post-block).
- Attention out-projection input mean strict leaving: 0.780 (+0.183 versus post-block).

SurfaceRot/SurfaceProtect is not promoted from the measured evidence. This is not a full kill of all possible internal surfaces: SSM input and B/C generation are local tensors inside the Mamba forward and remain unmeasured by the existing packet.
