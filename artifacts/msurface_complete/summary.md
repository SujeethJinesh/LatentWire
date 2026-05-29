# Complete M-SURFACE Internal Diagnostic

Decision: `KILL_MSURFACE_NO_LOWER_INTERNAL_SURFACE`.

The complete Granite internals run measured SSM input, B/C generation surfaces, Mamba out-projection input, attention o-projection input, and post-block output on the same two traces at decode positions 100 and 20000. No internal surface clears the promotion gate.

Key mean strict set-leaving values:

- Post-block output: 0.546.
- SSM input after convolution: 0.484 (only 0.061 below post-block).
- SSM C generation after convolution: 0.486 (only 0.060 below post-block).
- SSM B generation after convolution: 0.618.
- SSM B/C concat after convolution: 0.602.
- Mamba out-projection input: 0.887.
- Attention out-projection input: 0.799.

The best internal surfaces are slightly lower than post-block, but not below 0.30-0.40 and not at least 0.15 below post-block. SurfaceRot/SurfaceProtect is killed as a positive method for this sprint and becomes a stronger negative result: drift reaches SSM internals at meaningful magnitude.
