# Systems Cost Envelope

Generated: 2026-05-29T04:02:11Z
Git commit at generation time: `7f20ce0beb27d5bfcf04b33ff3ee70c725f495cd`

This directory contains the analytical systems envelope for hypothetical rotated-basis residual correction. K-RES did not pass the quality gate, so these are cost-model artifacts, not a deployed-kernel claim.

Energy constants are intentionally stated as ranges: 10--20 pJ/byte for HBM traffic and 1--5 pJ/MAC for arithmetic. The model is used only to show that residual correction is memory-bound at the tested K values.

The paper also reports the killed Granite top-8x32 aggregate as a separate calculation: 8 corrected modules, 10 active experts, output width 1536, and 32 corrected columns, giving 3.93M MACs/token and about 7.5 MiB/token of active residual-column traffic. This aggregate should not be read as the K=8 representative Granite-width row multiplied by modules.
