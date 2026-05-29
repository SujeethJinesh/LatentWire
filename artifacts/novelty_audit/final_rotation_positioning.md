# Final Rotation Positioning Audit

Positioning paragraph for the paper:

> Prior rotation methods construct static calibration-time transforms, learned rotations, or runtime smoothing rules that improve low-bit inference by changing the basis in which quantization error appears. Our question is different: under long reasoning decode, channel identity itself drifts. We measure when channel-set protection becomes ill-posed, then test whether rotation removes that basis dependence or whether drift-aware tail control, surface choice, branch-local rotation, or rotated-basis residual correction can improve on a fixed ParoQuant-style baseline.

| Method | Static vs decode-time | Rotation vs channel-set | Measures long-decode drift? | Safe distinction | Unsafe claim to avoid |
|---|---|---|---|---|---|
| ParoQuant | static calibration | rotation | no | strong prior-work rotation baseline; our packets test it under long-reasoning drift | calling ParoQuant our method |
| QuaRot | static calibration | rotation | no | rotation precedent for reducing quantization outliers | claiming we invent rotation quantization |
| SpinQuant | learned/static | rotation | no | adjacent learned rotation baseline | claiming learned rotation novelty |
| MambaQuant | static PTQ | rotation/smoothing for Mamba | no | Mamba-family static quantization context | claiming it tests long-decode protected sets |
| Quamba2 | static/offline SSM structure | channel/state grouping | not at our surface/horizon | our block-output and Granite-internal diagnostics test a distinct surface | saying Quamba2 is false for all internal SSM tensors |
| DecDEC | decode-time | channel-set selection | short horizon | qualitative precursor for dynamic salient channels | claiming our proxy is DecDEC itself |
| Our paper | decode-time measurement | channel-set tests plus rotation baseline | yes | long-decode drift, matched-control failures, regime protocol, residual cost envelope | claiming a globally positive quantizer |
