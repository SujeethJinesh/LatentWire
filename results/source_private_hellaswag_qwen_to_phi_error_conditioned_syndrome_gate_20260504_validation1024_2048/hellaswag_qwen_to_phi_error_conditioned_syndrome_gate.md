# HellaSwag Qwen-To-Phi Error-Conditioned Syndrome Gate

- pass gate: `False`
- calibration rows: `1487`
- eval rows: `768`
- fixed hybrid accuracy: `0.467448`
- error-conditioned accuracy: `0.463542`
- error-conditioned delta: `-0.003906`
- error-conditioned CI95 low: `-0.020833`
- overrides / helps / harms: `75 / 20 / 23`
- fixed-or-source-top2 oracle accuracy: `0.694010`
- best destructive: `target_derived_source_packet_receiver_control` (`0.450521`)

## Interpretation

This gate directly tests the branch promoted by the target-error repair audit: train on official rows where the fixed packet tends to fail and ask whether quantized source top-2 syndrome fields can safely trigger repairs on held-out Qwen-to-Phi HellaSwag. A negative result weakens score/top2 syndrome repair and promotes learned target-resonance soft-prefix encoders as the next highest-value branch.

## Lay Explanation

Qwen sends Phi only a tiny hint: its two favorite answer choices and a few coarse confidence bins. The receiver practices on training questions to decide when that hint means Phi should change the existing packet answer. The final test checks whether this rule improves held-out questions or just creates new mistakes.
