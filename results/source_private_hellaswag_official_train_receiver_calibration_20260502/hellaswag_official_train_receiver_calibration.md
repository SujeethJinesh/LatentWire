# HellaSwag Official-Train Receiver Calibration

- pass gate: `False`
- predeclared default pass gate: `False`
- scout pass gate: `False`
- official train calibration rows: `1487`
- validation rows: `10042`
- default eval accuracy: `0.618701`
- default delta vs packet-only: `-0.000498`
- best scout eval accuracy: `0.620594`
- best scout delta vs packet-only: `0.001394`
- full-validation Tiny+Qwen oracle: `0.686815`

## Lay Explanation

The earlier receiver used early validation rows for calibration. This experiment moves receiver training to official HellaSwag train rows. To avoid teaching the receiver from a packet that saw the same row during packet training, each train row is scored by packet models trained on other official-train samples only. The frozen receiver is then evaluated on full validation.

## Interpretation

This is the cleanest cached test of whether more legitimate receiver supervision fixes the packet-only blocker. If it passes, it promotes an official-train calibrated receiver. If it fails, the next branch should move beyond scalar acceptance into a richer learned common-basis or query bottleneck receiver. Because it reuses cached 512-row packet-training samples, even a pass should be repeated on a new disjoint official-train calibration sample before becoming the paper claim.
