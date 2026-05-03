# ARC Conditional-Innovation Packet Gate

- pass gate: `False`
- source-label heldout accuracy: `0.389`
- source-index decoder heldout accuracy: `0.362`
- quantized source-score packet heldout accuracy: `0.362`
- matched conditional-innovation heldout accuracy: `0.396`
- matched minus source-index decoder: `0.034`
- matched minus quantized source-score packet: `0.034`
- best control: `source_index_only_decoder` at `0.362`
- paired matched minus best control: `0.034` [-0.027, 0.094]

Lay explanation: the packet sends where the source model's answer scores differ from the receiver's own scores, then a small calibrated decoder asks whether that disagreement helps choose the answer.
