# ARC Llama-8B Disagreement Source Scout

- pass gate: `False`
- source family: `llama3.1_8b_instruct`
- validation rows: `144`
- test rows: `473`
- test source accuracy before packet: `0.553911`
- test matched mean: `0.368288`
- test Qwen-substituted mean: `0.317125`
- test cached Tiny mean: `0.269345`
- test delta vs Qwen-sub: `0.051163`
- test delta vs cached Tiny: `0.098943`
- test CI95 low vs Qwen-sub: `-0.034937`
- test rolled-source control mean: `0.205074`
- test random-source control mean: `0.245666`

## Lay Explanation

This run asks whether a much stronger non-Qwen local source model can choose better ARC answers on the hard rows where TinyLlama and Qwen disagreed. Only the chosen answer is converted into the same tiny 12-byte packet; Llama hidden states, text, and KV cache are not transmitted.

## Interpretation

A pass would promote Llama-8B to a full source-family gate on all ARC validation/test rows. A failure would cheaply rule out the only locally cached true non-Qwen stronger source as the next Mac-local ARC repair, leaving NVIDIA-scale connector training or a new cached source as the live branch.
