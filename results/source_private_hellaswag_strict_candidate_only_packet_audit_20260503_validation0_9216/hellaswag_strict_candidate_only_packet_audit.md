# HellaSwag Strict Candidate-Only Packet Audit

- pass gate: `True`
- source artifact: `results/source_private_hellaswag_hidden_innovation_multi_slice_stress_20260503_rank_score_channel_qwen05_validation0_9216/hellaswag_hidden_innovation_multi_slice_stress.json`
- total eval rows: `9216`
- slice count: `9`
- candidate-only weighted accuracy: `0.525499`
- source-rank/index-only control accuracy: `0.479384`
- score-only control accuracy: `0.479384`
- best label-copy control accuracy: `0.483941`
- previous packet: `2B` raw / `5B` framed
- candidate-only packet: `1B` raw / `4B` framed
- framed byte reduction: `1` bytes/request

## Interpretation

The strict Qwen HellaSwag positive surface does not need the packet's extra confidence/debug byte for the receiver-visible decision: the selected candidate id alone reproduces the same 9216-row accuracy while retaining the original rank, score-channel, label-copy, zero-hidden, and corrupted-hidden control separations. This strengthens the systems/privacy accounting row, but it also sharpens the limitation: the current evidence is candidate-id communication, not a learned receiver or general latent language.

## Limitations

- This is a multiple-choice candidate-id compaction result, not a positive receiver-fusion result.
- The receiver-visible payload is the selected candidate id; it does not prove a general latent language.
- Native latency, throughput, HBM traffic, and GPU-serving speedups still require NVIDIA/vLLM/SGLang rows.

## Lay Explanation

We checked whether the previous hint needed two bytes or whether it was enough to send only which of the four answer choices the source model picked. On the large frozen HellaSwag slice, sending just that choice gives exactly the same answers, so the packet can be smaller. This is useful for a systems table, but it does not prove that the receiving model learned to reason from a richer hidden message.

