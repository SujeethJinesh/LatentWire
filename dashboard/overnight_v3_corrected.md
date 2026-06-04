# Overnight V3 Corrected Re-Probe

- locality: CPU-only local run; no CUDA, foreground GPU, SSH, or confirm access.
- interpretation: broken or underpowered sanity gates are not negative method evidence.

## Summary

| probe | status | wall_clock_seconds | key n | sanity/gate note |
| --- | --- | ---: | ---: | --- |
| EXP1 corrected cache ceiling | `INCONCLUSIVE_UNDERPOWERED` | 124.6 | 107 | openbookqa=INCONCLUSIVE_UNDERPOWERED, arc_challenge=INCONCLUSIVE_UNDERPOWERED, mmlu_redux=INCONCLUSIVE_UNDERPOWERED |
| EXP4 corrected L-A2 rerank | `PARKED_NEEDS_STRONGER_GENERATOR` | 619.6 | 36 | correct-candidate/verifier gate |

## EXP1 Task Details

### openbookqa

- status: `INCONCLUSIVE_UNDERPOWERED`
- n total/dev/gate: `120` / `83` / `37`
- model own acc: `0.4864864864864865`
- receiver probe acc: `0.40540540540540543`
- baseline mode: `receiver_label_scores_repair`; hidden probe sane: `False`
- receiver+source acc: `0.4594594594594595`
- gain+CI+MDE: `{'ci95_high': 0.05405405405405406, 'ci95_low': -0.10810810810810811, 'delta': -0.02702702702702703, 'mde_half_width': 0.08108108108108109, 'n': 37}`
- dense fusion alpha/gain: `0.2` / `{'ci95_high': 0.05405405405405406, 'ci95_low': -0.10810810810810811, 'delta': -0.02702702702702703, 'mde_half_width': 0.08108108108108109, 'n': 37}`

### arc_challenge

- status: `INCONCLUSIVE_UNDERPOWERED`
- n total/dev/gate: `120` / `86` / `34`
- model own acc: `0.4411764705882353`
- receiver probe acc: `0.4411764705882353`
- baseline mode: `hidden_receiver_probe`; hidden probe sane: `True`
- receiver+source acc: `0.47058823529411764`
- gain+CI+MDE: `{'ci95_high': 0.17647058823529413, 'ci95_low': -0.11764705882352941, 'delta': 0.029411764705882353, 'mde_half_width': 0.14705882352941177, 'n': 34}`
- dense fusion alpha/gain: `0.7` / `{'ci95_high': 0.14705882352941177, 'ci95_low': -0.08823529411764706, 'delta': 0.029411764705882353, 'mde_half_width': 0.11764705882352941, 'n': 34}`

### mmlu_redux

- status: `INCONCLUSIVE_UNDERPOWERED`
- n total/dev/gate: `120` / `84` / `36`
- model own acc: `0.4722222222222222`
- receiver probe acc: `0.3611111111111111`
- baseline mode: `receiver_label_scores_repair`; hidden probe sane: `False`
- receiver+source acc: `0.4722222222222222`
- gain+CI+MDE: `{'ci95_high': 0.1388888888888889, 'ci95_low': -0.1388888888888889, 'delta': 0.0, 'mde_half_width': 0.1388888888888889, 'n': 36}`
- dense fusion alpha/gain: `1.0` / `{'ci95_high': 0.1388888888888889, 'ci95_low': -0.1388888888888889, 'delta': 0.0, 'mde_half_width': 0.1388888888888889, 'n': 36}`

## EXP4 Details

- status: `PARKED_NEEDS_STRONGER_GENERATOR`
- prompts/candidates: `100` / `1600`
- prompts with >=1 correct candidate: `36`
- verifier score present: `True`
- gain+CI+MDE: `None`
- GPU/stronger-generator command if parked: `GPU stronger-generator cache: generate >=300 dev/gate GSM8K/MATH prompts x16 candidates with a stronger generator, then score each candidate with source_score, target_score, and verifier_score; do not screen L_A2 until >=80 prompts have at least one correct candidate.`

## Artifacts

- `results/overnight_v3/20260604_corrected_reprobe/exp1_corrected_cache_ceiling/summary.json`
- `results/overnight_v3/20260604_corrected_reprobe/exp4_corrected_l_a2_rerank_ceiling/summary.json`
