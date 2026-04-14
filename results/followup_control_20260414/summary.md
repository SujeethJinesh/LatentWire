# Follow-up Control Pair Summary

Control pair:
- `Qwen/Qwen2.5-0.5B-Instruct -> Qwen/Qwen3-0.6B`
- Eval set: `data/gsm8k_100.jsonl`
- Calibration set: `data/calibration.txt` (`600` prompts)

## What Changed

- Calibration now masks padded positions and aligns source/target token positions
  per prompt before fitting.
- Evaluation now builds source and target caches over the same prompt prefix
  span before fusion.

## Alignment Quality

Overnight checkpoint:
- `K cos = 0.344`, `K rel_err = 0.935`
- `V cos = 0.439`, `V rel_err = 0.913`

Fixed checkpoint:
- `K cos = 0.408`, `K rel_err = 0.907`
- `V cos = 0.543`, `V rel_err = 0.831`

## Quantized Gate Sweep

From `qwen25_to_qwen3_masked_gate_sweep.log`:
- `target_alone = 0.04`
- `text_to_text = 0.10`
- `gate=0.00 -> 0.04`
- `gate=0.25 -> 0.04`
- `gate>=0.50 -> 0.00`

Interpretation:
- The code fixes removed the immediate collapse at low gate strength.
- Quantized translated KV is still not beating target-alone on this slice.

## Full-Precision Gate Sweep

From `qwen25_to_qwen3_masked_noquant_gate_sweep.log`:
- `gate=0.00 -> 0.04`
- `gate=0.25 -> 0.05`
- `gate>=0.50 -> 0.00`

From `qwen25_to_qwen3_masked_noquant_lowgate_sweep.log`:
- `gate=0.10 -> 0.04`
- `gate=0.15 -> 0.05`
- `gate=0.20 -> 0.04`
- `gate=0.25 -> 0.05`
- `gate=0.30 -> 0.05`

Interpretation:
- There is now a small positive signal in full precision at low gate values.
- The current bottleneck is not only fusion; quantization is also hurting.
- The method is still below `text_to_text = 0.10`.

## Recommended Next Step

Run selective transmission next on the same control pair:
- keep the fixed checkpoint path
- use `--no-quantize`
- try `layer_selection_ratio` in `{0.25, 0.5}`
- keep gate in the `0.15-0.30` range

Rationale:
- all 28 target layers are currently transmitted
- low-gate full-precision helps a little
- the next likely win is reducing noisy translated layers before further model expansion
