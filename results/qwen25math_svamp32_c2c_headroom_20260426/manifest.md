# Qwen2.5-Math -> Qwen3 SVAMP32 C2C Headroom Manifest

- date: `2026-04-26`
- git commit at run time: `ecd8b0946956636073875186add64d16c68af7a6`
- status: `clean_headroom_available`
- source model: `Qwen/Qwen2.5-Math-1.5B`
- target model: `Qwen/Qwen3-0.6B`
- eval file:
  `results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/_artifacts/svamp_eval_70_32_32.jsonl`
- ordered IDs: inherited from the SVAMP32 surface artifact

## Result Summary

- target-alone: `8/32`
- C2C teacher: `15/32`
- C2C-only over target: `9`
- source/text explained C2C-only IDs: `3`
- clean C2C-only IDs: `6`
- target-only vs C2C IDs: `2`

## Artifacts

| Path | SHA256 |
|---|---|
| `results/qwen25math_svamp32_c2c_headroom_20260426/c2c_teacher_innovation.json` | `c9f6d9dd4a4ef96999e243c5de412cb57bcd662a53b6f77733644fb4969e72a8` |
| `results/qwen25math_svamp32_c2c_headroom_20260426/c2c_teacher_innovation.md` | `4cd4abd00b94fdfc7892220448defa4b38ebf01ef681d2de5002d2fc799c6f10` |
| `results/qwen25math_svamp32_c2c_headroom_20260426/c2c_headroom_target_set.json` | `021b8b098c5fbc5a2b62193393bcf8da6bdba6c4eda2b1a411e32b94b6e81c32` |
| `results/qwen25math_svamp32_c2c_headroom_20260426/c2c_headroom_target_set.md` | `32cd9931c6320cb00e65b161d899bf8fc401f1e1a8ad29b6df1ffc34eaafcd34` |
| `results/qwen25math_svamp32_c2c_headroom_20260426/compatible_target_set.json` | `dfa26e421aca995fcf03f2eb9cf807d62de17e8b1601c85b4277328bd129d154` |
| `results/qwen25math_svamp32_c2c_headroom_20260426/compatible_target_set.md` | `c37bce08c4acc22e371acec253734a4fc9b6154ac06f4cd47da512580f3ddf04` |

## Decision

Use this as the strict-small clean C2C-headroom target set for the next
source-derived method gate. Do not count source/text-explained C2C-only IDs as
clean communication wins.

