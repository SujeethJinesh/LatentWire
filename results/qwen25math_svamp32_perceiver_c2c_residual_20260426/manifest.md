# Qwen2.5-Math SVAMP32 Perceiver C2C-Residual Manifest

- date: `2026-04-26`
- status: `failed_pre_generation_gate`
- checkpoint hash: `d50b00fd0b9f5b5afcb09af8f9ae89b868e913b0a0610ef8132e66f20c726759`
- calibration log hash: `495a782080e78fc1b40f59452dc25ec936207f93cb1945dbce4063196002d156`

## Gate Results

| Gate | JSON SHA256 | Markdown SHA256 | Matched-Only Clean | Control Leak Clean |
|---:|---|---|---:|---:|
| `0.125` | `c60c357ecf38a76479a94265296ec3a32905bd95d4b39787c83da84429c21503` | `c931ff46f0985524ee2fdc1a7170facf4ae21e4cde7699d40970e1446c74ce53` | 0/6 | 2/6 |
| `0.150` | `caf7de8d67de8fd06defc1bc71d68fa4c636877b05508f533fd1fe913cce690c` | `55ca28d4f33fbb68f7146e6959f04bf9cbd2008d7eabad59311e1ca12268744a` | 0/6 | 2/6 |
| `0.200` | `97d64048334ddc523b1a7303fb7e8922d46828c6d3bf3ee97e59445a81fc8eca` | `3d659f5ab0668215e957e6225b0975a95de1e7c3fe9729067d253bf0e2bf4903` | 0/6 | 2/6 |

## Decision

Do not run generation for this checkpoint. Matched source does not beat
zero-source, shuffled-source, target-only, and slots-only controls on any clean
C2C residual ID.
