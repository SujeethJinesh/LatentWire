# Cross-Family Interface Proxy Manifest

- date: `2026-04-26`
- scale-up rung: micro smoke / strict-surface scout
- status: `proxy_surface_fails`
- source model: `Qwen/Qwen2.5-0.5B-Instruct`
- target model for runnable proxy: `facebook/opt-350m`
- eval file: `data/gsm8k_gate_search_30.jsonl`

## Tracked Artifacts

- `tokenizer_interface_strict_small.json`
  - sha256: `0ef9295e064480350dc20f967da076d54a63c53d1876ba7424f5107586bbc665`
- `tokenizer_interface_strict_small.md`
  - sha256: `0edfb2c9129ac57620cf929bdb1da1aaf5f50815efbd925bc7b0f703b194c2c7`
- `qwen25_opt350m_tokenizer_interface.json`
  - sha256: `e35fb4e33efca327a084224201849e875f32238389bac09b710cb13c50861d4f`
- `qwen25_opt350m_tokenizer_interface.md`
  - sha256: `978189ca604e5cc3e48e792d8032da58058ae944634db9f93090b5790700b788`
- `quotient_gpa_sequence_sidecar_seed1.json`
  - sha256: `c5fed12c591e215ebfcd1e328b9b9fd8f32878cf330af84a2140f6bcd4c3c8d2`
- `quotient_gpa_sequence_sidecar_seed1.md`
  - sha256: `efef504d4de37edce2580631b2be029318376d6c1b4dbfadc8d3a96c738da909`
- `qwen25_to_opt350m_bytespan_gsm30_matched.jsonl`
  - sha256: `49659e18aa35a0d9deabb3208ccd6160fc62cf82cfdc928615a780eb9b6d663e`
- `qwen25_phi3_gsm30_target_source_t2t.jsonl`
  - sha256: `79ca67f7cb9259c46e132d3ff75e461bb066cfc3e6e4d7e29156d7dfc06fdfa4`
- `qwen25_phi3_svamp30_target_source_t2t.jsonl`
  - sha256: `5844c390496178e2312e52ee46b1a73a167e0ec195f542fdf548f61e39ccbb17`
- `qwen25_tinyllama_svamp30_target_source_t2t.jsonl`
  - sha256: `b7057406b5660d0dbaff95213575299bdaae50aacb6a18ffb2d8f092e215c7ef`

## Scratch Artifacts

- checkpoint:
  `.debug/qwen25_phi3_bytespan_interface_20260426/qwen25_to_opt350m_bytespan_r4_cal64.pt`
  - sha256:
    `3a6c0c2cf8aa46be91b58d5c36bab5477183111f66f2d08ef07699592554696c`
  - tracked: no, checkpoint tensor is kept in `.debug`
- calibration log:
  `.debug/qwen25_phi3_bytespan_interface_20260426/logs/calibrate_opt350m_after_width_patch.log`
  - sha256:
    `6832969d2845b97a6c1780b9245823e755f6dc8d02600b6ad87f32991539cde1`
- evaluation log:
  `.debug/qwen25_phi3_bytespan_interface_20260426/logs/evaluate_opt350m_matched_after_patch.log`
  - sha256:
    `a752bf7a46076dbe5c07921cbb2ddf2ccbbd18e6d2f9859fd5bc5632a6d49736`
- Phi-3 GSM30 baseline log:
  `.debug/cross_family_surface_baselines_20260426/logs/qwen25_phi3_gsm30_target_source_t2t.log`
  - sha256:
    `df26741eb0913382b89b48410d7f49a3fd5415e7f65f53edca3d3749ad4ccc0c`
- Phi-3 SVAMP30 baseline log:
  `.debug/cross_family_surface_baselines_20260426/logs/qwen25_phi3_svamp30_target_source_t2t.log`
  - sha256:
    `978e77827b1cb03b0738fc27d5a432893fa74ef550c63dac96f173e19203f533`
- TinyLlama SVAMP30 baseline log:
  `.debug/cross_family_surface_baselines_20260426/logs/qwen25_tinyllama_svamp30_target_source_t2t.log`
  - sha256:
    `b1f66b80c1574c162bdbd83a779ac3a28b8679d0ee0049c1e9a4a59a3884d14e`

## Decision

Kill OPT-350m as a cross-family decision surface. The matched proxy produced
`0/30` for rotalign, target-alone, and source-alone, while text relay produced
`3/30`. This is a surface failure rather than a decisive kill of the
sequence-aligned sidecar hypothesis.

Next gate: use a stronger target surface before running source-destroying
controls. Preferred targets are Phi-3 or TinyLlama after GQA/packed-QKV
compatibility repairs if baseline target/text accuracy is nonzero; fallback is
same-family Qwen2.5 -> Qwen3 with a real sidecar component and the full strict
control matrix.

Surface baseline update:

| Surface | Target | Source | Text Relay | Decision |
|---|---:|---:|---:|---|
| Qwen2.5 -> Phi-3, GSM30 | 3/30 | 0/30 | 1/30 | weak headroom |
| Qwen2.5 -> Phi-3, SVAMP30 | 5/30 | 1/30 | 2/30 | weak headroom |
| Qwen2.5 -> TinyLlama, SVAMP30 | 0/30 | 1/30 | 0/30 | dead surface |

Do not run a large GQA repair campaign on these exact surfaces unless a
stronger target/text baseline is found first.

