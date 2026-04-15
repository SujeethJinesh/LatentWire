# RotAlign Control Suite

| Checkpoint | Eval | Target | T2T | Best Metric | Value | Δ vs Target | Δ vs T2T | Bytes | TTFT (s) | Tok/s |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| cka_half_seed1 | fused_quant_brief | nan | nan | rotalign_kv_gate_0.15 | 0.0600 | nan | nan | 1197883.7 | 0.2698 | 31.326 |
| interp_full_seed0 | fused_noquant_plain | 0.0400 | 0.0100 | rotalign_kv_gate_0.15 | 0.0500 | 0.0100 | 0.0400 | 19017564.2 | 0.1874 | 31.753 |
| cka_half_seed0 | fused_quant_brief | nan | nan | rotalign_kv_gate_0.30 | 0.0500 | nan | nan | 1197883.7 | 0.2257 | 33.181 |
| cka_half_seed0 | fused_noquant_plain | 0.0400 | 0.0100 | target_alone | 0.0400 | 0.0000 | 0.0300 | nan | 0.0858 | 37.031 |
| cka_half_seed0 | fused_noquant_cot | nan | nan | rotalign_kv_gate_0.15 | 0.0400 | nan | nan | 9508782.1 | 0.2336 | 28.140 |
| cka_quarter_seed0 | fused_noquant_plain | 0.0400 | 0.0100 | target_alone | 0.0400 | 0.0000 | 0.0300 | nan | 0.0873 | 35.793 |
| cka_quarter_seed0 | fused_noquant_cot | nan | nan | rotalign_kv_gate_0.15 | 0.0400 | nan | nan | 4754391.0 | 0.1907 | 34.887 |
| cka_quarter_seed0 | fused_quant_brief | nan | nan | rotalign_kv_gate_0.25 | 0.0400 | nan | nan | 598941.8 | 0.2331 | 32.185 |
| cka_half_seed1 | fused_noquant_plain | 0.0400 | 0.0100 | target_alone | 0.0400 | 0.0000 | 0.0300 | nan | 0.0939 | 33.708 |
| cka_half_seed1 | fused_noquant_cot | nan | nan | rotalign_kv_gate_0.15 | 0.0400 | nan | nan | 9508782.1 | 0.2109 | 31.426 |
| interp_full_seed0 | fused_noquant_cot | nan | nan | rotalign_kv_gate_0.25 | 0.0200 | nan | nan | 19017564.2 | 0.1793 | 33.440 |
| interp_full_seed0 | fused_quant_brief | nan | nan | rotalign_kv_gate_0.25 | 0.0200 | nan | nan | 2395767.4 | 0.2255 | 34.379 |
| interp_full_seed0 | text_kv_noquant_brief | nan | nan | rotalign_text_kv_hybrid_gate_0.25 | 0.0100 | nan | nan | 21540700.2 | 1.6174 | 18.600 |
| cka_half_seed0 | text_kv_noquant_brief | nan | nan | rotalign_text_kv_hybrid_gate_0.15 | 0.0100 | nan | nan | 10770350.1 | 1.6072 | 19.008 |
| cka_half_seed1 | text_kv_noquant_brief | nan | nan | rotalign_text_kv_hybrid_gate_0.15 | 0.0100 | nan | nan | 10770350.1 | 1.6625 | 18.188 |
| cka_quarter_seed0 | text_kv_noquant_brief | nan | nan | rotalign_text_kv_hybrid_gate_0.15 | 0.0000 | nan | nan | 5385175.0 | 1.6007 | 19.253 |
