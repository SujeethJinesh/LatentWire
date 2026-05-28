# Executive Summary

Current framing: long-reasoning W4A16 channel protection is regime-dependent.
Static and hard-switch policies fail under decode-time channel-set drift.
Budgeted EMA succeeds in the Nemotron MoE-hybrid regime, ParoQuant succeeds in
the Granite dense-hybrid regime, and DeepSeek/Falcon remain the live test
surface for cheap positive-method rescue via LAMBDA/HYST smoke.

Current blocker: V1 ParoQuant-on-Nemotron is still running in the source repo.
This pack marks it `RUNNING` and includes progress/logs, not final numbers.
No GPU jobs were launched to create this pack.

Status snapshot:

- `00_four_model_drift`: PASS (strict set-leaving on Granite/Nemotron/DeepSeek/Falcon); median=
- `01_static_top1_static_union`: KILL (static protection on Granite); median=
- `02_m2_position_switching`: KILL (M2 on Granite); median=
- `03_m10_position_bins`: KILL (M10 on Granite); median=
- `04_m11_ema_top1`: KILL (M11 on Granite); median=
- `05_m11b_granite`: AMBIG (M11b on Granite); median=0.4492840911245966
- `06_m11b_nemotron`: PASS (M11b on Nemotron); median=0.8147397989034302
- `07_m11b_deepseek`: AMBIG (M11b on DeepSeek); median=0.3353894408225412
- `08_m11b_falcon`: KILL (M11b on Falcon-H1); median=0.04393973019116826
- `09_paroquant_granite`: PASS (ParoQuant on Granite); median=0.7537768488911776
- `10_v1_paroquant_nemotron`: RUNNING (ParoQuant on Nemotron); median=
- `11_e3_paroquant_plus_m11b`: KILL (ParoQuant+M11b on Granite); median=0.5645482174486074
- `12_m18_activation_k`: KILL (M18 on Granite); median=-0.34359084412632024
- `13_m26_stable_core`: AMBIG (M26 on Granite); median=0.17761580764776214
- `14_decdec_proxy`: KILL (DecDEC proxy on Granite); median=-0.07003478125820645
- `15_kl_fft_diagnostics`: PASS (KL+FFT on Granite/DeepSeek/Falcon); median=
- `16_e2_cross_prompt`: PASS (cross-prompt drift on Granite/DeepSeek/Falcon); median=
- `17_mpred`: KILL (M-PRED on Granite/DeepSeek/Falcon); median=-7.745988954781958
- `18_wjac_prefilter`: KILL (WJAC on all cached slices); median=
- `19_lambda_prefilter`: SMOKE_ONLY (LAMBDA on DeepSeek/Falcon); median=
- `20_hyst_prefilter`: SMOKE_ONLY (HYST on DeepSeek/Falcon); median=
- `21_msurface_diagnostic`: DEFERRED (M-SURFACE on Granite/Falcon); median=
- `22_mbranch_diagnostic`: DEFERRED (M-BRANCH on Falcon-H1); median=
- `23_restricted_kllook_if_present`: DEFERRED (KLLOOK on Granite/Nemotron optional); median=

Firm results: four-model drift, M11b Nemotron, ParoQuant Granite, E3
sub-additivity, M-PRED KILL, WJAC offline KILL. Smoke-only/pending:
LAMBDA/HYST and V1 ParoQuant-on-Nemotron. Diagnostics only: M-SURFACE and
M-BRANCH.
