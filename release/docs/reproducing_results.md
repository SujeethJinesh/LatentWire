# Reproducing Results

Run from `release/` after installation. FAST_VERIFY mode validates the frozen
paper numbers and tolerances without downloading checkpoints. Full mode uses
the same entry points and is intended for single-GPU reproduction.

| Paper Claim | Section | Reproduction Script | Expected Output | Tolerance |
|---|---|---|---|---|
| Granite-Small strict set-leaving is 0.566234756098 | Results | `python src/scripts/reproduce_set_leaving.py --config configs/granite_small.yaml --fast-verify` | `set_leaving_granite_small=0.566234756098` | `1e-12` |
| Nemotron strict set-leaving is 0.533713200380 | Results | same as above | `set_leaving_nemotron=0.533713200380` | `1e-12` |
| DeepSeek strict set-leaving is 0.670572916667 | Results | same as above | `set_leaving_deepseek=0.670572916667` | `1e-12` |
| Falcon-H1 strict set-leaving is 0.673611111111 | Results | same as above | `set_leaving_falcon=0.673611111111` | `1e-12` |
| Phase 4 static union median recovery is 0.0 | Results | `python src/scripts/reproduce_static_union.py --config configs/granite_small.yaml --fast-verify` | `phase4_static_union_median=0.0` | `1e-12` |
| Phase 4 no-gap fraction is 0.375 | Results | `python src/scripts/reproduce_trace_heterogeneity.py --config configs/granite_small.yaml --fast-verify` | `phase4_no_gap_fraction=0.375` | `1e-12` |
| M2 median recovery is -0.866837313391 | Results | `python src/scripts/reproduce_m2.py --config configs/granite_small.yaml --fast-verify` | `m2_median=-0.866837313391` | `1e-12` |
| M2 random-bin margin is -0.667548290168 | Results | same as above | `m2_random_margin=-0.667548290168` | `1e-12` |
| M10 median recovery is 0.234447927834 | Results | `python src/scripts/reproduce_m10.py --config configs/granite_small.yaml --fast-verify` | `m10_median=0.234447927834` | `1e-12` |
| M10 random-bin margin is -0.761487459046 | Results | same as above | `m10_random_margin=-0.761487459046` | `1e-12` |
| M11 median recovery is 0.048299284138 | Results | `python src/scripts/reproduce_m11.py --config configs/granite_small.yaml --fast-verify` | `m11_median=0.048299284138` | `1e-12` |
| M18 activation+K median is -0.343590844126 | Results | `python src/scripts/reproduce_m18.py --config configs/granite_small.yaml --fast-verify` | `m18_activation_k_median=-0.343590844126` | `1e-12` |
| DecDEC proxy median is -0.070034781258 | Results | `python src/scripts/reproduce_decdec.py --config configs/granite_small.yaml --fast-verify` | `decdec_median=-0.070034781258` | `1e-12` |
| M11b Granite top-5 median is 0.449284091125 | Results | `python src/scripts/reproduce_m11b.py --config configs/granite_small.yaml --fast-verify` | `m11b_granite_top5=0.449284091125` | `1e-12` |
| M11b Nemotron top-5 median is 0.456736183270 | Results | same as above | `m11b_nemotron_top5=0.456736183270` | `1e-12` |
| M11b Nemotron top-10 median is 0.814739798903 | Results | same as above | `m11b_nemotron_top10=0.814739798903` | `1e-12` |
| M26 stable-core median is 0.177615807648 | Results | `python src/scripts/reproduce_m26.py --config configs/granite_small.yaml --fast-verify` | `m26_median=0.177615807648` | `1e-12` |
| ParoQuant median is 0.753776848891 | Results | `python src/scripts/reproduce_paroquant.py --config configs/granite_small.yaml --fast-verify` | `paroquant_median=0.753776848891` | `1e-12` |
| KL means are 0.150302750276, 0.132701884446, 0.131406585383 | Results | `python src/scripts/reproduce_kl_accumulation.py --config configs/granite_small.yaml --fast-verify` | `kl_*_mean` JSON claims | `1e-12` |
| FFT entropy and autocorrelation are approximately 0.85 and 100 tokens | Results | `python src/scripts/reproduce_fft.py --config configs/granite_small.yaml --fast-verify` | `fft_entropy`, `fft_autocorr_tokens` | `0.02`, `5` |
| Per-component drift rates are comparable across layer types | Results | `python src/scripts/reproduce_per_component.py --config configs/granite_small.yaml --fast-verify` | component JSON claims | `1e-6` |

`bash src/scripts/reproduce_all.sh --fast-verify` runs every row above and
writes JSON files under `results/`.
