# Current Layer Localization (2026-04-20)

Telemetry source: `selector_trace` from controlled `gsm8k_eval_10` runs under the fair shared-chat / `enable_thinking=false` Qwen control.

| Method | Acc | Top target layers by layer_score | Mean keep frac (top layer) | Mean score top (top layer) | Mean score gap (top layer) |
|---|---:|---|---:|---:|---:|
| module_adapter | 0.1000 | L27<-S23, L5<-S4, L23<-S20, L22<-S19, L8<-S7 | 0.4992 | 0.9140 | 0.8785 |
| module_replace | 0.1000 | L27<-S23, L5<-S4, L23<-S20, L22<-S19, L8<-S7 | 0.4992 | 0.9140 | 0.8785 |
| shared_plus_private_asym_adapter | 0.1000 | L27<-S23, L5<-S4, L23<-S20, L22<-S19, L8<-S7 | 0.4992 | 0.9140 | 0.8785 |
| shared_plus_private_dynmap_adapter | 0.1000 | L27<-S23, L5<-S4, L23<-S20, L22<-S19, L8<-S7 | 0.4992 | 0.9140 | 0.8785 |
| spanalign_module_replace | 0.1000 | L27<-S23, L5<-S4, L23<-S20, L22<-S19, L8<-S7 | 0.4992 | 0.9140 | 0.8785 |
| tokenbasis_replace | 0.1000 | L27<-S23, L5<-S4, L23<-S20, L22<-S19, L8<-S7 | 0.4992 | 0.9140 | 0.8785 |
| xattn_adapter | 0.1000 | L27<-S23, L5<-S4, L23<-S20, L22<-S19, L8<-S7 | 0.4992 | 0.9140 | 0.8785 |
| xattn_dynmap_adapter | 0.1000 | L27<-S23, L5<-S4, L23<-S20, L22<-S19, L8<-S7 | 0.4992 | 0.9140 | 0.8785 |
