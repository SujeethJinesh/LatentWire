# Morning Brief

Row-safe Stage-1 screen completed on parseable cached families. It consumed `183653` dev/gate rows/traces and `0` confirm rows. Coverage is in `dashboard/cache_parse_coverage.md`; raw evidence is in `results/stage1/`.

Status counts: `{'AMBIGUOUS': 100, 'CPU_SCREENED': 57, 'KILLED': 12, 'PARKED_NEEDS_GPU': 2}`.

The held-out answer is unfavorable for final claims: many caches are prior test/validation/full-eval artifacts or lack explicit confirm naming, so Mac screens can kill or rank branches, but final confirmation needs quarantined row-specific confirm handling or fresh data.

## Overnight V3 Patch Status

- C_A1 is `BLOCKED_CONFIRM_CONTAMINATED` with `promotion_allowed=false`; the cached Granite tight-clip gate source references a confirmation path, so the GPU packet must wait for fresh non-confirm same-row ParoQuant-vs-tight-clip IDs on Granite, DeepSeek, and Falcon.
- L_SCORECOMP, L_B1, and L_Q1 are terminal killed score-packet branches and must not be queued.
- L_A2 is `INCONCLUSIVE_UNDERPOWERED_NEEDS_VERIFIER_CACHE`; the next cache must use at least 300 generated-solution prompts, include `verifier_score`, and screen only if the correct-candidate subset is large enough.
- `queues/gpu_foreground.yaml` remains empty.

## New 200 Ideation Triage - MPS-First Shortlist

- `L_PC1_cross_family_specialist_ceiling`: highest-priority LatentWire pivot; tests complementary-source conditional information beyond the receiver.
- `L_PC2_tool_augmented_source_ceiling`: private calculator/code-exec source signal; kill if visible/equal-byte tool controls explain it.
- `L_PC5_private_verifier_receiver_candidates`: fixes L-A2 degeneracy only if candidate pools have at least 80 prompts with one correct candidate.
- `L_C2_control_trained_lcf_lite_proxy`: cheap trained-fuser proxy; kill if wrong-row/zero-source controls retain the gain.
- `C_U1_drift_as_signal_router`: fresh Channel-Set positive path using drift trajectory as routing/difficulty signal.
- `C_W1_fixed_library_warmup_selector`: choose ParoQuant/C_A1/survival/reject from warmup features; kill if best fixed policy wins.
- `C_S1_clean_survival_stablecore_denominator`: only a fresh identical-row denominator can revive survival StableCore.
- `C_Y5_channel_set_defense_bundle`: audit packet for OSC/DecDEC/no-gap/ParoQuant parity before any Channel-Set claim.
- `C_A1` remains first in `queues/gpu_backfill.yaml`; the new GPU-after-MPS queue does not displace it, but it is blocked until the confirm-clean manifest exists.
- `queues/gpu_foreground.yaml` remains empty; this pass created ideation/filter/queue artifacts only.

## 2026-06-05 Consolidated Audit

- Confirm-path audit complete: `dashboard/confirm_path_audit.md` found `17` embedded confirm-looking source-path hits and marks `C_A1_GATE_STATUS: INVALID_CONFIRM_CONTAMINATED`.
- C_A1 remains first in `queues/gpu_backfill.yaml`, but both C_A1 backfill items are blocked until fresh non-confirm row IDs exist; `queues/gpu_foreground.yaml` is empty.
- L_PC1 cached strict evidence is a small powered ceiling signal only: `+0.042969`, CI `[+0.015625, +0.068359]`, but it lacks the fresh second pair/task and source-index/equal-byte text controls.
- L_PC5 and L_C2 strict positives are oracle-only: L_PC5 scores `cand == answer`; L_C2 uses SVAMP equation-derived `tool_answer`.
- Channel-Set C_U1 and C_W1 are schema-blocked, C_S1 remains below the local floor at `198/500`, and C_Y5 waits on native same-row pairing.

## 2026-06-05 Cheap Exhaustion Scan

- `scripts/cheap_exhaustion_scan.py` scanned `1294` files, used `820` non-confirm text artifacts, excluded `42` confirm-path files and `35` embedded-confirm files, and consumed `0` confirmation rows for method evidence.
- No listed Channel-Set method has a claim-clean CPU/cache positive: C_U1, C_W1, CE1, and C_C1 are parked for missing same-row feature/label/policy matrices; C_D1 is Granite-only underpowered defense evidence.
- C_A1 manifest remains incomplete: DeepSeek and Falcon have 12-row non-confirm baseline-vs-tight-clip pairs, but Granite tight-clip rows are embedded-confirm or diagnostic residual only. `queues/gpu_foreground.yaml` remains empty.
- The canonical next GPU action is materialization-only: `local_runner enqueue channel_set_c_a1_pair_materialization --models granite,deepseek,falcon --split dev,gate --prompt-file experimental/shared/prompts/aime_2025_indices_0_23.jsonl --policies paroquant_baseline,tight_clip_c_a1 --scale-clip-min 0.5 --scale-clip-max 2.0 --require-same-row --write-access-manifest --fail-on-confirm --out experimental/outlier_migrate/phase9/results/c_a1_nonconfirm_pair_matrix_${UTC_STAMP}`.
