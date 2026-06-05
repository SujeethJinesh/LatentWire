# C_A1 GPU Backfill Runbook

RUNBOOK_STATUS: BLOCKED_CONFIRM_CONTAMINATED
C_A1_GATE_IDS_CONFIRM_CLEAN: false
GPU_FOREGROUND_AUTHORIZED: false
PROMOTION_ALLOWED: false

This runbook is blocked. Do not launch the cached C_A1 paired replay packet.

## Blocking Audit

`dashboard/confirm_path_audit.md` marks `C_A1_GATE_STATUS: INVALID_CONFIRM_CONTAMINATED`.
The old Granite tight-clip gate rows reference `experimental/outlier_migrate/phase9/results/om_driftrot_granite_clip_tight_confirmation_20260528T1735Z/per_trace_metrics.json`.
Those row IDs cannot seed a GPU handoff, a parity card, or a promotion decision.

The old DeepSeek and Falcon cached tight-clip rows are not sufficient on their own. The required packet is a same-row, three-model native comparison, and the Granite member of that packet is currently invalid.

## Required Fresh Inputs

Before this backfill becomes runnable, create a fresh non-confirm dev/gate manifest with all of the following:

- Granite, DeepSeek, and Falcon row IDs from non-confirm dev/gate sources only.
- Same prompt IDs for `paroquant_baseline` and `tight_clip_c_a1` on each model.
- Native W4A16/ParoQuant runner provenance for each row.
- No-gap denominator counts and an access manifest showing no `confirm` or `confirmation` source paths.
- Pre-launch review record for the exact runner code hash before any claim-bearing Stage-2 job. For backfill-only cache materialization, at minimum repro and leakage review records are required.

## Safe Commands

These commands are audit/preflight only; they do not launch GPU experiments.

```bash
venv_arm64/bin/python scripts/audit_confirm_paths.py
venv_arm64/bin/python scripts/build_review_packet.py
venv_arm64/bin/python scripts/check_review_packet.py review_packet.zip
venv_arm64/bin/python scripts/check_handoff.py
```

## Runnable Command Status

There is no runnable C_A1 GPU command in this repo state. The local GPU runner should refuse the C_A1 backfill queue item until a fresh non-confirm manifest replaces the contaminated cached Granite gate IDs.

## Fresh Row-Materialization Command

This is a materialization-only command for the local runner after the repro + leakage review records exist. It must not authorize foreground promotion by itself.

```bash
local_runner enqueue channel_set_c_a1_pair_materialization \
  --models granite,deepseek,falcon \
  --split dev,gate \
  --prompt-file experimental/shared/prompts/aime_2025_indices_0_23.jsonl \
  --policies paroquant_baseline,tight_clip_c_a1 \
  --scale-clip-min 0.5 \
  --scale-clip-max 2.0 \
  --require-same-row \
  --write-access-manifest \
  --fail-on-confirm \
  --out experimental/outlier_migrate/phase9/results/c_a1_nonconfirm_pair_matrix_${UTC_STAMP}
```

Expected output contract: a non-confirm same-row manifest for Granite, DeepSeek, and Falcon with ParoQuant baseline and tight-clip C_A1 rows, plus an access manifest. Abort if any source path, access path, or split label contains `confirm` or `confirmation`.

## Promotion Rule

Promotion remains disabled. A future backfill report may recommend a foreground job only after a fresh confirm-clean three-model packet shows:

- tight-clip C_A1 improves tail/CVaR or worst-trace risk versus ParoQuant without median regression,
- DeepSeek and Falcon sentinels do not regress in median or tail,
- matched controls fail,
- no-gap denominator audit passes,
- ParoQuant parity reproduces on the same prompt IDs.

A Granite-only or confirm-contaminated win is invalid for the paper.
