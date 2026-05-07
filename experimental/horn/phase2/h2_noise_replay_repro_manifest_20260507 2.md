# HORN H2 Noise-Replay Stop Manifest

Status: `RESOURCE_LIMITED_NOT_PROMOTABLE_FAIL_REAL_HORN_H2_DIRECTIONAL_NOISE_PROPAGATION`.

This is a reproducibility manifest for the Granite Tiny H2 noisy-continuation
scout at:

`experimental/shared/results/horn_h2_noise_replay_scout_20260507/`

It is a contract-valid stop artifact and demotes HORN as a standalone branch. It
is not H2 promotion, not a precision-allocation recipe, not GPU evidence, and not
camera-ready positive evidence.

## Packet Identity

- project: `horn`
- gate: `horn_h2`
- model_id: `ibm-granite-4.0-h-tiny`
- served_model_id: `ibm-granite/granite-4.0-h-tiny`
- model_revision: `791e0d3d28c86e106c9b6e0b4cecdee0375b6124`
- tokenizer_revision: `791e0d3d28c86e106c9b6e0b4cecdee0375b6124`
- prompt_source: `experimental/shared/prompts/hybrid_reasoning_smoke_12_20260506.jsonl`
- prompt_source_sha256: `48e68434371a648c3984e85a7207d71d2ac68617c640b37da04bd1aaeea45fe0`
- prompt_limit: `2`
- max_input_tokens: `2`
- seed_list: `[1, 2, 3]`
- source_h1a_decision: `RESOURCE_LIMITED_NOT_PROMOTABLE_FAIL_REAL_H1A_DIRECTIONAL_ASYMMETRY_SCREEN`
- source_gate_packet_sha256: `sha256:f6588496123def548e5363cc5f0db061f198ee581f92886d4f5e3e4197e07ac0`
- preregistration_sha256: `sha256:fe71d4c0ecc117412731c7e5c30e866023dff3f06070355241438ecc0c6254f2`
- trace_plan_sha256: `a2df7d6485d376747ba179c80172882b3dddd440d1db3b5f765f777a857e75f0`
- review-base HEAD before this hardening pass: `e49353540474cceedfabe948e49fd1fa5f4da854`
- resource_limit_note: Mac-local HORN H2 scout over short Granite Tiny prompts; useful for demotion or follow-up planning only, not a promotable H2 packet.

## Readout

- raw gate status: `FAIL_REAL_HORN_H2_DIRECTIONAL_NOISE_PROPAGATION`
- row_count: `20`
- paired_unit_count: `6`
- expected_paired_unit_count: `6`
- selected_h2_direction: `ssm->attention`
- directional_drift_ratio: `1.0373710554468507`
- directional_ratio_ci_low: `0.32354204242892726`
- selected_direction_support_fraction: `0.5`
- hook_off_max_delta: `0`
- demotion_recommendation: `DEMOTE_HORN_STANDALONE_WEAK_H2`

The effect is near-null, the lower bound is far below the preregistered H2
directional-noise threshold, and support is split. Do not scale this to GPU under
the current HORN hypothesis.

## Exact Validation

Run from the repository root:

```bash
PYTHONPATH="$PWD" ./venv_arm64/bin/python -m experimental.shared.followup_gate_contracts \
  experimental/shared/results/horn_h2_noise_replay_scout_20260507 \
  --gate horn_h2
```

Expected checker output:

```json
{
  "decision": "FAIL_REAL_HORN_H2_DIRECTIONAL_NOISE_PROPAGATION",
  "errors": [],
  "gate": "horn_h2",
  "ok": true,
  "packet_dir": "experimental/shared/results/horn_h2_noise_replay_scout_20260507",
  "row_count": 20
}
```

The generic H1a/H1 `check_gate_packet --project horn` checker is not applicable
to this follow-up packet; H2 is validated by
`experimental.shared.followup_gate_contracts --gate horn_h2`.

## Artifact Hashes

| Artifact | SHA-256 |
|---|---|
| `config.json` | `26473d2c64c52ca4b05a3e9abee99cb9ceeb5e84d563edf9abf57dc6b6dfdc76` |
| `raw_rows.jsonl` | `def14c7df3c2ce4f2e84e3005df48bd2ce397795bf760811e7349ae7303834b6` |
| `summary.json` | `e718b1bc02eb425fea6d8772520aa7c38eef358341500363ab2967dbbd05eb38` |
| `decision.md` | `cdd096eb0faf5c367e06e2efe1bda3c07c9bda2bfa18c65bea0d34f88c270ef4` |

Source code hashes for this review pass:

| Source | SHA-256 |
|---|---|
| `experimental/shared/horn_h2_noise_replay_scout.py` | `32dc6c30926e79dd2312a3ff65af7e26f1c723befa23425e3cb721eb687e8954` |
| `experimental/shared/followup_gate_contracts.py` | `4e105379a799ffeb78b5af4917363388b07c42ff5b75d4389b8d12b3c808f14e` |
| `experimental/shared/check_gate_packet.py` | `3f6b52b82b7c3ab3eda695d2263f3ba15aaf5b2ed8203ced2a4207330eb72bb8` |

## Next Gate

HORN has no active next gate under the current scope. Reopen only with a new
preregistration that explains why this near-null H2 scout and the failing H1a
screen should reverse on larger surfaces.
