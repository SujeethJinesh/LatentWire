# HBSM Two-Prompt B1 Stop Manifest

Status: `RESOURCE_LIMITED_NOT_PROMOTABLE_FAIL_REAL_B1_SENSITIVITY_HETEROGENEITY`.

This is a reproducibility manifest for the Granite Tiny two-prompt B1
sensitivity scout at:

`experimental/shared/results/hbsm_prompt2_sensitivity_20260507/hbsm_gate_packet/`

It is a checker-passing resource-limited stop artifact. It weakens HBSM's broad
boundary-sensitivity hypothesis, but it is not B1 promotion, not cheap-predictor
evidence, not mechanism evidence, not GPU evidence, and not camera-ready
positive evidence.

## Packet Identity

- project: `hbsm`
- gate: `hbsm_b1`
- model_id: `ibm-granite-4.0-h-tiny`
- served_model_id: `ibm-granite/granite-4.0-h-tiny`
- model_revision: `791e0d3d28c86e106c9b6e0b4cecdee0375b6124`
- tokenizer_revision: `791e0d3d28c86e106c9b6e0b4cecdee0375b6124`
- prompt_source: `experimental/shared/prompts/hybrid_reasoning_smoke_12_20260506.jsonl`
- prompt_ids_hash: `sha256:48e68434371a648c3984e85a7207d71d2ac68617c640b37da04bd1aaeea45fe0`
- trace_plan_hash: `sha256:015e28d426aa4c11d00c67234c15e5cf5ed8f599a28de102fbc00aaccc84ed67`
- preregistration_sha256: `2f1d2ec3caab20eb3068dd5b7b42307ee1d61acbd3aecbc8551a1be494809592`
- review-base HEAD before this hardening pass: `e49353540474cceedfabe948e49fd1fa5f4da854`
- prompt_count: `2`
- layer_count: `8`
- max_input_tokens: `8`
- perturbation: `MXFP4 E2M1 block_size=32`
- resource_limit_note: HBSM local sensitivity runner used 2 short prompts, 8 layers, max_input_tokens=8, and MXFP4 E2M1 block_size=32; this validates forward-sensitivity plumbing only and cannot promote B1.

## Readout

- decision: `RESOURCE_LIMITED_NOT_PROMOTABLE_FAIL_REAL_B1_SENSITIVITY_HETEROGENEITY`
- gate_status: `FAIL_REAL_B1_SENSITIVITY_HETEROGENEITY`
- row_count: `64`
- prompt_count: `2`
- boundary_top_decile_count: `0`
- non_boundary_top_decile_count: `1`
- fisher_p_boundary_top_decile: `1.0`
- cheap_predictor_spearman: `-0.6666666666666666`

The second prompt removes the one-prompt boundary-top-decile hint and keeps the
cheap predictor in the wrong direction. Do not scale B1 to a long Mac sweep or
GPU validation unless a narrower mechanism hypothesis is preregistered first.

## Exact Validation

Run from the repository root:

```bash
PYTHONPATH="$PWD" ./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/shared/results/hbsm_prompt2_sensitivity_20260507/hbsm_gate_packet \
  --mode real --project hbsm
```

Expected checker output:

```json
{
  "decision": "RESOURCE_LIMITED_NOT_PROMOTABLE_FAIL_REAL_B1_SENSITIVITY_HETEROGENEITY",
  "errors": [],
  "mode": "real",
  "ok": true,
  "packet_dir": "experimental/shared/results/hbsm_prompt2_sensitivity_20260507/hbsm_gate_packet",
  "project": "hbsm",
  "row_count": 64,
  "surface": "real_hbsm_b1_sensitivity_packet"
}
```

## Artifact Hashes

| Artifact | SHA-256 |
|---|---|
| `hbsm_gate_packet/config.json` | `4d93ecd467aa7f973b4dda1e923397c4b6e2bc3a3d14ac3f8095da751553788d` |
| `hbsm_gate_packet/raw_rows.jsonl` | `4f845f7793649c2be1a1ed04ea51f7f489780026b6a76d8fd5abe9e831b7ed43` |
| `hbsm_gate_packet/summary.json` | `e84226aaa79bb44b7b4ae1f3ca69d6e8c64e774a500fe65c7349f1eb7c10f1a3` |
| `hbsm_gate_packet/decision.md` | `882b05e26c8ef24c53e2977229f7c7206f2e63da0202abd1881bca2ae4cf6eef` |
| `hbsm_row_packet.json` | `7e7dd37398ace957bc7bc93d74a04592ee53e014d71868882c48262f7cda5355` |

Source code hashes for this review pass:

| Source | SHA-256 |
|---|---|
| `experimental/shared/hbsm_local_sensitivity_runner.py` | `34ad97ac19ef15359430e17bbe5c30e5c451a33cb1b5f039385486626674b94f` |
| `experimental/shared/check_gate_packet.py` | `3f6b52b82b7c3ab3eda695d2263f3ba15aaf5b2ed8203ced2a4207330eb72bb8` |

## Next Gate

HBSM has no active next gate under the current broad hypothesis. Reopen only with
a new preregistered mechanism that explains why the one-prompt and two-prompt B1
failures should reverse, and keep any future B1 packet non-promotable if it is
resource-limited.
