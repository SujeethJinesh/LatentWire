# KILL Manifest: SSM-State Lifecycle Phase 0

Decision: `KILL_SSML_PHASE0_STATE_STABLE`

Date: 2026-05-08

Run directory:
`experimental/ssm_lifecycle/phase0/results/ssml_phase0_20260508T165752Z`

Checker:
`experimental/shared/check_phase0_gate.py --branch ssm_lifecycle --run-dir experimental/ssm_lifecycle/phase0/results/ssml_phase0_20260508T165752Z`

## Decision Summary

The Phase 0 gate killed the SSM-State Lifecycle branch under its original
preregistered hypothesis.

- `artifact_complete`: `true`
- `mamba_layer_pass_count`: `0`
- `mamba_layer_count`: `36`
- `mamba_layer_pass_fraction`: `0.0`
- Required pass fraction: `>= 0.5`
- Required per-layer criteria: combined KS p-value `<0.01` and median drift
  ratio `>=2.0`
- Checker reason: `state-stable kill: Mamba layer pass fraction 0.00000000 < 0.50`

All 36 Mamba layers had combined KS p-values below 0.01, but no layer reached
the preregistered `2x` median magnitude-drift threshold. This is therefore a
clean kill of the age-conditioned magnitude-drift compression hypothesis, not
an infrastructure failure.

This is NOT a re-kill of SSQ-LR. SSQ-LR failed as a cross-model static
mixed-precision recipe. This branch failed because Granite-4.0-H-Tiny SSM
states did not show the preregistered age-based magnitude drift required for a
lifecycle compression recipe.

## Artifact Integrity

Primary packet artifacts:

| Artifact | SHA-256 |
|---|---|
| `checker_result.json` | `cc7a139c586a00be345893bd6b2471a1d63a574e42c401aaa32754d4f1fc4510` |
| `artifact_check.json` | `65a37637b794228055ad76827498d4d6364e23761e6f27ba4329a8755633ff0c` |
| `metrics.json` | `7726db1607918671e7c3fd1960c05dda36c4e8016727bdc64f0b4a2caf2f392b` |
| `ssm_state_manifest.json` | `153aa3e643af03408c08e0692662e8a6da4b1536ec7bf3e6e8cb3b7cfa4245db` |

State artifacts:

| Artifact | SHA-256 |
|---|---|
| `ssm_states/layer_000.npz` | `2d148a92d95a5dab77813f55828e5d934f018957a8dbf237276721d75b445d62` |
| `ssm_states/layer_001.npz` | `9fb0fcebcaf46b1355d3a482cba9d61abc82f96d400ff157582cd0e6a886fdc7` |
| `ssm_states/layer_002.npz` | `4460e650f4f296287133c98804f89d1397140e5d6e89e4cad085bc23884b575d` |
| `ssm_states/layer_003.npz` | `3efc29a719e9186d6a330bd805459bd137f23be44b96dadf5818857fc590d60a` |
| `ssm_states/layer_004.npz` | `f65271aa910a72b582e9aa705473601ccaed6114be1cb46c5ec85f0a05bd41ff` |
| `ssm_states/layer_006.npz` | `eff0fcc727d9f1900947e781ca7a5bc1a2dc12ca8972acee7f0985618331f255` |
| `ssm_states/layer_007.npz` | `75a0564bb29927a4dcbc4bbd15a55ce23fcd5973010a80c6f0b79abbd7555d68` |
| `ssm_states/layer_008.npz` | `0ed0194fb347d7affdf0677b3f3b9d7fcd3a799004dd53d8b0da91bfb914d20e` |
| `ssm_states/layer_009.npz` | `835b15a93bebaa69ddca6073db15a8849cbba66193a79ad7567ac88cd2131439` |
| `ssm_states/layer_010.npz` | `04109440d58ab8fa8b1fee9bc2185b604f2d7a0a998d9390c8f18d1c989d454c` |
| `ssm_states/layer_011.npz` | `3c527057cd20f4c6409a2a2b69db72c5757859c85fbe00cc1fa3efe3f08e96e8` |
| `ssm_states/layer_012.npz` | `d31eed4c1c2da4a2cda26790021ad55a03b337da5c48fcc4cb5ab7c78c25594b` |
| `ssm_states/layer_013.npz` | `a153901e68e307c46536b7ecb601d9eef4661138dcbd81104d84c0293f95ea62` |
| `ssm_states/layer_014.npz` | `77c45eff81f6e2aebf24222deb33a1710b19a27f8d42802ba4c88921b0ed831f` |
| `ssm_states/layer_016.npz` | `acf5c9b205d657198ca83fdfa811fd702a0532bea713912d7566bed93c3fcc86` |
| `ssm_states/layer_017.npz` | `1961c3c35d0324665daa63bb677ed52c7b6a152d5f76233b9278b2ee16428594` |
| `ssm_states/layer_018.npz` | `cbbb91a4d78b92dc8adf2b806112406e67da9001caf82845d0bee2dabe7c62c3` |
| `ssm_states/layer_019.npz` | `fcc0c431dbe20752351803f9469e20da48beed39c3a7f3b34b22bffba91d56f7` |
| `ssm_states/layer_020.npz` | `5d6620e25bffa289d77d394152c04e4cc661b8c47285cc6952a259cc1f91f4d5` |
| `ssm_states/layer_021.npz` | `5d8c0cdd6fd11e7ca96997e71fdfedb785bb2bf84aabd0e49c9f44084a110ca1` |
| `ssm_states/layer_022.npz` | `2c81460f85e37c6f84ceb589b58eed13cd05f3815053c50169e2ce8aa49b8959` |
| `ssm_states/layer_023.npz` | `b60a6591d467f1b8fc93eceb21c855a7c5c2d5895c7c522876a1e3c714d85320` |
| `ssm_states/layer_024.npz` | `d10c9d864be6f858c19ae7518353524722b312c1cd7c9949641337c57d783021` |
| `ssm_states/layer_026.npz` | `564d9bb751081a4b3ad87e2a9bd6e40459d027dee330cfaa8bf0280cfe5145a0` |
| `ssm_states/layer_027.npz` | `0cef72e7ca1b1e0ebc012c1969e80c295a6043493ba7d51c1a5779f908770ac0` |
| `ssm_states/layer_028.npz` | `9439fc017cefd60127d5a32ed8244a978095244879ea1f6874e22ef6eb9a6cc7` |
| `ssm_states/layer_029.npz` | `9507b8eb7221d8d0499507fed1733560d272b3d7cd552442ad17f414253cc28e` |
| `ssm_states/layer_030.npz` | `23335d9850f3724da6dfe745d291d61a8b27399455d26efd207abe78786c9675` |
| `ssm_states/layer_031.npz` | `3069fe0d82c9ff05eb02a9a6df54c2a4ebc77664576b245901c38f4c5fd7ed94` |
| `ssm_states/layer_032.npz` | `20bd88c2bb23919cf8413d014970e17e11fb067a77f71a97c0cf9ddaceeae7c0` |
| `ssm_states/layer_033.npz` | `c8c4951444c9117cf372d59f7fce52ee3696c9f0e45ad8449fe8b31270718aa9` |
| `ssm_states/layer_034.npz` | `a475bf0dc1f3ac985ddfc74fc92d4ae7ccfb210abc3b9e198e136799cf92b624` |
| `ssm_states/layer_036.npz` | `edf5ebf28f2549d78ebc700a6c927b4a99efab72b24b0c60c9f58f0d6bd9c4de` |
| `ssm_states/layer_037.npz` | `b79d622ba22d0466df84e901e270c5671d341f852028f8744ecb80222baafef5` |
| `ssm_states/layer_038.npz` | `bae9fe157063339c56e433ec82ad5ff8e601866afa332d97b3d9eee91cfa4532` |
| `ssm_states/layer_039.npz` | `618320a4d59ce253d1e82cca23aa8638b19832f20e5b0d6dd3af1b0c7e71c00d` |

The complete artifact hash list is in
`experimental/ssm_lifecycle/phase0/results/ssml_phase0_20260508T165752Z/artifact_hashes.json`.

## Layer Readout

| Layer | Median drift ratio | Combined KS p-value | Passes |
|---:|---:|---:|---|
| 0 | 1.087189359022 | 0 | false |
| 1 | 1.158409392973 | 0 | false |
| 2 | 0.978222325424 | 0 | false |
| 3 | 1.015441876071 | 0 | false |
| 4 | 1.059967028314 | 0 | false |
| 6 | 0.999989296386 | 0 | false |
| 7 | 0.966182142785 | 0 | false |
| 8 | 0.971701011050 | 0 | false |
| 9 | 1.008133727018 | 0 | false |
| 10 | 0.965133792488 | 0 | false |
| 11 | 0.896229369907 | 0 | false |
| 12 | 1.031460846145 | 0 | false |
| 13 | 1.004885164044 | 0 | false |
| 14 | 0.967903040691 | 0 | false |
| 16 | 0.967352202519 | 0 | false |
| 17 | 0.871434957613 | 0 | false |
| 18 | 0.952506458686 | 0 | false |
| 19 | 1.099206838346 | 0 | false |
| 20 | 0.778845695718 | 0 | false |
| 21 | 0.850667967428 | 0 | false |
| 22 | 1.117700810327 | 0 | false |
| 23 | 0.836923209536 | 0 | false |
| 24 | 0.892207014835 | 0 | false |
| 26 | 0.835158278675 | 0 | false |
| 27 | 1.072773872593 | 0 | false |
| 28 | 0.815816799721 | 0 | false |
| 29 | 0.863638441251 | 0 | false |
| 30 | 0.942147724556 | 0 | false |
| 31 | 0.964382712079 | 0 | false |
| 32 | 0.704716177590 | 0 | false |
| 33 | 1.100402911757 | 0 | false |
| 34 | 0.882360713396 | 0 | false |
| 36 | 0.831511316699 | 0 | false |
| 37 | 0.992256516186 | 0 | false |
| 38 | 1.127682942518 | 0 | false |
| 39 | 1.129389346681 | 0 | false |

## Disposition

The original SSM-State Lifecycle Phase 0 and all downstream Phase 1 / Phase 2
entries gated by it are not active unless a fresh preregistered pivot is
authored under the swarm goal's pivot policy.
