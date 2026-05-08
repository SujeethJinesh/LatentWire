# KILL Manifest: SSM Shape-Conditioned Codec Phase 0

**Decision**: `KILL_SSC_PHASE0_NO_CODEC_GAIN`
**Date**: 2026-05-08
**Run dir**: `experimental/ssm_shape_codec/phase0/results/ssc_phase0_20260508T173705Z`
**Checker**: `experimental/ssm_shape_codec/phase0/check_ssc_phase0.py`
**Artifact completeness**: true

## Reason Classification

The branch is killed as **below preregistered effect-size shelf**. The
shape-conditioned 4-bit codec produced a real offline reconstruction gain, but
the mean relative NMSE reduction did not meet the preregistered `>=0.10`
threshold.

This was not an infrastructure failure, packet failure, or preregistration
ambiguity.

## Preregistered Decision Rule

`PASS_SSC_PHASE0_SHAPE_CODEC_GAIN` required all of:

- mean relative NMSE reduction `>=0.10`
- bootstrap 95% CI lower bound `>0.05`
- method non-worse than pooled baseline at both positions `{100, 10000}`
- artifact-complete packet

`KILL_SSC_PHASE0_NO_CODEC_GAIN` applies otherwise.

## Decision Metrics

- Mean relative NMSE reduction: `0.07133119147037506`
- Bootstrap 95% CI: `[0.06882070525106689, 0.073901076081768]`
- CI lower bound `>0.05`: true
- Mean reduction `>=0.10`: false
- Method non-worse at each position: true
- Row count: `432`
- Mamba layer count: `36`
- Model: `ibm-granite/granite-4.0-h-tiny`
- Model snapshot commit: recorded in `model_provenance.json`
- Prompt SHA-256: `sha256:436b1a4e46c9c14d89149db1cf8092ed112728bb5946eda1e1d6e1dbcb5b755b`

Position summaries:

| Position | Baseline NMSE | Method NMSE | Relative NMSE Reduction | Non-Worse |
|---:|---:|---:|---:|---|
| 100 | `0.6804698470428034` | `0.6095295439439791` | `0.09574283944620796` | true |
| 10000 | `0.7150237249831358` | `0.6783487142788039` | `0.04691954349454216` | true |

Largest layer-level reductions:

| Layer | Mean Relative NMSE Reduction |
|---:|---:|
| 39 | `0.25735352382423055` |
| 18 | `0.19733187866818575` |
| 7 | `0.17390405620457272` |
| 22 | `0.15861790456177824` |
| 17 | `0.13810289602443376` |
| 32 | `0.12609452006831884` |
| 12 | `0.10746164863601267` |
| 6 | `0.10474000755427944` |

Lowest layer-level reductions:

| Layer | Mean Relative NMSE Reduction |
|---:|---:|
| 34 | `0.005331185974826853` |
| 4 | `0.0115385962275001` |
| 26 | `0.015439609202715034` |
| 2 | `0.01721135223188027` |
| 28 | `0.018866436908439383` |
| 38 | `0.021685765892213972` |
| 33 | `0.02509944158455404` |
| 11 | `0.03203788808595366` |

These layer and position pockets are diagnostic only. They cannot be used to
rescue this preregistered branch.

## Artifact SHAs

- `artifact_check.json` (674 bytes): `sha256:98ef54dc09cdadf3d40c045c39fee695e5dfed0afa2bff644cae085484e78c6d`
- `artifact_hashes.json` (7843 bytes): `sha256:dd831c52dfb9d263ef31b77f7340baad6894f716e4f6390107e0085468bb0935`
- `checker_result.json` (1168 bytes): `sha256:05e4e4253a0f20497fd3c13a693660a31e465933a254d934ba812d84f3de916f`
- `codec_metrics.json` (106199 bytes): `sha256:aa2270b2b076e09217bfe62dd4e726c3bf98c5ed38044886961c197fa196bfd3`
- `command_metadata.json` (543 bytes): `sha256:e371d5a7ae984919614666ed88fc2358fc534f61cbe226818817e65d682ec819`
- `environment.json` (8291 bytes): `sha256:ee476833326149f336d2e9825becc207a8530aa978b4676287b31fe3b4dc2c1b`
- `logs/stderr.log` (652 bytes): `sha256:6df937b5f18cd11a1eb9871517fe72543d29d3ea261fe2d24edbd5c55dcba789`
- `logs/stdout.log` (113818 bytes): `sha256:88f63b6e09c67ab03ef947744154a1785698b83563c040244efdfbfb976deb45`
- `model_provenance.json` (5160 bytes): `sha256:545bace2cd289dbb9124eebca4f75067ef877ba20e357094ea824f5351fd59b3`
- `prompt_manifest.json` (8844 bytes): `sha256:710b956f0fe08b8e1d73cd03af3ce2918b37a74df83363a5aa8fb943d623d01b`
- `random_seed.json` (162 bytes): `sha256:c848499a758a5d282fa3582d33a38399ff85b656ef9a5c4ae98996206887fdc0`
- `raw_state_manifest.json` (27650 bytes): `sha256:0fe836c3d3219c09bcaccee47e351944dbcbd6376202c7f64b94301e6367b222`
- `run_events.jsonl` (311 bytes): `sha256:7cb0bba07bc744c39d0c9734fb32920db92a0deb7a09db1c7a635f8d9fcb706f`
- `ssm_states/layer_000.npz` (18545081 bytes): `sha256:581a097debca36b13b93f2da6a0c2fb204b791a483d8360f8a7404da7831db34`
- `ssm_states/layer_001.npz` (18570681 bytes): `sha256:a369588cf357a7965ae4aacaa7d30cbf6e7eaa1a7bcce0c94a2e622f3049c34d`
- `ssm_states/layer_002.npz` (18621617 bytes): `sha256:9eb94118d43aae9ff0187dc544d2ec2956f0b5d493086d1f5aae736785a6e0dc`
- `ssm_states/layer_003.npz` (18687697 bytes): `sha256:885d6b28120113a1cab56b052f046c37fcf3ad924412b68507e608d3be5194d7`
- `ssm_states/layer_004.npz` (18650827 bytes): `sha256:d5c4d3752a960a3c25bdda825f2c774286ec5bbc53d29b9a478a347cb17be6c0`
- `ssm_states/layer_006.npz` (18713053 bytes): `sha256:712f96748fd6410912263438b53bf1e960044b8dc18358e0d8c4cf740791e5dc`
- `ssm_states/layer_007.npz` (18373387 bytes): `sha256:9801c536c714fbe555e131b4f1a37d954987939ae342e03b232dee79db1add32`
- `ssm_states/layer_008.npz` (18729895 bytes): `sha256:7cdb769b9a692520b86cdc449c415b5df07f4e69d3b5f90598ded2c9189ddc1c`
- `ssm_states/layer_009.npz` (18392564 bytes): `sha256:e691095c7a61feb9771aba389579e901d589c565ff3cdc0f0edd535df0a46358`
- `ssm_states/layer_010.npz` (18788442 bytes): `sha256:d3678f24315ad78299c4850e53d82e868cb5b11c1254cdeaa1d281ce694621ab`
- `ssm_states/layer_011.npz` (18502381 bytes): `sha256:0b0e018142a5d33d34af102a8cc9007791f6d058af8dc6d71ccd0d9b767c6643`
- `ssm_states/layer_012.npz` (18646582 bytes): `sha256:1eecf1c64d60e47dd1bf55860d65b789181ce1dc1bc844031f5ec0670ee59d8d`
- `ssm_states/layer_013.npz` (18750986 bytes): `sha256:8d68f56db151f6923a95618e74198c13d3712b033b09b6b46f454c9d021dfb57`
- `ssm_states/layer_014.npz` (18824294 bytes): `sha256:2c4fd0a422a275edb4fe47a8dc2b5a047f0becf9f517b3000b4db0abbb2e8184`
- `ssm_states/layer_016.npz` (18727832 bytes): `sha256:ba92eacd7fb226ec513ba9380ca97a6493e5ecb5a08104415e05a36f6b67040d`
- `ssm_states/layer_017.npz` (18785678 bytes): `sha256:7487eee4c3594c859ba262891d4625e3b1f631eab50c9b70349c776da95c4619`
- `ssm_states/layer_018.npz` (18542548 bytes): `sha256:311c81a4dbb3ac3276fbdd0cc36188a35014a8a50edaaea378137d5178418228`
- `ssm_states/layer_019.npz` (18677592 bytes): `sha256:96c10ab121118d49501e9f138b3cf57224c5d09cfcde95956465086380ebfca0`
- `ssm_states/layer_020.npz` (18708808 bytes): `sha256:2ab4f763bb95e910d634ce3535db250c0f035605db6913b8eaf4beecce6aa512`
- `ssm_states/layer_021.npz` (18578083 bytes): `sha256:4b13c196558292aee094736cf5021f4e4b05448591946e00f105e1a69cd5146f`
- `ssm_states/layer_022.npz` (18710718 bytes): `sha256:83729391636a261c3527da4fc0b194a303e106958cb94aa7bdb16a2e469ec475`
- `ssm_states/layer_023.npz` (18603888 bytes): `sha256:172e59c18138baad40560c3817b56d42a7b4c195fb14361fd3189f3b6eb7b531`
- `ssm_states/layer_024.npz` (18756659 bytes): `sha256:fb8983d72f41d4b5436f4465e5804f4439416c9a7f0d756269af020bd41e42a0`
- `ssm_states/layer_026.npz` (18483801 bytes): `sha256:35b78e667ceb379df38f39b5f7733d0c06a0e45bc8654f5fa19fcce3932a86cf`
- `ssm_states/layer_027.npz` (18973417 bytes): `sha256:0516e3e243904322d9ceb7239f935168f82e45ab1b39d6e8c5e95454c339f194`
- `ssm_states/layer_028.npz` (18742973 bytes): `sha256:8c61b8a301c0f7b5f4bcdd1c34a2756dc578d852ab0a1f17ef1423a6ece484a9`
- `ssm_states/layer_029.npz` (18716762 bytes): `sha256:1c5895d4d8e9ad0fd8b021b7d4fe77204cc5fc7f706a6e64acd76a876c273584`
- `ssm_states/layer_030.npz` (18657066 bytes): `sha256:caca0401a8ce9ea3e7a580bde83d23c4891262a667c9e57f992eefa0a25e52fd`
- `ssm_states/layer_031.npz` (18843578 bytes): `sha256:c27e7c0064d68745803aceb67d24ded7fe12fe268efb1728ef4e7ca263444a58`
- `ssm_states/layer_032.npz` (18794774 bytes): `sha256:3c6faa3a1232f4b9d0becac64508f169161e0ff7c84d0f873b64673b8dc1b0b3`
- `ssm_states/layer_033.npz` (18977979 bytes): `sha256:bee4a438009a66275b2f8996e11eb147fba5e0d5701164acb28efb006ae0ebca`
- `ssm_states/layer_034.npz` (18625992 bytes): `sha256:1d5564524b7e9f59e2a489d5a1325f0878a0ec078c17fa9f536cacfdc65f7fff`
- `ssm_states/layer_036.npz` (18650859 bytes): `sha256:ffdbdf8f5ae0fc4556ae2f8b17c5cf0bf5e30b6ee99a8e310267fe7e1c39d8e5`
- `ssm_states/layer_037.npz` (18906493 bytes): `sha256:bf5d37861d11b45a0cb1e2ef04686fef02245605c0cfc6772c19db4b4361ab4f`
- `ssm_states/layer_038.npz` (19217457 bytes): `sha256:a98edb4a67b81f2d364818a9b4f1a1e744bc21604eeef51dd9521a8bb2dbf512`
- `ssm_states/layer_039.npz` (18665549 bytes): `sha256:9c6984bc79f08d79f4ddae388b5237e0e38d9b3c4bcd5815b869a798fe9b5ca3`

## Paper Status

No paper draft is created for this killed branch. The result remains available
as diagnostic evidence only.

## Diagnostic Outcome

The post-kill diagnostic classifies the failure as a weak effect-size miss. It
does not authorize continuing to mine the observed layer/position pockets under
this branch. Any future work would require a fresh depth-2 preregistration on a
new calibration-defined surface before inspecting fresh held-out data.
