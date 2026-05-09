# OutlierMigrate Phase 1 Migration Decomposition

## Scope

- Run directory: `experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z`
- Base decode position: 100
- Final decode position: 20000
- Trace count: 24
- Strict top-channel set: per prompt and layer, channels with rank <= `ceil(channel_count * 0.01) - 1` at decode position 100.
- Strict set-leaving definition: a strict base top-1% channel has final rank > `ceil(channel_count * 0.01) - 1`.
- Within-set rank shuffling definition: a base top-1% channel remains in the final top-1% set but moves by more than 2 rank positions.
- Original migration definition: the preregistered checker metric, which selects top-1% channels per layer by mean magnitude at position 100 across traces and counts movement by more than 2 rank positions.
- The first two rows are post-hoc interpretability readouts using the strict set-membership definition; the third row is the unchanged gate metric from `metrics.json`.

## Aggregate Readout

| Metric | Fraction | 95% bootstrap CI |
| --- | ---: | ---: |
| Strict set-leaving | 0.566234756098 | [0.550076219512, 0.581707317073] |
| Within-set rank shuffling | 0.270934959350 | [0.260797764228, 0.280614837398] |
| Original conflated migration | 0.843165650407 | [0.833434959350, 0.851143292683] |

## Consistency Check

- `metrics.json` original migration fraction: 0.843165650407
- Reported original migration fraction: 0.843165650407

## Per-Trace Readout

| Prompt index | Strict set-leaving | Within-set rank shuffling |
| ---: | ---: | ---: |
| 0 | 0.617073170732 | 0.245121951220 |
| 1 | 0.543292682927 | 0.295121951220 |
| 2 | 0.629268292683 | 0.229878048780 |
| 3 | 0.582926829268 | 0.248780487805 |
| 4 | 0.617682926829 | 0.247560975610 |
| 5 | 0.556707317073 | 0.271951219512 |
| 6 | 0.614024390244 | 0.236585365854 |
| 7 | 0.602439024390 | 0.237195121951 |
| 8 | 0.539634146341 | 0.265853658537 |
| 9 | 0.535975609756 | 0.300609756098 |
| 10 | 0.582317073171 | 0.260365853659 |
| 11 | 0.568902439024 | 0.270731707317 |
| 12 | 0.566463414634 | 0.292682926829 |
| 13 | 0.551219512195 | 0.276219512195 |
| 14 | 0.493292682927 | 0.292682926829 |
| 15 | 0.535975609756 | 0.313414634146 |
| 16 | 0.563414634146 | 0.282926829268 |
| 17 | 0.571951219512 | 0.289634146341 |
| 18 | 0.473170731707 | 0.293902439024 |
| 19 | 0.583536585366 | 0.239024390244 |
| 20 | 0.607317073171 | 0.243292682927 |
| 21 | 0.574390243902 | 0.271951219512 |
| 22 | 0.566463414634 | 0.282926829268 |
| 23 | 0.512195121951 | 0.314024390244 |

## Machine-Readable Payload

```json
{
  "aggregate": {
    "left_set_ci95": {
      "ci95_high": 0.5817073170731707,
      "ci95_low": 0.5500762195121951
    },
    "left_set_fraction": 0.566234756097561,
    "original_migration_ci95": {
      "ci95_high": 0.8511432926829268,
      "ci95_low": 0.8334349593495936
    },
    "original_migration_fraction": 0.843165650406504,
    "raw_count_left_set_fraction": 0.566234756097561,
    "raw_count_within_set_rank_shuffle_fraction": 0.2709349593495935,
    "within_set_rank_shuffle_ci95": {
      "ci95_high": 0.280614837398374,
      "ci95_low": 0.2607977642276423
    },
    "within_set_rank_shuffle_fraction": 0.2709349593495935
  },
  "base_position": 100,
  "final_position": 20000,
  "layer_metrics": [
    {
      "layer_index": 0,
      "left_set_fraction": 0.6290650406504065,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.22865853658536586
    },
    {
      "layer_index": 1,
      "left_set_fraction": 0.5304878048780488,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.3018292682926829
    },
    {
      "layer_index": 2,
      "left_set_fraction": 0.5294715447154471,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.3099593495934959
    },
    {
      "layer_index": 3,
      "left_set_fraction": 0.5619918699186992,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2926829268292683
    },
    {
      "layer_index": 4,
      "left_set_fraction": 0.5579268292682927,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.29776422764227645
    },
    {
      "layer_index": 5,
      "left_set_fraction": 0.551829268292683,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.3079268292682927
    },
    {
      "layer_index": 6,
      "left_set_fraction": 0.5528455284552846,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.29776422764227645
    },
    {
      "layer_index": 7,
      "left_set_fraction": 0.5630081300813008,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2855691056910569
    },
    {
      "layer_index": 8,
      "left_set_fraction": 0.5630081300813008,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2886178861788618
    },
    {
      "layer_index": 9,
      "left_set_fraction": 0.5782520325203252,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2703252032520325
    },
    {
      "layer_index": 10,
      "left_set_fraction": 0.5894308943089431,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2784552845528455
    },
    {
      "layer_index": 11,
      "left_set_fraction": 0.5823170731707317,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2764227642276423
    },
    {
      "layer_index": 12,
      "left_set_fraction": 0.573170731707317,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2845528455284553
    },
    {
      "layer_index": 13,
      "left_set_fraction": 0.5447154471544715,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.29065040650406504
    },
    {
      "layer_index": 14,
      "left_set_fraction": 0.540650406504065,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.3048780487804878
    },
    {
      "layer_index": 15,
      "left_set_fraction": 0.5284552845528455,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2936991869918699
    },
    {
      "layer_index": 16,
      "left_set_fraction": 0.5315040650406504,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.3018292682926829
    },
    {
      "layer_index": 17,
      "left_set_fraction": 0.5467479674796748,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2835365853658537
    },
    {
      "layer_index": 18,
      "left_set_fraction": 0.532520325203252,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.3008130081300813
    },
    {
      "layer_index": 19,
      "left_set_fraction": 0.5396341463414634,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2845528455284553
    },
    {
      "layer_index": 20,
      "left_set_fraction": 0.5548780487804879,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.28760162601626016
    },
    {
      "layer_index": 21,
      "left_set_fraction": 0.5619918699186992,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2733739837398374
    },
    {
      "layer_index": 22,
      "left_set_fraction": 0.5640243902439024,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.27947154471544716
    },
    {
      "layer_index": 23,
      "left_set_fraction": 0.5731707317073171,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2733739837398374
    },
    {
      "layer_index": 24,
      "left_set_fraction": 0.5792682926829268,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.28252032520325204
    },
    {
      "layer_index": 25,
      "left_set_fraction": 0.5762195121951219,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2703252032520325
    },
    {
      "layer_index": 26,
      "left_set_fraction": 0.5772357723577236,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2571138211382114
    },
    {
      "layer_index": 27,
      "left_set_fraction": 0.5670731707317073,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2693089430894309
    },
    {
      "layer_index": 28,
      "left_set_fraction": 0.576219512195122,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2540650406504065
    },
    {
      "layer_index": 29,
      "left_set_fraction": 0.5853658536585366,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.25203252032520324
    },
    {
      "layer_index": 30,
      "left_set_fraction": 0.592479674796748,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.241869918699187
    },
    {
      "layer_index": 31,
      "left_set_fraction": 0.5884146341463414,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.24898373983739838
    },
    {
      "layer_index": 32,
      "left_set_fraction": 0.5823170731707317,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.25609756097560976
    },
    {
      "layer_index": 33,
      "left_set_fraction": 0.5823170731707317,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.23780487804878048
    },
    {
      "layer_index": 34,
      "left_set_fraction": 0.5558943089430894,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2378048780487805
    },
    {
      "layer_index": 35,
      "left_set_fraction": 0.5721544715447154,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2571138211382114
    },
    {
      "layer_index": 36,
      "left_set_fraction": 0.5894308943089431,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.21951219512195122
    },
    {
      "layer_index": 37,
      "left_set_fraction": 0.6026422764227642,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.21138211382113822
    },
    {
      "layer_index": 38,
      "left_set_fraction": 0.5731707317073171,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.22052845528455284
    },
    {
      "layer_index": 39,
      "left_set_fraction": 0.568089430894309,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2266260162601626
    }
  ],
  "observed_trace_count": 24,
  "original_metrics_json_ci95": {
    "ci95_high": 0.8511432926829268,
    "ci95_low": 0.8334349593495936
  },
  "original_metrics_json_migration_fraction": 0.843165650406504,
  "phase": 1,
  "rank_delta_strictly_greater_than": 2,
  "run_dir": "experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z",
  "strict_set_membership_selection": "per prompt and layer, channels with position-100 rank <= ceil(channel_count * 0.01) - 1",
  "top_boundary_semantics": "zero-based rank <= ceil(channel_count * 0.01) - 1",
  "top_channel_fraction": 0.01,
  "trace_count": 24,
  "trace_metrics": [
    {
      "layer_count": 40,
      "left_set_fraction": 0.6170731707317073,
      "prompt_index": 0,
      "within_set_rank_shuffle_fraction": 0.2451219512195122
    },
    {
      "layer_count": 40,
      "left_set_fraction": 0.5432926829268293,
      "prompt_index": 1,
      "within_set_rank_shuffle_fraction": 0.2951219512195122
    },
    {
      "layer_count": 40,
      "left_set_fraction": 0.6292682926829268,
      "prompt_index": 2,
      "within_set_rank_shuffle_fraction": 0.22987804878048781
    },
    {
      "layer_count": 40,
      "left_set_fraction": 0.5829268292682926,
      "prompt_index": 3,
      "within_set_rank_shuffle_fraction": 0.24878048780487805
    },
    {
      "layer_count": 40,
      "left_set_fraction": 0.6176829268292683,
      "prompt_index": 4,
      "within_set_rank_shuffle_fraction": 0.2475609756097561
    },
    {
      "layer_count": 40,
      "left_set_fraction": 0.5567073170731708,
      "prompt_index": 5,
      "within_set_rank_shuffle_fraction": 0.27195121951219514
    },
    {
      "layer_count": 40,
      "left_set_fraction": 0.6140243902439024,
      "prompt_index": 6,
      "within_set_rank_shuffle_fraction": 0.23658536585365852
    },
    {
      "layer_count": 40,
      "left_set_fraction": 0.6024390243902439,
      "prompt_index": 7,
      "within_set_rank_shuffle_fraction": 0.2371951219512195
    },
    {
      "layer_count": 40,
      "left_set_fraction": 0.5396341463414634,
      "prompt_index": 8,
      "within_set_rank_shuffle_fraction": 0.2658536585365854
    },
    {
      "layer_count": 40,
      "left_set_fraction": 0.5359756097560976,
      "prompt_index": 9,
      "within_set_rank_shuffle_fraction": 0.30060975609756097
    },
    {
      "layer_count": 40,
      "left_set_fraction": 0.5823170731707317,
      "prompt_index": 10,
      "within_set_rank_shuffle_fraction": 0.2603658536585366
    },
    {
      "layer_count": 40,
      "left_set_fraction": 0.5689024390243903,
      "prompt_index": 11,
      "within_set_rank_shuffle_fraction": 0.2707317073170732
    },
    {
      "layer_count": 40,
      "left_set_fraction": 0.5664634146341464,
      "prompt_index": 12,
      "within_set_rank_shuffle_fraction": 0.2926829268292683
    },
    {
      "layer_count": 40,
      "left_set_fraction": 0.551219512195122,
      "prompt_index": 13,
      "within_set_rank_shuffle_fraction": 0.276219512195122
    },
    {
      "layer_count": 40,
      "left_set_fraction": 0.49329268292682926,
      "prompt_index": 14,
      "within_set_rank_shuffle_fraction": 0.2926829268292683
    },
    {
      "layer_count": 40,
      "left_set_fraction": 0.5359756097560976,
      "prompt_index": 15,
      "within_set_rank_shuffle_fraction": 0.31341463414634146
    },
    {
      "layer_count": 40,
      "left_set_fraction": 0.5634146341463415,
      "prompt_index": 16,
      "within_set_rank_shuffle_fraction": 0.2829268292682927
    },
    {
      "layer_count": 40,
      "left_set_fraction": 0.5719512195121951,
      "prompt_index": 17,
      "within_set_rank_shuffle_fraction": 0.2896341463414634
    },
    {
      "layer_count": 40,
      "left_set_fraction": 0.4731707317073171,
      "prompt_index": 18,
      "within_set_rank_shuffle_fraction": 0.2939024390243902
    },
    {
      "layer_count": 40,
      "left_set_fraction": 0.5835365853658536,
      "prompt_index": 19,
      "within_set_rank_shuffle_fraction": 0.23902439024390243
    },
    {
      "layer_count": 40,
      "left_set_fraction": 0.6073170731707317,
      "prompt_index": 20,
      "within_set_rank_shuffle_fraction": 0.24329268292682926
    },
    {
      "layer_count": 40,
      "left_set_fraction": 0.574390243902439,
      "prompt_index": 21,
      "within_set_rank_shuffle_fraction": 0.27195121951219514
    },
    {
      "layer_count": 40,
      "left_set_fraction": 0.5664634146341464,
      "prompt_index": 22,
      "within_set_rank_shuffle_fraction": 0.2829268292682927
    },
    {
      "layer_count": 40,
      "left_set_fraction": 0.5121951219512195,
      "prompt_index": 23,
      "within_set_rank_shuffle_fraction": 0.31402439024390244
    }
  ]
}
```
