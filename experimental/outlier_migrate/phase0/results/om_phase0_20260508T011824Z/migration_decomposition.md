# OutlierMigrate Phase 0 Migration Decomposition

## Scope

- Run directory: `experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z`
- Base decode position: 100
- Final decode position: 10000
- Trace count: 12
- Strict top-channel set: per prompt and layer, channels with rank <= `ceil(channel_count * 0.01) - 1` at decode position 100.
- Strict set-leaving definition: a strict base top-1% channel has final rank > `ceil(channel_count * 0.01) - 1`.
- Within-set rank shuffling definition: a base top-1% channel remains in the final top-1% set but moves by more than 2 rank positions.
- Original migration definition: the preregistered checker metric, which selects top-1% channels per layer by mean magnitude at position 100 across traces and counts movement by more than 2 rank positions.
- The first two rows are post-hoc interpretability readouts using the strict set-membership definition; the third row is the unchanged gate metric from `metrics.json`.

## Aggregate Readout

| Metric | Fraction | 95% bootstrap CI |
| --- | ---: | ---: |
| Strict set-leaving | 0.634244791667 | [0.605208333333, 0.664192708333] |
| Within-set rank shuffling | 0.175260416667 | [0.160156250000, 0.191015625000] |
| Original conflated migration | 0.817838541667 | [0.797265625000, 0.836848958333] |

## Consistency Check

- `metrics.json` original migration fraction: 0.817838541667
- Reported original migration fraction: 0.817838541667

## Per-Trace Readout

| Prompt index | Strict set-leaving | Within-set rank shuffling |
| ---: | ---: | ---: |
| 0 | 0.609375000000 | 0.229687500000 |
| 1 | 0.721875000000 | 0.142187500000 |
| 2 | 0.609375000000 | 0.175000000000 |
| 3 | 0.579687500000 | 0.181250000000 |
| 4 | 0.620312500000 | 0.190625000000 |
| 5 | 0.714062500000 | 0.154687500000 |
| 6 | 0.735937500000 | 0.125000000000 |
| 7 | 0.612500000000 | 0.160937500000 |
| 8 | 0.585937500000 | 0.181250000000 |
| 9 | 0.595312500000 | 0.189062500000 |
| 10 | 0.596875000000 | 0.212500000000 |
| 11 | 0.629687500000 | 0.160937500000 |

## Machine-Readable Payload

```json
{
  "aggregate": {
    "left_set_ci95": {
      "ci95_high": 0.6641927083333333,
      "ci95_low": 0.6052083333333333
    },
    "left_set_fraction": 0.6342447916666667,
    "original_migration_ci95": {
      "ci95_high": 0.8368489583333334,
      "ci95_low": 0.797265625
    },
    "original_migration_fraction": 0.8178385416666667,
    "raw_count_left_set_fraction": 0.6342447916666667,
    "raw_count_within_set_rank_shuffle_fraction": 0.17526041666666667,
    "within_set_rank_shuffle_ci95": {
      "ci95_high": 0.191015625,
      "ci95_low": 0.16015625
    },
    "within_set_rank_shuffle_fraction": 0.17526041666666667
  },
  "base_position": 100,
  "final_position": 10000,
  "layer_metrics": [
    {
      "layer_index": 0,
      "left_set_fraction": 0.71875,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.11458333333333333
    },
    {
      "layer_index": 1,
      "left_set_fraction": 0.6822916666666666,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.16666666666666666
    },
    {
      "layer_index": 2,
      "left_set_fraction": 0.640625,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.19270833333333334
    },
    {
      "layer_index": 3,
      "left_set_fraction": 0.6458333333333334,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.203125
    },
    {
      "layer_index": 4,
      "left_set_fraction": 0.6822916666666666,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.16145833333333334
    },
    {
      "layer_index": 5,
      "left_set_fraction": 0.6666666666666666,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.1875
    },
    {
      "layer_index": 6,
      "left_set_fraction": 0.6510416666666666,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.1875
    },
    {
      "layer_index": 7,
      "left_set_fraction": 0.6770833333333334,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.16145833333333334
    },
    {
      "layer_index": 8,
      "left_set_fraction": 0.6614583333333334,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.16666666666666666
    },
    {
      "layer_index": 9,
      "left_set_fraction": 0.671875,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.18229166666666666
    },
    {
      "layer_index": 10,
      "left_set_fraction": 0.6770833333333334,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.16666666666666666
    },
    {
      "layer_index": 11,
      "left_set_fraction": 0.6979166666666666,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.15625
    },
    {
      "layer_index": 12,
      "left_set_fraction": 0.6927083333333334,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.15104166666666666
    },
    {
      "layer_index": 13,
      "left_set_fraction": 0.6875,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.17708333333333334
    },
    {
      "layer_index": 14,
      "left_set_fraction": 0.671875,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.15625
    },
    {
      "layer_index": 15,
      "left_set_fraction": 0.6458333333333334,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.171875
    },
    {
      "layer_index": 16,
      "left_set_fraction": 0.6510416666666666,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.16145833333333334
    },
    {
      "layer_index": 17,
      "left_set_fraction": 0.6666666666666666,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.15104166666666666
    },
    {
      "layer_index": 18,
      "left_set_fraction": 0.671875,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.14583333333333334
    },
    {
      "layer_index": 19,
      "left_set_fraction": 0.6614583333333334,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.16145833333333334
    },
    {
      "layer_index": 20,
      "left_set_fraction": 0.6666666666666666,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.15625
    },
    {
      "layer_index": 21,
      "left_set_fraction": 0.6510416666666666,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.15625
    },
    {
      "layer_index": 22,
      "left_set_fraction": 0.65625,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.16666666666666666
    },
    {
      "layer_index": 23,
      "left_set_fraction": 0.640625,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.16666666666666666
    },
    {
      "layer_index": 24,
      "left_set_fraction": 0.640625,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.171875
    },
    {
      "layer_index": 25,
      "left_set_fraction": 0.609375,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.1875
    },
    {
      "layer_index": 26,
      "left_set_fraction": 0.5989583333333334,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.203125
    },
    {
      "layer_index": 27,
      "left_set_fraction": 0.59375,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.21354166666666666
    },
    {
      "layer_index": 28,
      "left_set_fraction": 0.5885416666666666,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.203125
    },
    {
      "layer_index": 29,
      "left_set_fraction": 0.5885416666666666,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.234375
    },
    {
      "layer_index": 30,
      "left_set_fraction": 0.6197916666666666,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.17708333333333334
    },
    {
      "layer_index": 31,
      "left_set_fraction": 0.6354166666666666,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.19270833333333334
    },
    {
      "layer_index": 32,
      "left_set_fraction": 0.578125,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.203125
    },
    {
      "layer_index": 33,
      "left_set_fraction": 0.5677083333333334,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.17708333333333334
    },
    {
      "layer_index": 34,
      "left_set_fraction": 0.546875,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.203125
    },
    {
      "layer_index": 35,
      "left_set_fraction": 0.5520833333333334,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.19791666666666666
    },
    {
      "layer_index": 36,
      "left_set_fraction": 0.5572916666666666,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.20833333333333334
    },
    {
      "layer_index": 37,
      "left_set_fraction": 0.5572916666666666,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.1875
    },
    {
      "layer_index": 38,
      "left_set_fraction": 0.578125,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.13541666666666666
    },
    {
      "layer_index": 39,
      "left_set_fraction": 0.5208333333333334,
      "trace_count": 12,
      "within_set_rank_shuffle_fraction": 0.14583333333333334
    }
  ],
  "observed_trace_count": 12,
  "original_metrics_json_ci95": {
    "ci95_high": 0.8368489583333334,
    "ci95_low": 0.797265625
  },
  "original_metrics_json_migration_fraction": 0.8178385416666667,
  "phase": 0,
  "rank_delta_strictly_greater_than": 2,
  "run_dir": "experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z",
  "strict_set_membership_selection": "per prompt and layer, channels with position-100 rank <= ceil(channel_count * 0.01) - 1",
  "top_boundary_semantics": "zero-based rank <= ceil(channel_count * 0.01) - 1",
  "top_channel_fraction": 0.01,
  "trace_count": 12,
  "trace_metrics": [
    {
      "layer_count": 40,
      "left_set_fraction": 0.609375,
      "prompt_index": 0,
      "within_set_rank_shuffle_fraction": 0.2296875
    },
    {
      "layer_count": 40,
      "left_set_fraction": 0.721875,
      "prompt_index": 1,
      "within_set_rank_shuffle_fraction": 0.1421875
    },
    {
      "layer_count": 40,
      "left_set_fraction": 0.609375,
      "prompt_index": 2,
      "within_set_rank_shuffle_fraction": 0.175
    },
    {
      "layer_count": 40,
      "left_set_fraction": 0.5796875,
      "prompt_index": 3,
      "within_set_rank_shuffle_fraction": 0.18125
    },
    {
      "layer_count": 40,
      "left_set_fraction": 0.6203125,
      "prompt_index": 4,
      "within_set_rank_shuffle_fraction": 0.190625
    },
    {
      "layer_count": 40,
      "left_set_fraction": 0.7140625,
      "prompt_index": 5,
      "within_set_rank_shuffle_fraction": 0.1546875
    },
    {
      "layer_count": 40,
      "left_set_fraction": 0.7359375,
      "prompt_index": 6,
      "within_set_rank_shuffle_fraction": 0.125
    },
    {
      "layer_count": 40,
      "left_set_fraction": 0.6125,
      "prompt_index": 7,
      "within_set_rank_shuffle_fraction": 0.1609375
    },
    {
      "layer_count": 40,
      "left_set_fraction": 0.5859375,
      "prompt_index": 8,
      "within_set_rank_shuffle_fraction": 0.18125
    },
    {
      "layer_count": 40,
      "left_set_fraction": 0.5953125,
      "prompt_index": 9,
      "within_set_rank_shuffle_fraction": 0.1890625
    },
    {
      "layer_count": 40,
      "left_set_fraction": 0.596875,
      "prompt_index": 10,
      "within_set_rank_shuffle_fraction": 0.2125
    },
    {
      "layer_count": 40,
      "left_set_fraction": 0.6296875,
      "prompt_index": 11,
      "within_set_rank_shuffle_fraction": 0.1609375
    }
  ]
}
```
