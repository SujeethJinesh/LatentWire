# OutlierMigrate Phase 2 Migration Decomposition

## Scope

- Run directory: `experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z`
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
| Strict set-leaving | 0.533713200380 | [0.467414529915, 0.599982193732] |
| Within-set rank shuffling | 0.269082383666 | [0.238841405508, 0.299056267806] |
| Original conflated migration | 0.820809591643 | [0.786532526116, 0.854493114910] |

## Consistency Check

- `metrics.json` original migration fraction: 0.820809591643
- Reported original migration fraction: 0.820809591643

## Per-Trace Readout

| Prompt index | Strict set-leaving | Within-set rank shuffling |
| ---: | ---: | ---: |
| 0 | 0.433048433048 | 0.325498575499 |
| 1 | 0.418091168091 | 0.309829059829 |
| 2 | 0.764957264957 | 0.148148148148 |
| 3 | 0.666666666667 | 0.212250712251 |
| 4 | 0.366809116809 | 0.346866096866 |
| 5 | 0.664529914530 | 0.237891737892 |
| 6 | 0.390313390313 | 0.328347578348 |
| 7 | 0.700854700855 | 0.195868945869 |
| 8 | 0.762820512821 | 0.168091168091 |
| 9 | 0.377492877493 | 0.334045584046 |
| 10 | 0.726495726496 | 0.185185185185 |
| 11 | 0.381054131054 | 0.335470085470 |
| 12 | 0.381054131054 | 0.346153846154 |
| 13 | 0.398148148148 | 0.327635327635 |
| 14 | 0.450142450142 | 0.269943019943 |
| 15 | 0.532051282051 | 0.307692307692 |
| 16 | 0.790598290598 | 0.136039886040 |
| 17 | 0.745014245014 | 0.181623931624 |
| 18 | 0.253561253561 | 0.383190883191 |
| 19 | 0.405270655271 | 0.326210826211 |
| 20 | 0.384615384615 | 0.341880341880 |
| 21 | 0.683048433048 | 0.214387464387 |
| 22 | 0.383903133903 | 0.321937321937 |
| 23 | 0.748575498575 | 0.173789173789 |

## Machine-Readable Payload

```json
{
  "aggregate": {
    "left_set_ci95": {
      "ci95_high": 0.5999821937321937,
      "ci95_low": 0.4674145299145299
    },
    "left_set_fraction": 0.533713200379867,
    "original_migration_ci95": {
      "ci95_high": 0.8544931149097815,
      "ci95_low": 0.7865325261158594
    },
    "original_migration_fraction": 0.820809591642925,
    "raw_count_left_set_fraction": 0.533713200379867,
    "raw_count_within_set_rank_shuffle_fraction": 0.269082383665717,
    "within_set_rank_shuffle_ci95": {
      "ci95_high": 0.2990562678062678,
      "ci95_low": 0.23884140550807217
    },
    "within_set_rank_shuffle_fraction": 0.26908238366571696
  },
  "base_position": 100,
  "final_position": 20000,
  "layer_metrics": [
    {
      "layer_index": 0,
      "left_set_fraction": 0.4074074074074074,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.21604938271604937
    },
    {
      "layer_index": 1,
      "left_set_fraction": 0.45987654320987653,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.23765432098765432
    },
    {
      "layer_index": 2,
      "left_set_fraction": 0.5092592592592592,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2345679012345679
    },
    {
      "layer_index": 3,
      "left_set_fraction": 0.41358024691358025,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.33796296296296297
    },
    {
      "layer_index": 4,
      "left_set_fraction": 0.5015432098765432,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2345679012345679
    },
    {
      "layer_index": 5,
      "left_set_fraction": 0.5262345679012346,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2685185185185185
    },
    {
      "layer_index": 6,
      "left_set_fraction": 0.5015432098765432,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.3533950617283951
    },
    {
      "layer_index": 7,
      "left_set_fraction": 0.5108024691358024,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.3132716049382716
    },
    {
      "layer_index": 8,
      "left_set_fraction": 0.5354938271604938,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2962962962962963
    },
    {
      "layer_index": 9,
      "left_set_fraction": 0.5385802469135802,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.30709876543209874
    },
    {
      "layer_index": 10,
      "left_set_fraction": 0.5493827160493827,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.29166666666666663
    },
    {
      "layer_index": 11,
      "left_set_fraction": 0.5817901234567902,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.24228395061728394
    },
    {
      "layer_index": 12,
      "left_set_fraction": 0.5756172839506173,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2638888888888889
    },
    {
      "layer_index": 13,
      "left_set_fraction": 0.5617283950617283,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2716049382716049
    },
    {
      "layer_index": 14,
      "left_set_fraction": 0.5462962962962963,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2716049382716049
    },
    {
      "layer_index": 15,
      "left_set_fraction": 0.5540123456790124,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.27932098765432095
    },
    {
      "layer_index": 16,
      "left_set_fraction": 0.5864197530864197,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2824074074074074
    },
    {
      "layer_index": 17,
      "left_set_fraction": 0.5679012345679012,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2901234567901234
    },
    {
      "layer_index": 18,
      "left_set_fraction": 0.595679012345679,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.28703703703703703
    },
    {
      "layer_index": 19,
      "left_set_fraction": 0.5864197530864197,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.28703703703703703
    },
    {
      "layer_index": 20,
      "left_set_fraction": 0.5910493827160493,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2824074074074074
    },
    {
      "layer_index": 21,
      "left_set_fraction": 0.595679012345679,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.26543209876543206
    },
    {
      "layer_index": 22,
      "left_set_fraction": 0.5848765432098765,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2623456790123457
    },
    {
      "layer_index": 23,
      "left_set_fraction": 0.6080246913580247,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2361111111111111
    },
    {
      "layer_index": 24,
      "left_set_fraction": 0.5925925925925926,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.22839506172839505
    },
    {
      "layer_index": 25,
      "left_set_fraction": 0.6033950617283951,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.24382716049382713
    },
    {
      "layer_index": 26,
      "left_set_fraction": 0.6049382716049383,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.22530864197530864
    },
    {
      "layer_index": 27,
      "left_set_fraction": 0.5848765432098765,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.22839506172839505
    },
    {
      "layer_index": 28,
      "left_set_fraction": 0.6033950617283951,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.21141975308641975
    },
    {
      "layer_index": 29,
      "left_set_fraction": 0.566358024691358,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.23302469135802467
    },
    {
      "layer_index": 30,
      "left_set_fraction": 0.5277777777777778,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2685185185185185
    },
    {
      "layer_index": 31,
      "left_set_fraction": 0.5138888888888888,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2716049382716049
    },
    {
      "layer_index": 32,
      "left_set_fraction": 0.5169753086419753,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.25308641975308643
    },
    {
      "layer_index": 33,
      "left_set_fraction": 0.529320987654321,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2685185185185185
    },
    {
      "layer_index": 34,
      "left_set_fraction": 0.5046296296296297,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2638888888888889
    },
    {
      "layer_index": 35,
      "left_set_fraction": 0.49537037037037035,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2716049382716049
    },
    {
      "layer_index": 36,
      "left_set_fraction": 0.5,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2561728395061728
    },
    {
      "layer_index": 37,
      "left_set_fraction": 0.4984567901234568,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2700617283950617
    },
    {
      "layer_index": 38,
      "left_set_fraction": 0.47993827160493824,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2901234567901234
    },
    {
      "layer_index": 39,
      "left_set_fraction": 0.5030864197530864,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.25771604938271603
    },
    {
      "layer_index": 40,
      "left_set_fraction": 0.49691358024691357,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.27314814814814814
    },
    {
      "layer_index": 41,
      "left_set_fraction": 0.5046296296296297,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.24691358024691357
    },
    {
      "layer_index": 42,
      "left_set_fraction": 0.5555555555555556,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.21450617283950615
    },
    {
      "layer_index": 43,
      "left_set_fraction": 0.5570987654320988,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.22530864197530864
    },
    {
      "layer_index": 44,
      "left_set_fraction": 0.5,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.31481481481481477
    },
    {
      "layer_index": 45,
      "left_set_fraction": 0.4845679012345679,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.31481481481481477
    },
    {
      "layer_index": 46,
      "left_set_fraction": 0.5046296296296297,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.30246913580246915
    },
    {
      "layer_index": 47,
      "left_set_fraction": 0.5015432098765432,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.30709876543209874
    },
    {
      "layer_index": 48,
      "left_set_fraction": 0.5061728395061729,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.27469135802469136
    },
    {
      "layer_index": 49,
      "left_set_fraction": 0.5447530864197531,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.24691358024691357
    },
    {
      "layer_index": 50,
      "left_set_fraction": 0.5200617283950617,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.2700617283950617
    },
    {
      "layer_index": 51,
      "left_set_fraction": 0.46296296296296297,
      "trace_count": 24,
      "within_set_rank_shuffle_fraction": 0.3472222222222222
    }
  ],
  "observed_trace_count": 24,
  "original_metrics_json_ci95": {
    "ci95_high": 0.8544931149097815,
    "ci95_low": 0.7865325261158594
  },
  "original_metrics_json_migration_fraction": 0.820809591642925,
  "phase": 2,
  "rank_delta_strictly_greater_than": 2,
  "run_dir": "experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z",
  "strict_set_membership_selection": "per prompt and layer, channels with position-100 rank <= ceil(channel_count * 0.01) - 1",
  "top_boundary_semantics": "zero-based rank <= ceil(channel_count * 0.01) - 1",
  "top_channel_fraction": 0.01,
  "trace_count": 24,
  "trace_metrics": [
    {
      "layer_count": 52,
      "left_set_fraction": 0.43304843304843305,
      "prompt_index": 0,
      "within_set_rank_shuffle_fraction": 0.32549857549857547
    },
    {
      "layer_count": 52,
      "left_set_fraction": 0.4180911680911681,
      "prompt_index": 1,
      "within_set_rank_shuffle_fraction": 0.30982905982905984
    },
    {
      "layer_count": 52,
      "left_set_fraction": 0.7649572649572649,
      "prompt_index": 2,
      "within_set_rank_shuffle_fraction": 0.14814814814814814
    },
    {
      "layer_count": 52,
      "left_set_fraction": 0.6666666666666666,
      "prompt_index": 3,
      "within_set_rank_shuffle_fraction": 0.21225071225071224
    },
    {
      "layer_count": 52,
      "left_set_fraction": 0.36680911680911676,
      "prompt_index": 4,
      "within_set_rank_shuffle_fraction": 0.3468660968660969
    },
    {
      "layer_count": 52,
      "left_set_fraction": 0.6645299145299145,
      "prompt_index": 5,
      "within_set_rank_shuffle_fraction": 0.23789173789173787
    },
    {
      "layer_count": 52,
      "left_set_fraction": 0.3903133903133903,
      "prompt_index": 6,
      "within_set_rank_shuffle_fraction": 0.32834757834757833
    },
    {
      "layer_count": 52,
      "left_set_fraction": 0.7008547008547008,
      "prompt_index": 7,
      "within_set_rank_shuffle_fraction": 0.19586894586894585
    },
    {
      "layer_count": 52,
      "left_set_fraction": 0.7628205128205128,
      "prompt_index": 8,
      "within_set_rank_shuffle_fraction": 0.16809116809116809
    },
    {
      "layer_count": 52,
      "left_set_fraction": 0.37749287749287747,
      "prompt_index": 9,
      "within_set_rank_shuffle_fraction": 0.33404558404558404
    },
    {
      "layer_count": 52,
      "left_set_fraction": 0.7264957264957265,
      "prompt_index": 10,
      "within_set_rank_shuffle_fraction": 0.18518518518518517
    },
    {
      "layer_count": 52,
      "left_set_fraction": 0.38105413105413105,
      "prompt_index": 11,
      "within_set_rank_shuffle_fraction": 0.33547008547008544
    },
    {
      "layer_count": 52,
      "left_set_fraction": 0.38105413105413105,
      "prompt_index": 12,
      "within_set_rank_shuffle_fraction": 0.34615384615384615
    },
    {
      "layer_count": 52,
      "left_set_fraction": 0.39814814814814814,
      "prompt_index": 13,
      "within_set_rank_shuffle_fraction": 0.3276353276353276
    },
    {
      "layer_count": 52,
      "left_set_fraction": 0.45014245014245013,
      "prompt_index": 14,
      "within_set_rank_shuffle_fraction": 0.26994301994301995
    },
    {
      "layer_count": 52,
      "left_set_fraction": 0.532051282051282,
      "prompt_index": 15,
      "within_set_rank_shuffle_fraction": 0.30769230769230765
    },
    {
      "layer_count": 52,
      "left_set_fraction": 0.7905982905982906,
      "prompt_index": 16,
      "within_set_rank_shuffle_fraction": 0.13603988603988604
    },
    {
      "layer_count": 52,
      "left_set_fraction": 0.745014245014245,
      "prompt_index": 17,
      "within_set_rank_shuffle_fraction": 0.18162393162393162
    },
    {
      "layer_count": 52,
      "left_set_fraction": 0.25356125356125353,
      "prompt_index": 18,
      "within_set_rank_shuffle_fraction": 0.3831908831908832
    },
    {
      "layer_count": 52,
      "left_set_fraction": 0.40527065527065526,
      "prompt_index": 19,
      "within_set_rank_shuffle_fraction": 0.3262108262108262
    },
    {
      "layer_count": 52,
      "left_set_fraction": 0.3846153846153846,
      "prompt_index": 20,
      "within_set_rank_shuffle_fraction": 0.3418803418803419
    },
    {
      "layer_count": 52,
      "left_set_fraction": 0.683048433048433,
      "prompt_index": 21,
      "within_set_rank_shuffle_fraction": 0.21438746438746437
    },
    {
      "layer_count": 52,
      "left_set_fraction": 0.3839031339031339,
      "prompt_index": 22,
      "within_set_rank_shuffle_fraction": 0.32193732193732194
    },
    {
      "layer_count": 52,
      "left_set_fraction": 0.7485754985754985,
      "prompt_index": 23,
      "within_set_rank_shuffle_fraction": 0.17378917378917377
    }
  ]
}
```
