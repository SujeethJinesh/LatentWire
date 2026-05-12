# OutlierMigrate Phase 7 Falcon-H1 Within-Lineage Replication Preregistration

**Frozen on**: 2026-05-12
**Frozen by**: Codex GPU swarm, under `swarm/goal.md` Phase 7 authorization
**Pivot depth**: 1 from OutlierMigrate Phase 0
**Status**: Frozen before any Phase 7 Falcon-H1 inference, activation rows,
or migration statistics were collected.

## Purpose

Phase 7 tests whether the OutlierMigrate decode-time top-channel migration
measurement replicates on Falcon-H1, a Mamba-2 hybrid model with a parallel
SSM/Attention topology. Granite-4 and Nemotron-3 use interleaved layer
topologies; Falcon-H1 runs SSM and Attention pathways in parallel before
combining them into the residual stream. A consistent Falcon-H1 result would
strengthen the Lineage-2 claim across hybridization topology.

## Model selection and compatibility precheck

Primary model:

- `tiiuae/Falcon-H1-0.5B-Instruct`
- HuggingFace snapshot commit observed before inference:
  `8f2587ca06bff78d8fa1adfccbe8c24d5f86b368`
- `AutoConfig` architecture: `FalconH1ForCausalLM`
- `AutoConfig` model type: `falcon_h1`
- Hidden size: `1024`
- Layers: `36`

Fallback model, only for documented infrastructure failure of the primary:

- `tiiuae/Falcon-H1-1.5B-Instruct`
- HuggingFace snapshot commit observed before inference:
  `80ebc50d7799a440b96c93bb6686a3924a09b0cb`
- `AutoConfig` architecture: `FalconH1ForCausalLM`
- Hidden size: `2048`
- Layers: `24`

Compatibility checks performed before freezing this preregistration:

- `AutoConfig.from_pretrained(..., trust_remote_code=True)` succeeded for
  both primary and fallback.
- `AutoTokenizer.from_pretrained(..., trust_remote_code=True)` succeeded for
  both primary and fallback.
- Local vLLM version `0.10.2` reports support for `FalconH1ForCausalLM`.

Because the primary model passed compatibility checks, Phase 7 uses
`tiiuae/Falcon-H1-0.5B-Instruct`. The fallback may be used only for an
infrastructure failure before any primary migration row is observed.

## Trace set

- Source: AIME-2025
- Count: 24 traces
- Selection: deterministic prompt indices 0-23
- Prompt SHA-256:
  `aa038b29332b6d137d558205ee441163e7ea4cb3cc323eb705a2f5928fd2fe4e`

The trace set matches OutlierMigrate Phase 1 and Phase 5'.

## Decode position grid

Measure at decode positions:

- `{100, 500, 1000, 5000, 10000, 20000}`

## Measurement procedure

1. Run deterministic greedy decoding for the fixed 24-prompt AIME-2025 trace
   set.
2. Capture post-block residual-stream output magnitudes at every Falcon-H1
   block at decode positions `{100, 500, 1000, 5000, 10000, 20000}`.
3. Compute the same migration metric as Phase 0/1/2/5': select top-1%
   channels at decode position 100 and count channels that move by more than
   2 rank positions at decode position 20000.
4. Bootstrap over traces with seed `20260512`.
5. Separately attempt layer/pathway stratification by hooking the SSM pathway
   output and Attention pathway output before their sum into the residual
   stream.
6. If Falcon-H1's local code layout makes separate pathway hooks impossible
   without source modification, fall back to post-sum residual-stream hooks
   only and record that limitation in the packet. Do not modify Falcon-H1
   source code.

## Reference for decision rule

The pooled Phase 0/1/2 trace-level median migration fraction, computed before
Phase 7 inference, is:

- `0.8427400914634147`

It pools the 12 Phase 0 trace metrics, 24 Phase 1 trace metrics, and 24
partial Phase 2 Nemotron-3 trace metrics.

## Decision rule

### WITHIN_LINEAGE_2_CONSISTENT

Return this decision if:

- Falcon-H1 median trace-level migration is within `0.10` absolute fraction
  of the pooled Phase 0/1/2 median `0.8427400914634147`.

This strengthens the Lineage-2 claim to cover both interleaved and parallel
Mamba-2 hybridization topologies.

### WITHIN_LINEAGE_2_DIVERGENT

Return this decision if:

- Falcon-H1 median trace-level migration differs from the pooled Phase 0/1/2
  median by more than `0.10` absolute fraction.

This narrows the Lineage-2 claim to the measured interleaved-topology models
and makes Falcon-H1 an honest exception or boundary case.

### FAIL_INFRA_PHASE7

Return this decision for model load failure, incomplete trace set, incomplete
packet, inaccessible post-block residual hooks, or any runner/checker failure
that prevents applying the decision mechanically.

## Required artifacts

The result packet must include:

- environment snapshot
- model provenance with HuggingFace snapshot commit
- prompt manifest and prompt SHA
- exact command line and stdout/stderr logs
- activation or reduced-rank evidence sufficient to recompute the metric
- metrics JSON with bootstrap CI
- migration decomposition JSON/Markdown
- checker result and artifact check
- artifact hashes
- pathway-stratification status documenting whether pre-sum SSM/Attention
  hooks were available without source modification

## Forbidden actions

- Modifying any prior preregistration.
- Modifying Falcon-H1 source code.
- Selecting a different scoring position post-hoc.
- Changing the trace set, decode grid, top-channel fraction, rank-delta
  threshold, bootstrap seed, or pooled-reference median after observing data.
- Skipping layer/pathway stratification status reporting. If separate
  pathway hooks are unavailable, report the limitation rather than omitting
  the field.

## Integration rule

If Phase 7 returns `WITHIN_LINEAGE_2_CONSISTENT`, the paper may state that
the Lineage-2 measurement covers both interleaved Mamba-2 hybrids
(Granite-4, Nemotron-3) and a parallel Mamba-2 hybrid (Falcon-H1).

If Phase 7 returns `WITHIN_LINEAGE_2_DIVERGENT`, the paper should report
Falcon-H1 as a boundary case and scope the Lineage-2 claim to interleaved
hybridization unless later human-authorized evidence changes that.

If Phase 7 returns `FAIL_INFRA_PHASE7`, the paper should omit Falcon-H1 from
the claim and mention only that Falcon-H1 was attempted but not measured if
the human wants an infrastructure limitation note.
