# Candidate-Local Residual Receiver Gate

- date: `2026-04-30`
- status: positive held-out method branch promoted to live ICLR candidate
- code: `scripts/run_source_private_learned_synonym_dictionary_packet_gate.py`
- summary: `results/source_private_candidate_local_residual_receiver_20260430/summary/`
- references: `references/547_candidate_local_residual_receiver_refs_20260430.md`

## What Changed

The public semantic-anchor adapter previously produced large matched-packet
lifts but failed strict controls: shuffled source packets, atom derangements, or
a permuted teacher could still help on some rows. This branch adds a
candidate-local residual decoder:

1. train the receiver's public adapter to predict semantic-anchor coordinates
   from public candidate surfaces;
2. build the four candidate vectors for the current example;
3. subtract the candidate-pool mean so generic public semantics are removed;
4. L2-normalize candidate residual rows and the source packet vector;
5. score candidates by the local residual dot product.

The pass rule now includes two additional strict controls:
`private_random_source_atoms`, which keeps the packet score distribution but
randomizes atom IDs, and `permuted_teacher_receiver`, which decodes the matched
source packet through a receiver trained on permuted public semantic-anchor
coordinates. `private_random_knockout` remains reported as a fragility
diagnostic, but it is no longer a hard veto because 2-4 atom packets are
expected to lose lift when a real transmitted atom is removed.

## Main Evidence

Aggregate artifact:
`results/source_private_candidate_local_residual_receiver_20260430/summary/hf_embedding_heldout_packet_summary.md`

Across three n256 seeds (`47`, `53`, `59`) plus one n512 scale run (`47`), the
aggregate summary reports:

- `36` rows total;
- `15` pass rows;
- bidirectional cross-family pass gate: `true`;
- all n256 seeds pass `core_to_holdout`, `holdout_to_core`, and
  `same_family_all`;
- n512 seed `47` also passes all three directions;
- family-qualified calibration/eval ID overlap is `0` in all promoted runs;
- transformed held-out eval surface overlap is `0` in all promoted runs.

Representative promoted rows:

| Run | Direction | Budget | Matched | Target | Best control | Delta | CI95 low | Oracle |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| n256 seed 47 | core->holdout | 8B | 0.625 | 0.250 | 0.250 | +0.375 | +0.316 | 0.875 |
| n256 seed 47 | holdout->core | 4B | 0.500 | 0.250 | 0.250 | +0.250 | +0.199 | 1.000 |
| n256 seed 53 | core->holdout | 8B | 0.625 | 0.250 | 0.258 | +0.375 | +0.320 | 0.875 |
| n256 seed 53 | holdout->core | 4B | 0.500 | 0.250 | 0.250 | +0.250 | +0.195 | 1.000 |
| n256 seed 59 | core->holdout | 8B | 0.625 | 0.250 | 0.250 | +0.375 | +0.316 | 0.875 |
| n256 seed 59 | holdout->core | 4B | 0.500 | 0.250 | 0.258 | +0.250 | +0.199 | 1.000 |
| n512 seed 47 | core->holdout | 8B | 0.625 | 0.250 | 0.250 | +0.375 | +0.332 | 0.875 |
| n512 seed 47 | holdout->core | 8B | 0.500 | 0.250 | 0.260 | +0.250 | +0.209 | 0.875 |

## Interpretation

This is the strongest positive held-out method so far. The receiver no longer
treats public semantic-anchor predictions as a global common space; it first
turns the current candidate set into a local basis. This matches the
side-information coding view: the source packet is a tiny syndrome whose meaning
depends on public receiver state.

Layman explanation: the source sends a tiny clue about what went wrong. The
target does not read that clue in isolation. It first looks at the four possible
answers, subtracts what those answers have in common, and only then asks which
answer the clue points toward. Fake clues and a scrambled receiver no longer
pass the strict rows.

## What Is Saturated

- Generic frozen embedding receivers alone remain insufficient.
- Public semantic-anchor teacher adapters without candidate-local residual
  normalization remain unsafe because permuted/shuffled controls can pass.
- Train-only calibration is too weak for this branch; broad public calibration
  helps, but promoted runs must exclude eval family-qualified IDs.

## What Is Alive

- Candidate-local residual decoding is now a live positive contribution.
- The systems story is compatible with prior packet-ISA accounting: the online
  payload is still byte-scale, and candidate features can be cached as public
  receiver state.
- The mathematical framing is now cleaner: local coordinate charts / receiver
  side information, not universal latent vectors.

## Remaining ICLR Gap

This is not yet enough for a comfortable ICLR full paper. Required next gates:

1. n512/n500 repeated seeds, not just seed `47`;
2. matched comparisons against C2C/KVComm/KVCOMM/Q-KVComm style communication
   on the same task or a carefully scoped proxy;
3. systems table separating offline public adapter fit, cold candidate feature
   build, resident candidate-cache decode, packet bytes, record bytes, and
   source text/KV exposure;
4. broader benchmarks beyond the held-out synonym repair task;
5. NVIDIA/vLLM-style serving counters before any production systems claim.

## COLM Workshop Readiness

This branch is enough to move the COLM workshop version from "negative-boundary
with systems artifact" to "positive Mac-scale method plus negative-boundary
map", assuming the paper is framed honestly:

- claim source-private byte packets decoded against local public candidate
  bases;
- include the all-public versus eval-disjoint audit;
- show the permuted-teacher and private-random atom controls;
- explicitly state that C2C/KV-cache baselines are future matched comparisons,
  not defeated baselines.
