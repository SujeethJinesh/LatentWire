# Benchmark Competitor Block Contract Refs (2026-04-22)

Purpose: freeze the next comparison order and keep the paper tables
 medium-split so the benchmark story stays interpretable.

## Main Same-Pair Block

1. target-alone
2. text-to-text
3. target self-repair
4. selected-route no-repair
5. selected-route repair
6. C2C
   Link: https://arxiv.org/abs/2510.03215
7. KVCOMM (online cross-context)
   Link: https://arxiv.org/abs/2510.12872

## Long-Context Control Block

- KIVI
  Link: https://arxiv.org/abs/2402.02750
- KVQuant
  Link: https://arxiv.org/abs/2401.18079
- Task-KV
  Link: https://arxiv.org/abs/2501.15113
- KVShare
  Link: https://arxiv.org/abs/2503.16525
- KVzip
  Link: https://arxiv.org/abs/2505.23416
- H2O
  Link: https://arxiv.org/abs/2306.14048
- SnapKV
  Link: https://arxiv.org/abs/2404.14469
- SALS
  Link: https://arxiv.org/abs/2510.24273

## Appendix-Only Direct-Communication Rows

- Direct Semantic Communication via Vector Translation
  Link: https://arxiv.org/abs/2511.03945
- Latent Space Communication via K-V Cache Alignment
  Link: https://arxiv.org/abs/2601.06123
- selective-sharing KVComm
  Link: https://arxiv.org/abs/2510.03346
- Q-KVComm
  Link: https://arxiv.org/abs/2512.17914
- LatentMAS
  Link: https://arxiv.org/abs/2511.20639
- Latent-DARM
  Link: https://arxiv.org/abs/2603.09184
- When Less Latent Leads to Better Relay
  Link: https://arxiv.org/abs/2604.13349

## Pass Gates Before Widening

- same example IDs as target-alone
- numeric extraction coverage `>= 31/32`
- zero empty predictions
- strictly above target-alone by at least `1/32`
- or matched accuracy with a clear bytes/latency win

## Current Read

- Do not widen the competitor tables before a new branch survives the frozen
  GSM8K32 contract.
- Keep same-pair, long-context, and appendix-only communication papers in
  separate blocks rather than a single mixed leaderboard.
