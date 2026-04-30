# Mac 128B Boundary Accounting References

- date: `2026-04-30`
- purpose: correct verifier packet-consumption accounting to use observed
  Mac-local cache-line size rather than a generic 64B assumption
- current blocker helped: systems reviewers will reject byte/traffic claims if
  the local hardware floor is hard-coded incorrectly

## Local Hardware Fact

`sysctl hw.cachelinesize` on this Mac reports `128`.

Mechanism/design impact: the verifier trace now auto-detects the local
cache-line floor and records the source as `sysctl hw.cachelinesize`. For the
current 5-byte packet record, single-request line traffic is `128B`, batch-64
traffic is `6.0B/request`, and batch-256 traffic is `5.0B/request`.

Role: systems correction / reproducibility guard.

## Systems Comparators Kept As Claim Boundaries

1. **C2C / Cache-to-Cache**
   - Sources: https://arxiv.org/abs/2510.03215 and
     https://fuvty.github.io/C2C_Project_Page/
   - Blocker helped: direct KV/cache communication is the closest
     cross-model systems comparator.
   - Mechanism/design idea: project and merge source KV/cache state into a
     target model rather than sending a source-private evidence packet.
   - Next experiment impact: compare against KV/cache byte floors and privacy
     exposure flags, not just raw payload bytes.
   - Role: competitor / systems boundary.

2. **KVCOMM**
   - Sources: https://openreview.net/forum?id=yGOytgjurF and
     https://arxiv.org/abs/2510.03346
   - Blocker helped: selective KV communication can reduce cache transfer, so
     naive fp16 KV baselines are weak.
   - Mechanism/design idea: communicate selected KV layers/segments.
   - Next experiment impact: report source-KV exposure separately from
     source-private packet exposure.
   - Role: competitor / systems baseline.

3. **KIVI and KVQuant**
   - Sources: https://arxiv.org/abs/2402.02750 and
     https://arxiv.org/abs/2401.18079
   - Blocker helped: compressed KV caches lower memory traffic substantially.
   - Mechanism/design idea: asymmetric low-bit KV quantization and outlier
     handling.
   - Next experiment impact: use compressed KV byte floors as the honest
     systems comparator, not only fp16 KV.
   - Role: systems baseline.

4. **QJL and TurboQuant**
   - Sources: https://arxiv.org/abs/2406.03482 and
     https://arxiv.org/abs/2504.19874
   - Blocker helped: one-bit sketching/protected rotations are the strongest
     compression-side inspiration for future packets.
   - Mechanism/design idea: random sign sketches and product/vector
     quantization of residual state.
   - Next experiment impact: keep QJL/TurboQuant as inspiration for
     TurboResidual packet branches while marking KV rows as source-KV-exposed.
   - Role: baseline / method inspiration.

5. **vLLM/PagedAttention and DistServe**
   - Sources: https://arxiv.org/abs/2309.06180 and
     https://arxiv.org/abs/2401.09670
   - Blocker helped: production serving claims require TTFT/TPOT/goodput, not
     only byte accounting.
   - Mechanism/design idea: paged KV memory management and prefill/decode
     disaggregation.
   - Next experiment impact: current Mac trace must be labeled as boundary
     traffic plus CPU receiver telemetry; native serving telemetry remains a
     separate gate.
   - Role: systems framing / non-claim.

## Decision

Promote the Mac-local verifier trace only after hardware-observed line-size
accounting is present in the artifact. The corrected seed31 trace passes, but
the systems claim remains boundary traffic and receiver consumption, not
production GPU serving speed.
