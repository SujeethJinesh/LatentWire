# ThoughtFlow-FP8 Phase 0 Setup Readout

Date: 2026-05-05

## Scope

Mac-only quick setup and source inventory for ThoughtFlow-FP8. No SSH, no GPU,
no global installs, no large model downloads, and no shared-file edits.

## Local State

- Project folder: `experimental/thoughtflow_fp8/`
- Existing project-local venv: `experimental/thoughtflow_fp8/.venv`
- Venv Python from prior scaffold: recorded in `progress.md` as Python 3.9.13.
- Requirements file exists: `experimental/thoughtflow_fp8/requirements.txt`
- No new packages installed in this pass.
- No competitor repositories cloned in this pass. This was intentional: the
  Phase 0/quick Phase 1 request prioritized primary-source forensics and avoided
  broad downloads.

## Primary Sources Located

| Topic | Primary source | Status |
|---|---|---|
| LongFlow paper and reviews | OpenReview forum `rz6WybXjgk`; PDF `https://openreview.net/pdf?id=rz6WybXjgk`; API notes endpoint | Read paper abstract/method claims and official reviews through API |
| Pitfalls of KV Cache Compression | OpenReview forum `dDgoYv2f7Q`; PDF `https://openreview.net/pdf?id=dDgoYv2f7Q`; API notes endpoint | Read abstract, core pitfall claims, author responses |
| DeepSeek V4 systems stack | LMSYS/SGLang launch post, 2026-04-25 | Read hybrid attention, C4/C128, ShadowRadix, Flash Compressor, Lightning TopK, training-stack notes |
| ThinKV | arXiv `2510.01290` | Read abstract/source metadata |
| R-KV | arXiv `2505.24133`; project/GitHub found | Read abstract/source metadata |
| R-KVHash | OpenReview MemAgents 2026 forum `UTRuEFJ57H` | Read abstract/source metadata |
| RaaS | ACL Findings 2025 PDF | Read abstract and method motivation |
| LazyEviction | arXiv `2506.15969` | Read abstract/source metadata |
| ForesightKV | arXiv `2602.03203` | Read abstract/source metadata |
| PM-KVQ | arXiv `2505.18610` | Read abstract/source metadata |

## Not Completed

- Did not clone LongFlow/ThinKV/R-KV/RaaS/LazyEviction/ForesightKV/PM-KVQ.
- Did not download datasets or Open-R1 traces.
- Did not run any model inference.
- Did not create Phase 2 traces or classifiers.

## Phase 0 Status

Partial pass for a quick Mac-only audit. The local scaffold exists and the
critical primary sources needed for Phase 1 were located. Full Phase 0 repo and
dataset mirroring remains incomplete by design.
