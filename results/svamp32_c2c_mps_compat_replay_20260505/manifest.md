# SVAMP32 C2C MPS Compatibility Replay Manifest

- date: `2026-05-05`
- status: `mps_c2c_compat_replay_ready`
- pass gate: `true`

## Summary

The local Mac MPS C2C path now reproduces the archived SVAMP32 dense-teacher
surface well enough to use as a compatibility teacher for trace diagnostics:

- replay C2C accuracy: `16/32`;
- archived C2C accuracy: `16/32`;
- correct-ID overlap with archived C2C: `15/16`;
- first four prediction fields match the archived C2C artifact exactly;
- replay-only correct ID: `b1200c32546a34a5`;
- archived-only correct ID: `575d7e83d84c1e67`;
- replay target set: `10` teacher-only rows, all `10` clean after
  source-alone and text-to-text controls.

This is compatibility evidence, not a native throughput or energy measurement.

## Artifact Hashes

| Artifact | SHA256 |
|---|---|
| `c2c_generate.jsonl` | `c57e99047faafee54f267fbb5cd538baeda83d251d2a26b3cec9117eb5a6653c` |
| `c2c_generate.jsonl.meta.json` | `a4e5bf9cd7a8add5049436147c7bc9f67fcb971c73222920e875d28a47d09699` |
| `c2c_replay_target_set.json` | `dc44d9d1c073c5e22317354a83826e4004a325306e175a92ca2abf0f722f9f0a` |
| `c2c_replay_target_set.md` | `7e37dae727455e997efc8bc8edf26ca7e8204838cc0f0081ccca07bcf3f230c2` |
| `c2c_teacher_innovation_probe.json` | `6aa73256f6451311e6a2660ca3b9d1d24b94164de4fb5b72d88e06f4d0450adb` |
| `c2c_teacher_innovation_probe.md` | `2cdf9e9edb70247dd6a3cfbe9592a9ed9003744025429d0972ab28e262e3723a` |
