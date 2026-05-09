# ThoughtFlow-FP8 Reproducibility Audit

Date: 2026-05-09

Scope: read-only audit for fallback camera-ready-candidate readiness under the
falsification-methodology framing. This does not mark the paper camera-ready
final and does not count as a positive-method candidate.

## Checks

1. Paper-polish checker:
   `experimental/thoughtflow_fp8/phase2/check_paper_buildable.py
   experimental/thoughtflow_fp8/phase2/results/thoughtflow_paper_polish_20260508T0050Z`

   Result: PASS. The checker returned `PASS_THOUGHTFLOW_PAPER_BUILDABLE`.

2. Owned tests:

   ```bash
   PYTHONPATH="$PWD" TRITON_CPU_BACKEND=1 TRITON_INTERPRET=1 \
     TRITON_HOME="$PWD/.debug/triton_home" \
     ./.venv_gpu/bin/python -m pytest \
     experimental/thoughtflow_fp8/phase2/tests \
     experimental/thoughtflow_fp8/phase4/tests -q
   ```

   Result: PASS. `70 passed, 1 warning`.

3. Reviewer-pack paths:

   Result: PASS. Referenced paths exist for the paper-polish packet, built
   PDF, committee reviews, current decision manifest, diagnostic packet,
   diagnostic packet manifest, and paper PDF/TeX.

4. Draft-marker and caveat audit:

   Result: PASS. The TeX has no `TODO`, `FIXME`, or `XXX`. It contains the
   required caveats: no positive method, no real FP8/CUDA/latency/throughput
   or live compression-method claim, Mac-local saved-trace fixtures, and proxy
   baselines that are not faithful implementations.

5. Committee threshold:

   Result: PASS under falsification-methodology framing. The 2026-05-09
   committee review records `7/10`, `7/10`, and `7/10`.

## Candidate Decision

ThoughtFlow-FP8 is a fallback camera-ready candidate only as a
falsification-methodology workshop diagnostic. It is not camera-ready final,
not a positive-method candidate, and not an MLSys systems paper. Human final
review remains required for title, venue framing, citation confidence, and
copyedit.

No stop condition fired: no p-hacking, post-hoc cherry-picking,
preregistration scope creep, citation hallucination, or paper-test regression
was found in this audit.
