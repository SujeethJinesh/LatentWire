# SA-CODE Release Code Audit

Date: 2026-05-26

Scope: `release/` code, docs, tests, release paper copy, and `swarm/final_report.md`.
This audit did not edit release code, paper, swarm, experiments, or existing docs.

Current paper readiness: not ICLR-ready as a positive-method paper. Current
story: OutlierMigrate/SA-CODE has a workshop-grade scoped mechanism result:
long-decode channel-set drift is robust, M11b is a partial budget remedy, and
ParoQuant remains a stronger Granite baseline. Blocking gap: no clean,
deployable cross-model positive method; release trustworthiness is also not yet
full-fidelity because `release/` is FAST_VERIFY replay plus small helpers.

Latest inputs read: prior code/repro/statistical/adversarial audits in `docs/`,
`swarm/final_report.md`, `release/VERIFICATION.md`, release docs/tests/code, and
`release/paper/paper.tex`.

## Commands Run

- `source ./.venv_gpu/bin/activate && cd release && PYTHONDONTWRITEBYTECODE=1 python -m pytest tests`
  - Result: `7 passed in 0.18s`.
- `PYTHONPATH=src python src/scripts/reproduce_set_leaving.py --config configs/granite_small.yaml --output-dir ../.debug/release_audit/set_leaving --fast-verify`
  - Result: passed.
- `PYTHONPATH=src python src/scripts/reproduce_m2.py --config configs/granite_small.yaml --output-dir ../.debug/release_audit/m2 --dry-run`
  - Result: passed.
- `PYTHONPATH=src python src/scripts/reproduce_m11b.py --config configs/granite_small.yaml --output-dir ../.debug/release_audit/m11b_full_mode_probe`
  - Result: failed loudly with `RuntimeError: Full-fidelity GPU reproduction is not implemented in release/.`
- CPU edge probes confirmed `kl_divergence([-1, 2], [0.5, 0.5])` returns a finite value, zero-mass KL returns `0.0`, `bootstrap_ci(np.array(...))` raises ambiguous truth-value `ValueError`, and empty autocorrelation emits NumPy warnings then returns `0`.

## Findings

### CRITICAL

1. Public release paper overclaims full reproduction and names a nonexistent
   script path.

   Evidence: `release/paper/paper.tex:403` says the release interface includes
   `release/scripts/reproduce_all.sh`, but the script is actually under
   `release/src/scripts/reproduce_all.sh:1`. The same sentence claims
   full-fidelity reproduction reruns every paper-referenced method on a GPU and
   that fast mode checks the same code paths with reduced trace counts. That
   contradicts `release/src/outlier_migrate/data.py:88-93`, which raises for
   non-fast/non-dry mode, `release/README.md:41-45`, which says full runners are
   not included, and `release/VERIFICATION.md:166-168`, which says full GPU
   reproduction is not implemented or verified. `release/paper/paper.tex:419`
   also says exact snapshot commits and file hashes are recorded in release
   documentation; I did not find those hashes in the release docs.

   Risk: reviewers can reasonably interpret the release PDF as claiming a
   standalone full-GPU artifact even though the executable release package only
   supports frozen-claim replay. This is the same failure mode prior audits
   marked critical, but it remains in the release paper copy.

   Concrete fix: either implement real full-fidelity runners in `release/` and
   make `release/scripts` or the paper path true, or revise the release paper
   reproducibility statement to say `release/src/scripts/reproduce_all.sh`
   performs FAST_VERIFY replay only, with full reruns delegated to archived
   experimental packet runners. Add checkpoint snapshot IDs and file hashes, or
   remove that claim.

### SUBSTANTIAL

2. FAST_VERIFY is hardcoded claim replay, and configs do not own the claims they
   appear to control.

   Evidence: all paper values live in module constants at
   `release/src/outlier_migrate/data.py:11-40`; `build_reproduction_payload`
   returns those values directly in fast mode at
   `release/src/outlier_migrate/data.py:94-98`. Individual scripts load YAML
   but do not use config fields to select model, traces, seed, positions, or
   output directory, e.g. `release/src/scripts/reproduce_m11b.py:20-22` and
   `release/src/scripts/reproduce_set_leaving.py:20-29`. `reproduce_all.sh`
   hardcodes `configs/granite_small.yaml` at `release/src/scripts/reproduce_all.sh:5`
   and passes it to every claim family at `release/src/scripts/reproduce_all.sh:12-25`,
   even for Nemotron, DeepSeek, and Falcon claims. Config fields such as
   `bootstrap_seed` and `output_dir` in `release/configs/granite_small.yaml:6-9`
   and `release/configs/nemotron_3_nano.yaml:6-9` are not enforced by the
   scripts.

   Risk: the release is deterministic, but config determinism is mostly
   decorative. A reviewer can change a config model, seed, trace count, or
   output directory and still get the same frozen paper claims.

   Concrete fix: move frozen expected values into explicit result manifests with
   provenance paths and hashes; have scripts validate that the selected config
   matches the claim family; use `output_dir`, `bootstrap_seed`, `traces`, and
   `decode_positions`; dispatch per-model configs in `reproduce_all.sh`; fail if
   a config cannot possibly support the requested claim.

3. Several method helpers do not match the paper algorithms at a defensible
   interface level.

   Evidence: the paper claims seven method classes in
   `release/paper/paper.tex:62` and discusses budget-tuned EMA in
   `release/paper/paper.tex:131`. The release `m11b_topk` is just
   `static_topk(scores, budget)` at `release/src/outlier_migrate/methods/m11b.py:19-22`;
   the only EMA helper is a single-step arithmetic function with no alpha bounds
   or state ownership at `release/src/outlier_migrate/methods/m11b.py:11-16`.
   `M26` selects the highest stability counts at
   `release/src/outlier_migrate/methods/m26.py:10-18`, not an explicit
   all-position stable-core intersection. ParoQuant is represented as a
   two-column Givens rotation at `release/src/outlier_migrate/methods/paroquant.py:8-18`,
   while rotation plus budget composition is a `NotImplementedError` stub at
   `release/src/outlier_migrate/methods/composition.py:8-15`.

   Risk: the helper names look like paper algorithms, but they are closer to toy
   mask/rotation interfaces. This is acceptable only if the release clearly
   labels them as minimal helpers and does not imply algorithmic reproduction.

   Concrete fix: either rename these helpers as interface sketches, or implement
   trace/layer/position-aware method APIs that consume the same inputs as the
   experimental packets and emit the same protected sets, budgets, and recovery
   metrics. Add method-level tests for EMA state evolution, budget sweeps, stable
   core construction, and ParoQuant composition.

4. Metric/analysis helpers silently accept invalid inputs or fail on common
   array inputs.

   Evidence: `kl_divergence` converts to arrays, clips all values below `eps`,
   and normalizes at `release/src/outlier_migrate/metrics.py:31-38`; negative
   probabilities and zero-mass vectors therefore produce finite outputs instead
   of failing. `bootstrap_ci` checks `if not values` at
   `release/src/outlier_migrate/metrics.py:42-50`, which raises for a nonempty
   NumPy array. `autocorrelation_length` lacks the dimensionality/size guard used
   by `spectral_entropy`; empty arrays warn and return `0` through
   `release/src/outlier_migrate/analysis.py:22-34`.

   Risk: invalid distributions, empty traces, or NumPy inputs can be masked or
   fail inconsistently, which is exactly the kind of silent/default behavior that
   weakens a release artifact.

   Concrete fix: require finite, nonnegative probability vectors with positive
   mass before KL normalization; use `arr.size == 0` in bootstrap; validate
   `samples > 0`; make autocorrelation reject non-1D or empty inputs with
   `ValueError`. Add tests that encode these failure modes.

5. Environment verification does not verify the optional full stack it documents.

   Evidence: `release/README.md:19-23` documents a full GPU install. The `full`
   extra pins `torch`, `transformers`, and `vllm` at `release/pyproject.toml:17-22`.
   `verify_environment.py` checks only `numpy` and `PyYAML` by default at
   `release/src/scripts/verify_environment.py:17-19`; with `--require-gpu`, it
   imports `torch` and checks CUDA at `release/src/scripts/verify_environment.py:21-26`,
   but never checks `transformers`, `vllm`, model config compatibility, disk,
   checkpoint accessibility, or snapshot hashes. `release/VERIFICATION.md:66-73`
   confirms no checkpoint download was exercised.

   Risk: a user can install `.[full,dev]` and pass environment verification
   without proving the documented inference stack is usable. Because full
   reproduction is not implemented, this should be framed as unverified rather
   than tested.

   Concrete fix: add `--mode fast|full` or `--require-full-stack`; in full mode,
   check all pinned packages, CUDA, model-loader availability, configured
   checkpoint IDs/snapshots, and writable output space. Until full runners exist,
   README should say the full extra is dependency pinning only, not a verified
   reproduction path.

6. Tests are too shallow to catch the release's main artifact risks.

   Evidence: method tests only check mask shape and budget at
   `release/tests/test_methods.py:12-18`; metric tests cover three happy paths at
   `release/tests/test_metrics.py:8-17`; quantization tests cover bounds and one
   mask example at `release/tests/test_quantization.py:8-20`. There are no tests
   that assert full mode fails loudly, that scripts use or reject config fields
   correctly, that frozen claims match `docs/reproducing_results.md`, that
   invalid metric inputs fail, or that the registry is isolated across tests.

   Risk: `7 passed` proves the small helper functions still run, not that the
   release package maps to paper claims or protects reviewers from misleading
   reproduction modes.

   Concrete fix: add subprocess tests for each script mode; add a manifest test
   comparing `PAPER_CLAIMS` keys and tolerances to `release/docs/reproducing_results.md`;
   add config mutation tests that must fail when model/seed/positions mismatch;
   add invalid-input tests for metrics and method masks; reset or namespace the
   method registry in tests.

### MINOR

7. Mask builders lack input-shape, finite-value, and deterministic tie policy
   validation.

   Evidence: `static_topk` uses `scores.size`, `np.argpartition`, and a flat mask
   at `release/src/outlier_migrate/methods/static.py:13-17`. It does not require
   one-dimensional finite scores or define tie-breaking. `m11b` and `decdec`
   inherit this behavior through `release/src/outlier_migrate/methods/m11b.py:19-22`
   and `release/src/outlier_migrate/methods/decdec.py:11-14`.

   Risk: multidimensional arrays, NaNs, or exact ties can produce surprising
   protected sets. This is not currently exposed by FAST_VERIFY, but it matters
   if these helpers become public extension points.

   Concrete fix: coerce and validate a 1-D finite array, reject NaNs/Infs, and
   use an explicit stable tie policy such as sorting by `(-score, channel_index)`.

8. Release docs are mostly honest after the prior fixes, but the opening README
   sentence is still broader than the executable scope.

   Evidence: `release/README.md:3-7` says the folder reproduces empirical claims,
   while `release/README.md:35-45` clarifies that fast verification replays
   frozen paper claims and full model runners are absent.

   Risk: lower than the release paper issue because the README later corrects
   the scope, but the first sentence can still be quoted out of context.

   Concrete fix: change the opening to "This folder provides FAST_VERIFY claim
   replay and minimal public interfaces for..." unless full runners are added.

9. Generated cache files are present in the working release tree, though ignored.

   Evidence: `release/.gitignore:1-6` ignores `results/`, `.pytest_cache/`,
   `__pycache__/`, and `*.pyc`; the working tree currently contains such files
   under `release/`. They are not tracked by `git ls-files`, so this is packaging
   hygiene rather than source-code risk.

   Risk: a manual archive of the working directory could accidentally include
   local caches or stale generated JSON outputs.

   Concrete fix: before creating a release archive, clean ignored generated
   files from `release/` or build from a fresh checkout.

## Saturated, Alive, Next Gate

- Saturated: FAST_VERIFY itself is deterministic and now fails loudly outside
  supported modes.
- Alive: release artifact honesty, especially the paper reproducibility
  statement and config/claim ownership.
- Highest-priority next gate: fix the CRITICAL release-paper reproducibility
  mismatch, then add script-level tests that prove full mode fails, configs are
  not decorative, and frozen claims map to documented paper values.
