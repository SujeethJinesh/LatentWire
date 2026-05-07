# Triton CPU Source Install Readout

- date: 2026-05-06
- environment: `./venv_arm64`
- source clone: `.debug/triton-cpu-src`
- source revision: `270e696`
- submodule: `third_party/sleef` at `93f04d8`
- installed package: `triton==3.7.0+git270e696d`

## Source Guidance Checked

- Official Triton installation docs: `https://triton-lang.org/main/getting-started/installation.html`
- Experimental Triton CPU backend: `https://github.com/triton-lang/triton-cpu`

The official package-index route is still unavailable on this Mac arm64 venv:
`pip install triton` and `pip index versions triton`, `triton-cpu`, and
`triton-nightly` do not expose a compatible wheel. The working local route was
therefore a source install of `triton-cpu`, following Triton's source-build
shape with the CPU backend repository.

## Commands That Mattered

```bash
PIP_CACHE_DIR="$PWD/.debug/pip_cache" ./venv_arm64/bin/python -m pip install ninja cmake
PIP_CACHE_DIR="$PWD/.debug/pip_cache" ./venv_arm64/bin/python -m pip install -r .debug/triton-cpu-src/python/requirements.txt
git -C .debug/triton-cpu-src submodule update --init --recursive --depth 1
SSL_CERT_FILE="$PWD/venv_arm64/lib/python3.11/site-packages/certifi/cacert.pem" \
REQUESTS_CA_BUNDLE="$PWD/venv_arm64/lib/python3.11/site-packages/certifi/cacert.pem" \
TRITON_HOME="$PWD/.debug/triton_home" \
PATH="$PWD/venv_arm64/bin:$PATH" \
MAX_JOBS=2 \
./venv_arm64/bin/python -m pip install -e .debug/triton-cpu-src --no-build-isolation
```

Two local fixes were required:

- putting `./venv_arm64/bin` first on `PATH`, otherwise Triton's build helper
  did not find the repo-local `ninja`;
- setting `SSL_CERT_FILE`/`REQUESTS_CA_BUNDLE` to the venv certifi bundle,
  otherwise the Python.org framework SSL store rejected Triton's third-party
  package download.

The first source build then failed because the shallow clone lacked the
`third_party/sleef` submodule. Initializing the submodule fixed that blocker.

## Validation

```bash
TRITON_CPU_BACKEND=1 TRITON_INTERPRET=1 TRITON_HOME="$PWD/.debug/triton_home" \
  ./venv_arm64/bin/python -m pytest \
  experimental/hybridkernel/phase0/tests experimental/hybridkernel/phase2/tests \
  experimental/hybridkernel/phase3/tests experimental/hybridkernel/phase4/tests \
  experimental/sinkaware/phase2/tests experimental/sinkaware/phase3/tests \
  experimental/sinkaware/phase4/tests experimental/thoughtflow_fp8/phase2/tests \
  experimental/thoughtflow_fp8/phase4/tests -rs
```

Result: `103 passed, 2 warnings`.

The narrower Phase 4 kernel-correctness suite also passes:

```bash
TRITON_CPU_BACKEND=1 TRITON_INTERPRET=1 TRITON_HOME="$PWD/.debug/triton_home" \
  ./venv_arm64/bin/python -m pytest \
  experimental/hybridkernel/phase4/tests \
  experimental/sinkaware/phase4/tests \
  experimental/thoughtflow_fp8/phase4/tests -rs
```

Result: `9 passed`.

## Interpretation

Mac-local Triton interpreter correctness is no longer dependency-blocked for
the three experimental projects. This is kernel logic evidence only. It is not
CUDA compilation evidence, native GPU timing, HBM traffic, energy, throughput,
or evidence of a serving-system speedup.

The broad recursive command

```bash
./venv_arm64/bin/python -m pytest experimental/hybridkernel experimental/sinkaware experimental/thoughtflow_fp8 -rs
```

is not the project-owned validation command because it collects vendored
FlashAttention, FlashInfer, FlashMLA, and other external CUDA/GPU test suites
under `experimental/sinkaware/external/`. That collection fails on missing
external packages such as `flash_mla`, `flash_attn_2_cuda`, `flashinfer`,
`cutlass`, and `apex`, and should not be used as a Mac readiness signal.
