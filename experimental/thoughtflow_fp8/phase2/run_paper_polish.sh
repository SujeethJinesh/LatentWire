#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RESULTS_DIR="$ROOT/experimental/thoughtflow_fp8/phase2/results"
RUN_ID="${RUN_ID:-thoughtflow_paper_polish_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$RESULTS_DIR/$RUN_ID"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv_gpu/bin/python}"
TECTONIC_BIN="${TECTONIC_BIN:-/workspace/bin/tectonic}"

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/build"

{
  echo "run_id=$RUN_ID"
  echo "run_dir=$RUN_DIR"
  echo "python=$PYTHON_BIN"
  echo "tectonic=$TECTONIC_BIN"
} | tee "$RUN_DIR/logs/stdout.log"

{
  cd "$ROOT"
  "$PYTHON_BIN" - <<'PY' > "$RUN_DIR/environment.json"
import json
import os
import platform
import subprocess
import sys

def command(cmd):
    completed = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True)
    return {
        "command": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }

payload = {
    "schema_version": "thoughtflow_paper_polish_environment_v1",
    "python": {"version": platform.python_version(), "executable": sys.executable},
    "platform": platform.platform(),
    "git_sha": command(["git", "rev-parse", "HEAD"])["stdout"],
    "pip_freeze": command([sys.executable, "-m", "pip", "freeze"]),
    "tectonic_help": command(["/workspace/bin/tectonic", "--help"]),
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

set +e
(
  cd "$ROOT"
  TECTONIC_CACHE_DIR=/workspace/tectonic_cache \
  XDG_CACHE_HOME=/workspace/.cache \
  "$TECTONIC_BIN" \
    --outdir "$RUN_DIR/build" \
    experimental/thoughtflow_fp8/paper/thoughtflow_fp8_colm2026.tex
) > "$RUN_DIR/logs/build_stdout.log" 2> "$RUN_DIR/logs/build_stderr.log"
BUILD_CODE=$?

(
  cd "$ROOT"
  TRITON_CPU_BACKEND=1 \
  TRITON_INTERPRET=1 \
  TRITON_HOME="$ROOT/.debug/triton_home" \
  "$PYTHON_BIN" -m pytest \
    experimental/thoughtflow_fp8/phase2/tests \
    experimental/thoughtflow_fp8/phase4/tests \
    -rs -q
) > "$RUN_DIR/logs/pytest_stdout.log" 2> "$RUN_DIR/logs/pytest_stderr.log"
PYTEST_CODE=$?
set -e

"$PYTHON_BIN" - <<'PY' "$ROOT" "$RUN_DIR" "$BUILD_CODE" "$PYTEST_CODE"
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_dir = Path(sys.argv[2])
build_code = int(sys.argv[3])
pytest_code = int(sys.argv[4])

def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()

paper_pdf = root / "experimental/thoughtflow_fp8/paper/thoughtflow_fp8_colm2026.pdf"
paper_tex = root / "experimental/thoughtflow_fp8/paper/thoughtflow_fp8_colm2026.tex"
reviewer_pack = root / "experimental/thoughtflow_fp8/paper/reviewer_pack.md"
built_pdf = run_dir / "build/thoughtflow_fp8_colm2026.pdf"
metrics = {
    "schema_version": "thoughtflow_paper_polish_metrics_v1",
    "decision_hint": "PASS_THOUGHTFLOW_PAPER_BUILDABLE"
    if build_code == 0 and pytest_code == 0 and reviewer_pack.is_file() and built_pdf.is_file()
    else "FAIL_INFRA_THOUGHTFLOW_PAPER_POLISH",
    "paper_mode": "copyedit_only_falsification_methodology",
    "paper_tex": str(paper_tex.relative_to(root)),
    "paper_tex_sha256": sha256(paper_tex),
    "tracked_pdf": str(paper_pdf.relative_to(root)),
    "tracked_pdf_sha256": sha256(paper_pdf),
    "built_pdf": str(built_pdf.relative_to(root)),
    "built_pdf_sha256": sha256(built_pdf),
    "reviewer_pack": str(reviewer_pack.relative_to(root)),
    "reviewer_pack_sha256": sha256(reviewer_pack),
    "build_returncode": build_code,
    "pytest_returncode": pytest_code,
    "tests": {
        "command": (
            "TRITON_CPU_BACKEND=1 TRITON_INTERPRET=1 "
            "TRITON_HOME=$PWD/.debug/triton_home ./.venv_gpu/bin/python -m pytest "
            "experimental/thoughtflow_fp8/phase2/tests "
            "experimental/thoughtflow_fp8/phase4/tests -rs -q"
        ),
        "stdout_path": "logs/pytest_stdout.log",
        "stderr_path": "logs/pytest_stderr.log",
    },
    "build": {
        "command": (
            "/workspace/bin/tectonic --outdir <run_dir>/build "
            "experimental/thoughtflow_fp8/paper/thoughtflow_fp8_colm2026.tex"
        ),
        "stdout_path": "logs/build_stdout.log",
        "stderr_path": "logs/build_stderr.log",
    },
}
(run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(json.dumps({"decision_hint": metrics["decision_hint"], "run_dir": str(run_dir.relative_to(root))}))
sys.exit(0 if metrics["decision_hint"] == "PASS_THOUGHTFLOW_PAPER_BUILDABLE" else 2)
PY
