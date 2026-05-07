from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]

os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ.setdefault("TRITON_CPU_BACKEND", "1")
os.environ.setdefault("TRITON_HOME", str(_REPO_ROOT / ".debug/triton_home"))
