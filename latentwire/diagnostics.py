# latentwire/diagnostics.py
from __future__ import annotations
import os, sys, json, platform, datetime

def _safe_import_version(modname: str) -> str:
    try:
        mod = __import__(modname)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "not-installed"

def capture_env_snapshot(out_dir: str, extras=None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    snap = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "python": sys.version.replace("\n"," "),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "versions": {
            "torch": _safe_import_version("torch"),
            "transformers": _safe_import_version("transformers"),
            "datasets": _safe_import_version("datasets"),
            "sentence_transformers": _safe_import_version("sentence_transformers"),
            "bitsandbytes": _safe_import_version("bitsandbytes"),
        },
        "argv": sys.argv,
    }
    dst = os.path.join(out_dir, "env_snapshot.json")
    with open(dst, "w") as f:
        json.dump(snap, f, indent=2)
    return dst