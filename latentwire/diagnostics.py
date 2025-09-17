# latentwire/diagnostics.py
# Lightweight environment snapshotter used by train/eval for post-hoc debugging.
from __future__ import annotations
import os, sys, json, platform, datetime
from typing import Optional, Dict, Any

def _safe_import_version(modname: str) -> str:
    try:
        mod = __import__(modname)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "not-installed"

def capture_env_snapshot(out_dir: str, extras: Optional[Dict[str, Any]] = None) -> str:
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
        "cuda": {},
        "argv": sys.argv,
    }
    try:
        import torch
        snap["cuda"]["is_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            snap["cuda"]["device_count"] = torch.cuda.device_count()
            devs = []
            for i in range(torch.cuda.device_count()):
                d = {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                }
                try:
                    free, total = torch.cuda.mem_get_info(i)
                    d["mem_free"] = int(free)
                    d["mem_total"] = int(total)
                except Exception:
                    pass
                devs.append(d)
            snap["cuda"]["devices"] = devs
    except Exception:
        pass
    if extras:
        snap["extras"] = extras
    dst = os.path.join(out_dir, "env_snapshot.json")
    with open(dst, "w") as f:
        json.dump(snap, f, indent=2)
    return dst