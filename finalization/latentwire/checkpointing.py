# latentwire/checkpointing.py
# Keep-only-latest checkpoint manager: atomic save + pruning.
import os
import json
import shutil
import tempfile
import re
from typing import Dict, Optional, Tuple, Iterable

try:
    import torch
except Exception:
    torch = None  # eval-time tools can still use the pruner without torch


# Canonical set we expect to keep inside a ckpt directory.
# NOTE: We also preserve training_stats.json by default (so eval-time calibration stays available).
CANONICAL_FILES = {
    "encoder.pt",
    "adapter_llama.pt",
    "adapter_qwen.pt",
    "state.pt",
    "optimizer.pt",   # optional; safe to leave here
    "scheduler.pt",   # optional; safe to leave here
    "scaler.pt",      # optional; safe to leave here
    "config.json",
    "training_stats.json",
}

# Things we consider "step-like" and temporary.
STEP_DIR_PREFIXES = ("step_", "ckpt_", "epoch_", "iter_", "global_step_")
TMP_SUFFIXES = (".tmp", ".new", ".partial", ".bak")
STEP_FILE_PATTERNS = (r".*_step\d+\.pt$", r"state_step\d+\.pt$")


def _is_step_dir(name: str) -> bool:
    name = name.lower()
    return any(name.startswith(p) for p in STEP_DIR_PREFIXES)


def _is_step_file(name: str) -> bool:
    return any(re.match(p, name) for p in STEP_FILE_PATTERNS)


def _is_tmp_file(name: str) -> bool:
    return any(name.endswith(suf) for suf in TMP_SUFFIXES)


def _human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def _safe_remove(path: str) -> int:
    """Remove file or dir; return freed bytes (best effort)."""
    freed = 0
    try:
        if os.path.isdir(path) and not os.path.islink(path):
            for root, _, files in os.walk(path):
                for f in files:
                    fp = os.path.join(root, f)
                    try:
                        freed += os.path.getsize(fp)
                    except Exception:
                        pass
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                freed += os.path.getsize(path)
            except Exception:
                pass
            os.remove(path)
    except FileNotFoundError:
        pass
    except Exception:
        # ignore but do not break saving
        pass
    return freed


def prune_save_dir(
    save_dir: str,
    keep_only: Optional[Iterable[str]] = None,
) -> int:
    """
    Delete everything in save_dir except the provided 'keep_only' filenames.
    Also removes step_* dirs and temp files. Returns freed bytes.
    """
    freed = 0
    os.makedirs(save_dir, exist_ok=True)
    keep_set = set(keep_only) if keep_only is not None else set()
    for name in os.listdir(save_dir):
        path = os.path.join(save_dir, name)

        # Always remove step-like directories
        if os.path.isdir(path) and _is_step_dir(name):
            freed += _safe_remove(path)
            continue

        # Remove step checkpoint files (e.g., encoder_step1000.pt)
        if os.path.isfile(path) and _is_step_file(name):
            freed += _safe_remove(path)
            continue

        # Remove stale temporary files
        if os.path.isfile(path) and (_is_tmp_file(name) or name.endswith(".pt.old")):
            freed += _safe_remove(path)
            continue

        # Apply keep-only policy when provided
        if keep_set:
            if name not in keep_set:
                freed += _safe_remove(path)
        else:
            # No keep list: remove anything not canonical
            if os.path.isfile(path) and (name not in CANONICAL_FILES):
                freed += _safe_remove(path)
    return freed


def _atomic_write_bytes(data: bytes, dst_path: str) -> None:
    """Write bytes atomically to dst_path via a temp file + os.replace."""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(dst_path) + ".", dir=os.path.dirname(dst_path))
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, dst_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _atomic_save_torch(obj, dst_path: str) -> None:
    """Atomic torch.save to dst_path."""
    if torch is None:
        raise RuntimeError("torch is not available but a torch save was requested.")
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(dst_path) + ".", dir=os.path.dirname(dst_path))
    os.close(tmp_fd)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, dst_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _atomic_save_json(obj: dict, dst_path: str) -> None:
    data = json.dumps(obj, indent=2).encode("utf-8")
    _atomic_write_bytes(data, dst_path)


def save_latest_checkpoint(
    save_dir: str,
    artifacts: Dict[str, object],
    pre_prune: bool = True,
    post_prune: bool = True,
    verbose: bool = True,
) -> Tuple[int, int]:
    """
    Save a set of artifacts to canonical filenames and keep only those files.
    - artifacts: mapping filename -> object (torch objects for *.pt, dict/str/bytes for *.json). None values are skipped.
    - pre_prune: delete non-canonical / step_* content before saving.
    - post_prune: after saving, enforce keep-only policy (hard guarantee).
    Returns (freed_bytes_pre, freed_bytes_post).
    """
    os.makedirs(save_dir, exist_ok=True)

    # Build list of target filenames we intend to keep
    keep_only = [name for name, obj in artifacts.items() if obj is not None]

    # Always preserve config.json if it already exists and caller didn't pass it
    cfg_path = os.path.join(save_dir, "config.json")
    if "config.json" not in keep_only and os.path.isfile(cfg_path):
        keep_only.append("config.json")

    # Always preserve training_stats.json if present
    stats_path = os.path.join(save_dir, "training_stats.json")
    if os.path.isfile(stats_path) and "training_stats.json" not in keep_only:
        keep_only.append("training_stats.json")

    freed_pre = 0
    if pre_prune:
        freed_pre = prune_save_dir(save_dir, keep_only=keep_only)

    # Save artifacts atomically
    for name, obj in artifacts.items():
        if obj is None:
            continue
        dst = os.path.join(save_dir, name)
        try:
            if name.endswith(".pt"):
                _atomic_save_torch(obj, dst)
            elif name.endswith(".json"):
                if isinstance(obj, (dict, list)):
                    _atomic_save_json(obj, dst)
                else:
                    data = obj if isinstance(obj, (bytes, bytearray)) else str(obj).encode("utf-8")
                    _atomic_write_bytes(data, dst)
            else:
                # Fallback: try JSON or raw bytes
                if isinstance(obj, (dict, list)):
                    _atomic_save_json(obj, dst)
                elif isinstance(obj, (bytes, bytearray)):
                    _atomic_write_bytes(bytes(obj), dst)
                else:
                    _atomic_write_bytes(str(obj).encode("utf-8"), dst)
        except Exception as e:
            if verbose:
                print(f"[checkpoint] ERROR saving {name} to {dst}: {e}")
            raise

    freed_post = 0
    if post_prune:
        freed_post = prune_save_dir(save_dir, keep_only=keep_only)

    if verbose:
        if pre_prune:
            print(f"[checkpoint] Freed {_human_bytes(freed_pre)} before save.")
        print(f"[checkpoint] Saved latest: {', '.join(keep_only)}")
        if post_prune:
            print(f"[checkpoint] Freed {_human_bytes(freed_post)} after save (non-canonical).")

    return freed_pre, freed_post
