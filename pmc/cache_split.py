from __future__ import annotations

import builtins
import csv
import gzip
import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator


DATA_EXTENSIONS = {
    ".csv",
    ".json",
    ".jsonl",
    ".npy",
    ".npz",
}

ROW_KEYS = (
    "example_id",
    "trace_id",
    "row_id",
    "id",
    "index",
    "question_id",
    "prompt_id",
)


@dataclass(frozen=True)
class CacheRow:
    unit_id: str
    path: str
    row_key: str
    split: str
    row_hash: str
    scan_status: str


@dataclass(frozen=True)
class CacheUnit:
    unit_id: str
    root: str
    path: str
    file_count: int
    row_count: int
    scan_statuses: tuple[str, ...]
    partition_status: str


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_for_key(key: str) -> str:
    bucket = int(stable_hash(key)[:12], 16) / float(16**12)
    if bucket < 0.60:
        return "dev"
    if bucket < 0.80:
        return "gate"
    return "confirm"


def is_data_file(path: Path) -> bool:
    if path.name.startswith("."):
        return False
    if path.suffix == ".gz" and path.name.endswith(".jsonl.gz"):
        return True
    return path.suffix in DATA_EXTENSIONS


def is_cache_path(path: Path) -> bool:
    parts = path.parts
    if not is_data_file(path):
        return False
    if "__pycache__" in parts:
        return False
    if "results" in parts and "experimental" not in parts:
        return True
    if "experimental" in parts:
        exp_idx = parts.index("experimental")
        return "results" in parts[exp_idx:]
    return False


def cache_unit_for(path: Path) -> Path:
    parts = path.parts
    if "experimental" in parts and "results" in parts:
        idx = parts.index("results")
        if idx + 1 < len(parts):
            return Path(*parts[: idx + 2])
        return Path(*parts[: idx + 1])
    if "results" in parts:
        idx = parts.index("results")
        if idx + 1 < len(parts):
            return Path(*parts[: idx + 2])
        return Path(*parts[: idx + 1])
    return path.parent


def discover_cache_files(roots: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            if is_cache_path(root):
                files.append(root)
            continue
        for path in root.rglob("*"):
            if path.is_file() and is_cache_path(path):
                files.append(path)
    return sorted(set(files), key=lambda p: p.as_posix())


def _open_text(path: Path):
    if path.name.endswith(".jsonl.gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _row_key_from_json(payload: object, fallback: str) -> str:
    if isinstance(payload, dict):
        for key in ROW_KEYS:
            if key in payload:
                return str(payload[key])
    return fallback


def _scan_jsonl(path: Path, max_scan_bytes: int) -> tuple[list[str], str]:
    if path.stat().st_size > max_scan_bytes:
        return [f"file:{path.as_posix()}"], "large_jsonl_file_level"
    rows: list[str] = []
    with _open_text(path) as handle:
        for idx, line in enumerate(handle):
            fallback = f"line:{idx}"
            try:
                row_key = _row_key_from_json(json.loads(line), fallback)
            except Exception:
                row_key = fallback
            rows.append(row_key)
    return rows or [f"empty:{path.as_posix()}"], "jsonl_rows"


def _scan_csv(path: Path, max_scan_bytes: int) -> tuple[list[str], str]:
    if path.stat().st_size > max_scan_bytes:
        return [f"file:{path.as_posix()}"], "large_csv_file_level"
    rows: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames:
            for idx, row in enumerate(reader):
                row_key = next((row[k] for k in ROW_KEYS if k in row and row[k]), None)
                rows.append(str(row_key) if row_key is not None else f"line:{idx}")
        else:
            handle.seek(0)
            for idx, _ in enumerate(handle):
                rows.append(f"line:{idx}")
    return rows or [f"empty:{path.as_posix()}"], "csv_rows"


def _scan_json(path: Path, max_scan_bytes: int) -> tuple[list[str], str]:
    if path.stat().st_size > max_scan_bytes:
        return [f"file:{path.as_posix()}"], "large_json_file_level"
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return [f"file:{path.as_posix()}"], "json_parse_failed_file_level"
    if isinstance(payload, list):
        return [_row_key_from_json(row, f"idx:{idx}") for idx, row in enumerate(payload)] or [
            f"empty:{path.as_posix()}"
        ], "json_list_rows"
    if isinstance(payload, dict):
        for key in ("rows", "records", "examples", "traces", "items", "predictions"):
            value = payload.get(key)
            if isinstance(value, list):
                rows = [_row_key_from_json(row, f"{key}:{idx}") for idx, row in enumerate(value)]
                return rows or [f"empty:{path.as_posix()}"], f"json_{key}_rows"
    return [f"file:{path.as_posix()}"], "json_file_level"


def scan_file_rows(path: Path, max_scan_bytes: int = 50_000_000) -> tuple[list[str], str]:
    if path.name.endswith(".jsonl.gz") or path.suffix == ".jsonl":
        return _scan_jsonl(path, max_scan_bytes)
    if path.suffix == ".csv":
        return _scan_csv(path, max_scan_bytes)
    if path.suffix == ".json":
        return _scan_json(path, max_scan_bytes)
    return [f"file:{path.as_posix()}"], f"{path.suffix.lstrip('.')}_file_level"


def infer_partition_status(unit: Path, files: Iterable[Path]) -> str:
    text = " ".join([unit.as_posix(), *(p.as_posix() for p in files)]).lower()
    has_dev = any(token in text for token in ("dev", "train", "calibration", "core"))
    has_gate = any(token in text for token in ("gate", "validation", "holdout"))
    has_confirm = "confirm" in text
    has_test = "test" in text
    if has_dev and has_gate and has_confirm:
        return "named_dev_gate_confirm_present"
    if has_dev and has_gate:
        return "named_dev_gate_no_confirm"
    if has_test or has_gate:
        return "prior_eval_rows_present_no_clean_confirm"
    return "no_named_partition_detected"


def build_inventory(
    roots: Iterable[Path], max_scan_bytes: int = 50_000_000
) -> tuple[list[CacheUnit], list[CacheRow]]:
    files = discover_cache_files(roots)
    grouped: dict[Path, list[Path]] = {}
    for path in files:
        grouped.setdefault(cache_unit_for(path), []).append(path)

    units: list[CacheUnit] = []
    cache_rows: list[CacheRow] = []
    for unit_path, unit_files in sorted(grouped.items(), key=lambda item: item[0].as_posix()):
        statuses: set[str] = set()
        unit_id = unit_path.as_posix()
        row_count = 0
        for path in sorted(unit_files, key=lambda p: p.as_posix()):
            row_keys, status = scan_file_rows(path, max_scan_bytes=max_scan_bytes)
            statuses.add(status)
            for row_key in row_keys:
                stable_key = f"{unit_id}|{path.as_posix()}|{row_key}"
                row_hash = stable_hash(stable_key)
                split = split_for_key(stable_key)
                cache_rows.append(
                    CacheRow(
                        unit_id=unit_id,
                        path=path.as_posix(),
                        row_key=str(row_key),
                        split=split,
                        row_hash=row_hash,
                        scan_status=status,
                    )
                )
            row_count += len(row_keys)
        units.append(
            CacheUnit(
                unit_id=unit_id,
                root=unit_path.parts[0] if unit_path.parts else "",
                path=unit_path.as_posix(),
                file_count=len(unit_files),
                row_count=row_count,
                scan_statuses=tuple(sorted(statuses)),
                partition_status=infer_partition_status(unit_path, unit_files),
            )
        )
    return units, cache_rows


def load_confirm_path_hashes(path: Path) -> set[str]:
    hashes: set[str] = set()
    if not path.exists():
        return hashes
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        hashes.add(line.split()[0])
    return hashes


def path_hash(path: Path) -> str:
    return stable_hash(path.resolve().as_posix())


@contextmanager
def confirm_open_guard(confirm_hashes_path: Path) -> Iterator[None]:
    blocked = load_confirm_path_hashes(confirm_hashes_path)
    original_open: Callable = builtins.open
    original_path_open = Path.open

    def guarded_open(file, *args, **kwargs):
        candidate = Path(file)
        if path_hash(candidate) in blocked:
            raise RuntimeError(f"confirm-split cache access blocked: {candidate}")
        return original_open(file, *args, **kwargs)

    def guarded_path_open(self: Path, *args, **kwargs):
        if path_hash(self) in blocked:
            raise RuntimeError(f"confirm-split cache access blocked: {self}")
        return original_path_open(self, *args, **kwargs)

    builtins.open = guarded_open
    Path.open = guarded_path_open
    try:
        yield
    finally:
        builtins.open = original_open
        Path.open = original_path_open
