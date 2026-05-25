"""Prompt and artifact data structures for release workflows."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PromptRecord:
    """A deterministic prompt row used by release reproductions."""

    index: int
    prompt_id: str
    text: str
    answer: str | None = None


@dataclass(frozen=True)
class PromptSet:
    """An ordered prompt collection with stable hashing."""

    records: tuple[PromptRecord, ...]

    def payload_sha256(self) -> str:
        """Return SHA-256 of concatenated prompt text in index order."""

        ordered = sorted(self.records, key=lambda row: row.index)
        payload = "".join(row.text for row in ordered).encode("utf-8")
        return "sha256:" + hashlib.sha256(payload).hexdigest()


def load_jsonl_prompts(path: Path, *, limit: int | None = None) -> PromptSet:
    """Load prompt records from JSONL."""

    records: list[PromptRecord] = []
    for row_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"prompt row {row_index} must be an object")
        records.append(
            PromptRecord(
                index=int(item.get("index", row_index)),
                prompt_id=str(item.get("prompt_id", item.get("id", row_index))),
                text=str(item.get("prompt", item.get("problem", item.get("question", "")))),
                answer=_optional_text(item.get("answer")),
            )
        )
        if limit is not None and len(records) >= limit:
            break
    return PromptSet(tuple(records))


def write_json(path: Path, payload: object) -> None:
    """Write an indented JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def records_to_rows(records: Iterable[PromptRecord]) -> list[dict[str, object]]:
    """Convert prompt records to serializable rows."""

    return [
        {
            "index": record.index,
            "prompt_id": record.prompt_id,
            "text": record.text,
            "answer": record.answer,
        }
        for record in records
    ]


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
