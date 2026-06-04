import json
from pathlib import Path

import pytest

from pmc.cache_split import build_inventory, confirm_open_guard, path_hash, split_for_key


def test_stage0_split_is_deterministic():
    key = "results/unit/source_prediction_cache.jsonl|example-7"
    assert split_for_key(key) == split_for_key(key)


def test_inventory_extracts_row_entities(tmp_path):
    root = tmp_path / "results" / "source_private_demo"
    root.mkdir(parents=True)
    cache = root / "source_prediction_cache.jsonl"
    cache.write_text(
        "\n".join(
            [
                json.dumps({"example_id": "a", "score": 1}),
                json.dumps({"example_id": "b", "score": 0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    units, rows = build_inventory([tmp_path / "results"])

    assert len(units) == 1
    assert units[0].row_count == 2
    assert {row.row_key for row in rows} == {"a", "b"}


def test_confirm_open_guard_blocks_planted_confirm_path(tmp_path):
    confirm = tmp_path / "confirm.json"
    confirm.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "confirm_hashes.txt"
    manifest.write_text(f"{path_hash(confirm)} {confirm}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="confirm-split cache access blocked"):
        with confirm_open_guard(manifest):
            confirm.open("r").close()


def test_confirm_open_guard_allows_unlisted_path(tmp_path):
    dev = tmp_path / "dev.json"
    dev.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "confirm_hashes.txt"
    manifest.write_text("", encoding="utf-8")

    with confirm_open_guard(manifest):
        assert dev.read_text(encoding="utf-8") == "{}"

