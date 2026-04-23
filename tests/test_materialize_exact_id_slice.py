from __future__ import annotations

import json
import pathlib

from scripts import materialize_exact_id_slice as slicer


def _record(example_id: str, *, method: str = "m", index: int = 0) -> dict:
    return {
        "correct": True,
        "example_id": example_id,
        "index": index,
        "method": method,
        "prediction": "1",
    }


def _write_jsonl(path: pathlib.Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")


def test_materialize_exact_id_slice_orders_like_reference() -> None:
    reference = [_record("b", index=0), _record("a", index=1)]
    source = [_record("a", method="source", index=10), _record("b", method="source", index=11)]

    rows = slicer.materialize_exact_id_slice(
        reference_records=reference,
        source_records=source,
    )

    assert [row["example_id"] for row in rows] == ["b", "a"]
    assert [row["index"] for row in rows] == [11, 10]


def test_materialize_exact_id_slice_rejects_missing_id() -> None:
    try:
        slicer.materialize_exact_id_slice(
            reference_records=[_record("a"), _record("b")],
            source_records=[_record("a")],
        )
    except ValueError as exc:
        assert "missing reference IDs" in str(exc)
    else:
        raise AssertionError("expected missing ID failure")


def test_cli_writes_slice_and_meta(tmp_path: pathlib.Path) -> None:
    reference = tmp_path / "reference.jsonl"
    source = tmp_path / "source.jsonl"
    output = tmp_path / "out.jsonl"
    _write_jsonl(reference, [_record("b", index=0), _record("a", index=1)])
    _write_jsonl(
        source,
        [
            _record("a", method="target_self_repair", index=10),
            _record("b", method="target_self_repair", index=11),
            _record("c", method="target_self_repair", index=12),
        ],
    )

    meta = slicer.main(
        [
            "--reference-jsonl",
            str(reference),
            "--source-jsonl",
            str(source),
            "--source-method",
            "target_self_repair",
            "--output-jsonl",
            str(output),
        ]
    )

    assert [json.loads(line)["example_id"] for line in output.read_text().splitlines()] == ["b", "a"]
    assert meta["source_artifact_n"] == 3
    assert meta["output_n"] == 2
    assert meta["dropped_source_rows"] == 1
    assert meta["exact_ordered_id_parity"] is True
    assert json.loads(output.with_suffix(".jsonl.meta.json").read_text())["output_n"] == 2
