import json

from scripts import materialize_jsonl_range as range_script


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_materialize_range_is_one_based_and_writes_manifest(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "slice.jsonl"
    manifest_json = tmp_path / "manifest.json"
    manifest_md = tmp_path / "manifest.md"
    _write_jsonl(
        source,
        [
            {"question": "q1", "metadata": {"id": "chal-1"}},
            {"question": "q2", "metadata": {"id": "chal-2"}},
            {"question": "q3", "metadata": {"id": "chal-3"}},
            {"question": "q4", "metadata": {"id": "chal-4"}},
        ],
    )

    payload = range_script.main(
        [
            "--source",
            str(source),
            "--output",
            str(output),
            "--start-index",
            "2",
            "--count",
            "2",
            "--manifest-json",
            str(manifest_json),
            "--manifest-md",
            str(manifest_md),
            "--run-date",
            "2026-04-26",
        ]
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["metadata"]["id"] for row in rows] == ["chal-2", "chal-3"]
    assert payload["first_metadata_id"] == "chal-2"
    assert payload["last_metadata_id"] == "chal-3"
    assert payload["output_sha256"]
    assert json.loads(manifest_json.read_text(encoding="utf-8"))["metadata_ids"] == ["chal-2", "chal-3"]
    assert "start index" in manifest_md.read_text(encoding="utf-8")
