import json
from pathlib import Path

from pmc.stage1_screens import ParsedRow, condition_accuracy, parse_coverage, read_jsonl_rows, summarize_latentwire


def test_row_split_filters_confirm_rows_without_consuming_them(tmp_path):
    path = tmp_path / "results" / "source_private_wyner_ziv_packet_gate_20260429" / "predictions_budget6.jsonl"
    path.parent.mkdir(parents=True)

    rows = []
    for index in range(200):
        row_id = f"row-{index}"
        rows.append(
            {
                "example_id": row_id,
                "condition": "matched_learned_syndrome",
                "correct": index % 2 == 0,
            }
        )
        rows.append({"example_id": row_id, "condition": "target_only", "correct": False})
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    parsed = read_jsonl_rows(path)
    assert any(row.split == "confirm" for row in parsed)

    raw_rows, summaries = summarize_latentwire({path.as_posix(): parsed})
    assert raw_rows
    assert all(row["split"] != "confirm" for row in raw_rows)
    assert summaries


def test_condition_accuracy_uses_only_supplied_rows(tmp_path):
    path = tmp_path / "results" / "source_private_wyner_ziv_packet_gate_20260429" / "predictions_budget6.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            [
                json.dumps({"example_id": "a", "condition": "target_only", "correct": True}),
                json.dumps({"example_id": "b", "condition": "target_only", "correct": False}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    parsed = read_jsonl_rows(path)
    acc = condition_accuracy(parsed)
    assert acc["target_only"]["n"] == 2
    assert acc["target_only"]["accuracy"] == 0.5


def test_fixed_packet_baseline_cannot_promote_to_gpu():
    path = "results/source_private_arc_challenge_fixed_packet_gate_qwen05_bge/test/predictions.jsonl"
    rows = []
    for index in range(32):
        row_id = f"row-{index}"
        rows.append(
            ParsedRow(
                "fixed_packet",
                path,
                row_id,
                "gate",
                {"example_id": row_id, "condition": "matched_source_private_packet", "correct": True},
            )
        )
        rows.append(
            ParsedRow(
                "fixed_packet",
                path,
                row_id,
                "gate",
                {"example_id": row_id, "condition": "target_only", "correct": False},
            )
        )

    _, summaries = summarize_latentwire({path: rows})
    assert summaries
    assert {row["method_id"] for row in summaries} == {"latentwire_cached_fixed_packet_baseline"}
    assert {row["status"] for row in summaries} == {"CPU_SCREENED"}


def test_stage1_outputs_are_not_rescanned_as_input_caches(tmp_path):
    output = tmp_path / "results" / "stage1" / "leaderboard.csv"
    output.parent.mkdir(parents=True)
    output.write_text("method_id,status\nx,CPU_SCREENED\n", encoding="utf-8")

    cache = (
        tmp_path
        / "results"
        / "source_private_wyner_ziv_packet_gate_20260429"
        / "predictions_budget2.jsonl"
    )
    cache.parent.mkdir(parents=True)
    cache.write_text(
        json.dumps({"example_id": "a", "condition": "target_only", "correct": True}) + "\n",
        encoding="utf-8",
    )

    coverage, parsed = parse_coverage([tmp_path / "results"])
    assert [row["path"] for row in coverage] == [cache.as_posix()]
    assert list(parsed) == [cache.as_posix()]
