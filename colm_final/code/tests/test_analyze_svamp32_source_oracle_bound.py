from __future__ import annotations

import json

from scripts import analyze_svamp32_source_oracle_bound as analyzer


def _write_jsonl(path, method: str, correct_ids: set[str]) -> None:
    rows = []
    for index, example_id in enumerate(["a", "b", "c", "d"]):
        rows.append(
            {
                "example_id": example_id,
                "index": index,
                "method": method,
                "correct": example_id in correct_ids,
                "prediction": str(index + 1),
                "normalized_prediction": str(index + 1),
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_svamp32_source_oracle_bound_reports_source_and_oracle_headroom(tmp_path) -> None:
    target = tmp_path / "target.jsonl"
    teacher = tmp_path / "teacher.jsonl"
    source = tmp_path / "source.jsonl"
    self_repair = tmp_path / "self_repair.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    target_set = tmp_path / "target_set.json"
    output_json = tmp_path / "out.json"
    output_md = tmp_path / "out.md"

    _write_jsonl(target, "target_alone", {"a", "b"})
    _write_jsonl(teacher, "c2c_generate", {"b", "c", "d"})
    _write_jsonl(source, "source_alone", {"c"})
    _write_jsonl(self_repair, "target_self_repair", {"a", "b", "c"})
    _write_jsonl(candidate, "candidate", {"a", "c", "d"})
    target_set.write_text(
        json.dumps(
            {
                "ids": {
                    "teacher_only": ["c", "d"],
                    "clean_residual_targets": ["c"],
                }
            }
        ),
        encoding="utf-8",
    )

    analyzer.main(
        [
            "--target",
            f"target=path={target},method=target_alone",
            "--teacher",
            f"teacher=path={teacher},method=c2c_generate",
            "--source",
            f"source=path={source},method=source_alone",
            "--baseline",
            f"target_self_repair=path={self_repair},method=target_self_repair",
            "--candidate",
            f"candidate=path={candidate},method=candidate",
            "--target-set-json",
            str(target_set),
            "--expected-n",
            "4",
            "--date",
            "2026-04-24",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))

    assert payload["reference"]["teacher_only_count"] == 2
    assert payload["reference"]["source_union_clean_residual_correct_count"] == 1
    assert payload["rows"]["sources"][0]["clean_residual_recovered_ids"] == ["c"]
    assert payload["rows"]["candidates"][0]["teacher_only_recovered_ids"] == ["c", "d"]

    oracle = {
        (row["base"], row["row"]): row
        for row in payload["oracle_bounds"]
    }
    assert oracle[("target_self_repair", "candidate")]["oracle_correct"] == 4
    assert oracle[("target_self_repair", "candidate")]["oracle_delta_vs_base"] == 1
    assert oracle[("target_self_repair", "candidate")]["clean_residual_added_count"] == 0

    md = output_md.read_text(encoding="utf-8")
    assert "SVAMP32 Source Informativeness And Oracle Bound" in md
    assert "| target_self_repair | candidate | 4/32 | +1 | 0 |" in md
