from __future__ import annotations

import json
import pathlib

from scripts import build_svamp32_c2c_teacher_sparse_packet_distillation_preflight as preflight


def _write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _rows(ids: list[str], *, method: str, correct_ids: set[str]) -> list[dict]:
    return [
        {
            "answer": ["1"],
            "correct": example_id in correct_ids,
            "example_id": example_id,
            "index": idx,
            "method": method,
            "normalized_prediction": "1" if example_id in correct_ids else "0",
            "prediction": "answer is 1" if example_id in correct_ids else "answer is 0",
        }
        for idx, example_id in enumerate(ids)
    ]


def _run(correct_ids: set[str], clean_ids: set[str], source_necessary: list[str]) -> dict:
    return {
        "condition_summaries": {
            "matched": {
                "correct_count": len(correct_ids),
                "correct_ids": sorted(correct_ids),
                "clean_correct_count": len(clean_ids & correct_ids),
                "clean_correct_ids": sorted(clean_ids & correct_ids),
                "teacher_only_correct_count": len(correct_ids - {f"id{i:02d}" for i in range(8)}),
                "teacher_only_correct_ids": sorted(correct_ids - {f"id{i:02d}" for i in range(8)}),
                "target_self_correct_count": 3,
                "target_self_correct_ids": ["id14", "id15", "id16"],
            }
        },
        "control_clean_union_ids": [],
        "source_necessary_clean_ids": source_necessary,
        "status": "synthetic",
        "syndrome_bits": 8,
        "syndrome_bytes": 1,
    }


def test_preflight_marks_oracle_bound_alive_but_deployable_failed(tmp_path: pathlib.Path) -> None:
    ids = [f"id{i:02d}" for i in range(32)]
    target_correct = set(ids[:8])
    teacher_only = set(ids[8:18])
    clean_residual = set(ids[8:14])
    teacher_correct = target_correct | teacher_only
    source_correct = set(ids[:5]) | {"id08"}
    text_correct = set(ids[:2])

    target_path = tmp_path / "target.jsonl"
    source_path = tmp_path / "source.jsonl"
    text_path = tmp_path / "text.jsonl"
    teacher_path = tmp_path / "teacher.jsonl"
    _write_jsonl(target_path, _rows(ids, method="target_alone", correct_ids=target_correct))
    _write_jsonl(source_path, _rows(ids, method="source_alone", correct_ids=source_correct))
    _write_jsonl(text_path, _rows(ids, method="text_to_text", correct_ids=text_correct))
    _write_jsonl(teacher_path, _rows(ids, method="c2c", correct_ids=teacher_correct))

    target_set = tmp_path / "target_set.json"
    teacher_probe = tmp_path / "teacher_probe.json"
    _write_json(
        target_set,
        {
            "ids": {
                "teacher_only": sorted(teacher_only),
                "clean_residual_targets": sorted(clean_residual),
            }
        },
    )
    _write_json(teacher_probe, {"teacher_only_ids": sorted(teacher_only)})

    oracle = tmp_path / "oracle.json"
    deployable = tmp_path / "deployable.json"
    _write_json(
        oracle,
        {
            "runs": [
                _run(
                    correct_ids=target_correct | {"id08", "id09", "id14", "id15", "id16", "id20"},
                    clean_ids=clean_residual,
                    source_necessary=["id08", "id09"],
                )
            ]
        },
    )
    _write_json(
        deployable,
        {
            "run": _run(
                correct_ids=target_correct | {"id14"},
                clean_ids=clean_residual,
                source_necessary=[],
            )
        },
    )

    output_json = tmp_path / "out" / "summary.json"
    output_md = tmp_path / "out" / "summary.md"
    manifest = tmp_path / "out" / "manifest.json"
    payload = preflight.analyze(
        target_spec=preflight.RowSpec("target", target_path, "target_alone"),
        source_spec=preflight.RowSpec("source", source_path, "source_alone"),
        text_spec=preflight.RowSpec("text", text_path, "text_to_text"),
        teacher_spec=preflight.RowSpec("c2c", teacher_path, "c2c_generate"),
        teacher_probe_path=teacher_probe,
        target_set_path=target_set,
        targetpool_oracle_path=oracle,
        augmented_oracle_path=oracle,
        source_latent_paths=[deployable],
        learned_syndrome_paths=[],
        c2c_mechanism_paths=[],
        output_json=output_json,
        output_md=output_md,
        manifest_path=manifest,
        run_date="2026-05-05",
        bootstrap_samples=20,
    )

    assert payload["headline"]["teacher_correct"] == 18
    assert payload["headline"]["oracle_sparse_sidecar_alive"] is True
    assert payload["headline"]["deployable_distillation_pass"] is False
    assert payload["status"].endswith("oracle_bound_alive")
    source_row = next(row for row in payload["evidence_rows"] if row["label"] == "source_alone")
    assert source_row["teacher_only_recovered_ids"] == ["id08"]
    oracle_row = next(row for row in payload["evidence_rows"] if row["label"] == "oracle_c2c_syndrome_targetpool")
    assert oracle_row["source_necessary_clean_count"] == 2
    assert "# SVAMP32 C2C Teacher" in output_md.read_text(encoding="utf-8")
    assert json.loads(manifest.read_text(encoding="utf-8"))["status"] == payload["status"]
