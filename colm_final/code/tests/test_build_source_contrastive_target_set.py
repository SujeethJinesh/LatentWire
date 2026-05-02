from __future__ import annotations

import json
import pathlib

from scripts import build_source_contrastive_target_set as target_set


def _row(example_id: str, method: str, correct: bool, pred: str = "1") -> dict:
    return {
        "example_id": example_id,
        "method": method,
        "prediction": f"final answer: {pred}",
        "normalized_prediction": pred,
        "correct": correct,
    }


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_build_source_contrastive_target_set_excludes_controls_and_baselines(
    tmp_path: pathlib.Path,
) -> None:
    target_path = tmp_path / "target.jsonl"
    source_path = tmp_path / "source.jsonl"
    control_path = tmp_path / "control.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    _write_jsonl(
        target_path,
        [
            _row("a", "target", False),
            _row("b", "target", False),
            _row("c", "target", False),
            _row("d", "target", True),
        ],
    )
    _write_jsonl(
        source_path,
        [
            _row("a", "source", True),
            _row("b", "source", True),
            _row("c", "source", True),
            _row("d", "source", True),
        ],
    )
    _write_jsonl(
        control_path,
        [
            _row("a", "zero_source", True),
            _row("b", "zero_source", False),
            _row("c", "zero_source", False),
            _row("d", "zero_source", False),
        ],
    )
    _write_jsonl(
        baseline_path,
        [
            _row("a", "target_self", False),
            _row("b", "target_self", True),
            _row("c", "target_self", False),
            _row("d", "target_self", False),
        ],
    )

    payload = target_set.build_target_set(
        target_spec=target_set.RowSpec("target", target_path, "target"),
        source_spec=target_set.RowSpec("source", source_path, "source"),
        control_specs=[target_set.RowSpec("zero_source", control_path, "zero_source")],
        baseline_specs=[target_set.RowSpec("target_self", baseline_path, "target_self")],
        min_source_only=1,
        run_date="2026-04-26",
    )

    assert payload["status"] == "source_contrastive_target_set_ready"
    assert payload["ids"]["source_only"] == ["a", "b", "c"]
    assert payload["ids"]["clean_source_only"] == ["c"]
    assert payload["ids"]["clean_residual_targets"] == ["c"]
    assert payload["ids"]["target_self_repair"] == ["b"]
    assert payload["counts"]["target_or_source_oracle"] == 4
    assert payload["provenance"]["exact_ordered_id_parity"] is True


def test_cli_writes_outputs(tmp_path: pathlib.Path) -> None:
    target_path = tmp_path / "target.jsonl"
    source_path = tmp_path / "source.jsonl"
    output_json = tmp_path / "out.json"
    output_md = tmp_path / "out.md"
    _write_jsonl(target_path, [_row("a", "target", False)])
    _write_jsonl(source_path, [_row("a", "source", True)])

    payload = target_set.main(
        [
            "--target",
            f"target=path={target_path},method=target",
            "--source",
            f"source=path={source_path},method=source",
            "--min-source-only",
            "1",
            "--date",
            "2026-04-26",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )

    assert payload["ids"]["clean_source_only"] == ["a"]
    assert json.loads(output_json.read_text())["status"] == "source_contrastive_target_set_ready"
    assert "Source-Contrastive Target Set" in output_md.read_text()
