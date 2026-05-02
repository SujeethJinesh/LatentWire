import json

from scripts import analyze_process_repair_source_controls as gate


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(example_id, method, correct):
    return {
        "example_id": example_id,
        "method": method,
        "answer": "1",
        "prediction": "1" if correct else "0",
        "normalized_prediction": "1" if correct else "0",
        "correct": bool(correct),
    }


def test_process_repair_source_control_passes_when_control_misses_route_specific_win(tmp_path):
    matched = tmp_path / "matched.jsonl"
    zero = tmp_path / "zero.jsonl"
    ids = ["base", "repair", "miss"]
    _write_jsonl(
        matched,
        [
            _row("base", "target_alone", True),
            _row("repair", "target_alone", False),
            _row("miss", "target_alone", False),
            _row("base", "target_self_repair", True),
            _row("repair", "target_self_repair", False),
            _row("miss", "target_self_repair", False),
            _row("base", "process_repair_selected_route", True),
            _row("repair", "process_repair_selected_route", True),
            _row("miss", "process_repair_selected_route", False),
        ],
    )
    _write_jsonl(
        zero,
        [
            _row(example_id, "process_repair_selected_route", example_id == "base")
            for example_id in ids
        ],
    )

    payload = gate.analyze(
        matched_path=matched,
        controls=[gate.ControlSpec("zero_source", zero)],
        method="process_repair_selected_route",
        target_method="target_alone",
        target_self_method="target_self_repair",
        run_date="2026-04-26",
    )

    assert payload["status"] == "process_repair_source_controls_support_matched_source"
    assert payload["matched_only_vs_target_self_ids"] == ["repair"]
    assert payload["source_specific_vs_target_self_ids"] == ["repair"]


def test_process_repair_source_control_fails_when_control_recovers_route_specific_win(tmp_path):
    matched = tmp_path / "matched.jsonl"
    zero = tmp_path / "zero.jsonl"
    ids = ["base", "repair"]
    _write_jsonl(
        matched,
        [
            _row("base", "target_alone", True),
            _row("repair", "target_alone", False),
            _row("base", "target_self_repair", True),
            _row("repair", "target_self_repair", False),
            _row("base", "process_repair_selected_route", True),
            _row("repair", "process_repair_selected_route", True),
        ],
    )
    _write_jsonl(
        zero,
        [
            _row(example_id, "process_repair_selected_route", example_id in {"base", "repair"})
            for example_id in ids
        ],
    )

    payload = gate.analyze(
        matched_path=matched,
        controls=[gate.ControlSpec("zero_source", zero)],
        method="process_repair_selected_route",
        target_method="target_alone",
        target_self_method="target_self_repair",
        run_date="2026-04-26",
    )

    assert payload["status"] == "process_repair_source_controls_do_not_clear_gate"
    assert payload["matched_only_vs_target_self_ids"] == ["repair"]
    assert payload["source_specific_vs_target_self_ids"] == []
