from __future__ import annotations

import json

from scripts import aggregate_stochastic_routes


def _record(index: int, method: str, normalized: str, correct: bool, *, salt: int | None = None) -> dict:
    record = {
        "index": index,
        "example_id": f"ex-{index}",
        "method": method,
        "prediction": normalized,
        "normalized_prediction": normalized,
        "answer": ["1"],
        "correct": correct,
    }
    if salt is not None:
        record["random_salt"] = salt
    return record


def test_aggregate_records_reports_majority_tiebreak_and_oracles() -> None:
    seed0 = [
        _record(0, "target_alone", "0", False),
        _record(1, "target_alone", "1", True),
        _record(0, "rotalign_kv_gate_0.10", "2", False, salt=0),
        _record(1, "rotalign_kv_gate_0.10", "2", False, salt=0),
    ]
    seed1 = [
        _record(0, "target_alone", "0", False),
        _record(1, "target_alone", "1", True),
        _record(0, "rotalign_kv_gate_0.10", "1", True, salt=1),
        _record(1, "rotalign_kv_gate_0.10", "3", False, salt=1),
    ]
    seed2 = [
        _record(0, "target_alone", "0", False),
        _record(1, "target_alone", "1", True),
        _record(0, "rotalign_kv_gate_0.10", "1", True, salt=2),
        _record(1, "rotalign_kv_gate_0.10", "4", False, salt=2),
    ]

    rows = aggregate_stochastic_routes.aggregate_records(
        [seed0, seed1, seed2],
        method="rotalign_kv_gate_0.10",
    )
    by_method = {}
    for row in rows:
        by_method.setdefault(row["method"], {})[row["index"]] = row

    assert by_method["stochastic_majority_vote"][0]["normalized_prediction"] == "1"
    assert by_method["stochastic_majority_vote"][0]["correct"] is True
    assert by_method["stochastic_majority_vote"][0]["vote_count"] == 2
    assert by_method["stochastic_target_tiebreak"][1]["method"] == "stochastic_target_tiebreak"
    assert by_method["stochastic_target_tiebreak"][1]["target_tiebreak_used"] is True
    assert by_method["stochastic_target_tiebreak"][1]["correct"] is True
    assert by_method["stochastic_any_seed_oracle"][0]["correct"] is True
    assert by_method["stochastic_target_or_seed_oracle"][1]["correct"] is True

    results = aggregate_stochastic_routes.summarize_results(rows)
    assert results["target_alone"] == 0.5
    assert results["stochastic_majority_vote"] == 0.5
    assert results["stochastic_target_tiebreak"] == 1.0
    assert results["stochastic_any_seed_oracle"] == 0.5
    assert results["stochastic_target_or_seed_oracle"] == 1.0
    assert results["paired_stochastic_target_tiebreak_vs_target_alone_delta_accuracy"] == 0.5


def test_cli_writes_jsonl_markdown_and_sidecar(tmp_path) -> None:
    paths = []
    for salt in range(3):
        path = tmp_path / f"salt{salt}.jsonl"
        rows = [
            _record(0, "target_alone", "0", False),
            _record(0, "rotalign_kv_gate_0.10", "1" if salt else "2", salt > 0, salt=salt),
        ]
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        paths.append(path)

    output = tmp_path / "agg.jsonl"
    markdown = tmp_path / "agg.md"
    args = [
        "--inputs",
        *(str(path) for path in paths),
        "--output-jsonl",
        str(output),
        "--output-md",
        str(markdown),
    ]
    import sys

    old_argv = sys.argv
    try:
        sys.argv = ["aggregate_stochastic_routes.py", *args]
        aggregate_stochastic_routes.main()
    finally:
        sys.argv = old_argv

    assert output.exists()
    assert markdown.exists()
    payload = json.loads((tmp_path / "agg.jsonl.meta.json").read_text())
    assert payload["run_config"]["method"] == "rotalign_kv_gate_0.10"
    assert payload["metric_summary"]["stochastic_majority_vote"] == 1.0
    assert "stochastic_target_or_seed_oracle" in markdown.read_text()
