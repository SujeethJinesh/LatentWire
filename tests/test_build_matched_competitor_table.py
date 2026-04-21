from __future__ import annotations

import json
from pathlib import Path

from scripts import build_matched_competitor_table as table


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_summarize_row_prefers_method_summary_meta(tmp_path: Path) -> None:
    artifact = tmp_path / "results/run.jsonl"
    _write_jsonl(artifact, [{"method": "target_alone", "correct": False}])
    artifact.with_suffix(".jsonl.meta.json").write_text(
        json.dumps(
            {
                "method_summary": {
                    "target_alone": {
                        "accuracy": 0.25,
                        "count": 4,
                        "avg_bytes": 128.0,
                    }
                },
                "metric_summary": {
                    "target_alone_latency_sec": 12.5,
                },
            }
        ),
        encoding="utf-8",
    )
    spec = table.RowSpec(
        "target_alone",
        "Target alone",
        "control",
        "results/run.jsonl",
        "target_alone",
    )

    stats = table.summarize_row(spec, tmp_path)

    assert stats.status == "present"
    assert stats.accuracy == 0.25
    assert stats.n == 4
    assert stats.latency_sec == 12.5
    assert stats.token_proxy == 128.0
    assert stats.source == "meta:method_summary"


def test_summarize_row_uses_run_meta_for_plain_competitor_meta(tmp_path: Path) -> None:
    artifact = tmp_path / "results/c2c.jsonl"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("", encoding="utf-8")
    artifact.with_suffix(".jsonl.meta.json").write_text(
        json.dumps(
            {
                "num_examples": 10,
                "limit": 999,
                "correct": 3,
                "latency_sec": 2.0,
                "generated_tokens_avg": 41.5,
            }
        ),
        encoding="utf-8",
    )
    spec = table.RowSpec("c2c", "C2C", "direct", "results/c2c.jsonl")

    stats = table.summarize_row(spec, tmp_path)

    assert stats.status == "present"
    assert stats.accuracy == 0.3
    assert stats.n == 10
    assert stats.latency_sec == 2.0
    assert stats.token_proxy == 41.5
    assert stats.source == "meta:run"


def test_summarize_row_uses_limit_as_count_when_num_examples_is_absent(tmp_path: Path) -> None:
    artifact = tmp_path / "results/kvpress.jsonl"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("", encoding="utf-8")
    artifact.with_suffix(".jsonl.meta.json").write_text(
        json.dumps({"limit": 20, "accuracy": 0.1, "latency_sec": 10.0}),
        encoding="utf-8",
    )
    spec = table.RowSpec("kvpress", "KVPress", "control", "results/kvpress.jsonl")

    stats = table.summarize_row(spec, tmp_path)

    assert stats.status == "present"
    assert stats.n == 20
    assert stats.accuracy == 0.1


def test_summarize_row_preserves_blocked_run_meta_status(tmp_path: Path) -> None:
    artifact = tmp_path / "results/latent.jsonl"
    artifact.parent.mkdir(parents=True)
    artifact.with_suffix(".jsonl.meta.json").write_text(
        json.dumps(
            {
                "status": "blocked",
                "num_examples": 1,
                "error_type": "IndexError",
                "error_message": "cache-position failure",
            }
        ),
        encoding="utf-8",
    )
    spec = table.RowSpec("latent", "LatentMAS latent", "probe", "results/latent.jsonl")

    stats = table.summarize_row(spec, tmp_path)

    assert stats.status == "blocked"
    assert stats.n == 1
    assert stats.accuracy is None


def test_summarize_row_falls_back_to_jsonl_records_and_filters_method(tmp_path: Path) -> None:
    artifact = tmp_path / "results/routes.jsonl"
    _write_jsonl(
        artifact,
        [
            {
                "example_id": "a",
                "method": "selected_route_no_repair",
                "correct": True,
                "generated_tokens": 10,
                "latency_sec": 0.2,
            },
            {
                "example_id": "b",
                "method": "selected_route_no_repair",
                "correct": False,
                "generated_tokens": 20,
                "latency_sec": 0.3,
            },
            {
                "example_id": "b",
                "method": "target_alone",
                "correct": True,
                "generated_tokens": 30,
                "latency_sec": 0.9,
            },
        ],
    )
    spec = table.RowSpec(
        "selected",
        "Selected route, no repair",
        "ours",
        "results/routes.jsonl",
        "selected_route_no_repair",
    )

    stats = table.summarize_row(spec, tmp_path)

    assert stats.status == "present"
    assert stats.n == 2
    assert stats.accuracy == 0.5
    assert stats.latency_sec == 0.5
    assert stats.token_proxy == 15.0
    assert stats.source == "jsonl"


def test_build_matrix_renders_mixed_present_and_missing_rows(tmp_path: Path) -> None:
    present = tmp_path / "results/present.jsonl"
    _write_jsonl(present, [{"id": "0", "correct": True, "trace": {"input_token_count": 7}}])
    rows = [
        table.RowSpec("present", "Present row", "direct", "results/present.jsonl"),
        table.RowSpec("missing", "LatentMAS placeholder", "latent", "results/missing.jsonl"),
    ]

    stats = table.build_matrix(tmp_path, rows)
    markdown = table.render_markdown(stats)

    assert [row.status for row in stats] == ["present", "missing"]
    assert "Present row" in markdown
    assert "LatentMAS placeholder" in markdown
    assert "artifact missing" in markdown
    assert "`results/missing.jsonl`" in markdown


def test_default_rows_include_latentmas_harness_probes() -> None:
    row_ids = {row.row_id for row in table.DEFAULT_ROWS}

    assert "latentmas_baseline_probe" in row_ids
    assert "latentmas_text_mas_probe" in row_ids
    assert "latentmas_latent_mas_blocker_probe" in row_ids
