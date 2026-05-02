from __future__ import annotations

import json
from pathlib import Path

from scripts import analyze_prompt_compression_control as control


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def test_analyze_source_tracks_budget_numbers_and_answer_spans(tmp_path: Path) -> None:
    source = tmp_path / "gsm.jsonl"
    rows = [
        {
            "prompt": "Question: Alice has 12 apples and gives away 5 apples. How many apples remain? Answer:",
            "answer_text": "7",
            "aliases": ["7", "#### 7"],
        },
        {
            "prompt": "Question: The box has 3 red balls, 4 blue balls, and 2 green balls. How many balls total? Answer:",
            "answer_text": "9",
            "aliases": ["9"],
        },
    ]
    _write_jsonl(source, rows)

    summary = control.analyze_source(source, budget_ratio=0.5, min_budget=8)

    assert summary.source == source.name
    assert summary.n_examples == 2
    assert summary.original_tokens_proxy_mean > summary.compressed_tokens_proxy_mean
    assert summary.bytes_saved_mean > 0
    assert summary.number_preservation_rate == 1.0
    assert summary.answer_span_coverage_rate == 0.0
    assert summary.answer_span_preservation_rate is None
    assert summary.examples[0].compressed_budget_tokens >= 8


def test_main_writes_json_and_markdown(tmp_path: Path) -> None:
    source = tmp_path / "svamp.jsonl"
    rows = [
        {
            "question": "Tiffany was collecting cans. She had 7 bags on monday and found 12 more bags on tuesday. How many more bags did she find?",
            "answer": "5",
            "aliases": ["5.0", "#### 5"],
            "metadata": {"id": "chal-31"},
        }
    ]
    _write_jsonl(source, rows)

    output_json = tmp_path / "control.json"
    output_md = tmp_path / "control.md"
    payload = control.main(
        [
            "--inputs",
            str(source),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--budget-ratio",
            "0.4",
            "--min-budget",
            "6",
        ]
    )

    assert json.loads(output_json.read_text()) == payload
    markdown = output_md.read_text()
    assert "# LLMLingua-Style Prompt Compression Control" in markdown
    assert source.name in markdown
    assert "Claim Risks" in markdown
