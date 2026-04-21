from __future__ import annotations

import json

from scripts import rerank_stochastic_routes


def _record(index: int, method: str, prediction: str, normalized: str, correct: bool) -> dict[str, object]:
    return {
        "index": index,
        "method": method,
        "prediction": prediction,
        "normalized_prediction": normalized,
        "correct": correct,
        "answer": [normalized if correct else "gold"],
        "generated_tokens": 20,
    }


def test_reranker_selects_agreement_and_format_policies() -> None:
    baseline = [_record(0, "target_alone", "target says 3", "3", False)]
    seed0 = [_record(0, "bridge", "messy answer 4", "4", False)]
    seed1 = [_record(0, "bridge", "Therefore, the answer is \\boxed{5}", "5", True)]
    seed2 = [_record(0, "bridge", "final answer is 5", "5", True)]

    records = rerank_stochastic_routes.rerank_records(
        [baseline + seed0, baseline + seed1, baseline + seed2],
        method="bridge",
    )
    by_method = {str(row["method"]): row for row in records}

    assert by_method["rerank_agreement_then_format"]["normalized_prediction"] == "5"
    assert by_method["rerank_format_then_agreement"]["normalized_prediction"] == "5"
    assert by_method["rerank_target_on_low_format"]["selected_candidate_source"] == "seed_1"
    assert by_method["rerank_agreement_then_format"]["candidate_vote_count"] == 2
    assert by_method["rerank_agreement_then_format"]["candidate_oracle_correct"] is True


def test_cli_writes_jsonl_markdown_and_sidecar(tmp_path) -> None:
    paths = []
    for salt, normalized in enumerate(["2", "3", "3"]):
        path = tmp_path / f"salt{salt}.jsonl"
        rows = [
            _record(0, "target_alone", "target answer is 2", "2", False),
            _record(0, "bridge", f"final answer is {normalized}", normalized, normalized == "3"),
        ]
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        paths.append(path)

    output = tmp_path / "reranked.jsonl"
    markdown = tmp_path / "reranked.md"
    records = rerank_stochastic_routes.rerank_records(
        [rerank_stochastic_routes.load_records(path) for path in paths],
        method="bridge",
    )
    results = rerank_stochastic_routes.summarize_results(records)
    rerank_stochastic_routes.write_prediction_records(str(output), records)
    rerank_stochastic_routes.write_prediction_sidecar(str(output), records, results, {"method": "bridge"})
    rerank_stochastic_routes.write_markdown_summary(results, markdown)

    loaded = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert {row["method"] for row in loaded} >= {"target_alone", "rerank_agreement_then_format"}
    assert (tmp_path / "reranked.jsonl.meta.json").exists()
    assert "rerank_agreement_then_format" in markdown.read_text(encoding="utf-8")
