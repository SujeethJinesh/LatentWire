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


def test_reranker_selects_format_and_numeric_policies() -> None:
    baseline = [_record(0, "target_alone", "target says seven", "7", False)]
    seed0 = [_record(0, "bridge", "Therefore, the answer is 5 because 2+2=4.", "5", False)]
    seed1 = [_record(0, "bridge", "4 + 4 = 8.", "8", True)]

    records = rerank_stochastic_routes.rerank_records(
        [baseline + seed0, baseline + seed1],
        method="bridge",
    )
    by_method = {str(row["method"]): row for row in records}

    assert by_method["rerank_format_then_agreement"]["selected_candidate_source"] == "seed_0"
    assert by_method["rerank_target_on_strict_format"]["selected_candidate_source"] == "seed_0"
    assert by_method["rerank_numeric_consistency_then_completion"]["selected_candidate_source"] == "seed_1"
    assert by_method["rerank_completion_then_numeric_consistency"]["selected_candidate_source"] == "seed_1"
    assert by_method["rerank_numeric_consistency_or_target"]["selected_candidate_source"] == "seed_1"
    assert by_method["rerank_numeric_consistency_then_completion"]["selected_candidate_tail_numeric_mention"] == "8"
    assert by_method["rerank_numeric_consistency_then_completion"]["selected_candidate_numeric_consistency_score"] > by_method[
        "rerank_format_then_agreement"
    ]["selected_candidate_numeric_consistency_score"]
    assert by_method["rerank_numeric_consistency_then_completion"]["candidate_scores"][2]["numeric_consistency_score"] > by_method[
        "rerank_numeric_consistency_then_completion"
    ]["candidate_scores"][1]["numeric_consistency_score"]


def test_strict_format_policy_keeps_target_when_seed_gain_is_small() -> None:
    baseline = [_record(0, "target_alone", "Therefore, the answer is 7.", "7", True)]
    seed = [_record(0, "bridge", "Therefore, the answer is 5.", "5", False)]

    records = rerank_stochastic_routes.rerank_records([baseline + seed], method="bridge")
    by_method = {str(row["method"]): row for row in records}

    assert by_method["rerank_target_on_strict_format"]["selected_candidate_source"] == "target"


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
    assert {row["method"] for row in loaded} >= {
        "target_alone",
        "rerank_agreement_then_format",
        "rerank_target_on_strict_format",
        "rerank_numeric_consistency_then_completion",
    }
    assert (tmp_path / "reranked.jsonl.meta.json").exists()
    assert "rerank_agreement_then_format" in markdown.read_text(encoding="utf-8")
    assert "Numeric ablation" in markdown.read_text(encoding="utf-8")
