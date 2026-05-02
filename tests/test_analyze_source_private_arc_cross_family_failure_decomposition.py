from __future__ import annotations

import json

from scripts import analyze_source_private_arc_cross_family_failure_decomposition as decomp


def _write_jsonl(path, rows) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _prediction(
    *,
    split: str,
    seed: int,
    condition: str,
    row_id: str,
    answer_index: int,
    source_selected_index: int,
    prediction_index: int,
) -> dict:
    return {
        "split": split,
        "seed": seed,
        "condition": condition,
        "row_id": row_id,
        "content_id": f"content-{row_id}",
        "answer_index": answer_index,
        "correct": prediction_index == answer_index,
        "prediction_index": prediction_index,
        "payload_bytes": 8,
        "metadata": {"source_selected_index": source_selected_index},
    }


def _wrapper(tmp_path):
    wrapper = tmp_path / "wrapper"
    wrapper.mkdir()
    payload = {
        "alternate_source_family": "toy_phi",
        "pass_gate": False,
        "basis": {"budget_bytes": 8, "seeds": [1]},
        "headline": {},
        "splits": {
            split: {
                "full_slice": {
                    "aggregate": {
                        "target_accuracy": 0.5,
                        "matched_accuracy_mean": 0.0,
                    }
                }
            }
            for split in ("validation", "test")
        },
    }
    (wrapper / "source_family_cache_falsification.json").write_text(
        json.dumps(payload, sort_keys=True), encoding="utf-8"
    )
    agreement = "\n".join(
        [
            "split,row_id,content_id,alt_source_selected_index,qwen_source_selected_index,agree,answer_index,alt_source_correct,qwen_source_correct",
            "validation,row-1,content-row-1,0,1,False,1,False,True",
            "test,row-1,content-row-1,0,1,False,1,False,True",
            "",
        ]
    )
    (wrapper / "source_cache_agreement.csv").write_text(agreement, encoding="utf-8")
    _write_jsonl(
        wrapper / "matched_predictions.jsonl",
        [
            _prediction(
                split="validation",
                seed=1,
                condition=decomp.MATCHED,
                row_id="row-1",
                answer_index=1,
                source_selected_index=0,
                prediction_index=0,
            ),
            _prediction(
                split="test",
                seed=1,
                condition=decomp.MATCHED,
                row_id="row-1",
                answer_index=1,
                source_selected_index=0,
                prediction_index=0,
            ),
        ],
    )
    _write_jsonl(
        wrapper / "qwen_disagreement_predictions.jsonl",
        [
            _prediction(
                split="validation",
                seed=1,
                condition=decomp.MATCHED,
                row_id="row-1",
                answer_index=1,
                source_selected_index=0,
                prediction_index=0,
            ),
            _prediction(
                split="validation",
                seed=1,
                condition=decomp.QWEN_SUB,
                row_id="row-1",
                answer_index=1,
                source_selected_index=1,
                prediction_index=1,
            ),
            _prediction(
                split="test",
                seed=1,
                condition=decomp.MATCHED,
                row_id="row-1",
                answer_index=1,
                source_selected_index=0,
                prediction_index=0,
            ),
            _prediction(
                split="test",
                seed=1,
                condition=decomp.QWEN_SUB,
                row_id="row-1",
                answer_index=1,
                source_selected_index=1,
                prediction_index=1,
            ),
        ],
    )
    return wrapper


def test_decomposition_marks_faithful_weak_source_as_source_endpoint_blocker(tmp_path) -> None:
    payload = decomp.build_decomposition(
        output_dir=tmp_path / "out",
        wrapper_dirs=[_wrapper(tmp_path)],
    )

    wrapper = payload["wrappers"][0]
    assert wrapper["source_family"] == "toy_phi"
    assert wrapper["source_agreement"]["test"]["full"]["alt_source_accuracy"] == 0.0
    assert wrapper["packet_summary"]["full_matched"]["test"][decomp.MATCHED][
        "prediction_matches_source_selected_mean"
    ] == 1.0
    assert wrapper["decision"]["primary_blocker"] == "source_endpoint_quality"
    assert payload["headline"]["selected_next_gate"] == "common_feature_connector_with_stronger_source"
    assert (tmp_path / "out" / "arc_cross_family_failure_decomposition.json").exists()
    assert (tmp_path / "out" / "family_failure_summary.csv").exists()
