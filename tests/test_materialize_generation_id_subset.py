from __future__ import annotations

import json

from scripts import materialize_generation_id_subset as subset


def _stable_row(question: str, answer: str) -> dict[str, object]:
    return {"question": question, "answer": answer, "aliases": [f"#### {answer}"]}


def test_materialize_generation_id_subset_preserves_requested_order(tmp_path) -> None:
    eval_path = tmp_path / "eval.jsonl"
    target_set = tmp_path / "target_set.json"
    output = tmp_path / "subset.jsonl"
    rows = [
        _stable_row("A has 1 apple. How many apples?", "1"),
        _stable_row("B has 2 apples. How many apples?", "2"),
        _stable_row("C has 3 apples. How many apples?", "3"),
    ]
    eval_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    examples = subset.load_generation(str(eval_path))
    ids = [subset._generation_example_id(example) for example in examples]
    target_set.write_text(
        json.dumps(
            {
                "ids": {
                    "clean_residual_targets": [ids[2], ids[0]],
                    "target_self_repair": [ids[1], ids[0]],
                }
            }
        ),
        encoding="utf-8",
    )

    subset.main(
        [
            "--eval-file",
            str(eval_path),
            "--target-set-json",
            str(target_set),
            "--id-fields",
            "clean_residual_targets",
            "target_self_repair",
            "--output-jsonl",
            str(output),
        ]
    )

    materialized = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["answer"] for row in materialized] == ["3", "1", "2"]
    meta = json.loads((tmp_path / "subset.jsonl.meta.json").read_text(encoding="utf-8"))
    assert meta["selected_ids"] == [ids[2], ids[0], ids[1]]
    assert meta["output_n"] == 3
