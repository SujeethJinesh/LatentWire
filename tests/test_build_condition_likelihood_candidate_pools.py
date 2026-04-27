import json

from scripts import build_condition_likelihood_candidate_pools as build


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_builds_condition_pools_and_recomputes_shuffled_correctness(tmp_path):
    target = tmp_path / "target.jsonl"
    text = tmp_path / "text.jsonl"
    source = tmp_path / "source.jsonl"
    out = tmp_path / "out"
    base = [
        {"example_id": "a", "index": 0, "answer": ["5", "5.0"], "method": "target_alone", "prediction": "0", "normalized_prediction": "0", "correct": False},
        {"example_id": "b", "index": 1, "answer": ["7", "7.0"], "method": "target_alone", "prediction": "7", "normalized_prediction": "7", "correct": True},
    ]
    _write_jsonl(target, base)
    _write_jsonl(text, [{**row, "method": "text_to_text"} for row in base])
    _write_jsonl(
        source,
        [
            {**base[0], "method": "source_alone", "prediction": "5", "normalized_prediction": "5", "correct": True},
            {**base[1], "method": "source_alone", "prediction": "5", "normalized_prediction": "5", "correct": False},
        ],
    )

    payload = build.build(
        build.parse_args(
            [
                "--target-jsonl",
                str(target),
                "--text-jsonl",
                str(text),
                "--source-jsonl",
                str(source),
                "--output-dir",
                str(out),
                "--shuffle-offset",
                "1",
                "--label-shuffle-offset",
                "1",
            ]
        )
    )

    assert payload["reference_n"] == 2
    shuffled = [json.loads(line) for line in (out / "shuffled_source" / "source.jsonl").read_text().splitlines()]
    assert shuffled[0]["example_id"] == "a"
    assert shuffled[0]["control_donor_example_id"] == "b"
    assert shuffled[0]["prediction"] == "5"
    assert shuffled[0]["correct"] is True
    assert shuffled[1]["example_id"] == "b"
    assert shuffled[1]["control_donor_example_id"] == "a"
    assert shuffled[1]["correct"] is False

    zero = [json.loads(line) for line in (out / "zero_source" / "source.jsonl").read_text().splitlines()]
    assert zero[0]["prediction"] == ""
    assert zero[0]["correct"] is False

    target_only_files = sorted(path.name for path in (out / "target_only").iterdir())
    assert target_only_files == ["source.jsonl", "target.jsonl", "text.jsonl"]
    target_only_source = [
        json.loads(line)
        for line in (out / "target_only" / "source.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert target_only_source[0]["prediction"] == "0"
    assert target_only_source[0]["correct"] is False

    label_shuffle_target = [
        json.loads(line)
        for line in (out / "label_shuffle" / "target.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    label_shuffle_source = [
        json.loads(line)
        for line in (out / "label_shuffle" / "source.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert label_shuffle_target[0]["prediction"] == "5"
    assert label_shuffle_source[0]["prediction"] == "0"
