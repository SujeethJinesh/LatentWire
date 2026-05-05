from __future__ import annotations

import json
import pathlib

import torch

from scripts import analyze_svamp32_syndrome_sidecar_probe as syndrome
from scripts import analyze_svamp32_c2c_preanswer_state_gate as gate


class SpaceTokenizer:
    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(str(int(token_id)) for token_id in token_ids)


def test_final_numeric_span_prefers_last_final_answer_number() -> None:
    text = "There were 7 bags, then 12 more. #### 19\nThe answer is: 19"

    value, start, end = gate._final_numeric_span(text)

    assert value == "19"
    assert text[start:end] == "19"
    assert start == text.rfind("19")


def test_locate_answer_span_maps_character_span_to_token_counts() -> None:
    tokenizer = SpaceTokenizer()
    generated = torch.tensor([7, 12, 19, 19])
    decoded = tokenizer.decode(generated.tolist())

    span = gate.locate_answer_span(tokenizer, generated, decoded)

    assert span.value == "19"
    assert span.pre_answer_token_count == 3
    assert span.answer_end_token_count == 4


def test_summarize_score_window_has_fixed_schema_for_empty_and_nonempty_windows() -> None:
    empty_features, empty_meta = gate.summarize_score_window(
        [],
        torch.tensor([], dtype=torch.long),
        prefix="pre",
    )
    logits = torch.tensor([[0.0, 2.0, 1.0]])
    full_features, full_meta = gate.summarize_score_window(
        [logits],
        torch.tensor([1], dtype=torch.long),
        prefix="pre",
    )

    assert empty_features.shape == full_features.shape
    assert empty_meta["feature_names"] == full_meta["feature_names"]
    assert full_meta["step_count"] == 1
    assert empty_meta["step_count"] == 0
    assert torch.any(full_features != 0)


def test_limit_helpers_write_debug_specs_and_target_set(tmp_path: pathlib.Path) -> None:
    records = [
        {"example_id": "a", "method": "m", "prediction": "1"},
        {"example_id": "b", "method": "m", "prediction": "2"},
    ]
    source = tmp_path / "rows.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    spec = syndrome.RowSpec(label="rows", path=source, method="m")

    limited = gate._limit_spec_to_ids(spec, reference_ids=["b"], output_dir=tmp_path / "debug")

    assert limited.path.read_text(encoding="utf-8").count("\n") == 1
    assert json.loads(limited.path.read_text(encoding="utf-8"))["example_id"] == "b"

    target_set = tmp_path / "target_set.json"
    target_set.write_text(
        json.dumps({"reference_ids": ["a", "b"], "ids": {"teacher_only": ["a", "b"]}}),
        encoding="utf-8",
    )
    limited_target_set = gate._limit_target_set_to_ids(
        target_set,
        reference_ids=["b"],
        output_dir=tmp_path / "debug",
    )

    payload = json.loads(limited_target_set.read_text(encoding="utf-8"))
    assert payload["reference_ids"] == ["b"]
    assert payload["ids"]["teacher_only"] == ["b"]
