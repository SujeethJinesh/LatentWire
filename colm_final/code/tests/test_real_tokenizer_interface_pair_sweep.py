from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts import run_real_tokenizer_interface_pair_sweep as sweep


class _FakeTokenizer:
    def __init__(self, *, split_digits: bool) -> None:
        self.split_digits = split_digits
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: list[str] = []

    def _segment(self, text: str) -> list[tuple[str, tuple[int, int]]]:
        tokens: list[tuple[str, tuple[int, int]]] = []
        current = []
        start = 0
        for index, ch in enumerate(text):
            if ch.isalpha():
                if current and not current[-1].isalpha():
                    tokens.append(("".join(current), (start, index)))
                    current = []
                    start = index
                elif not current:
                    start = index
                current.append(ch)
            elif ch.isdigit():
                if self.split_digits:
                    if current:
                        tokens.append(("".join(current), (start, index)))
                        current = []
                    tokens.append((ch, (index, index + 1)))
                    start = index + 1
                else:
                    if current and not current[-1].isdigit():
                        tokens.append(("".join(current), (start, index)))
                        current = []
                        start = index
                    elif not current:
                        start = index
                    current.append(ch)
            elif ch.isspace():
                if current:
                    tokens.append(("".join(current), (start, index)))
                    current = []
                tokens.append((ch, (index, index + 1)))
            else:
                if current:
                    tokens.append(("".join(current), (start, index)))
                    current = []
                tokens.append((ch, (index, index + 1)))
                start = index + 1
        if current:
            tokens.append(("".join(current), (start, len(text))))
        return tokens

    def _ensure_id(self, token: str) -> int:
        if token not in self._token_to_id:
            self._token_to_id[token] = len(self._id_to_token)
            self._id_to_token.append(token)
        return self._token_to_id[token]

    def __call__(self, text: str, *, add_special_tokens: bool = False, return_offsets_mapping: bool = False, return_tensors=None):
        del add_special_tokens, return_tensors
        segmented = self._segment(text)
        input_ids = [self._ensure_id(token) for token, _ in segmented]
        offsets = [offset for _, offset in segmented]
        if return_offsets_mapping:
            return SimpleNamespace(input_ids=input_ids, offset_mapping=offsets)
        return SimpleNamespace(input_ids=input_ids)

    def decode(self, token_ids, **kwargs) -> str:
        del kwargs
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join(self._id_to_token[int(token_id)] for token_id in token_ids)

    def convert_ids_to_tokens(self, token_ids):
        if isinstance(token_ids, int):
            return self._id_to_token[int(token_ids)]
        return [self._id_to_token[int(token_id)] for token_id in token_ids]


def test_real_tokenizer_interface_pair_sweep_cli_writes_json_and_markdown(tmp_path, monkeypatch) -> None:
    same = _FakeTokenizer(split_digits=False)
    cross = _FakeTokenizer(split_digits=True)

    def fake_from_pretrained(model_name: str, *args, **kwargs):
        del args, kwargs
        if model_name == "same-a":
            return same
        if model_name == "same-b":
            return same
        if model_name == "cross-b":
            return cross
        raise AssertionError(f"unexpected model: {model_name}")

    monkeypatch.setattr(sweep.AutoTokenizer, "from_pretrained", fake_from_pretrained)

    input_path = tmp_path / "prompts.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "Solve 12+3=15."}),
                json.dumps({"prompt": "Compute 101-22=79."}),
                json.dumps({"prompt": "Answer 256/2=128."}),
                json.dumps({"prompt": "Check 999-1=998."}),
            ]
        )
        + "\n"
    )
    output_json = tmp_path / "pair_sweep.json"
    output_md = tmp_path / "pair_sweep.md"

    payload = sweep.main(
        [
            "--input",
            str(input_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--calibration-examples",
            "2",
            "--remap-capacity",
            "8",
            "--pair",
            "same_pair",
            "same-a",
            "same-b",
            "--pair",
            "cross_pair",
            "same-a",
            "cross-b",
        ]
    )

    on_disk = json.loads(output_json.read_text())
    assert on_disk == payload
    rows = {row["label"]: row for row in on_disk["rows"]}
    assert set(rows) == {"same_pair", "cross_pair"}
    assert rows["same_pair"]["mean_shared_decoded_token_rate"] == 1.0
    assert rows["same_pair"]["mean_boundary_f1"] == 1.0
    assert rows["cross_pair"]["mean_target_token_count"] > rows["cross_pair"]["mean_source_token_count"]
    assert rows["cross_pair"]["mean_boundary_f1"] < 1.0
    assert rows["cross_pair"]["mean_byte_span_remap_coverage"] > 0.0

    md = output_md.read_text()
    assert "# Real Tokenizer Interface Pair Sweep" in md
    assert "| Pair | Src frag | Tgt frag | Frag delta | Shared decoded | Boundary F1 | Remap coverage |" in md
