from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts import audit_real_tokenizer_interface as audit


class _FakeTokenizer:
    def __init__(self, *, split_digits: bool) -> None:
        self.split_digits = split_digits
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: list[str] = []

    def _segment(self, text: str) -> list[str]:
        tokens: list[str] = []
        current = []
        for ch in text:
            if ch.isalpha():
                if current and not current[-1].isalpha():
                    tokens.append("".join(current))
                    current = []
                current.append(ch)
            elif ch.isdigit():
                if self.split_digits:
                    if current:
                        tokens.append("".join(current))
                        current = []
                    tokens.append(ch)
                else:
                    if current and not current[-1].isdigit():
                        tokens.append("".join(current))
                        current = []
                    current.append(ch)
            elif ch in "+-*/=:":
                if current:
                    tokens.append("".join(current))
                    current = []
                tokens.append(ch)
            elif ch.isspace():
                if current:
                    tokens.append("".join(current))
                    current = []
            else:
                if current:
                    tokens.append("".join(current))
                    current = []
                tokens.append(ch)
        if current:
            tokens.append("".join(current))
        return tokens

    def _ensure_id(self, token: str) -> int:
        if token not in self._token_to_id:
            self._token_to_id[token] = len(self._id_to_token)
            self._id_to_token.append(token)
        return self._token_to_id[token]

    def __call__(self, text: str, *, add_special_tokens: bool = False, return_tensors=None) -> SimpleNamespace:
        del add_special_tokens, return_tensors
        return SimpleNamespace(input_ids=[self._ensure_id(token) for token in self._segment(text)])

    def decode(self, token_ids, **kwargs) -> str:
        del kwargs
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join(self._id_to_token[int(token_id)] for token_id in token_ids)

    def convert_ids_to_tokens(self, token_ids):
        if isinstance(token_ids, int):
            return self._id_to_token[int(token_ids)]
        return [self._id_to_token[int(token_id)] for token_id in token_ids]


def test_real_tokenizer_interface_audit_cli_writes_json_and_markdown(tmp_path, monkeypatch) -> None:
    source = _FakeTokenizer(split_digits=False)
    target = _FakeTokenizer(split_digits=True)

    def fake_from_pretrained(model_name: str, *args, **kwargs):
        del args, kwargs
        if "Qwen2.5" in model_name:
            return source
        if "Qwen3" in model_name:
            return target
        raise AssertionError(f"unexpected model: {model_name}")

    monkeypatch.setattr(audit.AutoTokenizer, "from_pretrained", fake_from_pretrained)

    input_path = tmp_path / "gsm8k_gate_search_30.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prompt": "Solve: 12+3=15",
                        "answer_text": "15",
                        "aliases": [],
                        "source_question": "Solve: 12+3=15",
                    }
                ),
                json.dumps(
                    {
                        "prompt": "Answer: 8/2=4",
                        "answer_text": "4",
                        "aliases": [],
                        "source_question": "Answer: 8/2=4",
                    }
                ),
            ]
        )
        + "\n"
    )

    output_json = tmp_path / "audit.json"
    output_md = tmp_path / "audit.md"

    payload = audit.main(
        [
            "--input",
            str(input_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )

    on_disk = json.loads(output_json.read_text())
    assert on_disk == payload
    assert on_disk["summary"]["examples"] == 2
    assert on_disk["summary"]["mean_bytes_per_example"] > 0
    assert 0.0 < on_disk["summary"]["mean_source_fragmentation"] < on_disk["summary"]["mean_target_fragmentation"]
    assert 0.0 < on_disk["summary"]["mean_shared_decoded_token_rate"] <= 1.0
    assert 0.0 < on_disk["summary"]["mean_shared_digit_token_rate"] <= 1.0
    assert on_disk["rows"][0]["source_digit_token_count"] == 3
    assert on_disk["rows"][0]["target_digit_token_count"] == 5
    assert on_disk["rows"][0]["shared_decoded_token_rate"] == 0.5
    assert on_disk["rows"][0]["shared_digit_token_rate"] == 1 / 6

    md = output_md.read_text()
    assert "# Real Tokenizer Interface Audit" in md
    assert "| Example | Bytes | Src toks | Tgt toks | Src frag | Tgt frag | Shared rate |" in md
