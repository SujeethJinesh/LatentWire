from __future__ import annotations

from types import SimpleNamespace

from latent_bridge import c2c_eval


def test_build_c2c_messages_wraps_prompt_as_single_user_turn() -> None:
    assert c2c_eval.build_c2c_messages("hello") == [{"role": "user", "content": "hello"}]


def test_run_c2c_generation_eval_writes_prediction_records(tmp_path, monkeypatch) -> None:
    eval_file = tmp_path / "eval.jsonl"
    eval_file.write_text('{"prompt":"Q: 1+1?","answer_text":"2","aliases":[]}\n', encoding="utf-8")

    artifact = SimpleNamespace(
        repo_id="repo",
        subdir="subdir",
        config_path="cfg.json",
        checkpoint_dir="final",
        local_root="/tmp/root",
    )

    monkeypatch.setattr(
        c2c_eval,
        "load_c2c_model",
        lambda **kwargs: (object(), object(), artifact),
    )
    monkeypatch.setattr(
        c2c_eval,
        "generate_c2c_text",
        lambda *args, **kwargs: ("The answer is 2.", 4, 0.25),
    )

    output = tmp_path / "predictions.jsonl"
    payload = c2c_eval.run_c2c_generation_eval(
        source_model="src",
        target_model="tgt",
        eval_file=str(eval_file),
        device="cpu",
        max_new_tokens=8,
        prediction_output=str(output),
    )

    assert payload["metrics"]["c2c_accuracy"] == 1.0
    assert output.exists()
    assert output.with_suffix(".jsonl.meta.json").exists()
