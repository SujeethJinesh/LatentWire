import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

from latentwire.cli import train as train_cli


def test_cli_train_dry_run(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "model": {"llama_id": "A", "qwen_id": "B"},
        "features": {"use_lora": False},
        "checkpoint": {"save_dir": str(tmp_path / "runs")},
    }))

    monkeypatch.setattr(train_cli, "run_train", lambda argv: None)
    def _fail_eval(*args, **kwargs):
        raise AssertionError("run_eval_from_config should not be called for dry runs")
    monkeypatch.setattr(train_cli, "run_eval_from_config", _fail_eval)
    train_cli.main(["--config", str(config_path), "--dry-run"])
    out = capsys.readouterr().out
    assert "LatentWire Train CLI" in out
    assert "Derived argv" in out
    assert not list(tmp_path.glob("pipeline_*.log"))
    assert "--train_encoder" not in out


@pytest.fixture
def temp_history(tmp_path):
    history = tmp_path / "runs" / "metrics_history.jsonl"
    history.parent.mkdir(parents=True)
    history.touch()
    return history


def test_cli_train_records_history(tmp_path, monkeypatch, temp_history):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "model": {"llama_id": "A", "qwen_id": "B"},
        "features": {"use_lora": False},
        "checkpoint": {"save_dir": str(temp_history.parent)},
    }))

    monkeypatch.setattr(train_cli, "run_train", lambda argv: None)
    def _fake_eval(cfg, eval_cfg, tag, pipeline_log=None):
        record = {
            "kind": "eval",
            "tag": f"{tag}-eval",
            "config": {"out_dir": str(temp_history.parent)},
            "argv": ["--ckpt", "dummy"],
        }
        if pipeline_log is not None:
            record["pipeline_log"] = str(pipeline_log)
        return record
    monkeypatch.setattr(train_cli, "run_eval_from_config", _fake_eval)
    train_cli.main(["--config", str(config_path), "--tag", "unit"])
    lines = temp_history.read_text().strip().splitlines()
    assert len(lines) >= 2
    train_record = json.loads(lines[-2])
    eval_record = json.loads(lines[-1])
    assert train_record["kind"] == "train"
    assert train_record["tag"] == "unit"
    assert eval_record["kind"] == "eval"
    assert eval_record["tag"] == "unit-eval"
    assert "pipeline_log" in train_record
    log_path = pathlib.Path(train_record["pipeline_log"])
    assert log_path.exists()
