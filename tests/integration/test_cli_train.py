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
    train_cli.main(["--config", str(config_path), "--tag", "unit"])
    lines = temp_history.read_text().strip().splitlines()
    assert lines
    record = json.loads(lines[-1])
    assert record["tag"] == "unit"
    assert "pipeline_log" in record
    log_path = pathlib.Path(record["pipeline_log"])
    assert log_path.exists()
