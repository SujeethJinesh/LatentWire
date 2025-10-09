import json
import pathlib

from latentwire.cli import eval as eval_cli


def test_cli_eval_dry_run(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "eval.json"
    config_path.write_text(json.dumps({"ckpt": "runs/ckpt", "samples": 1}))
    monkeypatch.setattr(eval_cli, "run_eval", lambda argv: None)
    eval_cli.main(["--config", str(config_path), "--dry-run"])
    out = capsys.readouterr().out
    assert "LatentWire Eval CLI" in out


def test_cli_eval_records_history(tmp_path, monkeypatch):
    out_dir = tmp_path / "runs"
    out_dir.mkdir()
    history = out_dir / "metrics_history.jsonl"
    config_path = tmp_path / "eval.json"
    config_path.write_text(json.dumps({"ckpt": "runs/ckpt", "out_dir": str(out_dir)}))
    monkeypatch.setattr(eval_cli, "run_eval", lambda argv: None)
    eval_cli.main(["--config", str(config_path), "--tag", "eval-test"])
    lines = history.read_text().strip().splitlines()
    assert lines and json.loads(lines[-1])["tag"] == "eval-test"
