import json
import pathlib

from latentwire.cli import run_ablation


def test_ablation_dry_run(tmp_path, monkeypatch, capsys):
    config = {
        "base_config": str(tmp_path / "base.json"),
        "base_overrides": ["checkpoint.save_dir=runs/tmp"],
        "runs": [
            {"name": "baseline", "overrides": {}},
            {"name": "feature", "overrides": {"features.use_lora": True}},
        ],
        "sweeps": {"optimizer.lr": [0.001, 0.002]},
    }
    base_config = {
        "model": {"llama_id": "foo", "qwen_id": "bar"},
        "checkpoint": {"save_dir": "runs/default"},
    }
    pathlib.Path(config["base_config"]).write_text(json.dumps(base_config))
    config_path = tmp_path / "ablation.json"
    config_path.write_text(json.dumps(config))

    monkeypatch.setattr(run_ablation.train_cli, "config_to_argv_list", lambda config_path, overrides: {
        "config": base_config,
        "argv": ["--llama_id", "foo"],
    })
    monkeypatch.setattr(run_ablation.train_cli, "run_train", lambda argv: None)

    run_ablation.run_ablation(str(config_path), namespace(dry_run=True, tag_prefix="test"))
    captured = capsys.readouterr()
    assert "baseline" in captured.out
    assert "sweep_2" in captured.out


class namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


