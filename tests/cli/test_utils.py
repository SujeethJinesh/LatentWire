import json
import os
import tempfile
import pathlib

from latentwire.cli.utils import (
    append_metrics_history,
    apply_overrides,
    config_to_argv,
    ensure_metrics_history,
    flatten_training_config,
    load_training_config,
)


def test_config_round_trip(tmp_path):
    config_path = tmp_path / "config.json"
    sample = {
        "model": {"llama_id": "foo", "qwen_id": "bar"},
        "features": {"use_lora": True},
    }
    config_path.write_text(json.dumps(sample))
    cfg, cfg_dict = load_training_config(str(config_path))
    assert cfg.model.llama_id == "foo"
    assert cfg_dict["features"]["use_lora"] is True


def test_apply_overrides():
    config = {"model": {"llama_id": "foo"}, "data": {"samples": 1}}
    updated = apply_overrides(config, ["model.llama_id=baz", "data.samples=5", "checkpoint.save_dir=/tmp"])
    assert updated["model"]["llama_id"] == "baz"
    assert updated["data"]["samples"] == 5
    assert updated["checkpoint"]["save_dir"] == "/tmp"


def test_config_to_argv():
    flat = {"llama_id": "foo", "use_lora": True, "batch_size": 2, "adapter_metadata": False}
    argv = config_to_argv(flat)
    assert "--llama_id" in argv and "foo" in argv
    assert "--use_lora" in argv
    assert "--batch_size" in argv and "2" in argv
    assert "--no_adapter_metadata" in argv


def test_metrics_history_append(tmp_path):
    path = pathlib.Path(ensure_metrics_history(str(tmp_path)))
    record = {"tag": "test", "argv": ["--foo"]}
    append_metrics_history(str(tmp_path), record)
    data = path.read_text().strip().splitlines()
    assert len(data) == 1
    payload = json.loads(data[0])
    assert payload["tag"] == "test"


def test_flatten_training_config():
    config = {
        "model": {"llama_id": "foo"},
        "adapter": {"adapter_hidden_mult": 2},
        "misc": 42,
    }
    flat = flatten_training_config(config)
    assert flat["llama_id"] == "foo"
    assert flat["adapter_hidden_mult"] == 2
    assert flat["misc"] == 42
