from __future__ import annotations

import pathlib
import sys

from scripts import harness_common as harness


def test_resolve_materialized_eval_file_defaults_under_results_dir() -> None:
    path = harness.resolve_materialized_eval_file(
        None,
        results_dir=pathlib.Path("results/my_campaign"),
        slice_size=70,
    )
    assert path == pathlib.Path("results/my_campaign/_artifacts/gsm8k_eval_70.jsonl")


def test_python_executable_prefers_venv_arm64(tmp_path: pathlib.Path) -> None:
    py = tmp_path / "venv_arm64" / "bin" / "python"
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_text("")
    assert harness.python_executable(tmp_path) == str(py)


def test_python_executable_falls_back_to_dot_venv(tmp_path: pathlib.Path) -> None:
    py = tmp_path / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_text("")
    assert harness.python_executable(tmp_path) == str(py)


def test_python_executable_falls_back_to_running_interpreter(tmp_path: pathlib.Path) -> None:
    assert harness.python_executable(tmp_path) == sys.executable


def test_chat_template_cli_args_match_expected_flags() -> None:
    assert harness.chat_template_cli_args(enabled=False, thinking=False) == []
    assert harness.chat_template_cli_args(enabled=True, thinking=False) == [
        "--source-use-chat-template",
        "--target-use-chat-template",
        "--source-enable-thinking",
        "false",
        "--target-enable-thinking",
        "false",
    ]
