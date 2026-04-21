from __future__ import annotations

import json

from scripts import bootstrap_process_repair_holdout as bootstrap


def test_build_route_command_matches_random_kv_holdout_contract() -> None:
    command = bootstrap.build_route_command(
        python="python",
        checkpoint="ckpt.pt",
        source_model="source",
        target_model="target",
        eval_file="eval.jsonl",
        device="cpu",
        dtype="float32",
        max_new_tokens=64,
        fixed_gate=0.1,
        salt=2,
        output="out.jsonl",
    )

    assert command[:2] == ["python", "scripts/evaluate.py"]
    assert command[command.index("--position-selection-metric") + 1] == "attention"
    assert command[command.index("--position-selection-ratio") + 1] == "0.50"
    assert command[command.index("--kv-route-selection-metric") + 1] == "random"
    assert command[command.index("--kv-value-selection-metric") + 1] == "random"
    assert command[command.index("--random-salt") + 1] == "2"
    assert command[command.index("--prediction-output") + 1] == "out.jsonl"


def test_bootstrap_writes_manifest_shell_and_limited_eval_file(tmp_path) -> None:
    output_dir = tmp_path / "holdout"
    bootstrap.main(
        [
            "--splits",
            "gsm70",
            "--salts",
            "0",
            "1",
            "--output-dir",
            str(output_dir),
            "--python",
            "python",
            "--device",
            "cpu",
            "--limit",
            "2",
        ]
    )

    manifest_path = output_dir / "process_repair_holdout_manifest.json"
    shell_path = output_dir / "run_process_repair_holdout.sh"
    limited_eval = output_dir / "qwen_gsm70_n2.jsonl"

    assert manifest_path.exists()
    assert shell_path.exists()
    assert limited_eval.exists()
    assert len(limited_eval.read_text(encoding="utf-8").splitlines()) == 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["settings"]["claim_policy"].startswith("dev-smoke")
    assert manifest["settings"]["limit"] == 2
    plan = manifest["plans"][0]
    assert plan["split"] == "gsm70"
    assert plan["eval_file"] == str(limited_eval)
    assert len(plan["route_commands"]) == 2
    assert plan["repair_command"]["command"][1] == "scripts/process_repair_routes.py"
    assert plan["repair_command"]["command"].count("--inputs") == 1

    shell = shell_path.read_text(encoding="utf-8")
    assert "scripts/evaluate.py" in shell
    assert "scripts/process_repair_routes.py" in shell
    assert "--kv-route-selection-metric random" in shell
    assert "--position-selection-metric attention" in shell
