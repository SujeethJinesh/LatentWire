from __future__ import annotations

import json
from pathlib import Path

from experimental.hybridkernel.phase0 import preflight_environment


def test_collect_preflight_marks_triton_index_blocker(monkeypatch) -> None:
    def fake_import_status(module_name: str):
        if module_name == "torch":
            return {
                "importable": True,
                "version": "test-torch",
                "error": None,
                "cuda_version": None,
                "cuda_available": False,
                "cuda_device_count": 0,
                "mps_built": True,
                "mps_available": True,
            }
        return {"importable": False, "version": None, "error": "module not found"}

    monkeypatch.setattr(preflight_environment, "_torch_status", lambda: fake_import_status("torch"))
    monkeypatch.setattr(
        preflight_environment,
        "_import_status",
        lambda module_name: fake_import_status(module_name),
    )
    monkeypatch.setattr(
        preflight_environment,
        "_run_pip_index",
        lambda package, timeout_seconds: {
            "package": package,
            "command": ["python", "-m", "pip", "index", "versions", package],
            "returncode": 1,
            "available": False,
            "timed_out": False,
            "stdout": "",
            "stderr": "ERROR: No matching distribution found",
        },
    )

    payload = preflight_environment.collect_preflight(check_pip_index=True)

    assert payload["status"] == "BLOCKED_TRITON_UNAVAILABLE"
    assert payload["triton"]["install_possible_from_current_index"] is False
    assert "no matching package" in payload["triton"]["blocker"]
    assert len(payload["triton"]["pip_index"]) == 3


def test_write_outputs_records_json_and_markdown(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        preflight_environment,
        "_torch_status",
        lambda: {
            "importable": True,
            "version": "test-torch",
            "error": None,
            "cuda_version": None,
            "cuda_available": False,
            "cuda_device_count": 0,
            "mps_built": True,
            "mps_available": True,
        },
    )
    monkeypatch.setattr(
        preflight_environment,
        "_import_status",
        lambda module_name: {"importable": True, "version": "test-triton", "error": None},
    )

    payload = preflight_environment.collect_preflight(check_pip_index=False)
    json_path = tmp_path / "preflight.json"
    markdown_path = tmp_path / "preflight.md"
    preflight_environment.write_outputs(payload, json_path, markdown_path)

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert saved["status"] == "PASS"
    assert saved["triton"]["import"]["version"] == "test-triton"
    assert "HybridKernel Local Preflight" in markdown
    assert "install possible from current index" in markdown
