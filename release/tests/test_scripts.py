import json

from scripts.verify_environment import main as verify_environment_main


def test_verify_environment_dry_run_writes_manifest(tmp_path) -> None:
    exit_code = verify_environment_main(["--dry-run", "--output-dir", str(tmp_path)])
    manifest_path = tmp_path / "environment_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert "m11b" in payload["registered_methods"]
    assert len(payload["config_checks"]) == 4
