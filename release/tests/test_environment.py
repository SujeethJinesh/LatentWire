from pathlib import Path

from outlier_migrate.environment import collect_environment


def test_collect_environment_validates_configs() -> None:
    report = collect_environment((Path("configs/granite_tiny.yaml"),))
    assert report.python_version
    assert "m11b" in report.registered_methods
    assert report.config_checks[0].ok
    assert report.config_checks[0].model_id == "ibm-granite/granite-4.0-h-tiny"


def test_environment_report_is_serializable_mapping() -> None:
    report = collect_environment(())
    payload = report.to_dict()
    assert payload["registered_methods"]
    assert payload["config_checks"] == ()
