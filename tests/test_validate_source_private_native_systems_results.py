import csv

from scripts import validate_source_private_native_systems_results as gate


def test_metric_value_validation_handles_boolean_and_fraction() -> None:
    assert gate._validate_metric_value("source_kv_exposed", "bool", "true") is None
    assert gate._validate_metric_value("accuracy", "fraction", "0.75") is None
    assert gate._validate_metric_value("accuracy", "fraction", "1.5") == "accuracy must be in [0, 1]"
    assert gate._validate_metric_value("ttft_ms_p50", "ms", "-1") == "ttft_ms_p50 must be non-negative"


def test_measurement_errors_catch_exposure_mismatch() -> None:
    schema = [
        {"metric": "benchmark", "unit": "string", "required": "true"},
        {"metric": "accuracy", "unit": "fraction", "required": "true"},
        {"metric": "source_kv_exposed", "unit": "bool", "required": "true"},
    ]
    baseline_by_id = {
        "c2c_cache_to_cache": {
            "row_id": "c2c_cache_to_cache",
            "family": "cache_communication",
            "source_text_exposed": "false",
            "source_kv_exposed": "true",
        }
    }

    errors = gate._measurement_errors(
        row={
            "row_id": "c2c_cache_to_cache",
            "benchmark": "ARC-Challenge",
            "accuracy": "0.4",
            "source_text_exposed": "false",
            "source_kv_exposed": "false",
            "transferred_source_state_bytes": "0",
        },
        schema=schema,
        baseline_by_id=baseline_by_id,
    )

    assert "source_kv_exposed=False disagrees with baseline expectation True" in errors
    assert "source-state baseline must report transferred_source_state_bytes > 0" in errors


def test_validate_without_measurements_refuses_native_complete(tmp_path) -> None:
    schema_path = tmp_path / "schema.csv"
    baselines_path = tmp_path / "baselines.csv"
    with schema_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "unit", "required", "why"])
        writer.writeheader()
        for index in range(40):
            writer.writerow({"metric": f"metric_{index}", "unit": "string", "required": "true", "why": "test"})
    with baselines_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row_id",
                "family",
                "method",
                "source_url",
                "serving_substrate",
                "communicated_object",
                "source_private",
                "source_text_exposed",
                "source_kv_exposed",
                "required_for_native_gate",
                "claim_role",
            ],
        )
        writer.writeheader()
        for index in range(10):
            writer.writerow(
                {
                    "row_id": f"row_{index}",
                    "family": "serving_baseline",
                    "method": "test",
                    "source_url": "local",
                    "serving_substrate": "vLLM",
                    "communicated_object": "none",
                    "source_private": "true",
                    "source_text_exposed": "false",
                    "source_kv_exposed": "false",
                    "required_for_native_gate": "true",
                    "claim_role": "test",
                }
            )

    payload = gate.validate_native_systems_results(
        schema_path=schema_path,
        baseline_rows_path=baselines_path,
        measurement_inputs=(),
        output_dir=tmp_path / "out",
    )

    assert payload["validator_pass"] is True
    assert payload["native_systems_complete"] is False
    assert len(payload["missing_required_row_ids"]) == 10
