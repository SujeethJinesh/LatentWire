from __future__ import annotations

from scripts import audit_source_surface_answer_masking as audit


def test_source_answer_values_include_final_and_verified() -> None:
    profile = {"final": "7", "verified": {"9", "11"}}

    assert audit._source_answer_values(profile) == {"7", "9", "11"}


def test_clean_ids_prefers_clean_source_only() -> None:
    surface = type(
        "Surface",
        (),
        {
            "target_set": {
                "ids": {
                    "clean_source_only": ["a"],
                    "clean_residual_targets": ["b"],
                    "source_only": ["c"],
                }
            }
        },
    )()

    key, ids = audit._clean_ids(surface)  # type: ignore[arg-type]

    assert key == "clean_source_only"
    assert ids == {"a"}
