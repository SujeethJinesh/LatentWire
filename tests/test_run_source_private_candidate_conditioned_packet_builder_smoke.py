from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts import run_source_private_candidate_conditioned_packet_builder_smoke as smoke
from scripts import run_source_private_learned_synonym_dictionary_packet_gate as syn


class FakeDictionary:
    receiver_mode = "atom_ridge"

    def predict_vector(self, text: str, *, apply_top_k: bool = True) -> np.ndarray:
        vector = np.zeros(len(syn.ATOM_ORDER), dtype=np.float64)
        lowered = text.lower()
        if "empty" in lowered:
            vector[syn.ATOM_TO_ID["empty"]] = 1.0
            vector[syn.ATOM_TO_ID["guard"]] = 0.7
        if "missing" in lowered:
            vector[syn.ATOM_TO_ID["missing"]] = 1.0
            vector[syn.ATOM_TO_ID["key"]] = 0.7
        if "round" in lowered:
            vector[syn.ATOM_TO_ID["round"]] = 1.0
            vector[syn.ATOM_TO_ID["half_up"]] = 0.7
        return vector


def _example(example_id: str, log: str, answer_intent: str):
    candidates = (
        SimpleNamespace(label=f"{example_id}_answer", patch_intent=answer_intent, prior_score=0.9),
        SimpleNamespace(label=f"{example_id}_decoy_1", patch_intent="string coercion", prior_score=0.3),
        SimpleNamespace(label=f"{example_id}_decoy_2", patch_intent="sort before reading", prior_score=0.2),
        SimpleNamespace(label=f"{example_id}_decoy_3", patch_intent="uppercase only", prior_score=0.1),
    )
    return SimpleNamespace(
        example_id=example_id,
        family_name="toy",
        private_test_log=log,
        candidates=candidates,
        answer_label=f"{example_id}_answer",
    )


def _log(*, actual: str, hidden_input: str = "[]", expected: str = "0") -> str:
    return "\n".join(
        [
            "pytest session starts",
            f"hidden_input={hidden_input}",
            f"expected={expected}",
            f"actual={actual!r}",
            "failure_status=EXCEPTION",
            "private_tool_trace: REPAIR_DIAG=A0",
        ]
    )


def test_fit_packet_builder_learns_source_to_candidate_atoms() -> None:
    examples = [
        _example("empty", _log(actual="IndexError: list index out of range"), "empty-list guard"),
        _example("missing", _log(actual="KeyError: status", hidden_input="{}", expected="unknown"), "missing-key default"),
        _example("round", _log(actual="2", hidden_input="2.6", expected="3"), "half-up rounding"),
    ]
    builder = smoke._fit_packet_builder(
        examples=examples,
        dictionary=FakeDictionary(),
        candidate_atom_view="native",
        ridge=0.01,
    )
    payload, metadata = smoke._build_packet(
        source_atoms=syn._source_private_atoms(examples[0].private_test_log, mode="matched"),
        builder=builder,
        budget_bytes=4,
        packet_min_score=0.0,
    )
    decoded = syn._decode_payload_atoms(payload, budget_bytes=4)

    assert metadata["packet_builder"] == "source_to_candidate_ridge"
    assert metadata["packet_atom_count"] > 0
    assert "empty" in decoded or "guard" in decoded
    assert builder["fit_rows"] == 3


def test_empty_source_builds_empty_packet() -> None:
    builder = {"weights": np.ones((len(syn.ATOM_ORDER) + 1, len(syn.ATOM_ORDER)), dtype=np.float64)}
    payload, metadata = smoke._build_packet(
        source_atoms={},
        builder=builder,
        budget_bytes=8,
        packet_min_score=0.0,
    )
    assert payload == b""
    assert metadata["packet_atom_count"] == 0
    assert metadata["packet_vector_l2"] == 0.0


def test_project_mapped_packet_selects_nearest_public_candidate() -> None:
    example = _example(
        "empty",
        _log(actual="IndexError: list index out of range"),
        "empty-list guard",
    )
    weights = np.zeros((len(syn.ATOM_ORDER) + 1, len(syn.ATOM_ORDER)), dtype=np.float64)
    weights[1 + syn.ATOM_TO_ID["empty"], syn.ATOM_TO_ID["empty"]] = 1.0
    weights[1 + syn.ATOM_TO_ID["empty"], syn.ATOM_TO_ID["guard"]] = 0.7
    builder = {"weights": weights}

    payload, metadata = smoke._build_packet(
        source_atoms={"empty": 1.0},
        builder=builder,
        budget_bytes=4,
        packet_min_score=0.0,
        composition="project_mapped",
        example=example,
        dictionary=FakeDictionary(),
        candidate_atom_view="native",
    )
    decoded = syn._decode_payload_atoms(payload, budget_bytes=4)

    assert metadata["packet_projection"] == "nearest_public_candidate"
    assert metadata["packet_projection_selected_index"] == 0
    assert metadata["packet_projection_selected_is_answer"] is True
    assert "empty" in decoded


def test_train_donor_antishuffle_uses_builder_donor_pool() -> None:
    example = _example(
        "empty",
        _log(actual="IndexError: list index out of range"),
        "empty-list guard",
    )
    donor = _example(
        "missing",
        _log(actual="KeyError: status", hidden_input="{}", expected="unknown"),
        "missing-key default",
    )
    weights = np.zeros((len(syn.ATOM_ORDER) + 1, len(syn.ATOM_ORDER)), dtype=np.float64)
    weights[1 + syn.ATOM_TO_ID["empty"], syn.ATOM_TO_ID["empty"]] = 1.0
    weights[1 + syn.ATOM_TO_ID["empty"], syn.ATOM_TO_ID["guard"]] = 0.7
    builder = {
        "weights": weights,
        "train_mean_source_vector": [0.0 for _ in syn.ATOM_ORDER],
        "train_mean_prediction_vector": [0.0 for _ in syn.ATOM_ORDER],
        "train_donor_source_vectors": [
            smoke._source_vector(donor, mode="matched").tolist(),
        ],
        "train_donor_mapped_vectors": [
            np.zeros(len(syn.ATOM_ORDER), dtype=np.float64).tolist(),
        ],
        "train_donor_example_ids": [donor.example_id],
        "train_donor_answer_indices": [1],
    }

    payload, metadata = smoke._build_packet(
        source_atoms={"empty": 1.0, "guard": 0.7},
        builder=builder,
        budget_bytes=6,
        packet_min_score=0.0,
        composition="train_donor_antishuffle_innovation",
        source_identity_weight=0.75,
        antishuffle_train_donors=1,
        antishuffle_donor_weight=0.50,
        antishuffle_null_weight=0.75,
        antishuffle_generic_weight=0.10,
        antishuffle_carrier_mode="sum",
        example=example,
        eval_examples=[example, donor],
        index=0,
        dictionary=FakeDictionary(),
        null_dictionary=FakeDictionary(),
        candidate_atom_view="native",
        decoder_score_mode="candidate_local_residual_norm",
    )
    decoded = syn._decode_payload_atoms(payload, budget_bytes=6)

    assert metadata["packet_builder"] == "antishuffle_innovation"
    assert metadata["packet_builder_composition"] == "train_donor_antishuffle_innovation"
    assert metadata["contrast_source"].startswith("train_donor_mean:")
    assert metadata["antishuffle_train_donors"] == 1
    assert decoded


def test_leave_one_family_out_builder_bundle_excludes_eval_family() -> None:
    examples = [
        _example("fam_a_0", _log(actual="IndexError: list index out of range"), "empty-list guard"),
        _example("fam_b_0", _log(actual="KeyError: status", hidden_input="{}", expected="unknown"), "missing-key default"),
        _example("fam_c_0", _log(actual="2", hidden_input="2.6", expected="3"), "half-up rounding"),
    ]
    for example, family_name in zip(examples, ("fam_a", "fam_b", "fam_c"), strict=True):
        example.family_name = family_name

    bundle = {
        "mode": "leave_one_family_out_public",
        "builders": {
            example.family_name: smoke._fit_packet_builder(
                examples=[other for other in examples if other.family_name != example.family_name],
                dictionary=FakeDictionary(),
                candidate_atom_view="native",
                ridge=0.01,
            )
            for example in examples
        },
        "family_counts": {example.family_name: 2 for example in examples},
        "public_pool_examples": 3,
        "global_rows": [],
    }
    state = smoke._builder_state_for_json(bundle)

    assert smoke._builder_for_example(bundle, examples[0]) is bundle["builders"]["fam_a"]
    assert state["family_builder_count"] == 3
    assert state["fit_rows_min"] == 2
    assert state["fit_rows_max"] == 2


def _row(condition: str, example_id: str, correct: bool) -> dict[str, object]:
    return {
        "condition": condition,
        "example_id": example_id,
        "family_name": "toy",
        "budget_bytes": 8,
        "answer": "a",
        "prediction": "a" if correct else "b",
        "correct": correct,
        "strict_correct": correct,
        "payload_bytes": 8,
        "payload_tokens": 1,
        "latency_ms": 0.01,
        "metadata": {},
    }


def test_direction_summary_requires_base_improvement_and_clean_controls() -> None:
    rows: list[dict[str, object]] = []
    for idx in range(8):
        rows.append(_row("target_only", f"ex_{idx}", idx < 2))
        rows.append(_row(smoke.BASE_MATCHED_CONDITION, f"ex_{idx}", idx < 4))
        rows.append(_row(smoke.MATCHED_CONDITION, f"ex_{idx}", idx < 6))
        for condition in smoke.STRICT_CONTROLS:
            rows.append(_row(condition, f"ex_{idx}", idx < 2))
        rows.append(_row(smoke.ORACLE_CONDITION, f"ex_{idx}", True))

    summary = smoke._direction_summary(
        rows,  # type: ignore[arg-type]
        direction="toy",
        budget_bytes=8,
        seed=1,
        min_lift_over_target=0.15,
        min_gap_over_control=0.10,
        min_improvement_over_base=0.03,
        bootstrap_samples=50,
    )

    assert summary["target_accuracy"] == 0.25
    assert summary["base_matched_accuracy"] == 0.5
    assert summary["candidate_conditioned_packet_accuracy"] == 0.75
    assert summary["controls_ok"] is True
    assert summary["pass_gate"] is True
