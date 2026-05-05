from __future__ import annotations

from experimental.thoughtflow_fp8.phase2.policy_sweep import SweepConfig, _configs, _make_policy
from experimental.thoughtflow_fp8.phase2.simulate_phase_retention import Token


def test_configs_have_stable_names() -> None:
    configs = _configs()

    assert configs
    assert all(config.name.startswith("tf_sweep_") for config in configs)


def test_sweep_policy_respects_budget_and_keeps_recent() -> None:
    config = SweepConfig(recent_fraction=0.5, phase_bonus=0.1, math_bonus=0.3, protect_anchors=2)
    policy = _make_policy(config)
    trace = [
        Token("a", "anchor", 1.0),
        Token("b", "anchor", 0.9),
        Token("phase", "phase", 0.4),
        Token("math", "math_state", 0.7),
        Token("low", "reason", 0.1),
        Token("recent1", "reason", 0.1),
        Token("recent2", "reason", 0.1),
    ]

    kept = policy(trace, budget=5)

    assert len(kept) == 5
    assert {0, 1}.issubset(kept)
    assert {5, 6}.issubset(kept)


def test_sweep_policy_never_exceeds_budget_when_anchors_and_recent_overlap_pressure() -> None:
    config = SweepConfig(recent_fraction=0.8, phase_bonus=0.1, math_bonus=0.3, protect_anchors=4)
    policy = _make_policy(config)
    trace = [
        Token(f"anchor{i}", "anchor", 1.0)
        for i in range(4)
    ] + [
        Token(f"recent{i}", "reason", 0.1)
        for i in range(6)
    ]

    kept = policy(trace, budget=3)

    assert len(kept) == 3
