from __future__ import annotations

from experimental.thoughtflow_fp8.phase2.simulate_phase_retention import _run


def test_thoughtflow_improves_phase_recall_over_proxy_policies() -> None:
    summary = _run()["summary"]
    thoughtflow_phase = summary["thoughtflow"]["phase_recall"]
    assert thoughtflow_phase > summary["longflow_like"]["phase_recall"]
    assert thoughtflow_phase > summary["thin_kv_like"]["phase_recall"]
    assert thoughtflow_phase > summary["rkv_like"]["phase_recall"]
