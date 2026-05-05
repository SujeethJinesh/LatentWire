from experimental.hybridkernel.phase2.pre_gpu_threshold_model import _status


def test_threshold_status_alive_for_low_required_overhead() -> None:
    assert _status({"model": 0.08}).startswith("ALIVE")


def test_threshold_status_weakened_for_high_required_overhead() -> None:
    assert _status({"model": 0.40}).startswith("WEAKENED")
