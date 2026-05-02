from scripts import check_mps_blocker as blocker


def test_parse_ps_line_with_spaced_command():
    status = blocker._parse_ps_line(
        "31103     1 UE   12:06:25 /path/Python scripts/calibrate.py --device mps --seed 1"
    )

    assert status.pid == 31103
    assert status.ppid == 1
    assert status.stat == "UE"
    assert status.elapsed == "12:06:25"
    assert status.command == "/path/Python scripts/calibrate.py --device mps --seed 1"


def test_is_blocked_defaults_to_any_present_process():
    status = blocker.ProcessStatus(
        pid=31103,
        ppid=1,
        stat="UE",
        elapsed="00:01",
        command="python unrelated.py",
    )

    assert blocker._is_blocked(status, require_substring=None) is True
    assert blocker._is_blocked(None, require_substring=None) is False


def test_is_blocked_can_require_command_substring():
    status = blocker.ProcessStatus(
        pid=31103,
        ppid=1,
        stat="UE",
        elapsed="00:01",
        command="python scripts/calibrate.py --device mps",
    )

    assert blocker._is_blocked(status, require_substring="--device mps") is True
    assert blocker._is_blocked(status, require_substring="--device cpu") is False
