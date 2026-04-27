#!/usr/bin/env python3
"""Check whether the known orphaned MPS blocker process is still present."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class ProcessStatus:
    pid: int
    ppid: int
    stat: str
    elapsed: str
    command: str


def _parse_ps_line(line: str) -> ProcessStatus:
    parts = line.strip().split(None, 4)
    if len(parts) < 5:
        raise ValueError(f"Could not parse ps output line: {line!r}")
    return ProcessStatus(
        pid=int(parts[0]),
        ppid=int(parts[1]),
        stat=parts[2],
        elapsed=parts[3],
        command=parts[4],
    )


def _query_process(pid: int) -> ProcessStatus | None:
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "pid=,ppid=,stat=,etime=,command="],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return _parse_ps_line(result.stdout.strip().splitlines()[-1])


def _is_blocked(status: ProcessStatus | None, *, require_substring: str | None) -> bool:
    if status is None:
        return False
    if require_substring is None:
        return True
    return require_substring in status.command


def check(pid: int, *, require_substring: str | None) -> dict[str, object]:
    status = _query_process(pid)
    blocked = _is_blocked(status, require_substring=require_substring)
    return {
        "pid": pid,
        "present": status is not None,
        "blocked": blocked,
        "require_substring": require_substring,
        "process": asdict(status) if status is not None else None,
        "next_action": (
            "use_cpu_only_or_clear_os_session"
            if blocked
            else "mps_preflight_clear"
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, default=31103)
    parser.add_argument(
        "--require-substring",
        default=None,
        help="Only block when the present process command contains this text. Defaults to blocking whenever PID is present.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--require-clear", action="store_true", help="Exit nonzero when blocked.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    payload = check(int(args.pid), require_substring=args.require_substring)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        process = payload["process"]
        if process is None:
            print(f"PID {payload['pid']} absent; MPS preflight clear.")
        elif payload["blocked"]:
            print(f"PID {payload['pid']} present; use CPU-only work or clear OS/session before MPS.")
            print(f"command: {process['command']}")
        else:
            print(f"PID {payload['pid']} present but did not match blocker substring; MPS preflight clear.")
    if args.require_clear and payload["blocked"]:
        raise SystemExit(2)
    return payload


if __name__ == "__main__":
    main()
