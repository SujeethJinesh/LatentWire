#!/usr/bin/env python3
"""Create secret-free review SQLite slices from raw Nsight Systems exports."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from no_boundary_reduction import ROWS


RUN_DIR = Path(__file__).resolve().parents[1]


def client_window(run_id: str) -> tuple[int, int]:
    payload = json.loads((RUN_DIR / "logs" / f"client_{run_id}.log").read_text(encoding="utf-8"))
    requests = payload["requests"]
    return (
        min(int(row["start_epoch_ns"]) for row in requests),
        max(int(row["end_epoch_ns"]) for row in requests),
    )


def session_start(conn: sqlite3.Connection) -> int:
    return int(conn.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0])


def copy_review_slice(run_id: str) -> None:
    source = RUN_DIR / "nsys" / f"{run_id}.sqlite"
    dest = RUN_DIR / "nsys" / f"{run_id}.sanitized.sqlite"
    if dest.exists():
        dest.unlink()

    start_epoch_ns, end_epoch_ns = client_window(run_id)
    with sqlite3.connect(source) as src:
        start_rel_ns = start_epoch_ns - session_start(src)
        end_rel_ns = end_epoch_ns - session_start(src)

    with sqlite3.connect(dest) as out:
        out.execute("attach database ? as src", (str(source),))
        out.execute(
            "create table TARGET_INFO_SESSION_START_TIME as "
            "select * from src.TARGET_INFO_SESSION_START_TIME"
        )
        out.execute(
            """
            create table KERNEL_SUMMARY as
            select coalesce(s.value, printf('kernel_id:%d', k.demangledName)) as name,
                   count(*) as launches,
                   sum(k.end - k.start) as total_ns,
                   min(k.start) as min_start_ns,
                   max(k.end) as max_end_ns
            from src.CUPTI_ACTIVITY_KIND_KERNEL k
            left join src.StringIds s on s.id = k.demangledName
            where k.start >= ? and k.end <= ?
            group by name
            order by total_ns desc
            """,
            (start_rel_ns, end_rel_ns),
        )
        out.execute(
            """
            create table CUPTI_ACTIVITY_KIND_KERNEL_SAMPLE as
            select *
            from src.CUPTI_ACTIVITY_KIND_KERNEL
            where start >= ? and end <= ?
            order by (end - start) desc
            limit 512
            """,
            (start_rel_ns, end_rel_ns),
        )
        out.execute(
            "create table NVTX_EVENTS as "
            "select * from src.NVTX_EVENTS where start <= ? and coalesce(end, start) >= ?",
            (end_rel_ns, start_rel_ns),
        )
        string_ids: set[int] = set()
        for (value,) in out.execute("select demangledName from CUPTI_ACTIVITY_KIND_KERNEL_SAMPLE"):
            if value is not None:
                string_ids.add(int(value))
        for (value,) in out.execute("select shortName from CUPTI_ACTIVITY_KIND_KERNEL_SAMPLE"):
            if value is not None:
                string_ids.add(int(value))
        for (value,) in out.execute("select mangledName from CUPTI_ACTIVITY_KIND_KERNEL_SAMPLE"):
            if value is not None:
                string_ids.add(int(value))
        for (value,) in out.execute("select textId from NVTX_EVENTS"):
            if value is not None:
                string_ids.add(int(value))
        if string_ids:
            placeholders = ",".join("?" for _ in string_ids)
            out.execute(
                f"create table StringIds as select * from src.StringIds where id in ({placeholders})",
                tuple(sorted(string_ids)),
            )
        else:
            out.execute("create table StringIds as select * from src.StringIds where 0")
        out.execute(
            "create table SANITIZATION_NOTE (key text, value text)"
        )
        out.executemany(
            "insert into SANITIZATION_NOTE values (?, ?)",
            [
                ("source", f"nsys/{run_id}.sqlite"),
                ("method", "request-window slice; environment tables intentionally omitted"),
                ("start_epoch_ns", str(start_epoch_ns)),
                ("end_epoch_ns", str(end_epoch_ns)),
            ],
        )
        out.commit()


def main() -> None:
    for row in ROWS:
        copy_review_slice(str(row["run_id"]))


if __name__ == "__main__":
    main()
