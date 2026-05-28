import argparse, csv
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument("--trace_id", required=True); a=p.parse_args()
root=Path(__file__).resolve().parents[1]
with (root/"tables/per_trace_recovery.csv").open() as f:
    for r in csv.DictReader(f):
        if str(r.get("trace_id")) == str(a.trace_id): print(r)
