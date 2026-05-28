import argparse, csv
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument("--model", required=True); p.add_argument("--method_a", required=True); p.add_argument("--method_b", required=True); a=p.parse_args()
root=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((root/"tables/per_trace_recovery.csv").open()))
for method in [a.method_a, a.method_b]:
    vals=[float(r["recovery_unclipped"]) for r in rows if a.model.lower() in r["model"].lower() and method.lower() in r["method"].lower() and r["recovery_unclipped"]]
    print(method, vals)
