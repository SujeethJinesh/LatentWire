from pathlib import Path
import csv
root = Path(__file__).resolve().parents[1]
with (root / "tables/experiment_summary.csv").open() as f:
    rows = list(csv.DictReader(f))
print(f"experiments={len(rows)}")
for r in rows:
    print(f"{r['experiment_id']}: {r['status']} {r['method']} {r['model']} median={r['median_recovery']}")
