from pathlib import Path
import csv

def read_experiment_summary(root=Path(__file__).resolve().parents[1]):
    with (root / "tables/experiment_summary.csv").open() as f:
        return list(csv.DictReader(f))
