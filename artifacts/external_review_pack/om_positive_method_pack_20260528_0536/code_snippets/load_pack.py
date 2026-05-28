"""Load top-level summary tables from an unpacked review pack."""
from pathlib import Path
import csv

def read_summary(root=Path('.')):
    with (root / 'tables/experiment_summary.csv').open() as f:
        return list(csv.DictReader(f))
