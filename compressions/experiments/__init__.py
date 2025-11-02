"""Individual compression experiment modules.

Future home for modular experiment wrappers.
Currently experiments are run via the unified compression_ablations.py script.
"""

# Import from parent experimental directory
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "experimental" / "learning"))

from compression_ablations import run_single_ablation, run_all_ablations

__all__ = ['run_single_ablation', 'run_all_ablations']
