from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from latent_bridge.kvcomm_eval import main


if __name__ == "__main__":
    main()
