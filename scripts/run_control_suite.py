"""Repo-root entrypoint wrapper for the RotAlign control-suite harness."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from latent_bridge.control_suite import main


if __name__ == "__main__":
    main()
