"""Repo-root entrypoint wrapper for RotAlign-KV evaluation."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from latent_bridge.evaluate import main


if __name__ == "__main__":
    main()
