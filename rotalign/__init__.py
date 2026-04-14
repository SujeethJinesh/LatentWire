"""Compatibility shim for the flat `latent_bridge` source layout."""

from latent_bridge import *  # noqa: F401,F403
from latent_bridge import __all__ as _latent_bridge_all

__all__ = list(_latent_bridge_all)
