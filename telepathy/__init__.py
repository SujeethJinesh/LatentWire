# telepathy/__init__.py
"""
Latent Telepathy: Cross-model latent communication via neural adapters.

This module enables Llama 3.1 8B to inject its internal hidden states
directly into Mistral 0.3 7B, bypassing text-based communication.
"""
from latentwire.bridge import LatentBridge, PerceiverResampler

__all__ = ["LatentBridge", "PerceiverResampler"]
