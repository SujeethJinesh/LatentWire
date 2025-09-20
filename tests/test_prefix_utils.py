import pytest

try:
    import torch
except OSError as err:
    pytest.skip(f"Skipping Torch-dependent tests due to import error: {err}", allow_module_level=True)

from latentwire.core_utils import combine_latents


def test_combine_latents_shared_and_private():
    shared = torch.randn(5, 3, 8)
    private = {
        "llama": torch.randn(5, 2, 8),
        "qwen": torch.randn(5, 2, 8),
    }
    combined_llama = combine_latents({"shared": shared, "private": private}, "llama")
    combined_qwen = combine_latents({"shared": shared, "private": private}, "qwen")
    torch.testing.assert_close(combined_llama[:, :3], shared)
    torch.testing.assert_close(combined_llama[:, 3:], private["llama"])
    torch.testing.assert_close(combined_qwen[:, 3:], private["qwen"])


def test_combine_latents_raises_for_missing_model():
    shared = torch.zeros(1, 0, 4)
    private = {}
    with pytest.raises(KeyError):
        combine_latents({"shared": shared, "private": private}, "llama")
