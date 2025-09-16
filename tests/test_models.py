import pytest

try:
    import torch
except OSError as err:
    pytest.skip(f"Skipping Torch-dependent tests due to import error: {err}", allow_module_level=True)

from latentwire.models import InterlinguaEncoder


def test_interlingua_encoder_shared_only_cpu():
    encoder = InterlinguaEncoder(d_z=32, latent_shared_len=6, latent_private_len=0, model_keys=("llama", "qwen"))
    byte_inputs = torch.randint(0, 255, (3, 10), dtype=torch.long)
    out = encoder(byte_inputs)
    assert out.shape == (3, 6, 32)

    components = encoder(byte_inputs, return_components=True)
    assert components["shared"].shape == (3, 6, 32)
    assert components["private"]["llama"].numel() == 0
    assert components["private"]["qwen"].numel() == 0


def test_interlingua_encoder_shared_and_private_cpu():
    encoder = InterlinguaEncoder(
        d_z=16,
        latent_shared_len=4,
        latent_private_len=2,
        model_keys=("llama", "qwen"),
    )
    byte_inputs = torch.randint(0, 255, (2, 12), dtype=torch.long)

    components = encoder(byte_inputs, return_components=True)
    assert components["shared"].shape == (2, 4, 16)
    assert components["private"]["llama"].shape == (2, 2, 16)

    concatenated = encoder(byte_inputs)
    assert concatenated.shape == (2, 8, 16)
    reconstructed = torch.cat(
        [components["shared"], components["private"]["llama"], components["private"]["qwen"]], dim=1
    )
    torch.testing.assert_close(concatenated, reconstructed)

