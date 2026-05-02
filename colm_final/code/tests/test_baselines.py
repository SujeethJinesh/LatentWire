from __future__ import annotations

import pytest

from latent_bridge.baselines import (
    BaselineContext,
    C2CAdapter,
    KVCommAdapter,
    LatentMASAdapter,
    PublishedBaselineArtifact,
    available_baselines,
    get_baseline,
)


def test_baseline_registry_exposes_expected_placeholders() -> None:
    assert available_baselines() == ["c2c", "kvcomm", "latentmas"]
    assert get_baseline("c2c") is C2CAdapter
    assert get_baseline("kvcomm") is KVCommAdapter
    assert get_baseline("latentmas") is LatentMASAdapter


def test_unknown_baseline_raises_key_error() -> None:
    with pytest.raises(KeyError):
        get_baseline("missing")


def test_placeholder_adapters_fail_explicitly() -> None:
    context = BaselineContext(
        source_model="src",
        target_model="tgt",
        device="cpu",
        dtype="float32",
    )

    c2c = C2CAdapter()
    kvcomm = KVCommAdapter()
    latentmas = LatentMASAdapter()
    c2c.fit(context)
    kvcomm.fit(context)
    latentmas.fit(context)

    with pytest.raises(NotImplementedError):
        c2c.evaluate_generation()
    with pytest.raises(NotImplementedError):
        kvcomm.evaluate_mcq()
    with pytest.raises(NotImplementedError):
        latentmas.evaluate_generation()


def test_c2c_published_artifact_matches_qwen_pair() -> None:
    artifact = C2CAdapter.published_artifact(
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen3-0.6B",
    )

    assert isinstance(artifact, PublishedBaselineArtifact)
    assert artifact.repo_id == "nics-efc/C2C_Fuser"
    assert artifact.subdir == "qwen3_0.6b+qwen2.5_0.5b_Fuser"
    assert artifact.config_path.endswith("/config.json")
    assert artifact.checkpoint_dir.endswith("/final")


def test_c2c_local_paths_resolve_from_download_root() -> None:
    artifact = C2CAdapter.published_artifact(
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen3-0.6B",
        local_root="/tmp/c2c",
    )

    assert C2CAdapter.local_config_path(artifact) == "/tmp/c2c/qwen3_0.6b+qwen2.5_0.5b_Fuser/config.json"
    assert C2CAdapter.local_checkpoint_dir(artifact) == "/tmp/c2c/qwen3_0.6b+qwen2.5_0.5b_Fuser/final"
