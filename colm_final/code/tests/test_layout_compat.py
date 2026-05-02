from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rotalign_public_api_reexports():
    import rotalign
    import rotalign.procrustes as rotalign_procrustes
    import rotalign.quantize as rotalign_quantize
    import rotalign.rotation as rotalign_rotation
    import rotalign.translator as rotalign_translator

    from latent_bridge import (
        RotAlignKVTranslator,
        TranslatorConfig,
        random_orthogonal,
    )

    assert rotalign.RotAlignKVTranslator is RotAlignKVTranslator
    assert rotalign.TranslatorConfig is TranslatorConfig
    assert rotalign.random_orthogonal is random_orthogonal
    assert rotalign_rotation.random_orthogonal is random_orthogonal
    assert hasattr(rotalign_procrustes, "fit_alignment")
    assert hasattr(rotalign_quantize, "GaussianQuantizer")
    assert hasattr(rotalign_translator, "RotAlignKVTranslator")


def _run_wrapper_with_fake_latent_bridge(module_name: str) -> bool:
    called = {"value": False}

    def fake_main() -> None:
        called["value"] = True

    fake_pkg = types.ModuleType("latent_bridge")
    fake_pkg.__path__ = []  # mark as package
    fake_mod = types.ModuleType(f"latent_bridge.{module_name}")
    fake_mod.main = fake_main

    original = {name: sys.modules.get(name) for name in ["latent_bridge", f"latent_bridge.{module_name}"]}
    try:
        sys.modules["latent_bridge"] = fake_pkg
        sys.modules[f"latent_bridge.{module_name}"] = fake_mod
        runpy.run_path(str(ROOT / "scripts" / f"{module_name}.py"), run_name="__main__")
    finally:
        for name, mod in original.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
    return called["value"]


def test_script_wrappers_call_module_mains():
    for module_name in ["demo", "calibrate", "evaluate", "ablation_sweep"]:
        assert _run_wrapper_with_fake_latent_bridge(module_name)
