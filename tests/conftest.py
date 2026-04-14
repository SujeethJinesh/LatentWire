from __future__ import annotations

import importlib
import importlib.machinery
import pathlib
import types
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The scripts currently import the library as `rotalign`, while the checked-in
# package directory is `latent_bridge`. Expose the same module under both names
# so the test suite can exercise script imports without downloading anything.
latent_bridge = importlib.import_module("latent_bridge")
sys.modules.setdefault("rotalign", latent_bridge)

# `latent_bridge.evaluate` and `latent_bridge.calibrate` import `transformers`,
# and the installed Pillow wheel is not usable on this machine. Provide a tiny
# import-time stub so the modules stay importable for unit tests that only
# exercise parsing and helper logic.
pil_mod = types.ModuleType("PIL")
pil_mod.__path__ = []  # type: ignore[attr-defined]
pil_mod.__version__ = "0.0"
pil_mod.__spec__ = importlib.machinery.ModuleSpec("PIL", loader=None, is_package=True)
image_mod = types.ModuleType("PIL.Image")
image_mod.__spec__ = importlib.machinery.ModuleSpec("PIL.Image", loader=None)


class _StubImage:
    pass


image_mod.Image = _StubImage
pil_mod.Image = image_mod
sys.modules.setdefault("PIL", pil_mod)
sys.modules.setdefault("PIL.Image", image_mod)

transformers_mod = types.ModuleType("transformers")
transformers_mod.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)


class _UnavailableAutoClass:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise RuntimeError("transformers is stubbed in tests")


transformers_mod.AutoModelForCausalLM = _UnavailableAutoClass
transformers_mod.AutoTokenizer = _UnavailableAutoClass
sys.modules.setdefault("transformers", transformers_mod)

for submodule in ["rotation", "procrustes", "quantize", "translator", "evaluate", "calibrate", "ablation_sweep"]:
    module = importlib.import_module(f"latent_bridge.{submodule}")
    sys.modules.setdefault(f"rotalign.{submodule}", module)
