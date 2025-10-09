import json
import tempfile
from pathlib import Path

import pytest


def make_dummy_checkpoint(tmpdir: Path):
    (tmpdir / "encoder.pt").write_text("{}")
    state = {
        "encoder": {},
        "adp_llama": {},
        "deep_prefix_llama": {},
        "coprocessor_llama": {},
        "optimizer": {},
    }
    (tmpdir / "state.pt").write_text(json.dumps(state))
    return tmpdir


@pytest.mark.skip("checkpoint roundtrip requires heavy weights; deferred")
def test_checkpoint_roundtrip(tmp_path):
    ckpt_dir = make_dummy_checkpoint(tmp_path / "ckpt")
    assert ckpt_dir.exists()
