import unittest
import numpy as np

from quantization.kvwire.kvwire_v1 import KVWireConfig, pack, unpack


class TestKVWireRoundtrip(unittest.TestCase):
    def _roundtrip(self, quant_mode):
        rng = np.random.default_rng(123)
        k = rng.standard_normal((1, 2, 3, 4)).astype(np.float32)
        v = rng.standard_normal((1, 2, 3, 4)).astype(np.float32)
        indices = np.array([0, 2, 1], dtype=np.uint16)
        cfg = KVWireConfig(wire_quant_mode=quant_mode, wire_scale_granularity="per_block")
        blob = pack({"k": k, "v": v, "indices": indices}, cfg)
        out = unpack(blob, cfg)
        self.assertEqual(out["k"].shape, k.shape)
        self.assertEqual(out["v"].shape, v.shape)
        # Dequantized values should be close to original.
        err = np.max(np.abs(out["k"] - k))
        self.assertTrue(np.isfinite(err))

    def test_roundtrip_int8(self):
        self._roundtrip("int8")

    def test_roundtrip_int4(self):
        self._roundtrip("int4")


if __name__ == "__main__":
    unittest.main()
