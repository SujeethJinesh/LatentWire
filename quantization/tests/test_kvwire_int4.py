import unittest
import numpy as np

from quantization.kvwire.int4 import pack_int4, unpack_int4, int4_byte_length


class TestInt4Packing(unittest.TestCase):
    def test_pack_unpack_roundtrip(self):
        values = np.array([-8, -7, -1, 0, 1, 7, 3, -4, 2], dtype=np.int8)
        blob = pack_int4(values)
        out = unpack_int4(blob, values.shape)
        np.testing.assert_array_equal(out, values)

    def test_length(self):
        for n in range(1, 10):
            values = np.zeros((n,), dtype=np.int8)
            blob = pack_int4(values)
            self.assertEqual(len(blob), int4_byte_length(n))

    def test_empty(self):
        values = np.zeros((0,), dtype=np.int8)
        blob = pack_int4(values)
        self.assertEqual(blob, b"")
        out = unpack_int4(blob, values.shape)
        np.testing.assert_array_equal(out, values)


if __name__ == "__main__":
    unittest.main()
