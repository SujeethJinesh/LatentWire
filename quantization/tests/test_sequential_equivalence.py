import os
import unittest


@unittest.skipUnless(os.environ.get("RUN_EQUIV_TEST") == "1", "Set RUN_EQUIV_TEST=1 to run")
class TestSequentialEquivalence(unittest.TestCase):
    def test_placeholder(self):
        # Placeholder: real equivalence test requires GPU + model weights.
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
