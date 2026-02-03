import unittest
from quantization.utils.budget import compute_slack


class TestBudgetCap(unittest.TestCase):
    def test_slack(self):
        self.assertEqual(compute_slack(100, 50), 50.0)
        self.assertEqual(compute_slack(100, 150), 0.0)
        self.assertEqual(compute_slack(None, 10), 0.0)


if __name__ == "__main__":
    unittest.main()
