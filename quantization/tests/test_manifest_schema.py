import unittest
from quantization.utils.manifest_schema import validate_manifest, REQUIRED_FIELDS


class TestManifestSchema(unittest.TestCase):
    def test_required_fields_present(self):
        manifest = {k: None for k in REQUIRED_FIELDS}
        missing = validate_manifest(manifest)
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
