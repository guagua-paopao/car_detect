import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "retry_guarded_gdrive_download.py"
SPEC = importlib.util.spec_from_file_location("retry_guarded_gdrive_download", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RetryGuardedDownloadTest(unittest.TestCase):
    def test_exponential_delay_is_capped(self):
        self.assertEqual(MODULE.retry_delay(1, 30, 300), 30)
        self.assertEqual(MODULE.retry_delay(2, 30, 300), 60)
        self.assertEqual(MODULE.retry_delay(8, 30, 300), 300)

    def test_delay_never_exceeds_maximum(self):
        self.assertEqual(MODULE.retry_delay(1, 500, 300), 300)


if __name__ == "__main__":
    unittest.main()
