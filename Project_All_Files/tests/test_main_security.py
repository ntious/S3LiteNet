from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("s3litenet_main", PROJECT_ROOT / "main.py")
MAIN = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MAIN)


class ModelPathSecurityTests(unittest.TestCase):
    def test_model_path_is_anchored_to_repository(self):
        expected = (PROJECT_ROOT / "models" / "cnn_pruned.py").resolve()
        self.assertEqual(MAIN.resolve_model_path("cnn"), expected)

    def test_attacker_controlled_working_directory_is_ignored(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary_directory:
            malicious_model = Path(temporary_directory) / "models" / "cnn_pruned.py"
            malicious_model.parent.mkdir()
            malicious_model.write_text("raise RuntimeError('executed attacker file')\n", encoding="utf-8")
            try:
                os.chdir(temporary_directory)
                resolved = MAIN.resolve_model_path("cnn")
            finally:
                os.chdir(original_cwd)

        self.assertNotEqual(resolved, malicious_model.resolve())
        self.assertEqual(resolved.parent, (PROJECT_ROOT / "models").resolve())

    def test_unknown_model_is_rejected(self):
        with self.assertRaises(ValueError):
            MAIN.resolve_model_path("../../attacker")


if __name__ == "__main__":
    unittest.main()
