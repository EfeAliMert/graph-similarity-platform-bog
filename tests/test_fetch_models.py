from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.fetch_models import (
    INSTALL_MARKER,
    ModelInstallError,
    install_model,
    verify_model,
)


class ModelSourceInstallTests(unittest.TestCase):
    def setUp(self):
        if shutil.which("git") is None:
            self.skipTest("git is not installed")

    def test_clone_patch_verify_and_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream = root / "upstream"
            upstream.mkdir()
            self.run_git(upstream, "init")
            self.run_git(upstream, "config", "user.email", "test@example.com")
            self.run_git(upstream, "config", "user.name", "Test User")
            (upstream / "model.py").write_text("MODE = 'upstream'\n")
            self.run_git(upstream, "add", "model.py")
            self.run_git(upstream, "commit", "-m", "source")
            commit = self.run_git(upstream, "rev-parse", "HEAD").strip()

            patches = root / "patches"
            patches.mkdir()
            (patches / "compat.patch").write_text(
                "diff --git a/model.py b/model.py\n"
                "--- a/model.py\n"
                "+++ b/model.py\n"
                "@@ -1 +1 @@\n"
                "-MODE = 'upstream'\n"
                "+MODE = 'compatible'\n"
            )
            spec = {
                "id": "example",
                "name": "Example",
                "repository": str(upstream),
                "commit": commit,
                "path": "Models&Datasets/example",
                "patch": "patches/compat.patch",
                "required_files": ["model.py"],
                "content_checks": [
                    {"path": "model.py", "contains": "compatible"}
                ],
            }

            first = install_model(spec, root)
            second = install_model(spec, root)
            target = root / spec["path"]

            self.assertIn("downloaded", first)
            self.assertEqual(second, "already ready")
            self.assertIn("compatible", (target / "model.py").read_text())
            self.assertTrue((target / INSTALL_MARKER).is_file())
            verify_model(spec, root)

    def test_incompatible_non_git_folder_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "Models&Datasets/example"
            target.mkdir(parents=True)
            (target / "model.py").write_text("MODE = 'unknown'\n")
            spec = {
                "id": "example",
                "name": "Example",
                "repository": "https://example.invalid/model.git",
                "commit": "0" * 40,
                "path": "Models&Datasets/example",
                "patch": None,
                "required_files": ["model.py"],
                "content_checks": [
                    {"path": "model.py", "contains": "compatible"}
                ],
            }

            with self.assertRaisesRegex(ModelInstallError, "Move that folder aside"):
                install_model(spec, root)

            self.assertEqual((target / "model.py").read_text(), "MODE = 'unknown'\n")

    @staticmethod
    def run_git(path: Path, *args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout


if __name__ == "__main__":
    unittest.main()
