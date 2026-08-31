from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import zipfile

from scripts.fetch_checkpoints import CheckpointInstallError, install_checkpoints


class CheckpointInstallTests(unittest.TestCase):
    def test_install_verify_and_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            root, config, archive, target = self.fixture(Path(directory))

            first = install_checkpoints(
                config_path=config,
                root=root,
                archive_override=archive,
            )
            second = install_checkpoints(
                config_path=config,
                root=root,
                verify_only=True,
            )

            self.assertEqual(target.read_bytes(), b"checkpoint weights")
            self.assertIn("Installed and verified 1 checkpoints", first)
            self.assertIn("already installed", second)

    def test_archive_checksum_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root, config, archive, _ = self.fixture(Path(directory))
            archive.write_bytes(archive.read_bytes() + b"tampered")

            with self.assertRaisesRegex(CheckpointInstallError, "size mismatch"):
                install_checkpoints(
                    config_path=config,
                    root=root,
                    archive_override=archive,
                )

    def test_existing_mismatch_requires_force(self):
        with tempfile.TemporaryDirectory() as directory:
            root, config, archive, target = self.fixture(Path(directory))
            target.parent.mkdir(parents=True)
            target.write_bytes(b"my local run")

            with self.assertRaisesRegex(CheckpointInstallError, "--force"):
                install_checkpoints(
                    config_path=config,
                    root=root,
                    archive_override=archive,
                )
            self.assertEqual(target.read_bytes(), b"my local run")

            install_checkpoints(
                config_path=config,
                root=root,
                archive_override=archive,
                force=True,
            )
            self.assertEqual(target.read_bytes(), b"checkpoint weights")

    def test_manifest_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, config, archive, _ = self.fixture(Path(directory))
            manifest_path = root / "configs/checkpoint_bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files"][0]["path"] = "../outside.pt"
            manifest_path.write_text(json.dumps(manifest))

            with self.assertRaisesRegex(CheckpointInstallError, "Unsafe"):
                install_checkpoints(
                    config_path=config,
                    root=root,
                    archive_override=archive,
                )

    @staticmethod
    def fixture(directory: Path) -> tuple[Path, Path, Path, Path]:
        root = directory / "repo"
        model_root = root / "Models&Datasets/example"
        model_root.mkdir(parents=True)
        configs = root / "configs"
        configs.mkdir()
        relative = "Models&Datasets/example/checkpoints/model.pt"
        payload = b"checkpoint weights"
        digest = hashlib.sha256(payload).hexdigest()
        manifest = {
            "bundle_id": "test-v1",
            "checkpoints": [
                {
                    "dataset_id": "test",
                    "files": [relative],
                    "fingerprint": digest,
                    "model_id": "example",
                    "primary_path": relative,
                }
            ],
            "files": [{"bytes": len(payload), "path": relative, "sha256": digest}],
            "logical_checkpoint_count": 1,
            "origin": "test",
            "schema_version": 1,
        }
        (configs / "checkpoint_bundle_manifest.json").write_text(json.dumps(manifest))
        archive = directory / "bundle.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr(relative, payload)
        source = {
            "schema_version": 1,
            "bundle": {
                "asset_name": "bundle.zip",
                "bytes": archive.stat().st_size,
                "manifest": "configs/checkpoint_bundle_manifest.json",
                "required_model_roots": ["Models&Datasets/example"],
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "url": "https://example.invalid/bundle.zip",
            },
        }
        config = configs / "checkpoint_sources.json"
        config.write_text(json.dumps(source))
        return root, config, archive, root / relative


if __name__ == "__main__":
    unittest.main()
