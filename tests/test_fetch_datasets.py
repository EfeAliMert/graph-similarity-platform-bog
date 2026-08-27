from __future__ import annotations

import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.fetch_datasets import (
    DatasetInstallError,
    install_benchmark_file,
    load_manifest,
    validate_manifest,
    verify_tu_export,
)


class FakeResponse:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.offset = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, size: int) -> bytes:
        chunk = self.payload[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk


class DatasetDownloadTest(unittest.TestCase):
    def entry(self, payload: bytes, filename: str = "sample.zip") -> dict:
        return {
            "dataset_id": "sample",
            "kind": "graph_archive",
            "filename": filename,
            "google_drive_id": "upstream-file-id",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    def test_valid_existing_file_is_not_downloaded(self):
        payload = b"verified dataset"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            (target / "sample.zip").write_bytes(payload)

            def unexpected_opener(*args, **kwargs):
                raise AssertionError("valid file should not be downloaded")

            result = install_benchmark_file(
                self.entry(payload),
                target,
                opener=unexpected_opener,
            )

        self.assertEqual(result["status"], "verified")

    def test_invalid_existing_file_requires_force(self):
        payload = b"expected dataset"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            (target / "sample.zip").write_bytes(b"wrong")
            with self.assertRaisesRegex(DatasetInstallError, "--force"):
                install_benchmark_file(self.entry(payload), target)

    def test_force_replaces_invalid_file_after_verification(self):
        payload = b"replacement dataset"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            path = target / "sample.zip"
            path.write_bytes(b"wrong")

            result = install_benchmark_file(
                self.entry(payload),
                target,
                force=True,
                opener=lambda *args, **kwargs: FakeResponse(payload),
            )

            self.assertEqual(path.read_bytes(), payload)
            self.assertFalse((target / "sample.zip.part").exists())
        self.assertEqual(result["status"], "downloaded")

    def test_checksum_failure_does_not_install_partial_file(self):
        expected = b"expected"
        received = b"tampered"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            with self.assertRaisesRegex(DatasetInstallError, "SHA-256 mismatch"):
                install_benchmark_file(
                    self.entry(expected),
                    target,
                    opener=lambda *args, **kwargs: FakeResponse(received),
                )
            self.assertFalse((target / "sample.zip").exists())
            self.assertFalse((target / "sample.zip.part").exists())

    def test_manifest_rejects_parent_path_filename(self):
        payload = {
            "benchmark_files": [self.entry(b"data", filename="../sample.zip")],
            "tu_datasets": [],
        }
        with self.assertRaisesRegex(DatasetInstallError, "Unsafe"):
            validate_manifest(payload)

    def test_committed_manifest_has_unique_verified_entries(self):
        manifest = load_manifest(
            Path(__file__).resolve().parents[1] / "configs" / "dataset_sources.json"
        )
        entries = manifest["benchmark_files"]
        self.assertEqual(len(entries), 12)
        self.assertEqual(len({entry["filename"] for entry in entries}), 12)
        self.assertEqual(
            {entry["dataset_id"] for entry in entries},
            {"aids700nef", "linux", "imdbmulti", "ptc"},
        )

    def test_invalid_json_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text("not-json")
            with self.assertRaisesRegex(DatasetInstallError, "Invalid JSON"):
                load_manifest(path)

    def test_tu_export_checks_graph_and_label_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "TINY.zip"
            labels = root / "tiny_graph_labels.json"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("TINY/train/0.gexf", "<gexf />")
                handle.writestr("TINY/test/1.gexf", "<gexf />")
            labels.write_text('{"TINY/train/0.gexf": 0, "TINY/test/1.gexf": 1}')
            entry = {
                "name": "TINY",
                "graphs": 2,
                "train_graphs": 1,
                "test_graphs": 1,
            }

            verify_tu_export(entry, archive, labels)
            with self.assertRaisesRegex(DatasetInstallError, "Graph count mismatch"):
                verify_tu_export(
                    {
                        "name": "TINY",
                        "graphs": 3,
                        "train_graphs": 2,
                        "test_graphs": 1,
                    },
                    archive,
                    labels,
                )


if __name__ == "__main__":
    unittest.main()
