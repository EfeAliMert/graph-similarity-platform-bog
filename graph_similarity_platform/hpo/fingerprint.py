from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from scripts.universal_dataset import dataset_spec


FINGERPRINT_VERSION = "dataset-fingerprint-v1"


def hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    stat = path.stat()
    return _hash_file_cached(str(path.resolve()), stat.st_size, stat.st_mtime_ns, chunk_size)


@lru_cache(maxsize=64)
def _hash_file_cached(
    path_text: str,
    _size: int,
    _mtime_ns: int,
    chunk_size: int,
) -> str:
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(
    dataset_id: str,
    preprocessing: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Hash graph archive, target files, split metadata, and preprocessing."""
    spec = dataset_spec(dataset_id)
    archive = Path(spec["archive"])
    if not archive.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {archive}")

    target_files = []
    target_path = spec.get("ged")
    if target_path and Path(target_path).is_file():
        path = Path(target_path)
        target_files.append(
            {"name": path.name, "size": path.stat().st_size, "sha256": hash_file(path)}
        )

    manifest_path = archive.parent / "manifest.json"
    manifest = None
    if spec.get("uploaded") and manifest_path.is_file():
        manifest = {
            "name": manifest_path.name,
            "sha256": hash_file(manifest_path),
        }

    identity = {
        "fingerprint_version": FINGERPRINT_VERSION,
        "dataset_id": dataset_id,
        "archive": {
            "name": archive.name,
            "format": spec.get("format"),
            "size": archive.stat().st_size,
            "sha256": hash_file(archive),
        },
        "targets": target_files,
        "manifest": manifest,
        "split_strategy": spec.get("split_strategy") or "registered train/test",
        "split_seed": spec.get("split_seed"),
        "target_kind": spec.get("target_kind"),
        "preprocessing": dict(preprocessing or {}),
    }
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest(), identity
