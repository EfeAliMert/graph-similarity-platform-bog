from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import zipfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graph_similarity_platform.models.real_models import (  # noqa: E402
    MODELS,
    _preferred_checkpoint,
)
from scripts.checkpoint_provenance import checkpoint_fingerprint  # noqa: E402


DEFAULT_MANIFEST = ROOT / "configs" / "checkpoint_bundle_manifest.json"
DEFAULT_ARCHIVE = ROOT / "output" / "graph-similarity-checkpoints-v1.zip"


class CheckpointBundleError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the versioned bundle of registered local checkpoints."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(ROOT)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    write_archive(ROOT, manifest, args.archive)
    print(
        json.dumps(
            {
                "archive": str(args.archive),
                "archive_bytes": args.archive.stat().st_size,
                "archive_sha256": sha256_file(args.archive),
                "files": len(manifest["files"]),
                "logical_checkpoints": len(manifest["checkpoints"]),
                "manifest": str(args.manifest),
            },
            sort_keys=True,
        )
    )


def build_manifest(root: Path) -> dict[str, Any]:
    files: dict[Path, dict[str, Any]] = {}
    checkpoints = []
    for model in MODELS:
        model_root = root / model["local_path"]
        for dataset_id in model.get("datasets", []):
            checkpoint = _preferred_checkpoint(model_root, model, dataset_id)
            if checkpoint is None:
                raise CheckpointBundleError(
                    f"Missing registered checkpoint for {model['id']}/{dataset_id}."
                )
            members = checkpoint_members(checkpoint)
            if model["id"] == "simgnn":
                members.append(
                    model_root
                    / "original_datasets"
                    / dataset_id
                    / "manifest.json"
                )
            missing = [path for path in members if not path.is_file()]
            if missing:
                raise CheckpointBundleError(
                    f"Incomplete checkpoint for {model['id']}/{dataset_id}: "
                    + ", ".join(str(path) for path in missing)
                )
            members = sorted(set(members), key=lambda path: display_path(root, path))
            for path in members:
                files.setdefault(path, file_record(root, path))
            checkpoints.append(
                {
                    "dataset_id": dataset_id,
                    "files": [display_path(root, path) for path in members],
                    "fingerprint": checkpoint_fingerprint(checkpoint),
                    "model_id": model["id"],
                    "primary_path": display_path(root, checkpoint),
                }
            )

    if len(checkpoints) != 35:
        raise CheckpointBundleError(
            f"Expected 35 model/dataset checkpoints, found {len(checkpoints)}."
        )
    return {
        "bundle_id": "local-checkpoints-v1",
        "checkpoints": checkpoints,
        "files": sorted(files.values(), key=lambda row: row["path"]),
        "logical_checkpoint_count": len(checkpoints),
        "origin": "Locally trained checkpoints; not author-released pretrained weights.",
        "schema_version": 1,
    }


def checkpoint_members(checkpoint: Path) -> list[Path]:
    if checkpoint.is_file():
        members = [checkpoint]
    else:
        members = sorted(
            path
            for path in checkpoint.parent.glob(checkpoint.name + "*")
            if path.is_file()
        )
    hpo_sidecar = Path(str(checkpoint) + ".hpo.json")
    if hpo_sidecar.is_file():
        members.append(hpo_sidecar)
    return members


def write_archive(root: Path, manifest: dict[str, Any], archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as bundle:
        for record in manifest["files"]:
            source = safe_path(root, record["path"])
            info = zipfile.ZipInfo(record["path"], date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, source.read_bytes(), compresslevel=9)


def file_record(root: Path, path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "path": display_path(root, path),
        "sha256": sha256_file(path),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise CheckpointBundleError(f"Path escapes repository root: {relative}") from exc
    return path


def display_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


if __name__ == "__main__":
    main()
