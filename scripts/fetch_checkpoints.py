from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Any
from urllib.request import Request, urlopen
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "checkpoint_sources.json"


class CheckpointInstallError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install and verify the audited local checkpoint bundle."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--archive",
        type=Path,
        help="Use a local bundle instead of downloading the release asset.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify installed files without downloading anything.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace installed files whose checksums do not match the bundle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = install_checkpoints(
        config_path=args.config,
        root=args.root,
        archive_override=args.archive,
        verify_only=args.verify_only,
        force=args.force,
    )
    print(state)


def install_checkpoints(
    *,
    config_path: Path,
    root: Path,
    archive_override: Path | None = None,
    verify_only: bool = False,
    force: bool = False,
) -> str:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path, "checkpoint source manifest")
    if config.get("schema_version") != 1:
        raise CheckpointInstallError("Unsupported checkpoint source schema.")
    bundle = config.get("bundle")
    if not isinstance(bundle, dict):
        raise CheckpointInstallError("Checkpoint source manifest has no bundle entry.")

    manifest_path = safe_path(root, bundle.get("manifest", ""))
    manifest = read_json(manifest_path, "checkpoint bundle manifest")
    records = validate_manifest(manifest)
    verify_model_roots(root, bundle.get("required_model_roots"))

    problems = installed_problems(root, records)
    if not problems:
        return (
            f"Verified {manifest['logical_checkpoint_count']} checkpoints "
            f"across {len(records)} files; already installed."
        )
    if verify_only:
        raise CheckpointInstallError(
            "Checkpoint verification failed: " + "; ".join(problems[:8])
        )

    conflicts = [problem for problem in problems if problem.startswith("checksum:")]
    if conflicts and not force:
        raise CheckpointInstallError(
            "Installed checkpoint files differ from the audited bundle. "
            "Review them, then rerun with --force to replace them: "
            + "; ".join(conflicts[:5])
        )

    temporary_root = root / "tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="gsp-checkpoints-",
        dir=temporary_root,
    ) as directory:
        temporary = Path(directory)
        archive = temporary / bundle["asset_name"]
        if archive_override is not None:
            shutil.copyfile(archive_override.resolve(), archive)
        else:
            download(bundle["url"], archive)
        verify_archive(archive, bundle)
        staging = temporary / "staging"
        extract_verified(archive, staging, records)
        for record in records:
            source = safe_path(staging, record["path"])
            target = safe_path(root, record["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)

    remaining = installed_problems(root, records)
    if remaining:
        raise CheckpointInstallError(
            "Checkpoint installation did not verify: " + "; ".join(remaining[:8])
        )
    return (
        f"Installed and verified {manifest['logical_checkpoint_count']} checkpoints "
        f"across {len(records)} files."
    )


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise CheckpointInstallError(f"Missing {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CheckpointInstallError(f"Invalid {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CheckpointInstallError(f"Invalid {label}: expected a JSON object.")
    return payload


def validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != 1:
        raise CheckpointInstallError("Unsupported checkpoint bundle schema.")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise CheckpointInstallError("Checkpoint bundle manifest has no files.")
    if manifest.get("logical_checkpoint_count") != len(
        manifest.get("checkpoints", [])
    ):
        raise CheckpointInstallError("Logical checkpoint count is inconsistent.")
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise CheckpointInstallError("Invalid checkpoint file record.")
        path = record.get("path")
        if not isinstance(path, str) or path in seen:
            raise CheckpointInstallError(f"Invalid or duplicate bundle path: {path!r}")
        validate_relative_path(path)
        if not path.startswith("Models&Datasets/"):
            raise CheckpointInstallError(f"Unexpected checkpoint path: {path}")
        if not isinstance(record.get("bytes"), int) or record["bytes"] < 0:
            raise CheckpointInstallError(f"Invalid byte count for {path}.")
        digest = record.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise CheckpointInstallError(f"Invalid SHA-256 for {path}.")
        seen.add(path)
    return records


def verify_model_roots(root: Path, values: object) -> None:
    if not isinstance(values, list) or not values:
        raise CheckpointInstallError("No required model roots were declared.")
    missing = []
    for value in values:
        path = safe_path(root, value)
        if not path.is_dir():
            missing.append(value)
    if missing:
        raise CheckpointInstallError(
            "Model sources must be installed before checkpoints. Run `make models`. "
            "Missing: " + ", ".join(missing)
        )


def installed_problems(root: Path, records: list[dict[str, Any]]) -> list[str]:
    problems = []
    for record in records:
        path = safe_path(root, record["path"])
        if not path.is_file():
            problems.append(f"missing:{record['path']}")
        elif path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            problems.append(f"checksum:{record['path']}")
    return problems


def download(url: str, destination: Path) -> None:
    if not isinstance(url, str) or not url.startswith(("https://", "http://")):
        raise CheckpointInstallError("Checkpoint bundle URL must use HTTP(S).")
    print(f"Downloading {url} ...", flush=True)
    request = Request(url, headers={"User-Agent": "graph-similarity-platform/1"})
    try:
        with urlopen(request, timeout=120) as response, destination.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    except OSError as exc:
        raise CheckpointInstallError(f"Checkpoint download failed: {exc}") from exc


def verify_archive(path: Path, bundle: dict[str, Any]) -> None:
    expected_bytes = bundle.get("bytes")
    expected_sha256 = bundle.get("sha256")
    if path.stat().st_size != expected_bytes:
        raise CheckpointInstallError(
            f"Checkpoint archive size mismatch: expected {expected_bytes}, "
            f"found {path.stat().st_size}."
        )
    if sha256_file(path) != expected_sha256:
        raise CheckpointInstallError("Checkpoint archive SHA-256 mismatch.")


def extract_verified(
    archive_path: Path,
    staging: Path,
    records: list[dict[str, Any]],
) -> None:
    expected = {record["path"]: record for record in records}
    staging.mkdir(parents=True)
    try:
        archive = zipfile.ZipFile(archive_path)
    except zipfile.BadZipFile as exc:
        raise CheckpointInstallError("Checkpoint bundle is not a valid ZIP file.") from exc
    with archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        names = [info.filename for info in members]
        if len(names) != len(set(names)):
            raise CheckpointInstallError("Checkpoint bundle has duplicate paths.")
        if set(names) != set(expected):
            missing = sorted(set(expected) - set(names))
            extra = sorted(set(names) - set(expected))
            raise CheckpointInstallError(
                f"Checkpoint bundle contents differ from the manifest; "
                f"missing={missing[:3]}, extra={extra[:3]}."
            )
        for info in members:
            validate_relative_path(info.filename)
            mode = (info.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise CheckpointInstallError(
                    f"Checkpoint bundle contains a symlink: {info.filename}"
                )
            record = expected[info.filename]
            if info.file_size != record["bytes"]:
                raise CheckpointInstallError(
                    f"Checkpoint member size mismatch: {info.filename}"
                )
            target = safe_path(staging, info.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with archive.open(info) as source, target.open("wb") as handle:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    handle.write(chunk)
            if digest.hexdigest() != record["sha256"]:
                raise CheckpointInstallError(
                    f"Checkpoint member SHA-256 mismatch: {info.filename}"
                )


def validate_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts or "." in path.parts:
        raise CheckpointInstallError(f"Unsafe checkpoint path: {value}")


def safe_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise CheckpointInstallError("Empty checkpoint path.")
    validate_relative_path(relative)
    root = root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise CheckpointInstallError(f"Path escapes repository root: {relative}") from exc
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
