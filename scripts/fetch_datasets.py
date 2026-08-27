from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any, Callable
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "dataset_sources.json"
DEFAULT_TARGET = (
    ROOT
    / "Models&Datasets"
    / "drive-download-20260630T100606Z-3-001"
)
GOOGLE_DRIVE_DOWNLOAD = (
    "https://drive.usercontent.google.com/download"
    "?id={file_id}&export=download&confirm=t"
)
CHUNK_SIZE = 1024 * 1024


class DatasetInstallError(RuntimeError):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download the registered graph datasets from their upstream "
            "sources and verify the benchmark files."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET)
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Dataset ids to install (default: every registered dataset).",
    )
    parser.add_argument(
        "--skip-tu",
        action="store_true",
        help="Skip MUTAG, PROTEINS, and ENZYMES export.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check existing files without downloading or exporting anything.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace invalid or existing files from the registered source.",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    selected = normalize_selection(args.datasets, manifest)
    args.target_dir.mkdir(parents=True, exist_ok=True)

    benchmark_results = install_benchmark_files(
        manifest,
        args.target_dir,
        selected,
        verify_only=args.verify_only,
        force=args.force,
    )
    tu_results: list[dict[str, Any]] = []
    if not args.skip_tu:
        tu_results = install_tu_datasets(
            manifest,
            args.target_dir,
            selected,
            verify_only=args.verify_only,
            force=args.force,
        )

    print_summary(benchmark_results, tu_results)


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise DatasetInstallError(f"Dataset source manifest not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetInstallError(f"Invalid JSON in dataset source manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise DatasetInstallError("Dataset source manifest must contain a JSON object.")
    validate_manifest(payload)
    return payload


def validate_manifest(payload: dict[str, Any]) -> None:
    entries = payload.get("benchmark_files")
    if not isinstance(entries, list) or not entries:
        raise DatasetInstallError("Manifest benchmark_files must be a non-empty list.")

    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise DatasetInstallError("Every benchmark file entry must be an object.")
        filename = entry.get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise DatasetInstallError(f"Unsafe benchmark filename: {filename!r}")
        if filename in seen:
            raise DatasetInstallError(f"Duplicate benchmark filename: {filename}")
        seen.add(filename)
        if not isinstance(entry.get("bytes"), int) or entry["bytes"] <= 0:
            raise DatasetInstallError(f"Invalid byte count for {filename}")
        sha256 = entry.get("sha256")
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise DatasetInstallError(f"Invalid SHA-256 for {filename}")
        try:
            int(sha256, 16)
        except ValueError as exc:
            raise DatasetInstallError(f"Invalid SHA-256 for {filename}") from exc
        file_id = entry.get("google_drive_id")
        if not isinstance(file_id, str) or not file_id:
            raise DatasetInstallError(f"Missing Google Drive id for {filename}")

    tu_entries = payload.get("tu_datasets", [])
    if not isinstance(tu_entries, list):
        raise DatasetInstallError("Manifest tu_datasets must be a list.")
    for entry in tu_entries:
        if not isinstance(entry, dict) or not entry.get("name") or not entry.get("dataset_id"):
            raise DatasetInstallError("Every TU dataset entry needs name and dataset_id.")
        counts = (entry.get("graphs"), entry.get("train_graphs"), entry.get("test_graphs"))
        if not all(isinstance(value, int) and value >= 0 for value in counts):
            raise DatasetInstallError(
                f"Invalid graph counts for TU dataset {entry.get('name')}."
            )
        if entry["train_graphs"] + entry["test_graphs"] != entry["graphs"]:
            raise DatasetInstallError(
                f"TU split counts do not add up for {entry['name']}."
            )


def normalize_selection(
    requested: list[str] | None,
    manifest: dict[str, Any],
) -> set[str]:
    available = {
        str(entry["dataset_id"]).lower()
        for entry in manifest.get("benchmark_files", [])
    }
    available.update(
        str(entry["dataset_id"]).lower()
        for entry in manifest.get("tu_datasets", [])
    )
    if not requested:
        return available
    selected = {value.lower() for value in requested}
    unknown = sorted(selected - available)
    if unknown:
        raise DatasetInstallError(
            "Unknown dataset id(s): " + ", ".join(unknown)
        )
    return selected


def install_benchmark_files(
    manifest: dict[str, Any],
    target_dir: Path,
    selected: set[str],
    *,
    verify_only: bool,
    force: bool,
) -> list[dict[str, Any]]:
    results = []
    for entry in manifest["benchmark_files"]:
        if str(entry["dataset_id"]).lower() not in selected:
            continue
        results.append(
            install_benchmark_file(
                entry,
                target_dir,
                verify_only=verify_only,
                force=force,
            )
        )
    return results


def install_benchmark_file(
    entry: dict[str, Any],
    target_dir: Path,
    *,
    verify_only: bool = False,
    force: bool = False,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    filename = str(entry["filename"])
    target = target_dir / filename
    expected_bytes = int(entry["bytes"])
    expected_hash = str(entry["sha256"]).lower()

    if target.exists():
        valid, detail = verify_file(target, expected_bytes, expected_hash)
        if valid and not force:
            print(f"[verified] {filename}")
            return {"filename": filename, "status": "verified"}
        if verify_only:
            raise DatasetInstallError(f"Verification failed for {target}: {detail}")
        if not force:
            raise DatasetInstallError(
                f"Existing file failed verification: {target} ({detail}). "
                "Use --force to replace it."
            )
    elif verify_only:
        raise DatasetInstallError(f"Required dataset file is missing: {target}")

    target_dir.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.part")
    partial.unlink(missing_ok=True)
    url = GOOGLE_DRIVE_DOWNLOAD.format(file_id=entry["google_drive_id"])
    request = Request(url, headers={"User-Agent": "graph-similarity-platform/1.0"})
    print(f"[download] {filename} ({format_bytes(expected_bytes)})")

    digest = hashlib.sha256()
    received = 0
    next_progress = 25
    try:
        with opener(request, timeout=120) as response, partial.open("wb") as handle:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                received += len(chunk)
                percent = int(received * 100 / expected_bytes)
                if percent >= next_progress and expected_bytes >= CHUNK_SIZE:
                    print(f"  {min(percent, 100)}%")
                    next_progress += 25

        if received != expected_bytes:
            raise DatasetInstallError(
                f"Size mismatch for {filename}: expected {expected_bytes}, got {received}."
            )
        actual_hash = digest.hexdigest()
        if actual_hash != expected_hash:
            raise DatasetInstallError(
                f"SHA-256 mismatch for {filename}: expected {expected_hash}, "
                f"got {actual_hash}."
            )
        os.replace(partial, target)
    except DatasetInstallError:
        partial.unlink(missing_ok=True)
        raise
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise DatasetInstallError(
            f"Download failed for {filename}: {type(exc).__name__}: {exc}"
        ) from exc

    print(f"[installed] {filename}")
    return {"filename": filename, "status": "downloaded"}


def verify_file(path: Path, expected_bytes: int, expected_hash: str) -> tuple[bool, str]:
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        return False, f"expected {expected_bytes} bytes, got {actual_bytes}"
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash.lower():
        return False, f"expected SHA-256 {expected_hash}, got {actual_hash}"
    return True, "ok"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_tu_datasets(
    manifest: dict[str, Any],
    target_dir: Path,
    selected: set[str],
    *,
    verify_only: bool,
    force: bool,
) -> list[dict[str, Any]]:
    entries = [
        entry
        for entry in manifest.get("tu_datasets", [])
        if str(entry["dataset_id"]).lower() in selected
    ]
    if not entries:
        return []

    if verify_only:
        results = []
        for entry in entries:
            archive = target_dir / f'{entry["name"]}.zip'
            labels = target_dir / f'{entry["dataset_id"]}_graph_labels.json'
            verify_tu_export(entry, archive, labels)
            print(f"[verified] {entry['name']} export")
            results.append({"name": entry["name"], "status": "verified"})
        return results

    try:
        from download_real_graph_datasets import export_tu_dataset
    except ImportError as exc:
        raise DatasetInstallError(
            "TU dataset export requires the gnn-pyg environment. "
            "Run `make setup`, then run `make datasets` again."
        ) from exc

    results = []
    for entry in entries:
        print(f"[export] {entry['name']} via PyTorch Geometric")
        result = export_tu_dataset(
            str(entry["name"]),
            force=force,
            app_dataset_dir=target_dir,
        )
        archive = target_dir / f'{entry["name"]}.zip'
        labels = target_dir / f'{entry["dataset_id"]}_graph_labels.json'
        verify_tu_export(entry, archive, labels)
        result["verification"] = "passed"
        if result.get("status") == "exists":
            result["status"] = "verified"
        results.append(result)
    return results


def verify_tu_export(
    entry: dict[str, Any],
    archive: Path,
    labels_path: Path,
) -> None:
    name = str(entry["name"])
    if not archive.is_file() or not labels_path.is_file():
        raise DatasetInstallError(f"Missing exported TU dataset files for {name}.")

    expected_graphs = int(entry["graphs"])
    expected_train = int(entry["train_graphs"])
    expected_test = int(entry["test_graphs"])
    try:
        with zipfile.ZipFile(archive) as handle:
            corrupt_member = handle.testzip()
            graph_members = [
                member
                for member in handle.namelist()
                if member.lower().endswith(".gexf")
            ]
    except zipfile.BadZipFile as exc:
        raise DatasetInstallError(f"Invalid TU archive for {name}: {archive}") from exc
    if corrupt_member:
        raise DatasetInstallError(
            f"Corrupt member in TU archive for {name}: {corrupt_member}"
        )
    if len(graph_members) != expected_graphs:
        raise DatasetInstallError(
            f"Graph count mismatch for {name}: expected {expected_graphs}, "
            f"got {len(graph_members)}."
        )
    train_members = [member for member in graph_members if "/train/" in member]
    test_members = [member for member in graph_members if "/test/" in member]
    if len(train_members) != expected_train or len(test_members) != expected_test:
        raise DatasetInstallError(
            f"Split count mismatch for {name}: expected {expected_train}/{expected_test}, "
            f"got {len(train_members)}/{len(test_members)}."
        )

    try:
        labels = json.loads(labels_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetInstallError(f"Invalid label map for {name}: {labels_path}") from exc
    if not isinstance(labels, dict) or len(labels) != expected_graphs:
        actual = len(labels) if isinstance(labels, dict) else "non-object"
        raise DatasetInstallError(
            f"Label count mismatch for {name}: expected {expected_graphs}, got {actual}."
        )
    if set(labels) != set(graph_members):
        raise DatasetInstallError(
            f"Label keys do not match archive graph members for {name}."
        )


def print_summary(
    benchmark_results: list[dict[str, Any]],
    tu_results: list[dict[str, Any]],
) -> None:
    statuses: dict[str, int] = {}
    for result in [*benchmark_results, *tu_results]:
        status = str(result.get("status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1
    summary = ", ".join(
        f"{count} {status}" for status, count in sorted(statuses.items())
    )
    print(f"Dataset setup complete: {summary or 'nothing selected'}.")


def format_bytes(value: int) -> str:
    units = ("B", "KB", "MB", "GB")
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{value} B"


if __name__ == "__main__":
    try:
        main()
    except DatasetInstallError as exc:
        print(f"Dataset setup failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
