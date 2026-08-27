from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from typing import Any


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT))

from graph_similarity_platform.data import list_original_datasets
from graph_similarity_platform.models.real_models import (
    BASE_DIR,
    MODELS,
    _preferred_checkpoint,
)
from scripts.audit_checkpoints import (  # noqa: E402
    build_checkpoint_audit,
    write_checkpoint_audit,
)


OUTPUT_PATH = BASE_DIR / "artifacts.manifest.json"


def main() -> None:
    files: dict[Path, dict[str, Any]] = {}
    checkpoint_audit = build_checkpoint_audit()
    write_checkpoint_audit(checkpoint_audit)
    audit_by_key = {
        (row.get("model_id"), row.get("dataset_id")): row
        for row in checkpoint_audit.get("rows", [])
    }
    datasets = []
    for dataset in list_original_datasets():
        paths = [
            BASE_DIR / dataset["archive_path"],
            *[BASE_DIR / path for path in dataset.get("ground_truth_paths", [])],
        ]
        dataset_files = []
        for path in paths:
            for concrete in checkpoint_files(path):
                files.setdefault(concrete, file_record(concrete))
                dataset_files.append(display_path(concrete))
        datasets.append(
            {
                "id": dataset["id"],
                "name": dataset["name"],
                "graphs": dataset["graph_count"],
                "train_graphs": dataset["train_graphs"],
                "test_graphs": dataset["test_graphs"],
                "ground_truth_available": bool(dataset["ground_truth_available"]),
                "files": sorted(set(dataset_files)),
            }
        )

    checkpoints = []
    repositories = []
    for model in MODELS:
        local_path = BASE_DIR / model["local_path"]
        repositories.append(repository_record(model, local_path))
        for dataset_id in model.get("datasets", []):
            checkpoint = _preferred_checkpoint(local_path, model, dataset_id)
            if checkpoint is None:
                continue
            checkpoint_paths = checkpoint_files(checkpoint)
            for path in checkpoint_paths:
                files.setdefault(path, file_record(path))
            checkpoints.append(
                {
                    "model_id": model["id"],
                    "dataset_id": dataset_id,
                    "official_pretrained": bool(model.get("official_pretrained")),
                    "origin": model.get("checkpoint_origin"),
                    "files": [display_path(path) for path in checkpoint_paths],
                    "protocol_verified": audit_by_key.get(
                        (model["id"], dataset_id),
                        {},
                    ).get("protocol_verified"),
                    "hpo_status": audit_by_key.get(
                        (model["id"], dataset_id),
                        {},
                    ).get("hpo_status"),
                    "hpo_verified": audit_by_key.get(
                        (model["id"], dataset_id),
                        {},
                    ).get("hpo_verified"),
                    "seed": audit_by_key.get(
                        (model["id"], dataset_id),
                        {},
                    ).get("seed"),
                    "split_sha256": audit_by_key.get(
                        (model["id"], dataset_id),
                        {},
                    ).get("split_sha256"),
                }
            )

    research_paths = [
        BASE_DIR / "LICENSE",
        BASE_DIR / "configs" / "dataset_sources.json",
        BASE_DIR / "docs" / "ARTIFACT_SETUP.md",
        BASE_DIR / "docs" / "ADDING_NEW_DATASETS.md",
        BASE_DIR / "docs" / "DATASETS.md",
        BASE_DIR / "docs" / "THIRD_PARTY.md",
        BASE_DIR / "docs" / "PROJECT_STATUS.md",
        BASE_DIR / "docs" / "RESEARCH_PROTOCOL.md",
        BASE_DIR / "docs" / "HPO_ARCHITECTURE.md",
        BASE_DIR / "scripts" / "fetch_datasets.py",
        BASE_DIR / "reports" / "checkpoint_audit.json",
        BASE_DIR / "reports" / "checkpoint_audit.md",
        BASE_DIR / "reports" / "model_output_audit.json",
        BASE_DIR / "reports" / "model_output_audit.md",
        BASE_DIR / "reports" / "RESEARCH_RESULTS.json",
        BASE_DIR / "reports" / "RESEARCH_RESULTS.md",
        BASE_DIR / "reports" / "final_dataset_accuracy_audit.json",
        BASE_DIR / "reports" / "final_dataset_accuracy_audit.md",
        BASE_DIR / "reports" / "grouped_split_study.json",
        BASE_DIR / "reports" / "grouped_split_study.md",
    ]
    for directory in (
        BASE_DIR / "reports" / "research_matrices",
        BASE_DIR / "reports" / "retrieval_study",
        BASE_DIR / "reports" / "adapter_ablations",
        BASE_DIR / "reports" / "hpo",
        BASE_DIR / "configs" / "optimized",
    ):
        if directory.exists():
            research_paths.extend(
                sorted(path for path in directory.rglob("*") if path.is_file())
            )
    research_artifacts = []
    for path in research_paths:
        if not path.is_file():
            continue
        files.setdefault(path, file_record(path))
        research_artifacts.append(display_path(path))

    payload = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "warning": (
            "Checkpoint files are locally trained artifacts, not author-released "
            "pretrained weights, unless explicitly marked otherwise."
        ),
        "source_revision": (
            "Use the enclosing Git commit as the source revision. Its hash is "
            "not embedded here because this manifest is part of that commit."
        ),
        "checkpoint_protocol_audit": {
            "verified": checkpoint_audit.get("verified"),
            "hpo_verified": checkpoint_audit.get("hpo_verified"),
            "total": checkpoint_audit.get("total"),
            "complete": checkpoint_audit.get("complete"),
        },
        "datasets": datasets,
        "checkpoints": checkpoints,
        "research_artifacts": research_artifacts,
        "repositories": repositories,
        "files": sorted(files.values(), key=lambda row: row["path"]),
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "output": display_path(OUTPUT_PATH),
                "files": len(files),
                "datasets": len(datasets),
                "checkpoints": len(checkpoints),
            }
        )
    )


def checkpoint_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    candidates = sorted(
        candidate
        for candidate in path.parent.glob(f"{path.name}*")
        if candidate.is_file()
    )
    return candidates


def file_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": display_path(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def repository_record(model: dict[str, Any], path: Path) -> dict[str, Any]:
    return {
        "model_id": model["id"],
        "path": display_path(path),
        "declared_repository": model.get("repository_url"),
        "implementation_origin": model.get("implementation_origin"),
        "git_head": git_value(path, ["rev-parse", "HEAD"]),
        "git_remote": git_value(path, ["remote", "get-url", "origin"]),
    }


def git_value(path: Path, arguments: list[str]) -> str | None:
    if not (path / ".git").exists():
        return None
    completed = subprocess.run(
        ["git", "-C", str(path), *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    value = completed.stdout.strip()
    return value or None


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


if __name__ == "__main__":
    main()
