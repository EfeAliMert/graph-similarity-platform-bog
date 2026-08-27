from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORAGE_PATH = ROOT / "training_logs" / "hpo" / "optimization.db"
DEFAULT_PROGRESS_ROOT = ROOT / "training_logs" / "hpo" / "progress"


def sqlite_storage_url(path: Path = DEFAULT_STORAGE_PATH) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.resolve()}"


def stable_study_name(
    model_id: str,
    dataset_id: str,
    search_space_version: str,
    dataset_fingerprint: str,
) -> str:
    values = (
        model_id,
        dataset_id,
        "normalized_ged",
        search_space_version,
        dataset_fingerprint[:12],
    )
    return "__".join(_slug(value) for value in values)


class ProgressStore:
    def __init__(self, root: Path = DEFAULT_PROGRESS_ROOT):
        self.root = root

    def path(self, study_name: str) -> Path:
        return self.root / f"{_slug(study_name)}.json"

    def write(self, study_name: str, payload: dict[str, Any]) -> Path:
        path = self.path(study_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
        temporary.replace(path)
        return path

    def read(self, study_name: str) -> dict[str, Any] | None:
        path = self.path(study_name)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", str(value).lower()).strip("-")
