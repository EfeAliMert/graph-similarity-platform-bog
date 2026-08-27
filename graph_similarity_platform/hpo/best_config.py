from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from .types import DatasetProfile


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_ROOT = ROOT / "configs" / "optimized"


class BestConfigRegistry:
    def __init__(self, root: Path = DEFAULT_CONFIG_ROOT):
        self.root = root

    def path(self, dataset_id: str, model_id: str) -> Path:
        return self.root / dataset_id / f"{model_id}.json"

    def save(
        self,
        *,
        dataset_profile: DatasetProfile,
        model_id: str,
        search_space_version: str,
        study_name: str,
        best_trial: int,
        validation_mse: float,
        validation_spearman: float | None,
        validation_mse_std: float | None,
        seeds: list[int],
        hyperparameters: Mapping[str, Any],
        study_storage: str,
        split_seed: int,
        trial_checkpoint: str | None,
    ) -> dict[str, Any]:
        payload = {
            "schema_version": "optimized-config-v1",
            "dataset": dataset_profile.dataset_id,
            "dataset_name": dataset_profile.dataset_name,
            "model": model_id,
            "search_space_version": search_space_version,
            "study_name": study_name,
            "study_storage": _portable_path(study_storage),
            "best_trial": int(best_trial),
            "objective": "minimize validation normalized-GED similarity MSE",
            "validation_mse": float(validation_mse),
            "validation_mse_std": validation_mse_std,
            "validation_spearman": validation_spearman,
            "test_set_used_for_selection": False,
            "seeds": [int(seed) for seed in seeds],
            "split_seed": int(split_seed),
            "hyperparameters": dict(hyperparameters),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "dataset_fingerprint": dataset_profile.fingerprint,
            "dataset_profile_version": dataset_profile.profile_version,
            "target_kind": dataset_profile.target_kind,
            "target_source": dataset_profile.target_source,
            "trial_checkpoint": _portable_path(trial_checkpoint),
            "final_training": {
                "status": "not_started",
                "test_evaluation": "not_run",
            },
        }
        path = self.path(dataset_profile.dataset_id, model_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, payload)
        return payload

    def load(
        self,
        dataset_id: str,
        model_id: str,
        expected_fingerprint: str | None = None,
        expected_search_space_version: str | None = None,
    ) -> dict[str, Any] | None:
        path = self.path(dataset_id, model_id)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text())
        if expected_fingerprint and payload.get("dataset_fingerprint") != expected_fingerprint:
            return None
        if (
            expected_search_space_version
            and payload.get("search_space_version") != expected_search_space_version
        ):
            return None
        return payload

    def record_final_training(
        self,
        dataset_id: str,
        model_id: str,
        final_training: Mapping[str, Any],
    ) -> dict[str, Any]:
        path = self.path(dataset_id, model_id)
        if not path.is_file():
            raise FileNotFoundError(
                f"No optimized configuration exists for {model_id} on {dataset_id}."
            )
        payload = json.loads(path.read_text())
        payload["final_training"] = dict(final_training)
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_write(path, payload)
        return payload


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def _portable_path(value: str | Path | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None
