from __future__ import annotations

from pathlib import Path
from typing import Any

from .adapters import ModelHPOAdapter
from .storage import ProgressStore
from .trial_runner import TrialRunner
from .types import DatasetProfile, TrialContext, TrialResult


class TrialExecutionError(RuntimeError):
    pass


class ObjectiveEvaluator:
    def __init__(
        self,
        *,
        root: Path,
        study_name: str,
        study_dir: Path,
        adapter: ModelHPOAdapter,
        profile: DatasetProfile,
        seed: int,
        split_seed: int,
        resource: int,
        requested_trials: int,
        progress_store: ProgressStore,
        runner: TrialRunner | None = None,
    ) -> None:
        self.root = root
        self.study_name = study_name
        self.study_dir = study_dir
        self.adapter = adapter
        self.profile = profile
        self.seed = int(seed)
        self.split_seed = int(split_seed)
        self.resource = int(resource)
        self.requested_trials = int(requested_trials)
        self.progress_store = progress_store
        self.runner = runner or TrialRunner()

    def __call__(self, trial: Any) -> float:
        import optuna

        try:
            config = self.adapter.suggest(trial, self.profile)
        except Exception as exc:
            self._progress(
                {
                    "status": "running",
                    "current_trial": trial.number + 1,
                    "last_error": f"Search-space resolution failed: {exc}",
                }
            )
            raise TrialExecutionError(
                f"Search-space resolution failed: {exc}"
            ) from exc
        trial.set_user_attr("resolved_config", config)
        trial.set_user_attr("dataset_fingerprint", self.profile.fingerprint)
        trial.set_user_attr("search_space_version", self.adapter.search_space_version)
        trial_dir = self.study_dir / "trials" / f"trial_{trial.number:04d}"
        checkpoint = self.adapter.checkpoint_path(trial_dir)
        context = TrialContext(
            root=self.root,
            dataset_id=self.profile.dataset_id,
            model_id=self.adapter.model_id,
            seed=self.seed,
            split_seed=self.split_seed,
            resource=self.resource,
            trial_number=trial.number,
            trial_dir=trial_dir,
            checkpoint=checkpoint,
            profile=self.profile,
        )

        self._progress(
            {
                "status": "running",
                "current_trial": trial.number + 1,
                "requested_trials": self.requested_trials,
                "current_parameters": config,
                "current_validation_mse": None,
            }
        )
        result = self.runner.run(
            adapter=self.adapter,
            context=context,
            config=config,
            trial=trial,
            progress_callback=lambda update: self._progress(
                {
                    "status": "running",
                    "current_trial": trial.number + 1,
                    "requested_trials": self.requested_trials,
                    "current_parameters": config,
                    **update,
                }
            ),
        )
        trial.set_user_attr("trial_result", result.to_dict())
        if result.status == "pruned":
            raise optuna.TrialPruned(
                f"Pruned after validation step {result.best_step}."
            )
        if result.status != "completed" or result.validation_mse is None:
            raise TrialExecutionError(result.exception or "Trial failed.")
        trial.set_user_attr("validation_spearman", result.validation_spearman)
        trial.set_user_attr("duration_seconds", result.duration_seconds)
        trial.set_user_attr("peak_memory_mb", result.peak_memory_mb)
        trial.set_user_attr("checkpoint", result.checkpoint)
        return float(result.validation_mse)

    def _progress(self, update: dict[str, Any]) -> None:
        previous = self.progress_store.read(self.study_name) or {}
        self.progress_store.write(
            self.study_name,
            {
                **previous,
                "study_name": self.study_name,
                "model_id": self.adapter.model_id,
                "dataset_id": self.profile.dataset_id,
                "search_space_version": self.adapter.search_space_version,
                "dataset_fingerprint": self.profile.fingerprint,
                **update,
            },
        )


def result_from_trial(trial: Any) -> TrialResult | None:
    payload = trial.user_attrs.get("trial_result")
    if not isinstance(payload, dict):
        return None
    try:
        return TrialResult(**payload)
    except TypeError:
        return None
