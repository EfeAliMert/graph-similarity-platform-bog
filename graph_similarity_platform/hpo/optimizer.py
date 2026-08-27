from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
import subprocess
from typing import Any

from .best_config import BestConfigRegistry
from .budgets import BudgetSpec, get_budget
from .dataset_profile import DatasetProfiler
from .objective import ObjectiveEvaluator, TrialExecutionError, result_from_trial
from .registry import DEFAULT_REGISTRY, SearchSpaceRegistry
from .reproducibility import seed_everything
from .storage import ProgressStore, sqlite_storage_url, stable_study_name
from .trial_runner import TrialRunner
from .types import TrialContext


ROOT = Path(__file__).resolve().parents[2]
PYG_PYTHON = ROOT / ".venvs" / "gnn-pyg" / "bin" / "python"


@dataclass(frozen=True)
class OptimizationRequest:
    dataset_id: str
    model_id: str
    budget: str = "standard"
    seed: int = 379
    split_seed: int | None = None
    trials: int | None = None
    storage_path: Path | None = None
    refresh_profile: bool = False


class HyperparameterOptimizer:
    """Dataset-aware, validation-only HPO orchestration independent of Flask."""

    def __init__(
        self,
        *,
        root: Path = ROOT,
        registry: SearchSpaceRegistry = DEFAULT_REGISTRY,
        profiler: DatasetProfiler | None = None,
        config_registry: BestConfigRegistry | None = None,
        progress_store: ProgressStore | None = None,
        runner: TrialRunner | None = None,
    ) -> None:
        self.root = root
        self.registry = registry
        self.profiler = profiler or DatasetProfiler()
        self.config_registry = config_registry or BestConfigRegistry()
        self.progress_store = progress_store or ProgressStore()
        self.runner = runner or TrialRunner()

    def optimize(self, request: OptimizationRequest) -> dict[str, Any]:
        try:
            import optuna
        except ImportError as exc:
            raise RuntimeError(
                "Optuna is required for HPO. Install the project requirements first."
            ) from exc

        budget = get_budget(request.budget)
        adapter = self.registry.get(request.model_id)
        profile = self.profiler.profile(
            request.dataset_id,
            preprocessing={
                "canonical_node_order": True,
                "target_transform": "model-native normalized GED similarity",
            },
            refresh=request.refresh_profile,
        )
        split_seed = int(
            request.split_seed
            if request.split_seed is not None
            else _default_split_seed(profile, request.seed)
        )
        requested_trials = max(1, int(request.trials or budget.trials))
        resource = budget.resource_for(request.model_id)
        study_name = stable_study_name(
            request.model_id,
            request.dataset_id,
            adapter.search_space_version,
            profile.fingerprint,
        )
        study_dir = self.root / "training_logs" / "hpo" / "studies" / study_name
        study_dir.mkdir(parents=True, exist_ok=True)
        storage_path = request.storage_path or (
            self.root / "training_logs" / "hpo" / "optimization.db"
        )
        storage_url = sqlite_storage_url(storage_path)

        seed_everything(request.seed)
        if request.model_id == "simgnn":
            _prepare_simgnn_dataset(profile, split_seed)

        sampler = optuna.samplers.TPESampler(
            seed=int(request.seed),
            n_startup_trials=min(budget.startup_trials, requested_trials),
        )
        pruner = _build_pruner(optuna, budget, resource)
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )
        _validate_study_identity(study, adapter.search_space_version, profile.fingerprint)
        study.set_user_attr("model_id", request.model_id)
        study.set_user_attr("dataset_id", request.dataset_id)
        study.set_user_attr("dataset_fingerprint", profile.fingerprint)
        study.set_user_attr("search_space_version", adapter.search_space_version)
        study.set_user_attr("split_seed", split_seed)
        study.set_user_attr("test_set_used_for_selection", False)
        study.set_user_attr("objective", "validation normalized-GED similarity MSE")
        study.set_user_attr("dataset_profile", profile.to_dict())

        if not study.trials:
            study.enqueue_trial(adapter.default_config(profile))

        objective = ObjectiveEvaluator(
            root=self.root,
            study_name=study_name,
            study_dir=study_dir,
            adapter=adapter,
            profile=profile,
            seed=request.seed,
            split_seed=split_seed,
            resource=resource,
            requested_trials=requested_trials,
            progress_store=self.progress_store,
            runner=self.runner,
        )
        existing_effective = sum(
            trial.state
            in {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED}
            for trial in study.trials
        )
        remaining = max(0, requested_trials - existing_effective)
        self._write_progress(
            study_name,
            {
                "status": "running",
                "model_id": request.model_id,
                "dataset_id": request.dataset_id,
                "budget": budget.name,
                "requested_trials": requested_trials,
                "completed_trials": _state_count(study, optuna.trial.TrialState.COMPLETE),
                "pruned_trials": _state_count(study, optuna.trial.TrialState.PRUNED),
                "failed_trials": _state_count(study, optuna.trial.TrialState.FAIL),
                "test_set_used_for_selection": False,
            },
        )
        additional_failures = 0
        failure_limit = max(3, requested_trials)
        while remaining > 0 and additional_failures < failure_limit:
            failed_before = _state_count(study, optuna.trial.TrialState.FAIL)
            study.optimize(
                objective,
                n_trials=1,
                timeout=budget.timeout_seconds,
                catch=(TrialExecutionError,),
                callbacks=[
                    lambda current_study, _trial: self._study_callback(
                        current_study,
                        study_name,
                        requested_trials,
                        optuna,
                    )
                ],
                gc_after_trial=True,
            )
            failed_after = _state_count(study, optuna.trial.TrialState.FAIL)
            additional_failures += max(0, failed_after - failed_before)
            effective = sum(
                trial.state
                in {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED}
                for trial in study.trials
            )
            remaining = max(0, requested_trials - effective)

        completed = [
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
            and trial.value is not None
            and math.isfinite(float(trial.value))
        ]
        if not completed:
            self._write_progress(study_name, {"status": "failed"})
            raise RuntimeError("Every HPO trial failed; no configuration was selected.")

        confirmation = self._confirm_top_configs(
            adapter=adapter,
            profile=profile,
            study_name=study_name,
            study_dir=study_dir,
            trials=sorted(completed, key=lambda trial: (float(trial.value), trial.number)),
            budget=budget,
            primary_seed=request.seed,
            split_seed=split_seed,
            resource=resource,
        )
        winner = min(
            confirmation,
            key=lambda item: (
                item["validation_mse_mean"],
                item["validation_mse_std"],
                item["trial_number"],
            ),
        )
        source_trial = next(
            trial for trial in completed if trial.number == winner["trial_number"]
        )
        source_result = result_from_trial(source_trial)
        config_payload = self.config_registry.save(
            dataset_profile=profile,
            model_id=request.model_id,
            search_space_version=adapter.search_space_version,
            study_name=study_name,
            best_trial=winner["trial_number"],
            validation_mse=winner["validation_mse_mean"],
            validation_spearman=winner["validation_spearman_mean"],
            validation_mse_std=winner["validation_mse_std"],
            seeds=winner["seeds"],
            hyperparameters=winner["config"],
            study_storage=str(storage_path),
            split_seed=split_seed,
            trial_checkpoint=(source_result.checkpoint if source_result else None),
        )
        summary = {
            "study_name": study_name,
            "model_id": request.model_id,
            "dataset_id": request.dataset_id,
            "budget": budget.name,
            "strategy": "Optuna TPE with validation-only pruning",
            "objective": "minimum validation normalized-GED similarity MSE",
            "test_set_used_for_selection": False,
            "search_space_version": adapter.search_space_version,
            "dataset_fingerprint": profile.fingerprint,
            "dataset_profile": profile.to_dict(),
            "requested_trials": requested_trials,
            "completed_trials": len(completed),
            "pruned_trials": _state_count(study, optuna.trial.TrialState.PRUNED),
            "failed_trials": _state_count(study, optuna.trial.TrialState.FAIL),
            "best_trial": winner,
            "confirmation": confirmation,
            "best_config_path": str(
                self.config_registry.path(request.dataset_id, request.model_id).relative_to(self.root)
            ),
            "best_config": config_payload,
            "study_storage": str(storage_path.relative_to(self.root)),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "final_test": {
                "status": "not_run",
                "note": "Run final training first; the test set was not used by HPO.",
            },
        }
        summary_path = study_dir / "summary.json"
        _atomic_json(summary_path, summary)
        self._write_progress(
            study_name,
            {
                "status": "completed",
                "completed_trials": len(completed),
                "pruned_trials": summary["pruned_trials"],
                "failed_trials": summary["failed_trials"],
                "best_validation_mse": winner["validation_mse_mean"],
                "best_validation_spearman": winner["validation_spearman_mean"],
                "best_parameters": winner["config"],
                "summary_path": str(summary_path.relative_to(self.root)),
                "test_set_used_for_selection": False,
            },
        )
        return summary

    def _confirm_top_configs(
        self,
        *,
        adapter: Any,
        profile: Any,
        study_name: str,
        study_dir: Path,
        trials: list[Any],
        budget: BudgetSpec,
        primary_seed: int,
        split_seed: int,
        resource: int,
    ) -> list[dict[str, Any]]:
        candidates = trials[: min(budget.confirmation_top_k, len(trials))]
        rows: list[dict[str, Any]] = []
        for candidate_index, trial in enumerate(candidates):
            config = dict(trial.user_attrs.get("resolved_config") or trial.params)
            seed_results: list[dict[str, Any]] = []
            primary_result = result_from_trial(trial)
            for seed_index, seed in enumerate(budget.confirmation_seeds):
                reusable_primary_result = (
                    primary_result is not None
                    and primary_result.validation_spearman is not None
                    and primary_result.validation_mae is not None
                )
                if int(seed) == int(primary_seed) and reusable_primary_result:
                    result = primary_result
                else:
                    self._write_progress(
                        study_name,
                        {
                            "status": "confirming",
                            "confirmation_candidate": candidate_index + 1,
                            "confirmation_candidates": len(candidates),
                            "confirmation_seed": int(seed),
                            "confirmation_run": (
                                candidate_index * len(budget.confirmation_seeds)
                                + seed_index
                                + 1
                            ),
                            "confirmation_runs": (
                                len(candidates) * len(budget.confirmation_seeds)
                            ),
                            "current_step": 0,
                            "resource": int(resource),
                            "current_parameters": config,
                        },
                    )
                    trial_dir = (
                        study_dir
                        / "confirmation"
                        / f"candidate_{candidate_index:02d}"
                        / f"seed_{seed}"
                    )
                    context = TrialContext(
                        root=self.root,
                        dataset_id=profile.dataset_id,
                        model_id=adapter.model_id,
                        seed=int(seed),
                        split_seed=split_seed,
                        resource=resource,
                        trial_number=trial.number,
                        trial_dir=trial_dir,
                        checkpoint=adapter.checkpoint_path(trial_dir),
                        profile=profile,
                    )
                    result = self.runner.run(
                        adapter=adapter,
                        context=context,
                        config=config,
                        progress_callback=lambda update, seed=seed: self._write_progress(
                            study_name,
                            {
                                "status": "confirming",
                                "confirmation_seed": int(seed),
                                "current_parameters": config,
                                **update,
                            },
                        ),
                    )
                if result.status == "completed" and result.validation_mse is not None:
                    seed_results.append(
                        {
                            "seed": int(seed),
                            "validation_mse": float(result.validation_mse),
                            "validation_spearman": result.validation_spearman,
                            "duration_seconds": result.duration_seconds,
                            "peak_memory_mb": result.peak_memory_mb,
                        }
                    )
            if not seed_results:
                continue
            mse_values = [row["validation_mse"] for row in seed_results]
            spearman_values = [
                row["validation_spearman"]
                for row in seed_results
                if row["validation_spearman"] is not None
            ]
            rows.append(
                {
                    "trial_number": trial.number,
                    "config": config,
                    "seeds": [row["seed"] for row in seed_results],
                    "seed_results": seed_results,
                    "validation_mse_mean": float(fmean(mse_values)),
                    "validation_mse_std": float(pstdev(mse_values)) if len(mse_values) > 1 else 0.0,
                    "validation_spearman_mean": (
                        float(fmean(spearman_values)) if spearman_values else None
                    ),
                }
            )
        if not rows:
            raise RuntimeError("Multi-seed confirmation produced no valid result.")
        return rows

    def _study_callback(
        self,
        study: Any,
        study_name: str,
        requested_trials: int,
        optuna: Any,
    ) -> None:
        completed = _state_count(study, optuna.trial.TrialState.COMPLETE)
        pruned = _state_count(study, optuna.trial.TrialState.PRUNED)
        failed = _state_count(study, optuna.trial.TrialState.FAIL)
        best = study.best_trial if completed else None
        self._write_progress(
            study_name,
            {
                "status": "running",
                "requested_trials": requested_trials,
                "completed_trials": completed,
                "pruned_trials": pruned,
                "failed_trials": failed,
                "elapsed_trials": completed + pruned + failed,
                "best_validation_mse": float(best.value) if best else None,
                "best_validation_spearman": (
                    best.user_attrs.get("validation_spearman") if best else None
                ),
                "best_parameters": (
                    best.user_attrs.get("resolved_config") if best else None
                ),
            },
        )

    def _write_progress(self, study_name: str, update: dict[str, Any]) -> None:
        previous = self.progress_store.read(study_name) or {}
        self.progress_store.write(study_name, {**previous, **update})


def _build_pruner(optuna: Any, budget: BudgetSpec, resource: int) -> Any:
    if budget.name in {"standard", "research"} and resource >= 3:
        return optuna.pruners.HyperbandPruner(
            min_resource=max(1, resource // 10),
            max_resource=resource,
            reduction_factor=3,
        )
    return optuna.pruners.MedianPruner(
        n_startup_trials=budget.startup_trials,
        n_warmup_steps=max(0, resource // 4),
    )


def _validate_study_identity(study: Any, version: str, fingerprint: str) -> None:
    recorded_version = study.user_attrs.get("search_space_version")
    recorded_fingerprint = study.user_attrs.get("dataset_fingerprint")
    if recorded_version not in (None, version):
        raise ValueError(
            "Study search-space version mismatch; create a new versioned study."
        )
    if recorded_fingerprint not in (None, fingerprint):
        raise ValueError("Study dataset fingerprint mismatch; refusing unsafe resume.")


def _default_split_seed(profile: Any, seed: int) -> int:
    if profile.split_strategy == "subject_disjoint":
        return int(seed)
    return int(seed) + 1


def _prepare_simgnn_dataset(profile: Any, split_seed: int) -> None:
    train_pairs = min(8000, max(256, profile.train_graph_count * 8))
    validation_pairs = min(1200, max(64, profile.train_graph_count * 2))
    output_root = (
        ROOT
        / "training_logs"
        / "hpo"
        / "prepared"
        / "simgnn"
        / f"{profile.fingerprint[:12]}_split{split_seed}"
    )
    command = [
        str(PYG_PYTHON),
        "scripts/prepare_simgnn_original_dataset.py",
        "--dataset", profile.dataset_id,
        "--output-root", str(output_root),
        "--train-pairs", str(train_pairs),
        "--validation-pairs", str(validation_pairs),
        "--test-pairs", "0",
        "--seed", str(split_seed),
        "--clean",
    ]
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "SimGNN train/validation preparation failed: "
            + completed.stdout[-1200:]
            + completed.stderr[-1200:]
        )


def _state_count(study: Any, state: Any) -> int:
    return sum(trial.state == state for trial in study.trials)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)
