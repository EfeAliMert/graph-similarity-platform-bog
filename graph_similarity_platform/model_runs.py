from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import threading
import time
import uuid
from typing import Any

from scripts.checkpoint_provenance import checkpoint_fingerprint

from .data import original_pair_matches_graphs, pair_ground_truth
from .graph_utils import GraphData
from .hpo.best_config import BestConfigRegistry
from .hpo.service import verified_optimized_config
from .models.real_models import BASE_DIR, MODEL_BY_ID, inspect_model, run_models
from .training import (
    _checkpoint_target,
    running_training_job,
    start_training,
    training_job,
)


MODEL_RUN_JOBS: dict[str, dict[str, Any]] = {}
MODEL_RUN_LOCK = threading.Lock()
MODEL_RUN_DIR = BASE_DIR / "training_logs" / "model_runs" / "jobs"
HPO_MODES: dict[str, dict[str, Any]] = {
    "checkpoint": {
        "label": "Checkpoint only",
        "budget": None,
        "trials": 0,
    },
    "quick": {
        "label": "Quick HPO",
        "budget": "smoke",
        "trials": 2,
    },
    "balanced": {
        "label": "Balanced HPO",
        "budget": "standard",
        "trials": 24,
    },
    "research": {
        "label": "Research HPO",
        "budget": "research",
        "trials": 50,
    },
}
DEFAULT_HPO_MODE = "quick"
AUTOMATIC_SEED = 379
FINAL_TRAINING_BUDGET = 5
FINAL_BATCH_SIZE = 32
POLL_SECONDS = 1.0


def start_model_run(
    *,
    dataset_id: str,
    model_ids: list[str],
    left: GraphData,
    right: GraphData,
    meta: dict[str, Any] | None = None,
    hpo_mode: str = DEFAULT_HPO_MODE,
) -> dict[str, Any]:
    if not dataset_id:
        raise ValueError("Select a dataset before running models.")
    selected = list(dict.fromkeys(str(model_id) for model_id in model_ids))
    unknown = [model_id for model_id in selected if model_id not in MODEL_BY_ID]
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}")
    if not selected:
        raise ValueError("Select at least one model.")
    normalized_hpo_mode = str(hpo_mode or DEFAULT_HPO_MODE).strip().lower()
    if normalized_hpo_mode not in HPO_MODES:
        choices = ", ".join(HPO_MODES)
        raise ValueError(f"Unknown HPO mode {hpo_mode!r}. Choose one of: {choices}.")

    request_key = _model_run_request_key(
        dataset_id,
        selected,
        normalized_hpo_mode,
        left,
        right,
    )
    with MODEL_RUN_LOCK:
        duplicate = next(
            (
                existing
                for existing in MODEL_RUN_JOBS.values()
                if existing.get("status") == "running"
                and existing.get("_request_key") == request_key
            ),
            None,
        )
        if duplicate is not None:
            return _public_job(duplicate)

    job_id = uuid.uuid4().hex[:12]
    hpo_spec = HPO_MODES[normalized_hpo_mode]
    job = {
        "id": job_id,
        "dataset_id": dataset_id,
        "model_ids": selected,
        "hpo_mode": normalized_hpo_mode,
        "hpo_label": hpo_spec["label"],
        "hpo_trials": hpo_spec["trials"],
        "effective_hpo": {},
        "status": "running",
        "started_at": time.time(),
        "completed_models": 0,
        "total_models": len(selected),
        "failures": [],
        "progress": {
            "percent": 0.0,
            "stage": "dataset_check",
            "title": "Checking dataset",
            "detail": "Validating the selected graph pair and model bindings...",
            "current_model": None,
        },
        "_left": left,
        "_right": right,
        "_meta": dict(meta or {}),
        "_request_key": request_key,
    }
    thread = threading.Thread(
        target=_run_model_job,
        args=(job_id,),
        name=f"model-run-{job_id}",
        daemon=True,
    )
    job["_thread"] = thread
    with MODEL_RUN_LOCK:
        MODEL_RUN_JOBS[job_id] = job
        _persist_job(job)
    thread.start()
    return _public_job(job)


def model_run_job(job_id: str) -> dict[str, Any]:
    with MODEL_RUN_LOCK:
        job = MODEL_RUN_JOBS.get(job_id)
        if job is not None:
            return _public_job(job)
    path = MODEL_RUN_DIR / f"{job_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Model run {job_id!r} was not found.")
    payload = json.loads(path.read_text())
    if payload.get("status") == "running":
        payload["status"] = "interrupted"
        payload["error"] = "The server stopped before this automatic run completed."
    return payload


def _run_model_job(job_id: str) -> None:
    job = MODEL_RUN_JOBS[job_id]
    left: GraphData = job["_left"]
    right: GraphData = job["_right"]
    dataset_id = str(job["dataset_id"])
    model_ids = list(job["model_ids"])
    hpo_mode = str(job.get("hpo_mode") or DEFAULT_HPO_MODE)
    meta = dict(job["_meta"])
    results: list[dict[str, Any]] = []

    try:
        for index, model_id in enumerate(model_ids):
            model = MODEL_BY_ID[model_id]
            try:
                _update_progress(
                    job,
                    index,
                    0.03,
                    "dataset_check",
                    "Checking dataset",
                    f"{model['name']}: validating runtime, dataset, and checkpoint state...",
                )
                status = inspect_model(model, dataset_id=dataset_id)
                if status.get("dataset_supported") is False:
                    raise ValueError("The selected dataset is not supported by this model adapter.")
                if status.get("status") in {"missing", "repo_incomplete"}:
                    raise FileNotFoundError(
                        f"{status.get('status_label', 'Model files missing')}: "
                        f"{status.get('detail', 'Required model files were not found.')} "
                        "See docs/ARTIFACT_SETUP.md."
                    )
                if status.get("missing_runtime"):
                    raise RuntimeError(f"Required runtime is missing: {model.get('python')}.")
                if status.get("missing_requirements"):
                    missing = ", ".join(status["missing_requirements"])
                    raise RuntimeError(f"Required packages are missing: {missing}.")

                if hpo_mode == "checkpoint":
                    _update_progress(
                        job,
                        index,
                        0.9,
                        "checkpoint_only",
                        "Loading current checkpoint",
                        f"{model['name']}: skipping HPO and final training for this run...",
                    )
                else:
                    config = _ensure_optimized_config(job, index, model_id, dataset_id)
                    if config is not None:
                        _ensure_final_checkpoint(job, index, model_id, dataset_id, config)

                _update_progress(
                    job,
                    index,
                    0.94,
                    "pair_inference",
                    "Running pair inference",
                    f"{model['name']}: loading the selected checkpoint and scoring the graph pair...",
                )
                model_results = run_models(
                    left,
                    right,
                    [model_id],
                    dataset_id=dataset_id,
                    meta=meta,
                )
                if not model_results:
                    raise RuntimeError("The model adapter returned no result.")
                results.extend(model_results)
            except Exception as exc:  # Keep the remaining selected models running.
                detail = f"Automatic preparation failed: {type(exc).__name__}: {exc}"
                results.append(_failure_result(model_id, dataset_id, detail))
                job["failures"].append({"model_id": model_id, "detail": detail})
            finally:
                job["completed_models"] = index + 1
                _update_progress(
                    job,
                    index,
                    1.0,
                    "model_complete",
                    "Model complete",
                    f"Finished {model['name']}; continuing with the remaining selection...",
                )

        job["result"] = _comparison_payload(
            left,
            right,
            dataset_id=dataset_id,
            meta=meta,
            results=results,
        )
        job["status"] = "completed"
        job["finished_at"] = time.time()
        job["progress"] = {
            "percent": 100.0,
            "stage": "complete",
            "title": "Model run complete",
            "detail": (
                f"Completed {len(model_ids)} model run(s)"
                + (f" with {len(job['failures'])} failure(s)." if job["failures"] else ".")
            ),
            "current_model": None,
        }
        _persist_job(job)
    except Exception as exc:
        job["status"] = "failed"
        job["finished_at"] = time.time()
        job["error"] = f"{type(exc).__name__}: {exc}"
        job["progress"] = {
            "percent": float(job.get("progress", {}).get("percent", 0.0)),
            "stage": "failed",
            "title": "Model run failed",
            "detail": str(exc),
            "current_model": job.get("progress", {}).get("current_model"),
        }
        _persist_job(job)


def _ensure_optimized_config(
    job: dict[str, Any],
    index: int,
    model_id: str,
    dataset_id: str,
) -> dict[str, Any] | None:
    model = MODEL_BY_ID[model_id]
    mode_name = str(job.get("hpo_mode") or DEFAULT_HPO_MODE)
    mode = HPO_MODES[mode_name]
    requested_trials = int(mode["trials"])
    budget = str(mode["budget"])
    active = running_training_job(model_id, dataset_id, mode="optimization")
    config = _verified_config(dataset_id, model_id)
    if config is not None and _config_meets_hpo_mode(config, requested_trials):
        job["effective_hpo"][model_id] = {
            "requested_mode": mode_name,
            "effective_mode": "saved",
            "requested_trials": requested_trials,
            "attached_existing": False,
        }
        _update_progress(
            job,
            index,
            0.66,
            "hpo_reused",
            "Using optimized parameters",
            f"{model['name']}: compatible {mode['label']} result and dataset fingerprint verified.",
        )
        return config

    if (
        mode_name == "quick"
        and active is not None
        and int(active.get("trials") or 0) > requested_trials
        and _registered_checkpoint_exists(model_id, dataset_id)
    ):
        effective_trials = int(active.get("trials") or 0)
        effective_budget = str(active.get("budget_mode") or "active")
        job["effective_hpo"][model_id] = {
            "requested_mode": mode_name,
            "effective_mode": "checkpoint_fallback",
            "requested_trials": requested_trials,
            "background_mode": effective_budget,
            "background_trials": effective_trials,
            "attached_existing": False,
        }
        _update_progress(
            job,
            index,
            0.9,
            "background_hpo",
            "Using current checkpoint",
            (
                f"{model['name']}: active {effective_budget} HPO continues in the background; "
                "Quick mode is scoring with the current dataset checkpoint..."
            ),
        )
        return None

    if active is None:
        _update_progress(
            job,
            index,
            0.08,
            "hpo",
            "Optimizing hyperparameters",
            f"{model['name']}: starting {requested_trials} validation-only trials ({mode['label']})...",
        )
        try:
            active = start_training(
                model_id,
                dataset_id,
                seed=AUTOMATIC_SEED,
                optimize=True,
                trials=requested_trials,
                budget_mode=budget,
            )
        except ValueError:
            active = running_training_job(model_id, dataset_id, mode="optimization")
            if active is None:
                raise

    effective_trials = int(active.get("trials") or requested_trials)
    effective_budget = str(active.get("budget_mode") or budget)
    attached_existing = (
        effective_trials != requested_trials or effective_budget != budget
    )
    job["effective_hpo"][model_id] = {
        "requested_mode": mode_name,
        "effective_mode": effective_budget,
        "requested_trials": effective_trials,
        "attached_existing": attached_existing,
    }
    if attached_existing:
        _update_progress(
            job,
            index,
            0.08,
            "hpo",
            "Continuing active optimization",
            (
                f"{model['name']}: an existing {effective_budget} run is active "
                f"({effective_trials} trials), so it is reused instead of starting a duplicate..."
            ),
        )

    completed = _wait_for_training_job(
        job,
        index,
        active["id"],
        stage="hpo",
        model_name=model["name"],
        progress_start=0.08,
        progress_end=0.66,
    )
    config = _verified_config(dataset_id, model_id)
    if completed.get("status") != "completed" and not (
        config is not None and _config_meets_hpo_mode(config, requested_trials)
    ):
        raise RuntimeError(
            f"Hyperparameter optimization ended with status {completed.get('status', 'unknown')}."
        )
    if config is None or not _config_meets_hpo_mode(config, requested_trials):
        raise RuntimeError(
            f"HPO finished without a fingerprint-compatible {requested_trials}-trial configuration."
        )
    return config


def _ensure_final_checkpoint(
    job: dict[str, Any],
    index: int,
    model_id: str,
    dataset_id: str,
    config: dict[str, Any],
) -> None:
    model = MODEL_BY_ID[model_id]
    target_value = _checkpoint_target(model_id, dataset_id)
    if not target_value:
        raise RuntimeError("No dataset-specific checkpoint target is registered.")
    target = BASE_DIR / target_value
    if _final_checkpoint_ready(config, target):
        _update_progress(
            job,
            index,
            0.9,
            "checkpoint_reused",
            "Loading saved checkpoint",
            f"{model['name']}: optimized final checkpoint fingerprint verified.",
        )
        return

    active = running_training_job(model_id, dataset_id, mode="single")
    if active is None:
        _update_progress(
            job,
            index,
            0.7,
            "final_training",
            "Training final checkpoint",
            f"{model['name']}: fitting the selected configuration on the training split...",
        )
        try:
            active = start_training(
                model_id,
                dataset_id,
                epochs=FINAL_TRAINING_BUDGET,
                batch_size=FINAL_BATCH_SIZE,
                seed=AUTOMATIC_SEED,
                optimize=False,
            )
        except ValueError:
            active = running_training_job(model_id, dataset_id, mode="single")
            if active is None:
                raise

    completed = _wait_for_training_job(
        job,
        index,
        active["id"],
        stage="final_training",
        model_name=model["name"],
        progress_start=0.7,
        progress_end=0.9,
    )
    if completed.get("status") != "completed":
        raise RuntimeError(
            f"Final training ended with status {completed.get('status', 'unknown')}."
        )
    fingerprint = checkpoint_fingerprint(target)
    if fingerprint is None:
        raise RuntimeError(f"Final training completed but checkpoint {target_value} was not created.")

    _update_progress(
        job,
        index,
        0.92,
        "checkpoint_saved",
        "Saving checkpoint record",
        f"{model['name']}: recording optimized parameters and weight fingerprint...",
    )
    sidecar = _write_hpo_sidecar(target, config, completed, fingerprint)
    BestConfigRegistry().record_final_training(
        dataset_id,
        model_id,
        {
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "job_id": completed.get("id"),
            "seed": completed.get("seed", AUTOMATIC_SEED),
            "checkpoint": target_value,
            "checkpoint_fingerprint": fingerprint,
            "hpo_sidecar": str(sidecar.relative_to(BASE_DIR)),
            "test_evaluation": "not_run",
        },
    )


def _wait_for_training_job(
    model_job: dict[str, Any],
    model_index: int,
    training_job_id: str,
    *,
    stage: str,
    model_name: str,
    progress_start: float,
    progress_end: float,
) -> dict[str, Any]:
    while True:
        current = training_job(training_job_id)
        state = str(current.get("status", "unknown"))
        if state != "running":
            return current
        fraction, stage_label = _training_fraction(current, stage)
        local_progress = progress_start + (progress_end - progress_start) * fraction
        _update_progress(
            model_job,
            model_index,
            local_progress,
            stage_label,
            "Confirming top candidates" if stage_label == "seed_confirmation" else (
                "Optimizing hyperparameters" if stage == "hpo" else "Training final checkpoint"
            ),
            _training_detail(current, model_name, stage_label),
        )
        time.sleep(POLL_SECONDS)


def _training_fraction(job: dict[str, Any], stage: str) -> tuple[float, str]:
    if stage == "hpo":
        progress = job.get("hpo_progress") or {}
        requested = max(1, int(progress.get("requested_trials") or job.get("trials") or 1))
        elapsed = progress.get("elapsed_trials")
        if not isinstance(elapsed, (int, float)):
            elapsed = sum(
                int(progress.get(key) or 0)
                for key in ("completed_trials", "pruned_trials", "failed_trials")
            )
        if progress.get("status") == "confirming" or float(elapsed) >= requested:
            run_index = max(1, int(progress.get("confirmation_run") or 1))
            run_total = max(run_index, int(progress.get("confirmation_runs") or run_index))
            current_step = max(0, int(progress.get("current_step") or 0))
            resource = max(1, int(progress.get("resource") or 1))
            within_run = min(1.0, current_step / resource)
            confirmation_fraction = min(
                1.0,
                ((run_index - 1) + within_run) / run_total,
            )
            return 0.9 + 0.09 * confirmation_fraction, "seed_confirmation"
        return min(0.9, max(0.02, float(elapsed) / requested * 0.9)), "hpo"

    command = str(job.get("command", ""))
    log_tail = str(job.get("log_tail", ""))
    step_target = re.search(r"--steps\s+(\d+)", command)
    steps = [int(value) for value in re.findall(r"(?:^|\s)step=(\d+)", log_tail)]
    if step_target and steps:
        return min(0.97, max(0.03, steps[-1] / int(step_target.group(1)))), stage
    percentages = [int(value) for value in re.findall(r"(?:^|\s)(\d{1,3})%\|", log_tail)]
    if percentages:
        return min(0.97, max(0.03, percentages[-1] / 100.0)), stage
    return 0.08, stage


def _training_detail(job: dict[str, Any], model_name: str, stage: str) -> str:
    if stage == "seed_confirmation":
        progress = job.get("hpo_progress") or {}
        seed = progress.get("confirmation_seed")
        candidate = progress.get("confirmation_candidate")
        candidates = progress.get("confirmation_candidates")
        step = progress.get("current_step")
        resource = progress.get("resource")
        parts = [f"{model_name}: confirming top candidates"]
        if candidate is not None and candidates is not None:
            parts.append(f"candidate {candidate}/{candidates}")
        if seed is not None:
            parts.append(f"seed {seed}")
        if step is not None and resource is not None:
            parts.append(f"step {step}/{resource}")
        return " · ".join(parts) + "..."
    if stage == "hpo":
        progress = job.get("hpo_progress") or {}
        elapsed = progress.get("elapsed_trials")
        if elapsed is None:
            elapsed = sum(
                int(progress.get(key) or 0)
                for key in ("completed_trials", "pruned_trials", "failed_trials")
            )
        requested = progress.get("requested_trials") or job.get("trials") or 1
        return f"{model_name}: trial {min(int(elapsed), int(requested))}/{int(requested)}..."
    command = str(job.get("command", ""))
    log_tail = str(job.get("log_tail", ""))
    step_target = re.search(r"--steps\s+(\d+)", command)
    steps = [int(value) for value in re.findall(r"(?:^|\s)step=(\d+)", log_tail)]
    if step_target and steps:
        target = int(step_target.group(1))
        return f"{model_name}: step {min(steps[-1], target)}/{target}..."

    elapsed = max(0, int(time.time() - float(job.get("started_at") or time.time())))
    if elapsed >= 60:
        return f"{model_name}: preparing data and the first training batch ({elapsed // 60} min)..."
    return f"{model_name}: preparing data and the first training batch..."


def _final_checkpoint_ready(config: dict[str, Any], checkpoint: Path) -> bool:
    final = config.get("final_training") or {}
    fingerprint = checkpoint_fingerprint(checkpoint)
    return bool(
        final.get("status") == "completed"
        and fingerprint
        and final.get("checkpoint_fingerprint") == fingerprint
    )


def _write_hpo_sidecar(
    checkpoint: Path,
    config: dict[str, Any],
    training: dict[str, Any],
    fingerprint: str,
) -> Path:
    checkpoint_value = str(checkpoint.relative_to(BASE_DIR))
    payload = {
        "schema_version": "automatic-final-training-v1",
        "study_id": config.get("study_name"),
        "model_id": config.get("model"),
        "dataset_id": config.get("dataset"),
        "strategy": "Optuna TPE with validation-only pruning and multi-seed confirmation",
        "objective": config.get("objective"),
        "test_set_used_for_selection": False,
        "seed": training.get("seed", AUTOMATIC_SEED),
        "split_seed": config.get("split_seed"),
        "completed_trials": _completed_trial_count(config),
        "best_trial": {
            "number": config.get("best_trial"),
            "config": config.get("hyperparameters") or {},
            "validation_mse": config.get("validation_mse"),
            "validation_mse_std": config.get("validation_mse_std"),
            "validation_spearman": config.get("validation_spearman"),
            "checkpoint": config.get("trial_checkpoint"),
        },
        "active_checkpoint": checkpoint_value,
        "active_checkpoint_fingerprint": fingerprint,
        "dataset_fingerprint": config.get("dataset_fingerprint"),
        "search_space_version": config.get("search_space_version"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    sidecar = Path(str(checkpoint) + ".hpo.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(sidecar)
    return sidecar


def _completed_trial_count(config: dict[str, Any]) -> int | None:
    summary = _hpo_summary(config)
    if summary is None:
        return None
    value = summary.get("completed_trials")
    return int(value) if isinstance(value, (int, float)) else None


def _config_meets_hpo_mode(config: dict[str, Any], requested_trials: int) -> bool:
    summary = _hpo_summary(config)
    if summary is None:
        return False
    completed_budget = summary.get("requested_trials")
    return isinstance(completed_budget, (int, float)) and int(completed_budget) >= requested_trials


def _registered_checkpoint_exists(model_id: str, dataset_id: str) -> bool:
    target_value = _checkpoint_target(model_id, dataset_id)
    if not target_value:
        return False
    target = BASE_DIR / target_value
    if target.is_file() and target.stat().st_size > 0:
        return True
    if target.suffix == ".ckpt":
        return any(target.parent.glob(f"{target.name}.*"))
    return False


def _hpo_summary(config: dict[str, Any]) -> dict[str, Any] | None:
    summary_path = (
        BASE_DIR
        / "training_logs"
        / "hpo"
        / "studies"
        / str(config.get("study_name", ""))
        / "summary.json"
    )
    try:
        return json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        pass

    sidecar_value = (config.get("final_training") or {}).get("hpo_sidecar")
    if not sidecar_value:
        return None
    sidecar_path = Path(str(sidecar_value))
    if not sidecar_path.is_absolute():
        sidecar_path = BASE_DIR / sidecar_path
    try:
        sidecar = json.loads(sidecar_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    completed_trials = sidecar.get("completed_trials")
    if not isinstance(completed_trials, (int, float)):
        return None
    return {
        "completed_trials": int(completed_trials),
        "requested_trials": int(completed_trials),
        "source": "checkpoint_hpo_sidecar",
    }


def _model_run_request_key(
    dataset_id: str,
    model_ids: list[str],
    hpo_mode: str,
    left: GraphData,
    right: GraphData,
) -> tuple[Any, ...]:
    def graph_key(graph: GraphData) -> tuple[Any, ...]:
        return (
            tuple(graph.original_nodes),
            tuple(graph.edges),
            tuple(graph.labels),
        )

    return dataset_id, tuple(model_ids), hpo_mode, graph_key(left), graph_key(right)


def _verified_config(dataset_id: str, model_id: str) -> dict[str, Any] | None:
    try:
        return verified_optimized_config(dataset_id, model_id)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return None


def _comparison_payload(
    left: GraphData,
    right: GraphData,
    *,
    dataset_id: str,
    meta: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    input_matches_pair = original_pair_matches_graphs(
        dataset_id,
        meta.get("left_graph"),
        meta.get("right_graph"),
        left,
        right,
    )
    ground_truth = (
        pair_ground_truth(
            dataset_id,
            meta.get("left_graph"),
            meta.get("right_graph"),
            left.node_count,
            right.node_count,
        )
        if input_matches_pair is not False
        else None
    )
    return {
        "results": results,
        "ground_truth": ground_truth,
        "input_matches_dataset_pair": input_matches_pair,
        "stats": {"left": left.summary(), "right": right.summary()},
        "graphs": {"left": left.to_preview(), "right": right.to_preview()},
    }


def _failure_result(model_id: str, dataset_id: str, detail: str) -> dict[str, Any]:
    model = MODEL_BY_ID[model_id]
    inspection = inspect_model(model, dataset_id=dataset_id)
    failure_status = "preparation_failed"
    failure_label = "Preparation failed"
    if inspection.get("status") in {"missing", "repo_incomplete"}:
        failure_status = "model_files_missing"
        failure_label = "Model files missing"
        detail = (
            f"{inspection.get('status_label', 'Model files missing')}. "
            f"{inspection.get('detail', 'Required model files were not found.')} "
            "Install the model artifact bundle described in docs/ARTIFACT_SETUP.md."
        )
    return {
        "id": model_id,
        "name": model["name"],
        "family": model["family"],
        "paper": model["paper"],
        "accent": model["accent"],
        "latency_ms": None,
        "score": None,
        "model_score": None,
        "canonical_similarity": None,
        "comparable_similarity": None,
        "score_transformation": {},
        "distance": None,
        "status": failure_status,
        "status_label": failure_label,
        "detail": detail,
        "command": model["command"],
        "local_path": model.get("local_path") or "not found",
        "python": model.get("python") or "not configured",
        "environment": model.get("environment") or "",
        "implementation_origin": model.get("implementation_origin"),
        "implementation_note": model.get("implementation_note"),
        "architecture_class": model.get("architecture_class"),
        "runtime_architecture_class": None,
        "official_pretrained": bool(model.get("official_pretrained")),
        "checkpoint_note": model.get("checkpoint_note"),
        "selected_checkpoint": None,
        "score_semantics": model.get("score_semantics"),
        "input_binding": model.get("input_binding"),
        "input_matches_dataset_pair": None,
        "setup": model.get("setup") or [],
        "supported_datasets": model.get("datasets") or [],
        "runnable_datasets": model.get("runnable_datasets") or [],
        "dataset_supported": inspection.get("dataset_supported"),
        "dataset_runnable": inspection.get("dataset_runnable"),
        "missing_requirements": inspection.get("missing_requirements") or [],
        "missing_runtime": bool(inspection.get("missing_runtime")),
        "missing_files": inspection.get("missing_files") or [],
        "checkpoints": inspection.get("checkpoints") or [],
        "adapter_metrics": {},
    }


def _update_progress(
    job: dict[str, Any],
    model_index: int,
    local_fraction: float,
    stage: str,
    title: str,
    detail: str,
) -> None:
    total = max(1, int(job["total_models"]))
    percent = ((model_index + min(1.0, max(0.0, local_fraction))) / total) * 100.0
    model_id = job["model_ids"][model_index]
    job["progress"] = {
        "percent": round(percent, 1),
        "stage": stage,
        "title": title,
        "detail": detail,
        "current_model": model_id,
        "model_index": model_index + 1,
        "total_models": total,
    }
    _persist_job(job)


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in job.items() if not key.startswith("_")}


def _persist_job(job: dict[str, Any]) -> None:
    MODEL_RUN_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_RUN_DIR / f"{job['id']}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(_public_job(job), indent=2, sort_keys=True))
    temporary.replace(path)
