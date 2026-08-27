from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid
from shlex import quote
from pathlib import Path
from typing import Any

from .data import is_uploaded_dataset, list_original_datasets
from .hpo.budgets import load_budgets
from .hpo.service import latest_progress, verified_optimized_config
from .models.real_models import (
    BASE_DIR,
    GRAPHSIM_ENV,
    MODEL_BY_ID,
    MODELS,
    PYG_ENV,
    inspect_model,
)


TRAINING_JOBS: dict[str, dict[str, Any]] = {}
TRAINING_LOG_DIR = BASE_DIR / "training_logs"
TRAINING_LOCK = threading.Lock()


def training_catalog(dataset_id: str | None = None) -> dict[str, Any]:
    jobs_by_id = {
        job["id"]: job
        for job in _load_persisted_jobs()
        if isinstance(job.get("id"), str)
    }
    jobs_by_id.update(
        {
            job_id: _refresh_job(job)
            for job_id, job in TRAINING_JOBS.items()
        }
    )
    jobs = list(jobs_by_id.values())
    return {
        "dataset_id": dataset_id,
        "plans": [_training_plan(model, dataset_id) for model in MODELS],
        "jobs": sorted(
            jobs,
            key=lambda job: float(job.get("started_at", 0)),
            reverse=True,
        )[:50],
    }


def training_job(job_id: str) -> dict[str, Any]:
    """Return one refreshed training job from memory or persisted state."""
    if job_id in TRAINING_JOBS:
        return _refresh_job(TRAINING_JOBS[job_id])
    for job in _load_persisted_jobs():
        if job.get("id") == job_id:
            return job
    raise FileNotFoundError(f"Training job {job_id!r} was not found.")


def running_training_job(
    model_id: str,
    dataset_id: str,
    *,
    mode: str | None = None,
) -> dict[str, Any] | None:
    """Find an active job so automatic runs can resume instead of duplicating it."""
    jobs_by_id = {
        job["id"]: job
        for job in _load_persisted_jobs()
        if isinstance(job.get("id"), str)
    }
    jobs_by_id.update(
        {
            job_id: _refresh_job(job)
            for job_id, job in TRAINING_JOBS.items()
        }
    )
    candidates = [
        job
        for job in jobs_by_id.values()
        if job.get("status") == "running"
        and job.get("model_id") == model_id
        and job.get("dataset_id") == dataset_id
        and (mode is None or job.get("mode") == mode)
    ]
    return max(candidates, key=lambda job: float(job.get("started_at", 0))) if candidates else None


def start_training(
    model_id: str,
    dataset_id: str,
    epochs: int = 1,
    batch_size: int = 32,
    seed: int = 379,
    optimize: bool = False,
    trials: int = 6,
    budget_mode: str = "standard",
) -> dict[str, Any]:
    model = MODEL_BY_ID.get(model_id)
    if model is None:
        raise ValueError(f"Unknown model: {model_id}")

    plan = _training_plan(
        model,
        dataset_id,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        optimize=optimize,
        trials=trials,
        budget_mode=budget_mode,
    )
    if not plan["can_start"]:
        raise ValueError(plan["detail"])

    with TRAINING_LOCK:
        current = running_training_job(model_id, dataset_id)
        if current is not None:
            raise ValueError(
                f"Training is already running for {model['name']} on {dataset_id}."
            )

        TRAINING_LOG_DIR.mkdir(parents=True, exist_ok=True)
        job_id = uuid.uuid4().hex[:12]
        log_path = TRAINING_LOG_DIR / f"{job_id}.log"
        handle = log_path.open("w")
        try:
            process = subprocess.Popen(
                ["bash", "-lc", plan["command"]],
                cwd=str(BASE_DIR),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        finally:
            handle.close()
        job = {
            "id": job_id,
            "model_id": model_id,
            "model_name": model["name"],
            "dataset_id": dataset_id,
            "status": "running",
            "command": plan["command"],
            "target": plan["target"],
            "seed": plan.get("seed", seed),
            "mode": "optimization" if optimize else "single",
            "trials": plan.get("trials", 1),
            "budget_mode": plan.get("budget_mode"),
            "target_source": plan.get("target_source"),
            "validation_protocol": plan.get("validation_protocol"),
            "log_path": str(log_path.relative_to(BASE_DIR)),
            "started_at": time.time(),
            "pid": process.pid if isinstance(getattr(process, "pid", None), int) else None,
            "_process": process,
        }
        TRAINING_JOBS[job_id] = job
        _persist_job(job)
    return _refresh_job(job)


def _training_plan(
    model: dict[str, Any],
    dataset_id: str | None,
    epochs: int = 1,
    batch_size: int = 32,
    seed: int = 379,
    optimize: bool = False,
    trials: int = 6,
    budget_mode: str = "standard",
) -> dict[str, Any]:
    status = inspect_model(model, dataset_id=dataset_id)
    epochs = max(1, min(int(epochs or 1), 200))
    batch_size = max(1, min(int(batch_size or 32), 256))
    seed = max(0, min(int(seed if seed is not None else 379), 2_147_483_647))
    trials = max(1, min(int(trials or 6), 200))
    budgets = load_budgets()
    budget_mode = str(budget_mode or "standard").lower()
    if budget_mode not in budgets:
        budget_mode = "standard"
    base = {
        "id": model["id"],
        "name": model["name"],
        "dataset_id": dataset_id,
        "status": status["status"],
        "status_label": status["status_label"],
        "can_start": False,
        "command": "",
        "target": "",
        "detail": "",
        "setup": model["setup"],
        "seed": seed,
        "mode": "optimization" if optimize else "single",
        "trials": trials if optimize else 1,
        "budget_mode": budget_mode if optimize else None,
        "hpo_supported": True,
        "hpo_objective": "minimum validation similarity MSE",
        "validation_protocol": (
            "Deterministic pair holdout with canonical unordered pair keys; "
            "training/validation overlap is zero."
        ),
    }

    if not dataset_id:
        return {**base, "detail": "Select a dataset before training."}
    uploaded = is_uploaded_dataset(dataset_id)
    dataset_info = next(
        (dataset for dataset in list_original_datasets() if dataset["id"] == dataset_id),
        None,
    )
    target_kind = (
        dataset_info.get("ground_truth_kind", "structural_proxy")
        if dataset_info
        else "structural_proxy"
    )
    if uploaded and dataset_info and not dataset_info.get("training_ready"):
        target_kind = "structural_proxy"
    if target_kind == "exact":
        target_detail = " Target: exact registered A* GED benchmark."
        base["target_source"] = "exact A* GED"
    elif target_kind == "approximate_benchmark":
        target_detail = (
            " Target: published approximate GED benchmark formed from the "
            "minimum Beam/Hungarian/VJ upper bound; not exact GED."
        )
        base["target_source"] = "approximate GED benchmark upper bound"
    elif target_kind == "unverified_ged":
        target_detail = (
            " Target: user-provided GED reference; exactness and solver provenance "
            "have not been independently verified."
        )
        base["target_source"] = "user-provided unverified GED"
    else:
        target_detail = " Target: derived structural GED proxy; not a GED benchmark."
        base["target_source"] = "structural GED proxy"
    if dataset_info and dataset_info.get("split_strategy") == "subject_disjoint":
        base["validation_protocol"] = (
            "Subject-disjoint graph split before pair generation; graph and "
            "canonical pair overlap are both zero."
        )
    if not uploaded and dataset_id not in model.get("datasets", []):
        return {**base, "detail": "This paper architecture is not registered for the selected dataset."}
    if status.get("missing_runtime"):
        return {**base, "detail": f"Training runtime is missing: {model.get('python')}."}
    if status.get("missing_requirements"):
        return {**base, "detail": f"Training dependencies are missing: {', '.join(status['missing_requirements'])}."}
    if status.get("status") in {"missing", "repo_incomplete"}:
        return {
            **base,
            "detail": (
                f"Model source is not ready for training: {status.get('status_label', 'source missing')}. "
                f"{status.get('detail', '')}"
            ).strip(),
        }

    optimized_record = _verified_config(dataset_id, model["id"])
    optimized_parameters = (
        dict(optimized_record.get("hyperparameters") or {})
        if optimized_record
        else {}
    )
    base["optimized_config_available"] = bool(optimized_parameters)
    base["optimized_parameters"] = optimized_parameters
    base["optimized_config_path"] = (
        f"configs/optimized/{dataset_id}/{model['id']}.json"
        if optimized_parameters
        else None
    )
    optimized_detail = (
        " Uses the fingerprint-verified optimized configuration."
        if optimized_parameters
        else " Uses the model defaults; no compatible optimized configuration is registered."
    )

    if optimize:
        target = _checkpoint_target(model["id"], dataset_id)
        if target is None:
            return {**base, "detail": "Hyperparameter optimization is not wired for this model."}
        command = (
            f"{quote(PYG_ENV)} scripts/optimize.py "
            f"--model {quote(model['id'])} --dataset {quote(dataset_id)} "
            f"--trials {trials} --budget {quote(budget_mode)} --seed {seed}"
        )
        return {
            **base,
            "can_start": True,
            "command": command,
            "target": target,
            "detail": (
                f"Runs up to {trials} Optuna TPE trials using the {budget_mode} budget "
                "on one fixed validation split. Poor trials may be pruned, the best "
                "configuration is stored separately, and the test split is never used "
                f"for selection.{target_detail}"
            ),
        }

    if model["id"] == "simgnn" and (
        dataset_id in model.get("datasets", []) or uploaded
    ):
        if dataset_info is None:
            return {**base, "detail": "Dataset metadata could not be loaded."}
        train_graphs = max(1, int(dataset_info["train_graphs"]))
        test_graphs = max(1, int(dataset_info["test_graphs"]))
        validation_pairs = min(1200, max(1, train_graphs * train_graphs // 5))
        train_pairs = min(
            8000,
            max(1, train_graphs * train_graphs - validation_pairs),
        )
        test_pairs = min(1200, max(1, test_graphs * train_graphs))
        target = f"Models&Datasets/SimGNN-v_00001/checkpoints/simgnn_{dataset_id}.pt"
        effective_batch = int(optimized_parameters.get("batch_size", batch_size))
        optimized_flags = _optimized_cli_flags(model["id"], optimized_parameters)
        command = (
            f"{quote(PYG_ENV)} scripts/prepare_simgnn_original_dataset.py --dataset {dataset_id} "
            f"--train-pairs {train_pairs} --validation-pairs {validation_pairs} "
            f"--test-pairs {test_pairs} --seed {seed} --clean && "
            f"cd {quote('Models&Datasets/SimGNN-v_00001')} && "
            f"../../{quote(PYG_ENV)} src/main.py --training-graphs original_datasets/{dataset_id}/train/ "
            f"--validation-graphs original_datasets/{dataset_id}/validation/ "
            f"--testing-graphs original_datasets/{dataset_id}/test/ --epochs {max(epochs, 3)} --batch-size {effective_batch} "
            f"{optimized_flags} "
            f"--save-path checkpoints/simgnn_{dataset_id}.pt"
        )
        return {
            **base,
            "can_start": True,
            "command": command,
            "target": target,
            "detail": (
                "Trains SimGNN on graph-covering normalized-GED strata and restores "
                f"the best validation epoch.{optimized_detail}{target_detail}"
            ),
        }

    if model["id"] == "segmn" and (dataset_id in model.get("datasets", []) or uploaded):
        minimum_steps = (
            5000
            if target_kind in {"exact", "approximate_benchmark"}
            else 1500
        )
        target = (
            f"Models&Datasets/SEGMN-main/checkpoints/{dataset_id}/"
            f"segmn_{dataset_id}_best.pt"
        )
        effective_batch = min(int(optimized_parameters.get("batch_size", batch_size)), 8)
        optimized_flags = _optimized_cli_flags(model["id"], optimized_parameters)
        command = (
            f"{quote(PYG_ENV)} scripts/train_segmn_universal.py --dataset {dataset_id} "
            f"--checkpoint {quote(target)} --steps {max(epochs * 1000, minimum_steps)} "
            f"--batch-size {effective_batch} {optimized_flags} --seed {seed}"
        )
        return {
            **base,
            "can_start": True,
            "command": command,
            "target": target,
            "detail": (
                "Trains the real SEGMNNet with dataset features, the original AIDS "
                f"architecture profile, and validation-based checkpoint selection.{optimized_detail}{target_detail}"
            ),
        }

    if model["id"] == "graph-fusion" and (dataset_id in model.get("datasets", []) or uploaded):
        minimum_steps = (
            1000
            if target_kind in {"exact", "approximate_benchmark"}
            else 750
        )
        target = f"Models&Datasets/GFM-code/checkpoints/gfm_{dataset_id}.pt"
        effective_batch = int(optimized_parameters.get("batch_size", batch_size))
        optimized_flags = _optimized_cli_flags(model["id"], optimized_parameters)
        command = (
            f"{quote(PYG_ENV)} scripts/train_gfm_smoke.py --dataset {dataset_id} --checkpoint {quote(target)} "
            f"--steps {max(epochs * 100, minimum_steps)} --batch-size {effective_batch} "
            f"{optimized_flags} --seed {seed}"
        )
        return {
            **base,
            "can_start": True,
            "command": command,
            "target": target,
            "detail": (
                "Trains GFM with balanced normalized-GED batches and "
                f"validation-based checkpoint selection.{optimized_detail}{target_detail}"
            ),
        }

    if model["id"] == "graph2region" and (dataset_id in model.get("datasets", []) or uploaded):
        minimum_steps = (
            3000
            if target_kind in {"exact", "approximate_benchmark"}
            else 1500
        )
        target = (
            f"Models&Datasets/Graph2Region-main/checkpoints/{dataset_id}/"
            f"g2r_{dataset_id}_best.pt"
        )
        effective_batch = min(int(optimized_parameters.get("batch_size", batch_size)), 16)
        optimized_flags = _optimized_cli_flags(model["id"], optimized_parameters)
        command = (
            f"{quote(PYG_ENV)} scripts/train_graph2region_universal.py "
            f"--dataset {dataset_id} --checkpoint {quote(target)} "
            f"--steps {max(epochs * 200, minimum_steps)} --batch-size {effective_batch} "
            f"{optimized_flags} --seed {seed}"
        )
        return {
            **base,
            "can_start": True,
            "command": command,
            "target": target,
            "detail": (
                "Trains the real Graph2Region G2R architecture on registered local "
                f"graph pairs.{optimized_detail}{target_detail}"
            ),
        }

    if model["id"] == "multiscale-set" and (dataset_id in model.get("datasets", []) or uploaded):
        target = (
            "Models&Datasets/GraphSim-master/checkpoints/"
            f"{dataset_id}/graphsim.ckpt"
        )
        effective_batch = min(int(optimized_parameters.get("batch_size", batch_size)), 16)
        optimized_flags = _optimized_cli_flags(model["id"], optimized_parameters)
        command = (
            f"{quote(GRAPHSIM_ENV)} scripts/train_graphsim_compat.py "
            f"--dataset {dataset_id} --checkpoint {quote(target)} "
            f"--steps {max(epochs * 20, 500)} --batch-size {effective_batch} "
            f"{optimized_flags} --seed {seed}"
        )
        return {
            **base,
            "can_start": True,
            "command": command,
            "target": target,
            "detail": (
                "Trains the original GraphSim GCN, multi-scale matching, and CNN graph "
                "with stratified validation, best-checkpoint selection, and a "
                "validation-only isotonic output calibrator."
                f"{optimized_detail}{target_detail}"
            ),
        }

    return {**base, "detail": "No local training adapter is wired for this dataset yet."}


def _verified_config(dataset_id: str, model_id: str) -> dict[str, Any] | None:
    try:
        return verified_optimized_config(dataset_id, model_id)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return None


def _optimized_cli_flags(model_id: str, config: dict[str, Any]) -> str:
    if not config:
        return ""
    bindings = {
        "simgnn": {
            "learning_rate": "--learning-rate",
            "weight_decay": "--weight-decay",
            "dropout": "--dropout",
            "filters_1": "--filters-1",
            "filters_2": "--filters-2",
            "filters_3": "--filters-3",
            "tensor_neurons": "--tensor-neurons",
            "bottle_neck_neurons": "--bottle-neck-neurons",
            "bins": "--bins",
        },
        "multiscale-set": {
            "learning_rate": "--learning-rate",
            "patience": "--patience",
            "zero_fraction": "--zero-fraction",
            "identity_fraction": "--identity-fraction",
        },
        "segmn": {
            "learning_rate": "--learning-rate",
            "identity_probability": "--identity-probability",
            "node_cap": "--node-cap",
            "edge_cap": "--edge-cap",
        },
        "graph-fusion": {
            "learning_rate": "--learning-rate",
            "patience": "--patience",
            "identity_probability": "--identity-probability",
        },
        "graph2region": {
            "learning_rate": "--learning-rate",
            "identity_probability": "--identity-probability",
        },
    }
    flags = [
        f"{flag} {quote(str(config[key]))}"
        for key, flag in bindings.get(model_id, {}).items()
        if key in config
    ]
    if model_id == "simgnn" and config.get("histogram"):
        flags.append("--histogram")
    return " ".join(flags)


def _checkpoint_target(model_id: str, dataset_id: str) -> str | None:
    targets = {
        "simgnn": f"Models&Datasets/SimGNN-v_00001/checkpoints/simgnn_{dataset_id}.pt",
        "multiscale-set": f"Models&Datasets/GraphSim-master/checkpoints/{dataset_id}/graphsim.ckpt",
        "segmn": f"Models&Datasets/SEGMN-main/checkpoints/{dataset_id}/segmn_{dataset_id}_best.pt",
        "graph-fusion": f"Models&Datasets/GFM-code/checkpoints/gfm_{dataset_id}.pt",
        "graph2region": f"Models&Datasets/Graph2Region-main/checkpoints/{dataset_id}/g2r_{dataset_id}_best.pt",
    }
    return targets.get(model_id)


def _refresh_job(job: dict[str, Any]) -> dict[str, Any]:
    process = job.get("_process")
    if process is not None and job["status"] == "running":
        return_code = process.poll()
        if return_code is not None:
            job["status"] = "completed" if return_code == 0 else "failed"
            job["return_code"] = return_code
            job["finished_at"] = time.time()
            _persist_job(job)
    public = {key: value for key, value in job.items() if key != "_process"}
    public["log_tail"] = _log_tail(BASE_DIR / job["log_path"])
    if public.get("mode") == "optimization":
        public["optimization_result"] = _parse_optimization_result(public["log_tail"])
        public["hpo_progress"] = latest_progress(
            public.get("dataset_id", ""),
            public.get("model_id", ""),
            started_after=public.get("started_at"),
        )
    return public


def _persist_job(job: dict[str, Any]) -> None:
    job_dir = TRAINING_LOG_DIR / "jobs"
    job_dir.mkdir(parents=True, exist_ok=True)
    public = {
        key: value
        for key, value in job.items()
        if key != "_process"
    }
    (job_dir / f"{job['id']}.json").write_text(
        json.dumps(public, indent=2, sort_keys=True)
    )


def _load_persisted_jobs() -> list[dict[str, Any]]:
    job_dir = TRAINING_LOG_DIR / "jobs"
    if not job_dir.exists():
        return []
    jobs = []
    for path in job_dir.glob("*.json"):
        try:
            job = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        log_path = job.get("log_path")
        job["log_tail"] = (
            _log_tail(BASE_DIR / log_path)
            if isinstance(log_path, str) and log_path
            else ""
        )
        if (
            job.get("status") == "running"
            and job.get("id") not in TRAINING_JOBS
            and not _pid_is_running(job.get("pid"))
        ):
            optimization_result = (
                _parse_optimization_result(job["log_tail"])
                if job.get("mode") == "optimization"
                else None
            )
            job["status"] = "completed" if optimization_result else "interrupted"
            job["return_code"] = 0 if optimization_result else None
            job["finished_at"] = job.get("finished_at") or time.time()
            path.write_text(
                json.dumps(
                    {key: value for key, value in job.items() if key != "log_tail"},
                    indent=2,
                    sort_keys=True,
                )
            )
        if job.get("mode") == "optimization":
            job["optimization_result"] = _parse_optimization_result(job["log_tail"])
            job["hpo_progress"] = latest_progress(
                job.get("dataset_id", ""),
                job.get("model_id", ""),
                started_after=job.get("started_at"),
            )
        jobs.append(job)
    return jobs


def _pid_is_running(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except PermissionError:
        # EPERM means the process exists but this sandbox cannot signal it.
        return True
    except (OSError, ValueError):
        return False
    return True


def _log_tail(path: Path, max_chars: int = 4000) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(errors="replace")
    return text[-max_chars:]


def _parse_optimization_result(log_text: str) -> dict[str, Any] | None:
    for line in reversed(log_text.splitlines()):
        if not line.startswith("HPO_RESULT="):
            continue
        try:
            payload = json.loads(line.removeprefix("HPO_RESULT="))
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None
    return None
