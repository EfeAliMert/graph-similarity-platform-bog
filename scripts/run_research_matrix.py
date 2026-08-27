from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graph_similarity_platform.data import list_original_datasets  # noqa: E402
from graph_similarity_platform.evaluation import evaluate_models  # noqa: E402
from graph_similarity_platform.models.real_models import MODEL_BY_ID, MODELS  # noqa: E402
from graph_similarity_platform.research_summary import (  # noqa: E402
    finalize_matrix_artifacts,
    write_research_summary,
)
from graph_similarity_platform.training import _training_plan  # noqa: E402


DEFAULT_SEEDS = [379, 2026, 3407]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan or execute a reproducible multi-model, multi-seed training matrix."
    )
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--models", default="all")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--budget", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--exact-only", action="store_true")
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Include exact and published approximate GED benchmark datasets.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--evaluate-pairs", type=int, default=50)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument(
        "--evaluate-existing",
        action="store_true",
        help=(
            "Evaluate the currently stored checkpoints without retraining. "
            "Pair sampling uses --evaluation-seed; the checkpoint training seed "
            "is recorded from metadata."
        ),
    )
    parser.add_argument("--evaluation-seed", type=int, default=379)
    args = parser.parse_args()

    if args.exact_only and args.benchmark_only:
        parser.error("--exact-only and --benchmark-only are mutually exclusive")
    datasets = select_datasets(
        args.datasets,
        exact_only=args.exact_only,
        benchmark_only=args.benchmark_only,
    )
    models = select_models(args.models)
    seeds = positive_integers(args.seeds, allow_zero=True)
    if not seeds:
        raise ValueError("At least one seed is required.")

    evaluation_seeds = evaluation_seed_values(
        args.evaluate_existing,
        seeds,
        args.evaluation_seed,
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "training_logs" / "research_matrix" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "execute": args.execute,
        "mode": "evaluate-existing" if args.evaluate_existing else "train-and-evaluate",
        "budget": args.budget,
        "batch_size": args.batch_size,
        "datasets": datasets,
        "models": [model["id"] for model in models],
        "seeds": evaluation_seeds,
        "training_seeds_requested": seeds,
        "evaluation_pairs": args.evaluate_pairs,
        "evaluation_seed": args.evaluation_seed,
        "jobs": [],
    }
    manifest_path = run_dir / "manifest.json"

    for dataset_id in datasets:
        for model in models:
            for seed in evaluation_seeds:
                plan = _training_plan(
                    model,
                    dataset_id,
                    epochs=args.budget,
                    batch_size=args.batch_size,
                    seed=seed if not args.evaluate_existing else args.evaluation_seed,
                )
                existing_ready = checkpoint_exists(plan.get("target") or "")
                if args.evaluate_existing:
                    can_run = bool(plan.get("target") and existing_ready)
                    detail = (
                        f"Evaluate stored checkpoint {plan.get('target')}."
                        if can_run
                        else f"No stored checkpoint at {plan.get('target') or 'unknown path'}."
                    )
                else:
                    can_run = bool(plan["can_start"])
                    detail = plan["detail"]
                job = {
                    "dataset_id": dataset_id,
                    "model_id": model["id"],
                    "seed": seed,
                    "command": None if args.evaluate_existing else plan["command"],
                    "target": plan["target"],
                    "target_source": plan.get("target_source"),
                    "validation_protocol": plan.get("validation_protocol"),
                    "status": "planned" if can_run else "blocked",
                    "detail": detail,
                    "existing_checkpoint": existing_ready,
                    "training_skipped": bool(args.evaluate_existing),
                }
                manifest["jobs"].append(job)
                save_manifest(manifest_path, manifest)
                if not args.execute or not can_run:
                    print(
                        f"{job['status'].upper()} {model['id']} {dataset_id} "
                        f"seed={seed}: {job['command'] or job['detail']}"
                    )
                    continue

                started = time.perf_counter()
                log_path = run_dir / f"{model['id']}__{dataset_id}__seed-{seed}.log"
                job["status"] = "running"
                save_manifest(manifest_path, manifest)
                if args.evaluate_existing:
                    completed_code = 0
                    log_path.write_text(
                        f"evaluate-existing checkpoint={plan['target']}\n"
                    )
                else:
                    with log_path.open("w") as log:
                        completed = subprocess.run(
                            ["bash", "-lc", plan["command"]],
                            cwd=ROOT,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            text=True,
                            check=False,
                        )
                    completed_code = completed.returncode
                job["return_code"] = completed_code
                job["log_path"] = display_path(log_path)
                job["status"] = "completed" if completed_code == 0 else "failed"
                if completed_code == 0:
                    if args.evaluate_existing:
                        job["checkpoint_sha256"] = existing_checkpoint_digest(
                            ROOT / plan["target"]
                        )
                    else:
                        job["snapshots"] = snapshot_checkpoint(
                            ROOT / plan["target"],
                            run_dir / "checkpoints" / model["id"] / dataset_id / f"seed-{seed}",
                        )
                    dataset_info = next(
                        dataset
                        for dataset in list_original_datasets()
                        if dataset["id"] == dataset_id
                    )
                    if (
                        not args.skip_evaluation
                        and dataset_info["ground_truth_available"]
                    ):
                        try:
                            benchmark = evaluate_models(
                                dataset_id,
                                [model["id"]],
                                sample_size=max(1, min(args.evaluate_pairs, 200)),
                                scope="train-test",
                                sample_mode="stratified",
                                seed=seed,
                                top_k=min(10, max(1, args.evaluate_pairs)),
                            )
                            job["benchmark_run_id"] = benchmark["run_id"]
                            job["benchmark_artifact"] = benchmark["artifact_path"]
                            job["benchmark_status"] = benchmark["models"][0]["status"]
                            job["checkpoint_seeds"] = benchmark["models"][0].get(
                                "checkpoint_seeds"
                            )
                            job["reference_kind"] = benchmark["protocol"].get(
                                "reference_kind"
                            )
                            job["evaluated_pairs"] = benchmark.get("sample_size")
                        except Exception as exc:
                            job["benchmark_status"] = "failed"
                            job["benchmark_error"] = f"{type(exc).__name__}: {exc}"
                            if not args.continue_on_error:
                                save_manifest(manifest_path, manifest)
                                raise
                job["duration_seconds"] = round(time.perf_counter() - started, 3)
                save_manifest(manifest_path, manifest)
                print(
                    f"{job['status'].upper()} {model['id']} {dataset_id} "
                    f"seed={seed} duration={job['duration_seconds']}s"
                )
                if completed_code and not args.continue_on_error:
                    return completed_code

    manifest["snapshot_sidecars"] = finalize_matrix_artifacts(manifest_path)
    save_manifest(manifest_path, manifest)
    summary_paths = write_research_summary(manifest_path)
    manifest["summary_artifacts"] = summary_paths
    save_manifest(manifest_path, manifest)
    print(display_path(manifest_path))
    print(f"summary_json={summary_paths['json']}")
    print(f"summary_markdown={summary_paths['markdown']}")
    return 0


def select_datasets(
    value: str,
    exact_only: bool,
    benchmark_only: bool = False,
) -> list[str]:
    catalog = list_original_datasets()
    available = {dataset["id"]: dataset for dataset in catalog}
    selected = list(available) if value == "all" else split_values(value)
    unknown = [dataset_id for dataset_id in selected if dataset_id not in available]
    if unknown:
        raise ValueError(f"Unknown dataset: {unknown[0]}")
    if exact_only:
        selected = [
            dataset_id
            for dataset_id in selected
            if available[dataset_id].get("ground_truth_exact")
        ]
    elif benchmark_only:
        selected = [
            dataset_id
            for dataset_id in selected
            if available[dataset_id].get("ground_truth_benchmark")
        ]
    return selected


def evaluation_seed_values(
    evaluate_existing: bool,
    training_seeds: list[int],
    evaluation_seed: int,
) -> list[int]:
    if evaluate_existing:
        return [int(evaluation_seed)]
    return list(training_seeds)


def checkpoint_exists(target: str) -> bool:
    if not target:
        return False
    path = Path(target)
    if not path.is_absolute():
        path = ROOT / path
    return path.is_file() or Path(str(path) + ".index").is_file()


def existing_checkpoint_digest(path: Path) -> str | None:
    if path.is_file():
        return sha256(path)
    index_path = Path(str(path) + ".index")
    if index_path.is_file():
        return sha256(index_path)
    return None


def select_models(value: str) -> list[dict]:
    selected = list(MODEL_BY_ID) if value == "all" else split_values(value)
    unknown = [model_id for model_id in selected if model_id not in MODEL_BY_ID]
    if unknown:
        raise ValueError(f"Unknown model: {unknown[0]}")
    return [MODEL_BY_ID[model_id] for model_id in selected]


def positive_integers(value: str, allow_zero: bool = False) -> list[int]:
    lower = 0 if allow_zero else 1
    return sorted(
        {
            integer
            for integer in (int(item) for item in split_values(value))
            if integer >= lower
        }
    )


def split_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def snapshot_checkpoint(source: Path, destination: Path) -> list[dict]:
    paths = [source] if source.is_file() else sorted(
        path for path in source.parent.glob(f"{source.name}*") if path.is_file()
    )
    destination.mkdir(parents=True, exist_ok=True)
    snapshots = []
    for path in paths:
        target = destination / path.name
        shutil.copy2(path, target)
        snapshots.append(
            {
                "path": display_path(target),
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
            }
        )
    return snapshots


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
