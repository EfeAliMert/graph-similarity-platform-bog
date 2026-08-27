from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venvs" / "gnn-pyg" / "bin" / "python"
REPORT_DIR = ROOT / "reports" / "hpo"

NON_AIDS_DATASETS = (
    "linux",
    "imdbmulti",
    "ptc",
    "mutag",
    "proteins",
    "enzymes",
)
DEFAULT_MODELS = (
    "simgnn",
    "multiscale-set",
    "segmn",
    "graph-fusion",
    "graph2region",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a dataset/model HPO matrix serially for datasets besides AIDS700nef."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(NON_AIDS_DATASETS),
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--budget", default="quick")
    parser.add_argument("--seed", type=int, default=379)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    python = str(PYTHON if PYTHON.exists() else Path(sys.executable))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jobs: list[dict[str, Any]] = []

    for dataset_id in args.datasets:
        for model_id in args.models:
            job = {
                "dataset_id": dataset_id,
                "model_id": model_id,
                "budget": args.budget,
                "status": "planned",
            }
            if args.skip_existing and existing_optimized_config(dataset_id, model_id):
                job["status"] = "skipped"
                job["detail"] = "Compatible optimized configuration already exists."
                jobs.append(job)
                print(f"SKIP {model_id} {dataset_id}")
                continue
            command = [
                python,
                "scripts/optimize.py",
                "--dataset",
                dataset_id,
                "--model",
                model_id,
                "--budget",
                args.budget,
                "--seed",
                str(args.seed),
            ]
            job["command"] = _display_command(command)
            print("RUN", " ".join(command), flush=True)
            completed = subprocess.run(command, cwd=str(ROOT), check=False)
            job["return_code"] = completed.returncode
            job["status"] = "completed" if completed.returncode == 0 else "failed"
            jobs.append(job)
            print(f"{job['status'].upper()} {model_id} {dataset_id}", flush=True)
            if completed.returncode and not args.continue_on_error:
                _write_report(run_id, args, jobs)
                return completed.returncode

    report_path = _write_report(run_id, args, jobs)
    print(report_path.relative_to(ROOT), flush=True)
    failed = [job for job in jobs if job["status"] == "failed"]
    return 1 if failed else 0


def existing_optimized_config(dataset_id: str, model_id: str) -> dict[str, Any] | None:
    try:
        from graph_similarity_platform.hpo.service import verified_optimized_config
    except Exception:
        return None
    try:
        return verified_optimized_config(dataset_id, model_id)
    except Exception:
        return None


def _display_command(command: list[str]) -> list[str]:
    """Store a repository-relative command while executing with absolute paths."""
    if not command:
        return []
    try:
        executable_path = Path(command[0])
        if not executable_path.is_absolute():
            executable_path = ROOT / executable_path
        executable = str(executable_path.relative_to(ROOT))
    except ValueError:
        executable = command[0]
    return [executable, *command[1:]]


def _write_report(
    run_id: str,
    args: argparse.Namespace,
    jobs: list[dict[str, Any]],
) -> Path:
    payload = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "budget": args.budget,
        "seed": args.seed,
        "datasets": list(args.datasets),
        "models": list(args.models),
        "jobs": jobs,
        "completed": sum(job["status"] == "completed" for job in jobs),
        "skipped": sum(job["status"] == "skipped" for job in jobs),
        "failed": sum(job["status"] == "failed" for job in jobs),
    }
    path = REPORT_DIR / f"matrix_{run_id}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


if __name__ == "__main__":
    raise SystemExit(main())
