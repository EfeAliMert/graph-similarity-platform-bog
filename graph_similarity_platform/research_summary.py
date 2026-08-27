from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
from typing import Any

from .models.real_models import BASE_DIR, MODEL_BY_ID


SUMMARY_DIR = BASE_DIR / "reports" / "research_matrices"
MATRIX_DIR = BASE_DIR / "training_logs" / "research_matrix"
METRICS = (
    "mse_similarity_x1e3",
    "mae_ged",
    "rmse_ged",
    "spearman_ged",
    "kendall_ged",
    "precision_at_10",
    "ndcg_at_k",
    "latency_p50_ms",
    "latency_p95_ms",
    "throughput_pairs_per_second",
    "peak_rss_mb",
)
RUN_EVIDENCE_FIELDS = (
    "evaluated_samples",
    "mae_ged_ci95",
    "mae_similarity_ci95",
)


def build_research_summary(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for job in manifest.get("jobs", []):
        key = (job.get("dataset_id", ""), job.get("model_id", ""))
        groups.setdefault(key, []).append(job)

    rows = []
    expected_seeds = sorted(int(seed) for seed in manifest.get("seeds", []))
    for (dataset_id, model_id), jobs in sorted(groups.items()):
        run_rows = []
        for job in jobs:
            benchmark = _load_artifact(job.get("benchmark_artifact"))
            model = (
                next(
                    (
                        row
                        for row in benchmark.get("models", [])
                        if row.get("id") == model_id
                    ),
                    None,
                )
                if benchmark
                else None
            )
            run_rows.append(
                {
                    "seed": job.get("seed"),
                    "training_status": job.get("status"),
                    "training_duration_seconds": job.get("duration_seconds"),
                    "target_source": job.get("target_source"),
                    "benchmark_run_id": job.get("benchmark_run_id"),
                    "benchmark_status": model.get("status") if model else None,
                    "metrics": {
                        metric: model.get(metric)
                        for metric in METRICS
                    } if model else {},
                    "evidence": {
                        field: model.get(field)
                        for field in RUN_EVIDENCE_FIELDS
                    } if model else {},
                    "size_generalization": model.get("size_generalization", []) if model else [],
                    "pair_split_verified": model.get("pair_split_verified") if model else None,
                }
            )
        evaluated = [
            run
            for run in run_rows
            if run["benchmark_status"] == "evaluated"
        ]
        completed_seeds = sorted(
            int(run["seed"])
            for run in evaluated
            if isinstance(run.get("seed"), (int, float))
        )
        rows.append(
            {
                "dataset_id": dataset_id,
                "model_id": model_id,
                "model_name": MODEL_BY_ID.get(model_id, {}).get("name", model_id),
                "target_source": next(
                    (job.get("target_source") for job in jobs if job.get("target_source")),
                    None,
                ),
                "reference_kind": next(
                    (job.get("reference_kind") for job in jobs if job.get("reference_kind")),
                    None,
                ),
                "expected_seeds": expected_seeds,
                "evaluated_seeds": completed_seeds,
                "checkpoint_seeds": sorted(
                    {
                        int(seed)
                        for job in jobs
                        for seed in (job.get("checkpoint_seeds") or [])
                        if isinstance(seed, (int, float))
                    }
                ),
                "training_skipped": all(
                    bool(job.get("training_skipped")) for job in jobs
                ),
                "seed_coverage": (
                    round(len(set(completed_seeds)) / len(expected_seeds), 6)
                    if expected_seeds
                    else 0.0
                ),
                "complete": set(completed_seeds) == set(expected_seeds),
                "pair_split_verified": bool(evaluated) and all(
                    run.get("pair_split_verified") for run in evaluated
                ),
                "metrics": {
                    metric: _aggregate(
                        [
                            run["metrics"].get(metric)
                            for run in evaluated
                            if isinstance(run["metrics"].get(metric), (int, float))
                        ]
                    )
                    for metric in METRICS
                },
                "pair_bootstrap_ci95_by_seed": [
                    {
                        "seed": run.get("seed"),
                        "evaluated_samples": run.get("evidence", {}).get("evaluated_samples"),
                        "mae_ged": run.get("evidence", {}).get("mae_ged_ci95"),
                        "mae_similarity": run.get("evidence", {}).get("mae_similarity_ci95"),
                    }
                    for run in evaluated
                ],
                "training_duration_seconds": _aggregate(
                    [
                        run["training_duration_seconds"]
                        for run in run_rows
                        if isinstance(run.get("training_duration_seconds"), (int, float))
                    ]
                ),
                "size_generalization": _aggregate_size_buckets(evaluated),
                "runs": run_rows,
            }
        )
    return {
        "run_id": manifest.get("run_id"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": _display_path(manifest_path),
        "mode": manifest.get("mode", "train-and-evaluate"),
        "expected_seeds": expected_seeds,
        "evaluation_pairs": manifest.get("evaluation_pairs"),
        "matrix_complete": bool(rows) and all(row["complete"] for row in rows),
        "rows": rows,
    }


def write_research_summary(manifest_path: Path) -> dict[str, str]:
    payload = build_research_summary(manifest_path)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    stem = payload.get("run_id") or manifest_path.parent.name
    json_path = SUMMARY_DIR / f"{stem}.json"
    markdown_path = SUMMARY_DIR / f"{stem}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    exact_rows = [
        row for row in payload["rows"] if _reference_group(row) == "exact"
    ]
    approximate_rows = [
        row for row in payload["rows"] if _reference_group(row) == "approximate"
    ]
    other_rows = [
        row
        for row in payload["rows"]
        if _reference_group(row) not in {"exact", "approximate"}
    ]
    lines = [
        f"# Research Matrix {stem}",
        "",
        f"Complete: `{payload['matrix_complete']}`",
        f"Mode: `{payload.get('mode', 'train-and-evaluate')}`",
        f"Evaluation pairs: `{payload.get('evaluation_pairs')}`",
        "",
        "Exact A* GED, approximate solver upper bounds, and proxy targets are "
        "reported in separate tables. `Complete` means every declared evaluation "
        "seed has a held-out artifact; it is not a paper-table reproduction.",
        "",
    ]
    lines.extend(_summary_table("Exact A* GED", exact_rows))
    lines.extend(_summary_table("Approximate GED benchmark", approximate_rows))
    if other_rows:
        lines.extend(_summary_table("Other targets", other_rows))
    markdown_path.write_text("\n".join(lines).rstrip() + "\n")
    return {
        "json": _display_path(json_path),
        "markdown": _display_path(markdown_path),
    }


def _summary_table(title: str, rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    lines = [
        f"## {title}",
        "",
        "| Dataset | Model | Eval seeds | Checkpoint seeds | MSE x1e3 mean +/- std | Spearman mean +/- std | NDCG@k mean +/- std | Split verified |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        checkpoint_seeds = ",".join(str(seed) for seed in row.get("checkpoint_seeds") or []) or "-"
        lines.append(
            "| {dataset} | {model} | {done}/{expected} | {ckpts} | {mse} | {rho} | {ndcg} | {split} |".format(
                dataset=row["dataset_id"],
                model=row["model_name"],
                done=len(row["evaluated_seeds"]),
                expected=len(row["expected_seeds"]),
                ckpts=checkpoint_seeds,
                mse=_mean_std(row["metrics"]["mse_similarity_x1e3"]),
                rho=_mean_std(row["metrics"]["spearman_ged"]),
                ndcg=_mean_std(row["metrics"]["ndcg_at_k"]),
                split=row["pair_split_verified"],
            )
        )
    lines.append("")
    return lines


def _reference_group(row: dict[str, Any]) -> str:
    kind = str(row.get("reference_kind") or "")
    source = str(row.get("target_source") or "").lower()
    if kind == "exact" or source.startswith("exact"):
        return "exact"
    if kind == "approximate_benchmark" or "approximate" in source:
        return "approximate"
    return "other"


def finalize_matrix_artifacts(manifest_path: Path) -> dict[str, int]:
    manifest = json.loads(manifest_path.read_text())
    written = 0
    skipped = 0
    for job in manifest.get("jobs", []):
        snapshots = job.get("snapshots") or []
        if not snapshots:
            skipped += 1
            continue
        benchmark = _load_artifact(job.get("benchmark_artifact"))
        model = (
            next(
                (
                    row
                    for row in benchmark.get("models", [])
                    if row.get("id") == job.get("model_id")
                ),
                None,
            )
            if benchmark
            else None
        )
        first_snapshot = BASE_DIR / snapshots[0]["path"]
        sidecar = first_snapshot.parent / "run_metadata.json"
        sample = (
            next(
                (
                    row
                    for row in (model or {}).get("samples", [])
                    if row.get("status") == "executed"
                ),
                {},
            )
            if model
            else {}
        )
        payload = {
            "matrix_run_id": manifest.get("run_id"),
            "dataset_id": job.get("dataset_id"),
            "model_id": job.get("model_id"),
            "seed": job.get("seed"),
            "budget": manifest.get("budget"),
            "batch_size": manifest.get("batch_size"),
            "target_source": job.get("target_source"),
            "validation_protocol": job.get("validation_protocol"),
            "training_duration_seconds": job.get("duration_seconds"),
            "checkpoint_files": snapshots,
            "benchmark_run_id": job.get("benchmark_run_id"),
            "benchmark_artifact": job.get("benchmark_artifact"),
            "benchmark_status": job.get("benchmark_status"),
            "pair_split": sample.get("pair_split"),
            "checkpoint_seed": sample.get("checkpoint_seed"),
            "metrics": {
                metric: model.get(metric)
                for metric in METRICS
            } if model else {},
        }
        sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True))
        written += 1
    return {"written": written, "skipped": skipped}


def latest_research_summary() -> dict[str, Any] | None:
    if not SUMMARY_DIR.exists():
        return None
    candidates = []
    for path in SUMMARY_DIR.glob("*.json"):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        evaluated_seeds = sum(
            len(row.get("evaluated_seeds") or [])
            for row in payload.get("rows", [])
        )
        candidates.append(
            (
                bool(payload.get("matrix_complete")),
                evaluated_seeds,
                path.stat().st_mtime,
                payload,
            )
        )
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate[:3])[3]


def latest_research_matrix_status() -> dict[str, Any] | None:
    if not MATRIX_DIR.exists():
        return None
    manifest_paths = sorted(
        MATRIX_DIR.glob("*/manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not manifest_paths:
        return None
    manifests = []
    for path in manifest_paths:
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        manifests.append((path, manifest))
    if not manifests:
        return None

    active = next(
        (
            item
            for item in manifests
            if item[1].get("execute")
            and _manifest_is_active(item[1])
        ),
        None,
    )
    if active is not None:
        path, manifest = active
    else:
        preferred_run_id = (latest_research_summary() or {}).get("run_id")
        path, manifest = next(
            (
                item
                for item in manifests
                if item[1].get("run_id") == preferred_run_id
            ),
            manifests[0],
        )
    jobs = manifest.get("jobs", [])
    total_expected = (
        len(manifest.get("datasets", []))
        * len(manifest.get("models", []))
        * len(manifest.get("seeds", []))
    )
    completed = sum(job.get("status") == "completed" for job in jobs)
    failed = sum(job.get("status") == "failed" for job in jobs)
    current = next(
        (
            job
            for job in reversed(jobs)
            if job.get("status") in {"planned", "running"}
        ),
        None,
    )
    finished = (
        len(jobs) == total_expected
        and current is None
        and completed + failed == total_expected
    )
    return {
        "run_id": manifest.get("run_id"),
        "manifest_path": _display_path(path),
        "total_expected": total_expected,
        "jobs_created": len(jobs),
        "completed": completed,
        "failed": failed,
        "finished": finished,
        "current": current,
    }


def _manifest_is_active(manifest: dict[str, Any]) -> bool:
    jobs = manifest.get("jobs", [])
    total_expected = (
        len(manifest.get("datasets", []))
        * len(manifest.get("models", []))
        * len(manifest.get("seeds", []))
    )
    terminal = sum(
        job.get("status") in {"completed", "failed", "blocked"}
        for job in jobs
    )
    return (
        any(job.get("status") == "running" for job in jobs)
        or len(jobs) < total_expected
        or terminal < total_expected
    )


def checkpoint_audit_summary() -> dict[str, Any] | None:
    path = BASE_DIR / "reports" / "checkpoint_audit.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return {
        "generated_at": payload.get("generated_at"),
        "verified": payload.get("verified"),
        "total": payload.get("total"),
        "complete": payload.get("complete"),
        "artifact_path": _display_path(path),
        "unverified": [
            {
                "model_id": row.get("model_id"),
                "dataset_id": row.get("dataset_id"),
            }
            for row in payload.get("rows", [])
            if not row.get("protocol_verified")
        ],
    }


def _load_artifact(path_value: Any) -> dict[str, Any] | None:
    if not isinstance(path_value, str) or not path_value:
        return None
    path = BASE_DIR / path_value
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _aggregate(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 6),
        "std": round(statistics.stdev(values), 6) if len(values) > 1 else None,
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def _aggregate_size_buckets(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        for row in run.get("size_generalization", []):
            buckets.setdefault(row["bucket"], []).append(row)
    return [
        {
            "bucket": bucket,
            "runs": len(rows),
            "mae_ged": _aggregate(
                [float(row["mae_ged"]) for row in rows if isinstance(row.get("mae_ged"), (int, float))]
            ),
            "mse_similarity_x1e3": _aggregate(
                [
                    float(row["mse_similarity_x1e3"])
                    for row in rows
                    if isinstance(row.get("mse_similarity_x1e3"), (int, float))
                ]
            ),
        }
        for bucket, rows in sorted(buckets.items())
    ]


def _mean_std(row: dict[str, Any]) -> str:
    if row.get("mean") is None:
        return "-"
    if row.get("std") is None:
        return f"{row['mean']} (std N/A; n={row.get('count', 1)})"
    return f"{row['mean']} +/- {row['std']}"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)
