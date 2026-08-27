from __future__ import annotations

import heapq
import json
import math
from pathlib import Path
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations, product
from typing import Any
import uuid

from .data import (
    ground_truth_kind,
    list_original_graphs,
    load_ground_truth_distances,
    load_original_graph_collection,
    load_original_pair,
)
from .graph_utils import GraphData, graph_from_payload
from .models.real_models import BASE_DIR, MODEL_BY_ID, inspect_model, run_models


RETRIEVAL_DIR = BASE_DIR / "training_logs" / "retrieval_ablation"
RERANK_JOBS: dict[str, dict[str, Any]] = {}
RERANK_LOCK = threading.Lock()
ENSEMBLE_METHOD_ID = "ensemble-all"
ENSEMBLE_NAME = "All-model canonical ensemble"


class BestPairSearchError(ValueError):
    def __init__(self, message: str, status_code: int = 400, payload: dict[str, Any] | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload or {}


def _require_exact_ged_dataset(dataset_id: str, operation: str) -> None:
    try:
        reference_kind = ground_truth_kind(dataset_id)
    except ValueError:
        # Unit-level callers may inject an in-memory dataset. Normal API requests
        # still fail in the dataset registry before any result is returned.
        return
    if reference_kind not in {"exact", "approximate_benchmark"}:
        raise BestPairSearchError(
            f"{operation} requires a registered GED benchmark. This dataset contains "
            "only a structural proxy target.",
            status_code=422,
            payload={
                "dataset_id": dataset_id,
                "ground_truth_available": False,
                "structural_proxy_available": True,
                "ground_truth_exact": False,
                "ground_truth_kind": reference_kind,
            },
        )


def start_reranking_ablation(
    dataset_id: str,
    method_id: str,
    budgets: list[int],
    scope: str = "train-test",
    top_k: int = 10,
) -> dict[str, Any]:
    _require_exact_ged_dataset(dataset_id, "GNN reranking ablation")
    if method_id not in {*MODEL_BY_ID, ENSEMBLE_METHOD_ID}:
        raise BestPairSearchError("Select one checkpoint-backed GNN reranker.")
    requested_methods = (
        list(MODEL_BY_ID)
        if method_id == ENSEMBLE_METHOD_ID
        else [method_id]
    )
    if set(_runnable_methods(dataset_id, requested_methods)) != set(requested_methods):
        raise BestPairSearchError(
            "Every selected real model must be runnable for this dataset.",
            status_code=422,
        )
    normalized_budgets = _normalize_budgets(budgets, maximum=100)
    with RERANK_LOCK:
        for job in RERANK_JOBS.values():
            if (
                job["status"] in {"queued", "running"}
                and job["dataset_id"] == dataset_id
                and job["method_id"] == method_id
            ):
                raise BestPairSearchError(
                    "A reranking experiment is already running for this model and dataset.",
                    status_code=409,
                    payload={"job": _public_rerank_job(job)},
                )
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id,
            "dataset_id": dataset_id,
            "method_id": method_id,
            "model_name": _reranker_name(method_id),
            "budgets": normalized_budgets,
            "scope": scope,
            "top_k": int(top_k),
            "status": "queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        RERANK_JOBS[job_id] = job
        _persist_rerank_job(job)
        thread = threading.Thread(
            target=_run_reranking_job,
            args=(job,),
            daemon=True,
            name=f"rerank-{job_id}",
        )
        job["_thread"] = thread
        thread.start()
    return _public_rerank_job(job)


def reranking_job(job_id: str) -> dict[str, Any]:
    if not job_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in job_id
    ):
        raise ValueError("Invalid reranking job id.")
    job = RERANK_JOBS.get(job_id)
    if job is not None:
        return _public_rerank_job(job)
    path = RETRIEVAL_DIR / "jobs" / f"{job_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Reranking job not found: {job_id}")
    persisted = json.loads(path.read_text())
    if persisted.get("status") in {"queued", "running"}:
        persisted["status"] = "interrupted"
        persisted["error"] = "The Flask process restarted before this in-process job finished."
        persisted["finished_at"] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(persisted, indent=2, sort_keys=True))
    return persisted


def _run_reranking_job(job: dict[str, Any]) -> None:
    job["status"] = "running"
    job["started_at"] = datetime.now(timezone.utc).isoformat()
    _persist_rerank_job(job)
    try:
        result = evaluate_reranking_ablation(
            job["dataset_id"],
            job["method_id"],
            job["budgets"],
            scope=job["scope"],
            top_k=job["top_k"],
        )
        job["status"] = "completed"
        job["result"] = result
        job["artifact_path"] = result["artifact_path"]
    except Exception as exc:  # The job API must preserve a terminal diagnostic.
        job["status"] = "failed"
        job["error"] = str(exc)
    finally:
        job["finished_at"] = datetime.now(timezone.utc).isoformat()
        _persist_rerank_job(job)


def _public_rerank_job(job: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in job.items() if key != "_thread"}


def _persist_rerank_job(job: dict[str, Any]) -> None:
    job_dir = RETRIEVAL_DIR / "jobs"
    job_dir.mkdir(parents=True, exist_ok=True)
    public = _public_rerank_job(job)
    (job_dir / f"{job['id']}.json").write_text(
        json.dumps(public, indent=2, sort_keys=True)
    )


def evaluate_prefilter_ablation(
    dataset_id: str,
    budgets: list[int],
    scope: str = "train-test",
    top_k: int = 10,
) -> dict[str, Any]:
    started = time.perf_counter()
    _require_exact_ged_dataset(dataset_id, "Retrieval ablation")
    if scope not in {"train-test", "all"}:
        raise ValueError("Retrieval scope must be 'train-test' or 'all'.")
    records = _load_records(dataset_id)
    record_by_id = {
        int(record["id"]): record
        for record in records
        if str(record.get("id", "")).isdigit()
    }
    train_ids = {
        graph_id for graph_id, record in record_by_id.items() if record["split"] == "train"
    }
    test_ids = {
        graph_id for graph_id, record in record_by_id.items() if record["split"] == "test"
    }
    all_ids = set(record_by_id)
    try:
        distances = load_ground_truth_distances(dataset_id, task="ged")
    except FileNotFoundError as exc:
        raise BestPairSearchError(
            "Retrieval ablation requires a registered GED benchmark.",
            status_code=422,
            payload={"dataset_id": dataset_id, "ground_truth_available": False},
        ) from exc

    candidates = []
    seen: set[tuple[int, int]] = set()
    for (left_id, right_id), ged in distances.items():
        normalized = _normalize_pair_ids(
            left_id,
            right_id,
            scope,
            train_ids,
            test_ids,
            all_ids,
        )
        if normalized is None:
            continue
        left_id, right_id = normalized
        key = tuple(sorted((left_id, right_id)))
        if key in seen or left_id not in record_by_id or right_id not in record_by_id:
            continue
        seen.add(key)
        left = record_by_id[left_id]
        right = record_by_id[right_id]
        candidates.append(
            {
                "key": key,
                "left_graph": left["member"],
                "right_graph": right["member"],
                "exact_ged": float(ged),
                "prefilter_score": _prefilter_score(
                    left["signature"],
                    right["signature"],
                ),
            }
        )
    if not candidates:
        raise BestPairSearchError(
            "No GED benchmark pairs matched the retrieval scope.",
            status_code=404,
        )

    exact_order = sorted(
        candidates,
        key=lambda row: (row["exact_ged"], row["left_graph"], row["right_graph"]),
    )
    prefilter_order = sorted(
        candidates,
        key=lambda row: (-row["prefilter_score"], row["left_graph"], row["right_graph"]),
    )
    effective_top_k = min(max(1, int(top_k or 10)), len(candidates))
    relevant, relevance_cutoff_ged = _tie_aware_relevant_keys(
        exact_order,
        effective_top_k,
    )
    normalized_budgets = sorted(
        {
            min(max(1, int(budget)), len(candidates))
            for budget in budgets
            if isinstance(budget, (int, float)) or str(budget).isdigit()
        }
    )
    if not normalized_budgets:
        normalized_budgets = [min(8, len(candidates))]

    rows = []
    best_ged = exact_order[0]["exact_ged"]
    for budget in normalized_budgets:
        selected = prefilter_order[:budget]
        hits = sum(row["key"] in relevant for row in selected)
        candidate_best_ged = min(row["exact_ged"] for row in selected)
        rows.append(
            {
                "budget": budget,
                "candidate_fraction": round(budget / len(candidates), 6),
                "reduction_percent": round((1.0 - budget / len(candidates)) * 100.0, 3),
                "recall_at_k": round(hits / len(relevant), 6),
                "precision": round(hits / budget, 6),
                "exact_best_recalled": any(
                    math.isclose(
                        float(row["exact_ged"]),
                        float(best_ged),
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                    for row in selected
                ),
                "best_ged_in_candidates": candidate_best_ged,
                "best_ged_regret": round(candidate_best_ged - best_ged, 6),
            }
        )

    run_id = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    payload = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "scope": scope,
        "total_pairs": len(candidates),
        "top_k": effective_top_k,
        "metric_semantics_version": "tie-aware-v1",
        "relevant_pair_count": len(relevant),
        "relevance_cutoff_ged": relevance_cutoff_ged,
        "exact_best_pair": {
            key: value
            for key, value in exact_order[0].items()
            if key != "key"
        },
        "budgets": rows,
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "protocol": {
            "prefilter": "labels, degree histogram, size, density, components",
            "ground_truth_relevance": (
                f"all pairs at or below the GED of rank {effective_top_k}; "
                "ties at the cutoff are included"
            ),
            "reference_kind": ground_truth_kind(dataset_id),
            "purpose": "candidate-recall ceiling before checkpoint-backed GNN reranking",
        },
    }
    RETRIEVAL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = RETRIEVAL_DIR / f"{run_id}.json"
    artifact.write_text(json.dumps(payload, indent=2, sort_keys=True))
    payload["artifact_path"] = str(artifact.relative_to(BASE_DIR))
    return payload


def evaluate_reranking_ablation(
    dataset_id: str,
    method_id: str,
    budgets: list[int],
    scope: str = "train-test",
    top_k: int = 10,
) -> dict[str, Any]:
    started = time.perf_counter()
    _require_exact_ged_dataset(dataset_id, "GNN reranking ablation")
    if scope not in {"train-test", "all"}:
        raise ValueError("Retrieval scope must be 'train-test' or 'all'.")
    method_ids = (
        list(MODEL_BY_ID)
        if method_id == ENSEMBLE_METHOD_ID
        else [method_id]
    )
    if not method_ids or set(_runnable_methods(dataset_id, method_ids)) != set(method_ids):
        raise BestPairSearchError(
            "Every selected checkpoint-backed model must execute on this dataset.",
            status_code=422,
        )
    candidates, record_by_id = _ground_truth_retrieval_candidates(dataset_id, scope)
    exact_order = sorted(
        candidates,
        key=lambda row: (row["exact_ged"], row["left_graph"], row["right_graph"]),
    )
    prefilter_order = sorted(
        candidates,
        key=lambda row: (-row["prefilter_score"], row["left_graph"], row["right_graph"]),
    )
    effective_top_k = min(max(1, int(top_k or 10)), len(candidates))
    normalized_budgets = _normalize_budgets(budgets, maximum=min(100, len(candidates)))
    maximum_budget = max(normalized_budgets)
    relevant, relevance_cutoff_ged = _tie_aware_relevant_keys(
        exact_order,
        effective_top_k,
    )
    exact_best = exact_order[0]
    scored = []

    for candidate in prefilter_order[:maximum_budget]:
        left_id, right_id = candidate["oriented_ids"]
        left = record_by_id[left_id]
        right = record_by_id[right_id]
        pair_payload = load_original_pair(dataset_id, left["member"], right["member"])
        results = run_models(
            left["data"],
            right["data"],
            method_ids,
            dataset_id=dataset_id,
            meta=pair_payload["meta"],
        )
        member_scores = {
            result.get("id"): _reranker_result_score(result)
            for result in results
        }
        complete = all(
            isinstance(member.get("score"), (int, float))
            for member in member_scores.values()
        ) and len(member_scores) == len(method_ids)
        score = (
            sum(float(member["score"]) for member in member_scores.values())
            / len(member_scores)
            if complete
            else None
        )
        scored.append(
            {
                **candidate,
                "model_score": score,
                "model_status": "executed" if complete else "ensemble_incomplete",
                "model_status_label": (
                    "Executed" if complete else "Ensemble member failed"
                ),
                "latency_ms": sum(
                    float(result["latency_ms"])
                    for result in results
                    if isinstance(result.get("latency_ms"), (int, float))
                ),
                "member_scores": member_scores,
                "member_statuses": {
                    result.get("id"): result.get("status")
                    for result in results
                },
                "selected_checkpoints": {
                    result.get("id"): result.get("selected_checkpoint")
                    for result in results
                },
            }
        )

    rows = []
    for budget in normalized_budgets:
        selected = scored[:budget]
        executed = [
            row for row in selected if isinstance(row.get("model_score"), (int, float))
        ]
        reranked = sorted(
            executed,
            key=lambda row: (
                -float(row["model_score"]),
                row["left_graph"],
                row["right_graph"],
            ),
        )
        retrieved = reranked[:effective_top_k]
        hits = sum(row["key"] in relevant for row in retrieved)
        candidate_hits = sum(row["key"] in relevant for row in selected)
        winner = reranked[0] if reranked else None
        rows.append(
            {
                "budget": budget,
                "executed_candidates": len(executed),
                "candidate_fraction": round(budget / len(candidates), 6),
                "reduction_percent": round(
                    (1.0 - budget / len(candidates)) * 100.0,
                    3,
                ),
                "candidate_recall_at_k": round(
                    candidate_hits / len(relevant),
                    6,
                ),
                "reranked_precision_at_k": (
                    round(hits / len(retrieved), 6) if retrieved else None
                ),
                "reranked_recall_at_k": round(hits / len(relevant), 6),
                "reranked_ndcg_at_k": _retrieval_ndcg(retrieved, exact_order, effective_top_k),
                "exact_best_in_candidates": any(
                    math.isclose(
                        float(row["exact_ged"]),
                        float(exact_best["exact_ged"]),
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                    for row in selected
                ),
                "model_selected_pair": (
                    {
                        "left_graph": winner["left_graph"],
                        "right_graph": winner["right_graph"],
                        "model_score": round(float(winner["model_score"]), 6),
                        "exact_ged": winner["exact_ged"],
                    }
                    if winner
                    else None
                ),
                "model_selected_ged_regret": (
                    round(float(winner["exact_ged"]) - float(exact_best["exact_ged"]), 6)
                    if winner
                    else None
                ),
                "latency_total_ms": round(
                    sum(
                        float(row["latency_ms"])
                        for row in selected
                        if isinstance(row.get("latency_ms"), (int, float))
                    ),
                    3,
                ),
            }
        )

    run_id = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    payload = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "method_id": method_id,
        "method_ids": method_ids,
        "model_name": _reranker_name(method_id),
        "scope": scope,
        "total_pairs": len(candidates),
        "top_k": effective_top_k,
        "metric_semantics_version": "tie-aware-v1",
        "relevant_pair_count": len(relevant),
        "relevance_cutoff_ged": relevance_cutoff_ged,
        "maximum_scored_candidates": maximum_budget,
        "candidate_diagnostics": [
            {
                "left_graph": row["left_graph"],
                "right_graph": row["right_graph"],
                "exact_ged": row["exact_ged"],
                "model_status": row["model_status"],
                "member_statuses": row["member_statuses"],
                "member_scores": row["member_scores"],
            }
            for row in scored
        ],
        "exact_best_pair": {
            key: value
            for key, value in exact_best.items()
            if key not in {"key", "oriented_ids"}
        },
        "budgets": rows,
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "protocol": {
            "stage_1": "Deterministic structural prefilter",
            "stage_2": f"Real checkpoint-backed {_reranker_name(method_id)} reranking",
            "ground_truth_relevance": (
                f"all pairs at or below the GED of rank {effective_top_k}; "
                "ties at the cutoff are included"
            ),
            "reference_kind": ground_truth_kind(dataset_id),
            "score": (
                "canonical exp(-predicted GED / average graph size); a bounded "
                "native similarity is retained for a direct-similarity regressor "
                "when inverse GED is undefined"
            ),
            "ensemble": (
                "Arithmetic mean of all canonical member scores; every member must execute"
                if method_id == ENSEMBLE_METHOD_ID
                else "single model"
            ),
            "candidate_reuse": "Each candidate is scored once at the maximum budget",
        },
    }
    RETRIEVAL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = RETRIEVAL_DIR / f"{run_id}-gnn.json"
    artifact.write_text(json.dumps(payload, indent=2, sort_keys=True))
    payload["artifact_path"] = str(artifact.relative_to(BASE_DIR))
    return payload


def _tie_aware_relevant_keys(
    exact_order: list[dict[str, Any]],
    top_k: int,
) -> tuple[set[Any], float]:
    """Return all pairs tied at the requested GED cutoff."""
    effective_top_k = min(max(1, int(top_k)), len(exact_order))
    cutoff = float(exact_order[effective_top_k - 1]["exact_ged"])
    tolerance = max(1e-12, abs(cutoff) * 1e-12)
    relevant = {
        row["key"]
        for row in exact_order
        if float(row["exact_ged"]) <= cutoff + tolerance
    }
    return relevant, cutoff


def _reranker_name(method_id: str) -> str:
    if method_id == ENSEMBLE_METHOD_ID:
        return ENSEMBLE_NAME
    return MODEL_BY_ID[method_id]["name"]


def _reranker_result_score(result: dict[str, Any]) -> dict[str, Any]:
    canonical = result.get("canonical_similarity")
    if result.get("status") == "executed" and isinstance(
        canonical,
        (int, float),
    ):
        return {"score": float(canonical), "source": "canonical"}
    return {"score": None, "source": "unavailable"}


def _ground_truth_retrieval_candidates(
    dataset_id: str,
    scope: str,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    _require_exact_ged_dataset(dataset_id, "GNN reranking ablation")
    records = _load_records(dataset_id)
    record_by_id = {
        int(record["id"]): record
        for record in records
        if str(record.get("id", "")).isdigit()
    }
    train_ids = {
        graph_id for graph_id, record in record_by_id.items() if record["split"] == "train"
    }
    test_ids = {
        graph_id for graph_id, record in record_by_id.items() if record["split"] == "test"
    }
    all_ids = set(record_by_id)
    try:
        distances = load_ground_truth_distances(dataset_id, task="ged")
    except FileNotFoundError as exc:
        raise BestPairSearchError(
            "GNN reranking ablation requires a registered GED benchmark.",
            status_code=422,
            payload={"dataset_id": dataset_id, "ground_truth_available": False},
        ) from exc
    candidates = []
    seen: set[tuple[int, int]] = set()
    for (left_id, right_id), ged in distances.items():
        normalized = _normalize_pair_ids(
            left_id,
            right_id,
            scope,
            train_ids,
            test_ids,
            all_ids,
        )
        if normalized is None:
            continue
        oriented_left, oriented_right = normalized
        key = tuple(sorted((oriented_left, oriented_right)))
        if (
            key in seen
            or oriented_left not in record_by_id
            or oriented_right not in record_by_id
        ):
            continue
        seen.add(key)
        left = record_by_id[oriented_left]
        right = record_by_id[oriented_right]
        candidates.append(
            {
                "key": key,
                "oriented_ids": (oriented_left, oriented_right),
                "left_graph": left["member"],
                "right_graph": right["member"],
                "exact_ged": float(ged),
                "prefilter_score": _prefilter_score(
                    left["signature"],
                    right["signature"],
                ),
            }
        )
    if not candidates:
        raise BestPairSearchError(
            "No GED benchmark pairs matched the reranking scope.",
            status_code=404,
        )
    return candidates, record_by_id


def _normalize_budgets(budgets: list[int], maximum: int) -> list[int]:
    normalized = sorted(
        {
            min(max(1, int(budget)), maximum)
            for budget in budgets
            if isinstance(budget, (int, float)) or str(budget).isdigit()
        }
    )
    return normalized or [min(8, maximum)]


def _retrieval_ndcg(
    retrieved: list[dict[str, Any]],
    exact_order: list[dict[str, Any]],
    k: int,
) -> float | None:
    if not retrieved:
        return None
    relevance = {
        row["key"]: 1.0 / (1.0 + float(row["exact_ged"]))
        for row in exact_order
    }
    dcg = sum(
        (2.0 ** relevance[row["key"]] - 1.0) / math.log2(rank + 2.0)
        for rank, row in enumerate(retrieved)
    )
    ideal_dcg = sum(
        (2.0 ** relevance[row["key"]] - 1.0) / math.log2(rank + 2.0)
        for rank, row in enumerate(exact_order[:k])
    )
    return round(dcg / ideal_dcg, 6) if ideal_dcg else None


def find_best_pair(
    dataset_id: str,
    method_ids: list[str],
    max_pairs: int = 8,
    scope: str = "train-test",
) -> dict[str, Any]:
    started = time.perf_counter()
    if method_ids == ["exact-ged"]:
        return _find_best_pair_by_exact_ged(dataset_id, max_pairs=max_pairs, scope=scope, started=started)
    if method_ids == ["structure-search"]:
        return _find_best_pair_by_structure(dataset_id, max_pairs=max_pairs, scope=scope, started=started)

    method_ids = _runnable_methods(dataset_id, method_ids)
    if not method_ids:
        raise BestPairSearchError("No selected real checkpoint-backed model can run on this dataset.", status_code=422)

    max_pairs = max(1, min(int(max_pairs or 8), 100))
    records = _load_records(dataset_id)
    candidate_pairs, total_pairs = _prefilter_candidates(records, scope=scope, limit=max_pairs)
    if not candidate_pairs:
        raise BestPairSearchError("No graph pairs were available for this dataset.", status_code=404)

    scored = []
    for candidate in candidate_pairs:
        left = candidate["left"]
        right = candidate["right"]
        pair_payload = load_original_pair(dataset_id, left["member"], right["member"])
        results = run_models(
            left["data"],
            right["data"],
            method_ids,
            dataset_id=dataset_id,
            meta=pair_payload["meta"],
        )
        score = _aggregate_score(results)
        scored.append(
            {
                "left_graph": left["member"],
                "right_graph": right["member"],
                "prefilter_score": round(candidate["prefilter_score"], 6),
                "score": score,
                "results": results,
            }
        )

    executed = [candidate for candidate in scored if candidate["score"] is not None]
    if not executed:
        raise BestPairSearchError(
            "Selected checkpoint-backed model did not execute for any candidate pair.",
            status_code=422,
            payload={"candidates": scored, "method_ids": method_ids},
        )

    winner = max(executed, key=lambda candidate: candidate["score"])
    pair_payload = load_original_pair(dataset_id, winner["left_graph"], winner["right_graph"])
    left_data = graph_from_payload(pair_payload["left"], name="Best Graph A")
    right_data = graph_from_payload(pair_payload["right"], name="Best Graph B")
    pair_payload["meta"] = {
        **pair_payload["meta"],
        "best_pair": True,
        "best_pair_score": winner["score"],
        "best_pair_methods": method_ids,
    }

    return {
        **pair_payload,
        "stats": {
            "left": left_data.summary(),
            "right": right_data.summary(),
        },
        "graphs": {
            "left": left_data.to_preview(),
            "right": right_data.to_preview(),
        },
        "search": {
            "dataset_id": dataset_id,
            "method_ids": method_ids,
            "scope": scope,
            "total_pairs": total_pairs,
            "scored_pairs": len(scored),
            "displayed_pairs": len(scored),
            "exhaustive": len(scored) >= total_pairs,
            "selection_label": "Best scored candidate" if len(scored) < total_pairs else "Best pair",
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "winner": winner,
            "candidates": sorted(executed, key=lambda candidate: candidate["score"], reverse=True)[:8],
        },
    }


def _find_best_pair_by_structure(
    dataset_id: str,
    max_pairs: int = 8,
    scope: str = "train-test",
    started: float | None = None,
) -> dict[str, Any]:
    started = time.perf_counter() if started is None else started
    max_pairs = max(1, min(int(max_pairs or 8), 100))
    records = _load_records(dataset_id)
    candidate_pairs, total_pairs = _prefilter_candidates(records, scope=scope, limit=max_pairs)
    if not candidate_pairs:
        raise BestPairSearchError("No graph pairs were available for this dataset.", status_code=404)

    candidates = []
    for candidate in candidate_pairs:
        left = candidate["left"]
        right = candidate["right"]
        score = round(float(candidate["prefilter_score"]), 6)
        candidates.append(
            {
                "left_graph": left["member"],
                "right_graph": right["member"],
                "prefilter_score": score,
                "score": score,
                "results": [_structure_search_result(dataset_id, score)],
            }
        )

    winner = max(candidates, key=lambda candidate: candidate["score"])
    pair_payload = load_original_pair(dataset_id, winner["left_graph"], winner["right_graph"])
    left_data = graph_from_payload(pair_payload["left"], name="Best Graph A")
    right_data = graph_from_payload(pair_payload["right"], name="Best Graph B")
    pair_payload["meta"] = {
        **pair_payload["meta"],
        "best_pair": True,
        "best_pair_score": winner["score"],
        "best_pair_methods": ["structure-search"],
    }

    return {
        **pair_payload,
        "stats": {
            "left": left_data.summary(),
            "right": right_data.summary(),
        },
        "graphs": {
            "left": left_data.to_preview(),
            "right": right_data.to_preview(),
        },
        "search": {
            "dataset_id": dataset_id,
            "method_ids": ["structure-search"],
            "scope": scope,
            "total_pairs": total_pairs,
            "scored_pairs": total_pairs,
            "displayed_pairs": len(candidates),
            "exhaustive": True,
            "selection_label": "Most similar by graph structure",
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "winner": winner,
            "candidates": candidates[:8],
        },
    }


def _find_best_pair_by_exact_ged(
    dataset_id: str,
    max_pairs: int = 8,
    scope: str = "train-test",
    started: float | None = None,
) -> dict[str, Any]:
    _require_exact_ged_dataset(dataset_id, "GED benchmark search")
    reference_kind = ground_truth_kind(dataset_id)
    reference_exact = reference_kind == "exact"
    started = time.perf_counter() if started is None else started
    max_pairs = max(1, min(int(max_pairs or 8), 100))
    graph_lists = list_original_graphs(dataset_id)
    member_by_id = {int(graph["id"]): graph for graph in graph_lists["graphs"] if str(graph["id"]).isdigit()}
    train_ids = {int(graph["id"]) for graph in graph_lists["train"] if str(graph["id"]).isdigit()}
    test_ids = {int(graph["id"]) for graph in graph_lists["test"] if str(graph["id"]).isdigit()}
    all_ids = set(member_by_id)
    try:
        distances = load_ground_truth_distances(dataset_id, task="ged")
    except FileNotFoundError as exc:
        raise BestPairSearchError(
            "This dataset has no GED benchmark. Use Structure Search for datasets "
            "that only have structural proxy targets.",
            status_code=422,
            payload={
                "dataset_id": dataset_id,
                "method_ids": ["exact-ged"],
                "scope": scope,
                "ground_truth_available": False,
                "selection_label": "GED benchmark unavailable",
                "candidates": [],
            },
        ) from exc

    heap: list[tuple[float, int, int, int]] = []
    considered = 0
    seen_pairs: set[tuple[int, int]] = set()
    for index, ((left_id, right_id), ged) in enumerate(distances.items()):
        pair_ids = _normalize_pair_ids(left_id, right_id, scope, train_ids, test_ids, all_ids)
        if pair_ids is None:
            continue
        normalized_left_id, normalized_right_id = pair_ids
        if normalized_left_id not in member_by_id or normalized_right_id not in member_by_id:
            continue
        pair_key = tuple(sorted((normalized_left_id, normalized_right_id)))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        considered += 1
        item = (-float(ged), index, normalized_left_id, normalized_right_id)
        if len(heap) < max_pairs:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)

    if not heap:
        raise BestPairSearchError("No GED ground-truth pairs matched this dataset/scope.", status_code=404)

    candidates = []
    for neg_ged, _index, left_id, right_id in sorted(heap, reverse=True):
        left_member = member_by_id[left_id]["member"]
        right_member = member_by_id[right_id]["member"]
        pair_payload = load_original_pair(dataset_id, left_member, right_member)
        left_data = graph_from_payload(pair_payload["left"], name=left_member)
        right_data = graph_from_payload(pair_payload["right"], name=right_member)
        ged = -neg_ged
        score = _ged_similarity(ged, left_data, right_data)
        candidates.append(
            {
                "left_graph": left_member,
                "right_graph": right_member,
                "exact_ged": ged,
                "score": score,
                "results": [
                    {
                        "id": "exact-ged",
                        "name": (
                            "Exact GED Ground Truth"
                            if reference_exact
                            else "Approximate GED Benchmark"
                        ),
                        "accent": "#125f70",
                        "paper": "Original benchmark",
                        "family": (
                            "A* GED label"
                            if reference_exact
                            else "minimum Beam/Hungarian/VJ upper bound"
                        ),
                        "latency_ms": 0,
                        "status": "ground_truth",
                        "status_label": (
                            "Exact benchmark label"
                            if reference_exact
                            else "Approximate benchmark label"
                        ),
                        "score": score,
                        "distance": ged,
                        "detail": (
                            f"Benchmark GED reference is {ged:g}. Similarity is "
                            "exp(-GED / average graph size)."
                        ),
                        "command": "load_ground_truth_distances(dataset_id, task='ged')",
                        "local_path": "Models&Datasets/drive-download-20260630T100606Z-3-001/*_ged_astar_gidpair_dist_map.pickle",
                        "entrypoint": "ground-truth pickle",
                        "python": "not required",
                        "requirements": [],
                        "environment": (
                            "Original benchmark GED map loaded from the local archive. "
                            + (
                                "The reference is exact A*."
                                if reference_exact
                                else (
                                    "The reference is an approximate upper bound, not exact GED."
                                )
                            )
                        ),
                        "setup": [],
                        "supported_datasets": ["aids700nef", "linux", "imdbmulti", "ptc"],
                        "runnable_datasets": ["aids700nef", "linux", "imdbmulti", "ptc"],
                        "dataset_supported": True,
                        "dataset_runnable": True,
                        "missing_requirements": [],
                        "missing_runtime": False,
                        "missing_files": [],
                        "checkpoints": [],
                        "adapter_metrics": {
                            "reference_ged": ged,
                            "reference_exact": reference_exact,
                            "reference_kind": reference_kind,
                        },
                    }
                ],
            }
        )

    winner = max(candidates, key=lambda candidate: (candidate["score"], -candidate["exact_ged"]))
    pair_payload = load_original_pair(dataset_id, winner["left_graph"], winner["right_graph"])
    left_data = graph_from_payload(pair_payload["left"], name="Best Graph A")
    right_data = graph_from_payload(pair_payload["right"], name="Best Graph B")
    pair_payload["meta"] = {
        **pair_payload["meta"],
        "best_pair": True,
        "best_pair_score": winner["score"],
        "best_pair_methods": ["exact-ged"],
        "exact_ged": winner["exact_ged"],
        "reference_exact": reference_exact,
        "reference_kind": reference_kind,
    }

    return {
        **pair_payload,
        "stats": {
            "left": left_data.summary(),
            "right": right_data.summary(),
        },
        "graphs": {
            "left": left_data.to_preview(),
            "right": right_data.to_preview(),
        },
        "search": {
            "dataset_id": dataset_id,
            "method_ids": ["exact-ged"],
            "scope": scope,
            "total_pairs": considered,
            "scored_pairs": considered,
            "displayed_pairs": len(candidates),
            "exhaustive": True,
            "selection_label": "Best pair",
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "winner": winner,
            "candidates": candidates[:8],
        },
    }


def _normalize_pair_ids(
    left_id: int,
    right_id: int,
    scope: str,
    train_ids: set[int],
    test_ids: set[int],
    all_ids: set[int],
) -> tuple[int, int] | None:
    if left_id == right_id:
        return None
    if scope == "all":
        if left_id in all_ids and right_id in all_ids:
            return left_id, right_id
        return None
    if left_id in train_ids and right_id in test_ids:
        return left_id, right_id
    if left_id in test_ids and right_id in train_ids:
        return right_id, left_id
    return None


def _runnable_methods(dataset_id: str, method_ids: list[str]) -> list[str]:
    runnable = []
    for method_id in method_ids:
        model = MODEL_BY_ID.get(method_id)
        if model is None:
            continue
        status = inspect_model(model, dataset_id=dataset_id)
        if status["status"] == "adapter_required" and status.get("dataset_runnable") is not False:
            runnable.append(method_id)
    return runnable


def _load_records(dataset_id: str) -> list[dict[str, Any]]:
    records = []
    for item in load_original_graph_collection(dataset_id):
        data = graph_from_payload(item["graph"], name=item["member"])
        records.append(
            {
                **item,
                "data": data,
                "signature": _signature(data),
            }
        )
    return records


def _prefilter_candidates(records: list[dict[str, Any]], scope: str, limit: int) -> tuple[list[dict[str, Any]], int]:
    if scope == "all":
        pairs = combinations(records, 2)
        total_pairs = (len(records) * (len(records) - 1)) // 2
    else:
        train = [record for record in records if record["split"] == "train"] or records
        test = [record for record in records if record["split"] == "test"] or records
        pairs = ((left, right) for left, right in product(train, test) if left["member"] != right["member"])
        total_pairs = len(train) * len(test)

    heap: list[tuple[float, int, dict[str, Any], dict[str, Any]]] = []
    for index, (left, right) in enumerate(pairs):
        score = _prefilter_score(left["signature"], right["signature"])
        item = (score, index, left, right)
        if len(heap) < limit:
            heapq.heappush(heap, item)
        elif score > heap[0][0]:
            heapq.heapreplace(heap, item)

    candidates = [
        {"left": left, "right": right, "prefilter_score": score}
        for score, _index, left, right in sorted(heap, key=lambda item: item[0], reverse=True)
    ]
    return candidates, total_pairs


def _signature(graph: GraphData) -> dict[str, Any]:
    return {
        "nodes": graph.node_count,
        "edges": graph.edge_count,
        "density": graph.density,
        "components": len(graph.connected_components()),
        "degrees": graph.degree_counter,
        "labels": graph.label_counter,
    }


def _prefilter_score(left: dict[str, Any], right: dict[str, Any]) -> float:
    return (
        0.24 * _counter_similarity(left["labels"], right["labels"])
        + 0.24 * _counter_similarity(left["degrees"], right["degrees"])
        + 0.18 * _closeness(left["nodes"], right["nodes"])
        + 0.14 * _closeness(left["edges"], right["edges"])
        + 0.12 * (1.0 - abs(left["density"] - right["density"]))
        + 0.08 * _closeness(left["components"], right["components"])
    )


def _counter_similarity(left: Counter, right: Counter) -> float:
    union = left | right
    if not union:
        return 1.0
    return sum((left & right).values()) / sum(union.values())


def _closeness(left: float, right: float) -> float:
    denominator = max(float(left), float(right), 1.0)
    return max(0.0, 1.0 - abs(float(left) - float(right)) / denominator)


def _structure_search_result(dataset_id: str, score: float) -> dict[str, Any]:
    return {
        "id": "structure-search",
        "name": "Structure Search",
        "accent": "#125f70",
        "paper": "Dataset utility",
        "family": "No-GED structural search",
        "latency_ms": 0,
        "status": "structural_search",
        "status_label": "Structure search",
        "score": score,
        "distance": None,
        "detail": (
            "No exact GED label is required. This utility ranks pairs with a transparent structural score "
            "from node labels, degree histogram, graph size, density, and connected components. It is not a GNN prediction."
        ),
        "command": "_prefilter_score(label, degree, size, density, components)",
        "local_path": f"dataset registry: {dataset_id}",
        "entrypoint": "graph_similarity_platform/search.py",
        "python": "not required",
        "requirements": [],
        "environment": "Built-in deterministic graph-pair search used only when exact GED/model scoring is not available.",
        "setup": [],
        "supported_datasets": ["aids700nef", "linux", "imdbmulti", "ptc", "mutag", "proteins", "enzymes"],
        "runnable_datasets": ["aids700nef", "linux", "imdbmulti", "ptc", "mutag", "proteins", "enzymes"],
        "dataset_supported": True,
        "dataset_runnable": True,
        "missing_requirements": [],
        "missing_runtime": False,
        "missing_files": [],
        "checkpoints": [],
        "adapter_metrics": {"structural_score": score},
    }


def _aggregate_score(results: list[dict[str, Any]]) -> float | None:
    scores = [
        float(result["canonical_similarity"])
        for result in results
        if (
            result.get("status") == "executed"
            and isinstance(result.get("canonical_similarity"), (int, float))
            and math.isfinite(float(result["canonical_similarity"]))
        )
    ]
    if not scores:
        return None
    return sum(scores) / len(scores)


def _ged_similarity(ged: float, left: GraphData, right: GraphData) -> float:
    graph_size = max(0.5 * (left.node_count + right.node_count), 1.0)
    return float(math.exp(-float(ged) / graph_size))
