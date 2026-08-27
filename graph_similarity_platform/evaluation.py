from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any
import uuid

from .data import (
    ground_truth_kind,
    list_original_graphs,
    load_ground_truth_distances,
    load_original_pair,
)
from .graph_utils import graph_from_payload
from .models.real_models import BASE_DIR, MODEL_BY_ID, run_models


MAX_EVALUATION_PAIRS = 200
BENCHMARK_DIR = BASE_DIR / "training_logs" / "benchmarks"


def evaluate_models(
    dataset_id: str,
    method_ids: list[str],
    sample_size: int = 12,
    scope: str = "train-test",
    sample_mode: str = "stratified",
    seed: int = 379,
    top_k: int = 5,
    persist: bool = True,
    adapter_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    reference_kind = ground_truth_kind(dataset_id)
    if reference_kind not in {"exact", "approximate_benchmark"}:
        raise ValueError(
            "This dataset contains a structural proxy target, not exact GED ground truth. "
            "Use its checkpoint-backed pair comparison instead of GED benchmark evaluation."
        )
    method_ids = [method_id for method_id in method_ids if method_id in MODEL_BY_ID]
    if not method_ids:
        raise ValueError("No valid checkpoint-backed model was selected for evaluation.")
    if scope not in {"train-test", "all"}:
        raise ValueError("Evaluation scope must be 'train-test' or 'all'.")
    if sample_mode not in {"stratified", "random", "all"}:
        raise ValueError("Sampling mode must be stratified, random, or all.")

    requested_size = max(1, min(int(sample_size or 12), MAX_EVALUATION_PAIRS))
    pairs, candidate_count = _sample_ground_truth_pairs(
        dataset_id,
        requested_size,
        scope,
        sample_mode,
        int(seed),
    )
    effective_top_k = max(1, min(int(top_k or 5), len(pairs)))
    model_rows: dict[str, dict[str, Any]] = {
        method_id: {
            "id": method_id,
            "name": MODEL_BY_ID[method_id]["name"],
            "paper": MODEL_BY_ID[method_id]["paper"],
            "samples": [],
        }
        for method_id in method_ids
    }

    for pair in pairs:
        pair_payload = load_original_pair(dataset_id, pair["left_graph"], pair["right_graph"])
        left = graph_from_payload(pair_payload["left"], name=pair["left_graph"])
        right = graph_from_payload(pair_payload["right"], name=pair["right_graph"])
        graph_size = max(0.5 * (left.node_count + right.node_count), 1.0)
        exact_ged = float(pair["exact_ged"])
        exact_similarity = math.exp(-exact_ged / graph_size)
        pair_meta = dict(pair_payload.get("meta") or {})
        if adapter_options:
            pair_meta.update(adapter_options)
        results = run_models(
            left,
            right,
            method_ids,
            dataset_id=dataset_id,
            meta=pair_meta,
        )

        pair.update(
            {
                "left_nodes": left.node_count,
                "right_nodes": right.node_count,
                "average_graph_size": graph_size,
                "exact_normalized_ged": exact_ged / graph_size,
                "exact_similarity": exact_similarity,
                "reference_exact": reference_kind == "exact",
                "reference_kind": reference_kind,
            }
        )

        for result in results:
            row = model_rows[result["id"]]
            sample = {
                "left_graph": pair["left_graph"],
                "right_graph": pair["right_graph"],
                "average_graph_size": graph_size,
                "exact_ged": exact_ged,
                "exact_normalized_ged": exact_ged / graph_size,
                "exact_similarity": exact_similarity,
                "latency_ms": result.get("latency_ms"),
                "status": result["status"],
                "status_label": result.get("status_label"),
                "detail": result.get("detail", ""),
            }
            if result.get("status") == "executed" and isinstance(result.get("score"), (int, float)):
                model_score = float(result["score"])
                predicted_ged = _predicted_ged(result, graph_size)
                predicted_similarity = _canonical_similarity(result, predicted_ged, graph_size)
                sample.update(
                    {
                        "model_score": model_score,
                        "predicted_similarity": predicted_similarity,
                        "predicted_ged": predicted_ged,
                        "predicted_normalized_ged": (
                            predicted_ged / graph_size if predicted_ged is not None else None
                        ),
                        "score_semantics": result.get("score_semantics"),
                        "checkpoint_seed": (result.get("adapter_metrics") or {}).get("seed"),
                        "pair_split": (result.get("adapter_metrics") or {}).get("pair_split"),
                        "peak_rss_bytes": (result.get("adapter_metrics") or {}).get("peak_rss_bytes"),
                        "projection_applied": (result.get("adapter_metrics") or {}).get("projection_applied", False),
                        "input_projection": (result.get("adapter_metrics") or {}).get("input_projection"),
                        "selected_checkpoint": result.get("selected_checkpoint"),
                    }
                )
                if predicted_similarity is not None:
                    sample["abs_similarity_error"] = abs(
                        predicted_similarity - exact_similarity
                    )
                if predicted_ged is not None:
                    sample["abs_ged_error"] = abs(predicted_ged - exact_ged)
                    sample["abs_normalized_ged_error"] = abs(
                        predicted_ged / graph_size - exact_ged / graph_size
                    )
            row["samples"].append(sample)

    for row in model_rows.values():
        _summarize_model(row, effective_top_k, int(seed))

    completed_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "run_id": (
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
            f"{uuid.uuid4().hex[:8]}"
        ),
        "dataset_id": dataset_id,
        "scope": scope,
        "sample_size": len(pairs),
        "candidate_pairs": candidate_count,
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "completed_at": completed_at,
        "protocol": {
            "task": "GED regression and pair ranking",
            "ground_truth": (
                "registered exact A* GED"
                if reference_kind == "exact"
                else "registered approximate GED upper-bound benchmark"
            ),
            "reference_kind": reference_kind,
            "reference_exact": reference_kind == "exact",
            "approximate_reference_note": (
                None
                if reference_kind == "exact"
                else (
                    "IMDBMulti/PTC reference values are the minimum upper bound "
                    "from Beam, Hungarian, and VJ; they are not exact GED."
                )
            ),
            "target": "exp(-GED / average graph size)",
            "sampling": sample_mode,
            "seed": int(seed),
            "requested_pairs": requested_size,
            "evaluated_pairs": len(pairs),
            "top_k": effective_top_k,
            "max_pairs_per_run": MAX_EVALUATION_PAIRS,
            "latency_boundary": (
                "End-to-end local adapter subprocess, including process startup "
                "and graph preparation."
            ),
            "checkpoint_warning": (
                "Local checkpoints are not author-released pretrained weights; "
                "results do not reproduce paper benchmark tables."
            ),
        },
        "pairs": pairs,
        "models": list(model_rows.values()),
    }
    if persist:
        payload["artifact_path"] = _persist_benchmark(payload)
    return payload


def benchmark_catalog() -> list[dict[str, Any]]:
    if not BENCHMARK_DIR.exists():
        return []
    rows = []
    for path in sorted(BENCHMARK_DIR.glob("*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rows.append(
            {
                "run_id": payload.get("run_id", path.stem),
                "dataset_id": payload.get("dataset_id"),
                "completed_at": payload.get("completed_at"),
                "sample_size": payload.get("sample_size"),
                "models": [model.get("id") for model in payload.get("models", [])],
                "artifact_path": str(path.relative_to(BASE_DIR)),
            }
        )
    return rows


def load_benchmark(run_id: str) -> dict[str, Any]:
    if not run_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in run_id):
        raise ValueError("Invalid benchmark run id.")
    path = BENCHMARK_DIR / f"{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Benchmark run not found: {run_id}")
    return json.loads(path.read_text())


def _persist_benchmark(payload: dict[str, Any]) -> str:
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    path = BENCHMARK_DIR / f"{payload['run_id']}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return str(path.relative_to(BASE_DIR))


def _sample_ground_truth_pairs(
    dataset_id: str,
    limit: int,
    scope: str,
    sample_mode: str,
    seed: int,
) -> tuple[list[dict[str, Any]], int]:
    if ground_truth_kind(dataset_id) not in {"exact", "approximate_benchmark"}:
        raise ValueError(
            "This dataset contains a structural proxy target, not exact GED ground truth."
        )
    graph_lists = list_original_graphs(dataset_id)
    member_by_id = {
        int(graph["id"]): graph
        for graph in graph_lists["graphs"]
        if str(graph["id"]).isdigit()
    }
    train_ids = {
        int(graph["id"])
        for graph in graph_lists["train"]
        if str(graph["id"]).isdigit()
    }
    test_ids = {
        int(graph["id"])
        for graph in graph_lists["test"]
        if str(graph["id"]).isdigit()
    }
    all_ids = set(member_by_id)
    try:
        distances = load_ground_truth_distances(dataset_id, task="ged")
    except FileNotFoundError as exc:
        raise ValueError(
            "This dataset has no registered GED benchmark, so GED evaluation is unavailable. "
            "Use AIDS700nef or LINUX for exact GED, and IMDBMulti or PTC for "
            "the published approximate upper-bound benchmark."
        ) from exc
    candidates = []
    seen_pairs: set[tuple[int, int]] = set()

    for (left_id, right_id), ged in distances.items():
        pair_ids = _normalize_pair_ids(
            left_id,
            right_id,
            scope,
            train_ids,
            test_ids,
            all_ids,
        )
        if pair_ids is None:
            continue
        normalized_left_id, normalized_right_id = pair_ids
        if normalized_left_id not in member_by_id or normalized_right_id not in member_by_id:
            continue
        pair_key = tuple(sorted((normalized_left_id, normalized_right_id)))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        candidates.append((float(ged), normalized_left_id, normalized_right_id))

    if not candidates:
        raise ValueError("No ground-truth GED pairs matched this dataset/scope.")

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    selected_limit = min(limit, len(candidates))
    if sample_mode == "all" or selected_limit >= len(candidates):
        selected = candidates[:selected_limit]
    elif sample_mode == "random":
        selected = random.Random(seed).sample(candidates, selected_limit)
        selected.sort(key=lambda item: (item[0], item[1], item[2]))
    elif selected_limit == 1:
        selected = [candidates[len(candidates) // 2]]
    else:
        selected = [
            candidates[round(index * (len(candidates) - 1) / (selected_limit - 1))]
            for index in range(selected_limit)
        ]

    return (
        [
            {
                "left_graph": member_by_id[left_id]["member"],
                "right_graph": member_by_id[right_id]["member"],
                "exact_ged": ged,
            }
            for ged, left_id, right_id in selected
        ],
        len(candidates),
    )


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
        return (left_id, right_id) if left_id in all_ids and right_id in all_ids else None
    if left_id in train_ids and right_id in test_ids:
        return left_id, right_id
    if left_id in test_ids and right_id in train_ids:
        return right_id, left_id
    return None


def _predicted_ged(result: dict[str, Any], graph_size: float) -> float | None:
    metrics = result.get("adapter_metrics") or {}
    if "predicted_ged" in metrics and metrics["predicted_ged"] is None:
        return None
    if isinstance(metrics.get("predicted_ged"), (int, float)):
        return float(metrics["predicted_ged"])
    score = max(float(result.get("score") or 0.0), 1e-12)
    return -math.log(score) * graph_size


def _canonical_similarity(
    result: dict[str, Any],
    predicted_ged: float | None,
    graph_size: float,
) -> float | None:
    if predicted_ged is not None:
        return math.exp(-predicted_ged / graph_size)
    value = result.get("canonical_similarity")
    return float(value) if isinstance(value, (int, float)) else None


def _summarize_model(row: dict[str, Any], top_k: int = 5, seed: int = 379) -> None:
    runtime_executed = [
        sample for sample in row["samples"] if sample.get("status") == "executed"
    ]
    evaluated = [
        sample
        for sample in row["samples"]
        if "abs_similarity_error" in sample
        and isinstance(sample.get("predicted_ged"), (int, float))
    ]
    row["attempted_samples"] = len(row["samples"])
    row["executed_samples"] = len(runtime_executed)
    row["evaluated_samples"] = len(evaluated)
    row["projected_samples"] = sum(
        sample.get("projection_applied") is True for sample in evaluated
    )
    if not evaluated:
        row["status"] = "not_evaluable" if runtime_executed else "not_executed"
        for key in (
            "agreement_percent",
            "mae_similarity",
            "mse_similarity",
            "mse_similarity_x1e3",
            "rmse_similarity",
            "mae_ged",
            "rmse_ged",
            "mae_normalized_ged",
            "spearman_ged",
            "kendall_ged",
            "precision_at_k",
            "recall_at_k",
            "ndcg_at_k",
            "precision_at_10",
            "precision_at_20",
            "latency_p50_ms",
            "latency_p95_ms",
            "throughput_pairs_per_second",
            "peak_rss_mb",
            "mae_ged_ci95",
            "mae_similarity_ci95",
            "size_generalization",
        ):
            row[key] = None
        return

    similarity_errors = [sample["abs_similarity_error"] for sample in evaluated]
    ged_errors = [sample["abs_ged_error"] for sample in evaluated]
    normalized_errors = [sample["abs_normalized_ged_error"] for sample in evaluated]
    exact_ged = [sample["exact_ged"] for sample in evaluated]
    predicted_ged = [sample["predicted_ged"] for sample in evaluated]
    exact_normalized_ged = [sample["exact_normalized_ged"] for sample in evaluated]
    predicted_normalized_ged = [
        sample["predicted_normalized_ged"] for sample in evaluated
    ]
    latencies = [
        float(sample["latency_ms"])
        for sample in runtime_executed
        if isinstance(sample.get("latency_ms"), (int, float))
    ]
    peak_rss_values = [
        int(sample["peak_rss_bytes"])
        for sample in evaluated
        if isinstance(sample.get("peak_rss_bytes"), (int, float))
    ]
    mae_similarity = statistics.fmean(similarity_errors)
    mse_similarity = statistics.fmean([error**2 for error in similarity_errors])
    mae_ged = statistics.fmean(ged_errors)
    k = min(max(1, top_k), len(evaluated))
    ranking = _ranking_metrics(
        exact_normalized_ged,
        predicted_normalized_ged,
        [sample["exact_similarity"] for sample in evaluated],
        k,
    )

    row.update(
        {
            "status": "evaluated",
            "agreement_percent": round(max(0.0, 1.0 - mae_similarity) * 100.0, 2),
            "mae_similarity": round(mae_similarity, 6),
            "mse_similarity": round(mse_similarity, 9),
            "mse_similarity_x1e3": round(mse_similarity * 1000.0, 6),
            "rmse_similarity": round(math.sqrt(mse_similarity), 6),
            "mae_ged": round(mae_ged, 6),
            "rmse_ged": round(
                math.sqrt(statistics.fmean([error**2 for error in ged_errors])),
                6,
            ),
            "mae_normalized_ged": round(statistics.fmean(normalized_errors), 6),
            "spearman_ged": _round_or_none(_spearman(exact_ged, predicted_ged)),
            "kendall_ged": _round_or_none(_kendall_tau_b(exact_ged, predicted_ged)),
            "top_k": k,
            "precision_at_k": round(ranking["precision"], 6),
            "recall_at_k": round(ranking["recall"], 6),
            "ndcg_at_k": round(ranking["ndcg"], 6),
            "precision_at_10": _fixed_precision(
                exact_normalized_ged,
                predicted_normalized_ged,
                10,
            ),
            "precision_at_20": _fixed_precision(
                exact_normalized_ged,
                predicted_normalized_ged,
                20,
            ),
            "latency_p50_ms": (
                round(_percentile(latencies, 0.5), 3) if latencies else None
            ),
            "latency_p95_ms": (
                round(_percentile(latencies, 0.95), 3) if latencies else None
            ),
            "throughput_pairs_per_second": (
                round(1000.0 / statistics.fmean(latencies), 3)
                if latencies and statistics.fmean(latencies) > 0
                else None
            ),
            "peak_rss_mb": (
                round(max(peak_rss_values) / (1024.0 * 1024.0), 3)
                if peak_rss_values
                else None
            ),
            "mae_ged_ci95": _bootstrap_mae_ci(
                exact_ged,
                predicted_ged,
                seed + _stable_id_seed(row.get("id", "")),
            ),
            "mae_similarity_ci95": _bootstrap_error_ci(
                similarity_errors,
                seed + _stable_id_seed(row.get("id", "")) + 1,
            ),
            "size_generalization": _size_generalization(evaluated),
            "checkpoint_seeds": sorted(
                {
                    int(sample["checkpoint_seed"])
                    for sample in evaluated
                    if isinstance(sample.get("checkpoint_seed"), (int, float))
                }
            ),
            "pair_split_verified": all(
                isinstance(sample.get("pair_split"), dict)
                and sample["pair_split"].get(
                    "pair_overlap_count",
                    sample["pair_split"].get("pair_overlap"),
                ) == 0
                for sample in evaluated
            ),
            "unprojected_metrics": _compact_metrics(
                [
                    sample
                    for sample in evaluated
                    if sample.get("projection_applied") is not True
                ],
                top_k,
                seed,
                row.get("id", ""),
            ),
            "projected_metrics": _compact_metrics(
                [
                    sample
                    for sample in evaluated
                    if sample.get("projection_applied") is True
                ],
                top_k,
                seed,
                row.get("id", ""),
            ),
        }
    )


def _compact_metrics(
    samples: list[dict[str, Any]],
    top_k: int,
    seed: int,
    model_id: str,
) -> dict[str, Any] | None:
    if not samples:
        return None
    ged_errors = [float(sample["abs_ged_error"]) for sample in samples]
    similarity_errors = [float(sample["abs_similarity_error"]) for sample in samples]
    exact_ged = [float(sample["exact_ged"]) for sample in samples]
    predicted_ged = [float(sample["predicted_ged"]) for sample in samples]
    exact_normalized_ged = [float(sample["exact_normalized_ged"]) for sample in samples]
    predicted_normalized_ged = [
        float(sample["predicted_normalized_ged"]) for sample in samples
    ]
    k = min(max(1, top_k), len(samples))
    ranking = _ranking_metrics(
        exact_normalized_ged,
        predicted_normalized_ged,
        [float(sample["exact_similarity"]) for sample in samples],
        k,
    )
    return {
        "samples": len(samples),
        "mae_ged": round(statistics.fmean(ged_errors), 6),
        "mse_similarity": round(
            statistics.fmean([error**2 for error in similarity_errors]),
            9,
        ),
        "spearman_ged": _round_or_none(_spearman(exact_ged, predicted_ged)),
        "ndcg_at_k": round(ranking["ndcg"], 6),
        "mae_ged_ci95": _bootstrap_mae_ci(
            exact_ged,
            predicted_ged,
            seed + _stable_id_seed(model_id) + 17,
        ),
    }


def _fixed_precision(
    exact_distances: list[float],
    predicted_distances: list[float],
    k: int,
) -> float | None:
    if len(exact_distances) < k:
        return None
    ranking = _ranking_metrics(
        exact_distances,
        predicted_distances,
        [math.exp(-distance) for distance in exact_distances],
        k,
    )
    return round(ranking["precision"], 6)


def _size_generalization(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not samples:
        return []
    ordered_sizes = sorted(_sample_graph_size(sample) for sample in samples)
    low = _percentile(ordered_sizes, 1.0 / 3.0)
    high = _percentile(ordered_sizes, 2.0 / 3.0)
    buckets = [
        ("small", float("-inf"), low),
        ("medium", low, high),
        ("large", high, float("inf")),
    ]
    rows = []
    for label, minimum, maximum in buckets:
        selected = [
            sample
            for sample in samples
            if (
                (_sample_graph_size(sample) <= maximum if label == "small" else True)
                and (_sample_graph_size(sample) > minimum if label != "small" else True)
                and (_sample_graph_size(sample) <= maximum if label == "medium" else True)
                and (_sample_graph_size(sample) > minimum if label == "large" else True)
            )
        ]
        if not selected:
            continue
        ged_errors = [float(sample["abs_ged_error"]) for sample in selected]
        similarity_errors = [
            float(sample["abs_similarity_error"]) for sample in selected
        ]
        rows.append(
            {
                "bucket": label,
                "samples": len(selected),
                "average_size_min": round(
                    min(_sample_graph_size(sample) for sample in selected),
                    3,
                ),
                "average_size_max": round(
                    max(_sample_graph_size(sample) for sample in selected),
                    3,
                ),
                "mae_ged": round(statistics.fmean(ged_errors), 6),
                "mse_similarity_x1e3": round(
                    statistics.fmean([error**2 for error in similarity_errors])
                    * 1000.0,
                    6,
                ),
            }
        )
    return rows


def _sample_graph_size(sample: dict[str, Any]) -> float:
    if isinstance(sample.get("average_graph_size"), (int, float)):
        return max(float(sample["average_graph_size"]), 1.0)
    normalized = sample.get("exact_normalized_ged")
    exact = sample.get("exact_ged")
    if (
        isinstance(normalized, (int, float))
        and float(normalized) > 0
        and isinstance(exact, (int, float))
    ):
        return max(float(exact) / float(normalized), 1.0)
    return 1.0


def _ranking_metrics(
    exact_distances: list[float],
    predicted_distances: list[float],
    exact_relevance: list[float],
    k: int,
) -> dict[str, float | int]:
    exact_order = sorted(range(len(exact_distances)), key=lambda index: (exact_distances[index], index))
    predicted_order = sorted(
        range(len(predicted_distances)),
        key=lambda index: (predicted_distances[index], index),
    )
    effective_k = min(max(1, int(k)), len(exact_order))
    cutoff_distance = exact_distances[exact_order[effective_k - 1]]
    tolerance = max(1e-12, abs(float(cutoff_distance)) * 1e-12)
    relevant = {
        index
        for index, distance in enumerate(exact_distances)
        if float(distance) <= float(cutoff_distance) + tolerance
    }
    retrieved = predicted_order[:effective_k]
    hits = sum(index in relevant for index in retrieved)
    dcg = sum(
        (2.0 ** exact_relevance[index] - 1.0) / math.log2(rank + 2.0)
        for rank, index in enumerate(retrieved)
    )
    ideal_order = sorted(
        range(len(exact_relevance)),
        key=lambda index: (-exact_relevance[index], index),
    )
    ideal_dcg = sum(
        (2.0 ** exact_relevance[index] - 1.0) / math.log2(rank + 2.0)
        for rank, index in enumerate(ideal_order[:effective_k])
    )
    return {
        "precision": hits / len(retrieved),
        "recall": hits / len(relevant),
        "ndcg": dcg / ideal_dcg if ideal_dcg else 0.0,
        "relevance_count": len(relevant),
        "relevance_cutoff_distance": float(cutoff_distance),
    }


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    return _pearson(_average_ranks(left), _average_ranks(right))


def _kendall_tau_b(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    concordant = discordant = ties_left = ties_right = 0
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
            left_sign = _sign(left[first] - left[second])
            right_sign = _sign(right[first] - right[second])
            if left_sign == 0 and right_sign == 0:
                continue
            if left_sign == 0:
                ties_left += 1
            elif right_sign == 0:
                ties_right += 1
            elif left_sign == right_sign:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_left)
        * (concordant + discordant + ties_right)
    )
    return (concordant - discordant) / denominator if denominator else None


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average = (position + end - 1) / 2.0 + 1.0
        for cursor in range(position, end):
            ranks[order[cursor]] = average
        position = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_sum = sum((value - left_mean) ** 2 for value in left)
    right_sum = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_sum * right_sum)
    return numerator / denominator if denominator else None


def _bootstrap_mae_ci(
    exact: list[float],
    predicted: list[float],
    seed: int,
    iterations: int = 500,
) -> list[float] | None:
    errors = [abs(predicted_value - exact_value) for exact_value, predicted_value in zip(exact, predicted)]
    return _bootstrap_error_ci(errors, seed, iterations)


def _bootstrap_error_ci(
    errors: list[float],
    seed: int,
    iterations: int = 500,
) -> list[float] | None:
    if not errors:
        return None
    if len(errors) == 1:
        value = round(errors[0], 6)
        return [value, value]
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(rng.choice(errors) for _ in errors)
        for _ in range(iterations)
    )
    return [
        round(_percentile(means, 0.025), 6),
        round(_percentile(means, 0.975), 6),
    ]


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _stable_id_seed(value: str) -> int:
    return sum((index + 1) * ord(character) for index, character in enumerate(value))


def _sign(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def _round_or_none(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None
