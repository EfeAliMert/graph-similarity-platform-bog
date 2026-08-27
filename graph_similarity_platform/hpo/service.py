from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .best_config import BestConfigRegistry
from .budgets import load_budgets
from .dataset_profile import DatasetProfiler
from .registry import DEFAULT_REGISTRY
from .storage import DEFAULT_PROGRESS_ROOT


def verified_optimized_config(dataset_id: str, model_id: str) -> dict[str, Any] | None:
    """Return a config only when its dataset fingerprint and space version match."""
    adapter = DEFAULT_REGISTRY.get(model_id)
    profile = DatasetProfiler().profile(
        dataset_id,
        preprocessing={
            "canonical_node_order": True,
            "target_transform": "model-native normalized GED similarity",
        },
    )
    return BestConfigRegistry().load(
        dataset_id,
        model_id,
        expected_fingerprint=profile.fingerprint,
        expected_search_space_version=adapter.search_space_version,
    )


def optimization_catalog(dataset_id: str, model_id: str) -> dict[str, Any]:
    adapter = DEFAULT_REGISTRY.get(model_id)
    profile = DatasetProfiler().profile(
        dataset_id,
        preprocessing={
            "canonical_node_order": True,
            "target_transform": "model-native normalized GED similarity",
        },
    )
    optimized = BestConfigRegistry().load(
        dataset_id,
        model_id,
        expected_fingerprint=profile.fingerprint,
        expected_search_space_version=adapter.search_space_version,
    )
    return {
        "dataset_profile": profile.to_dict(),
        "search_space": adapter.search_space_summary(profile),
        "budgets": {
            name: {
                "trials": budget.trials,
                "resource": budget.resource_for(model_id),
                "confirmation_top_k": budget.confirmation_top_k,
                "confirmation_seeds": list(budget.confirmation_seeds),
            }
            for name, budget in load_budgets().items()
        },
        "optimized_config": optimized,
        "test_set_used_for_selection": False,
    }


def latest_progress(
    dataset_id: str,
    model_id: str,
    started_after: float | None = None,
    root: Path = DEFAULT_PROGRESS_ROOT,
) -> dict[str, Any] | None:
    if not root.is_dir():
        return None
    candidates = sorted(root.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        if started_after is not None and path.stat().st_mtime + 1 < started_after:
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("dataset_id") == dataset_id and payload.get("model_id") == model_id:
            return {**payload, "path": str(path)}
    return None
