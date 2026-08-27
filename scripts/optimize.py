from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graph_similarity_platform.hpo import (  # noqa: E402
    HyperparameterOptimizer,
    OptimizationRequest,
    SearchSpaceRegistry,
    load_budgets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dataset-adaptive, validation-only hyperparameter optimization for "
            "the local graph similarity models."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True, help="Model id or 'all'.")
    parser.add_argument(
        "--budget",
        choices=tuple(load_budgets()),
        default="standard",
    )
    parser.add_argument("--seed", type=int, default=379)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--storage", type=Path)
    parser.add_argument("--refresh-profile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = SearchSpaceRegistry()
    model_ids = registry.model_ids() if args.model == "all" else (args.model,)
    optimizer = HyperparameterOptimizer(registry=registry)
    summaries = []
    for model_id in model_ids:
        summary = optimizer.optimize(
            OptimizationRequest(
                dataset_id=args.dataset,
                model_id=model_id,
                budget=args.budget,
                seed=args.seed,
                split_seed=args.split_seed,
                trials=args.trials,
                storage_path=args.storage,
                refresh_profile=args.refresh_profile,
            )
        )
        summaries.append(summary)
        result = {
            "study_name": summary["study_name"],
            "dataset_id": summary["dataset_id"],
            "model_id": summary["model_id"],
            "best_validation_mse": summary["best_trial"]["validation_mse_mean"],
            "best_validation_spearman": summary["best_trial"]["validation_spearman_mean"],
            "best_config": summary["best_trial"]["config"],
            "best_config_path": summary["best_config_path"],
            "completed_trials": summary["completed_trials"],
            "pruned_trials": summary["pruned_trials"],
            "failed_trials": summary["failed_trials"],
            "test_set_used_for_selection": False,
            "promoted": False,
        }
        print("HPO_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    if len(summaries) > 1:
        print(
            "HPO_ALL_RESULT="
            + json.dumps(
                {
                    "dataset_id": args.dataset,
                    "models": [summary["model_id"] for summary in summaries],
                    "test_set_used_for_selection": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
