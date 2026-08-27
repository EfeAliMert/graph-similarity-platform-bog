from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graph_similarity_platform.models.real_models import MODEL_BY_ID  # noqa: E402
from graph_similarity_platform.search import (  # noqa: E402
    ENSEMBLE_METHOD_ID,
    evaluate_prefilter_ablation,
    evaluate_reranking_ablation,
)
from scripts.run_research_matrix import select_datasets  # noqa: E402


REPORT_DIR = ROOT / "reports" / "retrieval_study"
DEFAULT_BUDGETS = [1, 4, 8, 16, 32, 64]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run structural prefilter and GNN reranking retrieval studies."
    )
    parser.add_argument("--prefilter-datasets", default="aids700nef,linux,imdbmulti,ptc")
    parser.add_argument("--rerank-datasets", default="aids700nef,linux")
    parser.add_argument("--models", default="all")
    parser.add_argument("--budgets", default=",".join(str(value) for value in DEFAULT_BUDGETS))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument("--include-ensemble", action="store_true")
    args = parser.parse_args()

    budgets = [int(item) for item in args.budgets.split(",") if item.strip()]
    prefilter_datasets = select_datasets(args.prefilter_datasets, False, True)
    rerank_datasets = (
        []
        if args.skip_rerank
        else select_datasets(args.rerank_datasets, False, True)
    )
    model_ids = list(MODEL_BY_ID) if args.models == "all" else [
        item.strip() for item in args.models.split(",") if item.strip()
    ]
    if args.include_ensemble:
        model_ids = [*model_ids, ENSEMBLE_METHOD_ID]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metric_semantics_version": "tie-aware-v1",
        "budgets": budgets,
        "top_k": args.top_k,
        "prefilter": [],
        "rerank": [],
    }

    for dataset_id in prefilter_datasets:
        print(f"PREFILTER {dataset_id}")
        result = evaluate_prefilter_ablation(
            dataset_id,
            budgets,
            scope="train-test",
            top_k=args.top_k,
        )
        payload["prefilter"].append(_compact_prefilter(result))

    for dataset_id in rerank_datasets:
        for model_id in model_ids:
            print(f"RERANK {model_id} {dataset_id}")
            result = evaluate_reranking_ablation(
                dataset_id,
                model_id,
                budgets,
                scope="train-test",
                top_k=args.top_k,
            )
            payload["rerank"].append(_compact_rerank(result))

    json_path = REPORT_DIR / f"{run_id}.json"
    markdown_path = REPORT_DIR / f"{run_id}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    markdown_path.write_text(render_markdown(payload))
    print(json_path.relative_to(ROOT))
    print(markdown_path.relative_to(ROOT))
    return 0


def _compact_prefilter(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": result.get("run_id"),
        "dataset_id": result.get("dataset_id"),
        "artifact_path": result.get("artifact_path"),
        "total_pairs": result.get("total_pairs"),
        "metric_semantics_version": result.get("metric_semantics_version"),
        "relevant_pair_count": result.get("relevant_pair_count"),
        "relevance_cutoff_ged": result.get("relevance_cutoff_ged"),
        "reference_kind": (result.get("protocol") or {}).get("reference_kind"),
        "budgets": result.get("budgets"),
    }


def _compact_rerank(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": result.get("run_id"),
        "dataset_id": result.get("dataset_id"),
        "method_id": result.get("method_id"),
        "model_name": result.get("model_name"),
        "artifact_path": result.get("artifact_path"),
        "total_pairs": result.get("total_pairs"),
        "metric_semantics_version": result.get("metric_semantics_version"),
        "relevant_pair_count": result.get("relevant_pair_count"),
        "relevance_cutoff_ged": result.get("relevance_cutoff_ged"),
        "reference_kind": (result.get("protocol") or {}).get("reference_kind"),
        "budgets": [
            {
                "budget": row.get("budget"),
                "candidate_recall_at_k": row.get("candidate_recall_at_k"),
                "reranked_ndcg_at_k": row.get("reranked_ndcg_at_k"),
                "reranked_recall_at_k": row.get("reranked_recall_at_k"),
                "model_selected_ged_regret": row.get("model_selected_ged_regret"),
                "latency_total_ms": row.get("latency_total_ms"),
            }
            for row in result.get("budgets") or []
        ],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Retrieval Study {payload['run_id']}",
        "",
        "Structural prefilter ranks every registered train-test pair. GNN reranking "
        "scores only the surviving candidate budget. A reranker cannot recover a "
        "pair removed by the prefilter.",
        "Recall uses every pair tied at the GED cutoff; it is not affected by "
        "graph identifier ordering.",
        "",
        "## Structural prefilter",
        "",
        "| Dataset | Reference | Budget | Recall@k | Best GED regret | Reduction % |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for study in payload.get("prefilter") or []:
        for row in study.get("budgets") or []:
            lines.append(
                f"| {study['dataset_id']} | {study.get('reference_kind')} | "
                f"{row.get('budget')} | {row.get('recall_at_k')} | "
                f"{row.get('best_ged_regret')} | {row.get('reduction_percent')} |"
            )
    lines.extend(
        [
            "",
            "## GNN reranking",
            "",
            "| Dataset | Model | Budget | Candidate recall@k | Reranked NDCG@k | GED regret | Latency ms |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for study in payload.get("rerank") or []:
        for row in study.get("budgets") or []:
            lines.append(
                f"| {study['dataset_id']} | {study.get('model_name') or study.get('method_id')} | "
                f"{row.get('budget')} | {row.get('candidate_recall_at_k')} | "
                f"{row.get('reranked_ndcg_at_k')} | {row.get('model_selected_ged_regret')} | "
                f"{row.get('latency_total_ms')} |"
            )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
