from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graph_similarity_platform.evaluation import evaluate_models  # noqa: E402
from scripts.run_research_matrix import select_datasets  # noqa: E402


REPORT_DIR = ROOT / "reports" / "adapter_ablations"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ablate SEGMN input projection and Graph2Region compatibility corrections."
    )
    parser.add_argument("--datasets", default="aids700nef,linux")
    parser.add_argument("--pairs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=379)
    args = parser.parse_args()
    datasets = select_datasets(args.datasets, False, True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_pairs": args.pairs,
        "seed": args.seed,
        "rows": [],
    }

    for dataset_id in datasets:
        print(f"SEGMN projection {dataset_id}")
        segmn = evaluate_models(
            dataset_id,
            ["segmn"],
            sample_size=args.pairs,
            scope="train-test",
            sample_mode="stratified",
            seed=args.seed,
            top_k=min(10, args.pairs),
        )
        payload["rows"].append(_segmn_row(segmn))

        print(f"Graph2Region correction on {dataset_id}")
        corrected = evaluate_models(
            dataset_id,
            ["graph2region"],
            sample_size=args.pairs,
            scope="train-test",
            sample_mode="stratified",
            seed=args.seed,
            top_k=min(10, args.pairs),
        )
        print(f"Graph2Region correction off {dataset_id}")
        original = evaluate_models(
            dataset_id,
            ["graph2region"],
            sample_size=args.pairs,
            scope="train-test",
            sample_mode="stratified",
            seed=args.seed,
            top_k=min(10, args.pairs),
            adapter_options={"disable_compatibility_correction": True},
        )
        payload["rows"].append(_g2r_row(corrected, original))

    json_path = REPORT_DIR / f"{run_id}.json"
    markdown_path = REPORT_DIR / f"{run_id}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    markdown_path.write_text(render_markdown(payload))
    print(json_path.relative_to(ROOT))
    print(markdown_path.relative_to(ROOT))
    return 0


def _segmn_row(benchmark: dict[str, Any]) -> dict[str, Any]:
    model = benchmark["models"][0]
    return {
        "ablation": "segmn-projection",
        "dataset_id": benchmark["dataset_id"],
        "reference_kind": benchmark["protocol"]["reference_kind"],
        "artifact_path": benchmark.get("artifact_path"),
        "all_pairs": _metric_view(model),
        "unprojected_pairs": model.get("unprojected_metrics"),
        "projected_pairs": model.get("projected_metrics"),
        "projected_samples": model.get("projected_samples"),
        "evaluated_samples": model.get("evaluated_samples"),
    }


def _g2r_row(corrected: dict[str, Any], original: dict[str, Any]) -> dict[str, Any]:
    return {
        "ablation": "graph2region-compatibility",
        "dataset_id": corrected["dataset_id"],
        "reference_kind": corrected["protocol"]["reference_kind"],
        "corrected": {
            "artifact_path": corrected.get("artifact_path"),
            **_metric_view(corrected["models"][0]),
        },
        "original_equation": {
            "artifact_path": original.get("artifact_path"),
            **_metric_view(original["models"][0]),
        },
    }


def _metric_view(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "evaluated_samples": model.get("evaluated_samples"),
        "mae_ged": model.get("mae_ged"),
        "mse_similarity": model.get("mse_similarity"),
        "spearman_ged": model.get("spearman_ged"),
        "ndcg_at_k": model.get("ndcg_at_k"),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Adapter Ablations {payload['run_id']}",
        "",
        "SEGMN projection compares the full held-out sample with the subset that "
        "fits the assignment-graph caps. Graph2Region compares the local "
        "GED-volume/positional corrections with the original equations on the "
        "same pairs.",
        "",
        "## SEGMN projection",
        "",
        "| Dataset | All MAE | Unprojected MAE | Projected pairs | Spearman all |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        if row["ablation"] != "segmn-projection":
            continue
        unprojected = row.get("unprojected_pairs") or {}
        lines.append(
            f"| {row['dataset_id']} | {row['all_pairs'].get('mae_ged')} | "
            f"{unprojected.get('mae_ged')} | {row.get('projected_samples')} | "
            f"{row['all_pairs'].get('spearman_ged')} |"
        )
    lines.extend(
        [
            "",
            "## Graph2Region compatibility correction",
            "",
            "| Dataset | Corrected MAE | Original MAE | Corrected Spearman | Original Spearman |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["rows"]:
        if row["ablation"] != "graph2region-compatibility":
            continue
        lines.append(
            f"| {row['dataset_id']} | {row['corrected'].get('mae_ged')} | "
            f"{row['original_equation'].get('mae_ged')} | "
            f"{row['corrected'].get('spearman_ged')} | "
            f"{row['original_equation'].get('spearman_ged')} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
