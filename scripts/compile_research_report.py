from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = ROOT / "reports" / "research_matrices"
OUTPUT_JSON = ROOT / "reports" / "RESEARCH_RESULTS.json"
OUTPUT_MD = ROOT / "reports" / "RESEARCH_RESULTS.md"
ACCURACY_JSON = ROOT / "reports" / "final_dataset_accuracy_audit.json"
ACCURACY_MD = ROOT / "reports" / "final_dataset_accuracy_audit.md"


def main() -> None:
    summary = latest_json(SUMMARY_DIR)
    retrieval = latest_json(ROOT / "reports" / "retrieval_study")
    ablations = latest_json(ROOT / "reports" / "adapter_ablations")
    grouped = _read(ROOT / "reports" / "grouped_split_study.json")
    checkpoints = _read(ROOT / "reports" / "checkpoint_audit.json")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": (
            "These are local-checkpoint results. They are not reproductions of "
            "the five papers' published accuracy tables."
        ),
        "matrix": summary,
        "retrieval": retrieval,
        "adapter_ablations": ablations,
        "grouped_split": grouped,
        "checkpoint_protocol": {
            "verified": (checkpoints or {}).get("verified"),
            "hpo_verified": (checkpoints or {}).get("hpo_verified"),
            "total": (checkpoints or {}).get("total"),
            "complete": (checkpoints or {}).get("complete"),
        },
    }
    if summary:
        write_accuracy_audit(summary)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True))
    OUTPUT_MD.write_text(render_markdown(payload))
    print(OUTPUT_JSON.relative_to(ROOT))
    print(OUTPUT_MD.relative_to(ROOT))


def write_accuracy_audit(summary: dict[str, Any]) -> None:
    datasets: dict[str, Any] = {}
    for row in summary.get("rows") or []:
        dataset_id = row["dataset_id"]
        datasets.setdefault(
            dataset_id,
            {
                "dataset_id": dataset_id,
                "reference_kind": row.get("reference_kind"),
                "evaluation_pairs": summary.get("evaluation_pairs"),
                "mode": summary.get("mode"),
                "models": [],
            },
        )
        metrics = row.get("metrics") or {}
        mae_ci95 = _single_seed_ci(row, "mae_ged")
        datasets[dataset_id]["models"].append(
            {
                "id": row["model_id"],
                "name": row["model_name"],
                "status": "evaluated" if row.get("complete") else "incomplete",
                "evaluated_seeds": row.get("evaluated_seeds"),
                "checkpoint_seeds": row.get("checkpoint_seeds"),
                "mae_ged": (metrics.get("mae_ged") or {}).get("mean"),
                "mae_ged_pair_bootstrap_ci95": mae_ci95,
                "mse_similarity_x1e3": (metrics.get("mse_similarity_x1e3") or {}).get("mean"),
                "spearman_ged": (metrics.get("spearman_ged") or {}).get("mean"),
                "ndcg_at_k": (metrics.get("ndcg_at_k") or {}).get("mean"),
                "pair_split_verified": row.get("pair_split_verified"),
            }
        )
    ACCURACY_JSON.write_text(json.dumps(datasets, indent=2, sort_keys=True))
    lines = [
        "# Dataset Accuracy Audit",
        "",
        "This is a local-checkpoint study, not a reproduction of author-released paper results.",
        "",
        f"Matrix complete: `{summary.get('matrix_complete')}`. "
        f"Mode: `{summary.get('mode')}`. "
        f"Held-out pairs: `{summary.get('evaluation_pairs')}`.",
        "",
    ]
    for dataset_id, block in datasets.items():
        lines.extend(
            [
                f"## {dataset_id}",
                "",
                f"Reference: **{block.get('reference_kind')}**.",
                "",
                "| Model | Checkpoint seeds | GED MAE | Pair-bootstrap 95% CI | MSE x1e3 | Spearman | NDCG@k | Split verified |",
                "|---|---|---:|---|---:|---:|---:|---|",
            ]
        )
        for model in block["models"]:
            lines.append(
                f"| {model['name']} | {','.join(str(seed) for seed in model.get('checkpoint_seeds') or []) or '-'} | "
                f"{_number(model.get('mae_ged'))} | {_format_ci(model.get('mae_ged_pair_bootstrap_ci95'))} | "
                f"{_number(model.get('mse_similarity_x1e3'), 3)} | "
                f"{_number(model.get('spearman_ged'))} | {_number(model.get('ndcg_at_k'))} | "
                f"{model.get('pair_split_verified')} |"
            )
        lines.append("")
    ACCURACY_MD.write_text("\n".join(lines))


def render_markdown(payload: dict[str, Any]) -> str:
    matrix = payload.get("matrix") or {}
    lines = [
        "# Research Results",
        "",
        payload["claim_boundary"],
        "",
        f"Generated: `{payload['generated_at']}`.",
        f"Matrix complete: `{matrix.get('matrix_complete')}`.",
        f"Mode: `{matrix.get('mode')}`.",
        f"Held-out pairs: `{matrix.get('evaluation_pairs')}`.",
        "",
        "Exact A* GED and approximate solver upper bounds stay in separate tables. "
        "A standard deviation is not estimable from one evaluation seed; the table "
        "therefore reports pair-bootstrap 95% confidence intervals for GED MAE. Paper-level MCS "
        "and graph-classification experiments remain out of scope.",
        "",
    ]
    lines.extend(_matrix_section(matrix, "exact", "Exact A* GED"))
    lines.extend(_matrix_section(matrix, "approximate", "Approximate GED benchmark"))
    lines.extend(_retrieval_section(payload.get("retrieval")))
    lines.extend(_ablation_section(payload.get("adapter_ablations")))
    lines.extend(_grouped_section(payload.get("grouped_split")))
    protocol = payload.get("checkpoint_protocol") or {}
    lines.extend(
        [
            "## Checkpoint protocol",
            "",
            f"Verified `{protocol.get('verified')}/{protocol.get('total')}`.",
            f"HPO-to-checkpoint hash bindings: `{protocol.get('hpo_verified')}/{protocol.get('total')}`.",
            "",
            "## Still not a paper reproduction",
            "",
            "- The current matrix uses one evaluation seed; between-seed variation is not estimated.",
            "- All model-dataset HPO selections are tracked, but only the completed final trainings are hash-bound to active checkpoints.",
            "- Full-corpus evaluation was not run.",
            "",
        ]
    )
    return "\n".join(lines)


def _matrix_section(matrix: dict[str, Any], group: str, title: str) -> list[str]:
    rows = [
        row
        for row in matrix.get("rows") or []
        if _row_group(row) == group
    ]
    if not rows:
        return []
    lines = [
        f"## {title}",
        "",
        "| Dataset | Model | Eval seeds | Checkpoint seeds | GED MAE | Pair-bootstrap 95% CI | MSE x1e3 | Spearman | NDCG@k | Split verified |",
        "|---|---|---|---|---:|---|---:|---:|---:|---|",
    ]
    for row in rows:
        metrics = row.get("metrics") or {}
        seeds = ",".join(str(seed) for seed in row.get("checkpoint_seeds") or []) or "-"
        evaluation_seeds = ",".join(str(seed) for seed in row.get("evaluated_seeds") or []) or "-"
        lines.append(
            f"| {row['dataset_id']} | {row['model_name']} | {evaluation_seeds} | {seeds} | "
            f"{_mean(metrics.get('mae_ged'))} | {_format_ci(_single_seed_ci(row, 'mae_ged'))} | "
            f"{_mean(metrics.get('mse_similarity_x1e3'))} | "
            f"{_mean(metrics.get('spearman_ged'))} | {_mean(metrics.get('ndcg_at_k'))} | "
            f"{row.get('pair_split_verified')} |"
        )
    lines.append("")
    return lines


def _row_group(row: dict[str, Any]) -> str:
    kind = str(row.get("reference_kind") or "")
    source = str(row.get("target_source") or "").lower()
    if kind == "exact" or source.startswith("exact"):
        return "exact"
    if kind == "approximate_benchmark" or "approximate" in source:
        return "approximate"
    return "other"


def _retrieval_section(retrieval: dict[str, Any] | None) -> list[str]:
    if not retrieval:
        return []
    if retrieval.get("metric_semantics_version") != "tie-aware-v1":
        return [
            "## Retrieval",
            "",
            "The stored retrieval study predates tie-aware relevance and is excluded "
            "from current claims. Rerun `scripts/run_retrieval_study.py` before reporting retrieval metrics.",
            "",
        ]
    lines = [
        "## Retrieval",
        "",
        "Structural prefilter first. GNN reranking scores only the surviving budget.",
        "",
    ]
    if retrieval.get("prefilter"):
        lines.extend(
            [
                "### Tie-aware structural prefilter",
                "",
                "| Dataset | Relevant pairs | Budget | Recall@k | GED regret | Reduction % |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for study in retrieval.get("prefilter") or []:
            for row in study.get("budgets") or []:
                lines.append(
                    f"| {study.get('dataset_id')} | {study.get('relevant_pair_count')} | "
                    f"{row.get('budget')} | {row.get('recall_at_k')} | "
                    f"{row.get('best_ged_regret')} | {row.get('reduction_percent')} |"
                )
        lines.extend(["", "### Checkpoint-backed reranking", ""])
    if retrieval.get("rerank"):
        lines.extend(
            [
                "| Dataset | Model | Budget | Candidate recall@k | Reranked NDCG@k | GED regret |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
    for study in retrieval.get("rerank") or []:
        for row in study.get("budgets") or []:
            lines.append(
                f"| {study.get('dataset_id')} | {study.get('model_name') or study.get('method_id')} | "
                f"{row.get('budget')} | {row.get('candidate_recall_at_k')} | "
                f"{row.get('reranked_ndcg_at_k')} | {row.get('model_selected_ged_regret')} |"
            )
    if not retrieval.get("rerank"):
        lines.extend(
            [
                "No current tie-aware GNN reranking run is included. Do not quote "
                "the legacy reranking metrics.",
                "",
            ]
        )
    else:
        lines.append("")
    return lines


def _ablation_section(ablations: dict[str, Any] | None) -> list[str]:
    if not ablations:
        return []
    lines = [
        "## Adapter ablations",
        "",
        "| Dataset | Ablation | Setting A | Setting B | MAE A | MAE B |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in ablations.get("rows") or []:
        if row.get("ablation") == "segmn-projection":
            unprojected = row.get("unprojected_pairs") or {}
            lines.append(
                f"| {row.get('dataset_id')} | SEGMN projection | all pairs | unprojected | "
                f"{row.get('all_pairs', {}).get('mae_ged')} | {unprojected.get('mae_ged')} |"
            )
        elif row.get("ablation") == "graph2region-compatibility":
            lines.append(
                f"| {row.get('dataset_id')} | Graph2Region correction | corrected | original equation | "
                f"{row.get('corrected', {}).get('mae_ged')} | "
                f"{row.get('original_equation', {}).get('mae_ged')} |"
            )
    lines.append("")
    return lines


def _grouped_section(grouped: dict[str, Any] | None) -> list[str]:
    if not grouped:
        return []
    synthetic = grouped.get("synthetic_grouped_subjects") or {}
    lines = [
        "## Grouped split",
        "",
        "Pair-disjoint validation can share graphs. Subject-disjoint validation does not.",
        "",
    ]
    pair_row = synthetic.get("pair_disjoint")
    subject_row = synthetic.get("subject_disjoint")
    if pair_row and subject_row:
        lines.extend(
            [
                "| Strategy | Graph overlap | Pair overlap |",
                "|---|---:|---:|",
                f"| {pair_row['strategy']} | {pair_row['graph_overlap']} | {pair_row['pair_overlap']} |",
                f"| {subject_row['strategy']} | {subject_row['graph_overlap']} | {subject_row['pair_overlap']} |",
                "",
            ]
        )
    return lines


def _mean(row: dict[str, Any] | None) -> str:
    if not isinstance(row, dict) or row.get("mean") is None:
        return "-"
    return f"{row['mean']}"


def _single_seed_ci(row: dict[str, Any], metric: str) -> list[float] | None:
    intervals = row.get("pair_bootstrap_ci95_by_seed") or []
    if len(intervals) != 1:
        return None
    value = intervals[0].get(metric)
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, (int, float)) for item in value)
    ):
        return [float(value[0]), float(value[1])]
    return None


def _format_ci(value: object) -> str:
    if not isinstance(value, list) or len(value) != 2:
        return "N/A"
    if not all(isinstance(item, (int, float)) for item in value):
        return "N/A"
    return f"[{float(value[0]):.3f}, {float(value[1]):.3f}]"


def latest_json(directory: Path) -> dict[str, Any] | None:
    if not directory.exists():
        return None
    candidates = sorted(
        directory.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        payload = _read(path)
        if payload:
            return payload
    return None


def _read(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _number(value: object, digits: int = 3) -> str:
    return "-" if not isinstance(value, (int, float)) else f"{float(value):.{digits}f}"


if __name__ == "__main__":
    main()
