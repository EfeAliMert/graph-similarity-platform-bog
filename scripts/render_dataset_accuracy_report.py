from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACCURACY_PATH = ROOT / "reports" / "final_dataset_accuracy_audit.json"
TARGET_PATH = ROOT / "reports" / "dataset_target_audit.json"
CHECKPOINT_PATH = ROOT / "reports" / "checkpoint_audit.json"
OUTPUT_PATH = ROOT / "reports" / "final_dataset_accuracy_audit.md"


def main() -> None:
    accuracy = _read_json(ACCURACY_PATH)
    targets = _read_json(TARGET_PATH)
    checkpoints = _read_json(CHECKPOINT_PATH)
    target_rows = {row["dataset_id"]: row for row in targets.get("datasets", [])}
    lines = [
        "# Dataset Accuracy Audit",
        "",
        "This is a local-checkpoint audit, not a reproduction of author-released paper results. "
        "Checkpoint selection used validation data; the reported pairs are blind train-test pairs.",
        "",
        f"Checkpoint protocol: **{checkpoints.get('verified', 0)}/{checkpoints.get('total', 0)} verified**.",
        "",
        "## Target provenance",
        "",
        "| Dataset | Graphs | Reference | Status | Scientific claim |",
        "|---|---:|---|---|---|",
    ]
    for dataset_id in (
        "aids700nef",
        "linux",
        "imdbmulti",
        "ptc",
        "mutag",
        "proteins",
        "enzymes",
    ):
        row = target_rows.get(dataset_id, {})
        lines.append(
            f"| {row.get('name', dataset_id)} | {row.get('graphs', '-')} | "
            f"{row.get('reference_kind', '-')} | {row.get('status', '-')} | "
            f"{row.get('scientific_claim', '-')} |"
        )

    for dataset_id in ("aids700nef", "linux", "imdbmulti", "ptc"):
        run = accuracy.get(dataset_id)
        if not run:
            continue
        reference_kind = run.get("protocol", {}).get("reference_kind", "unknown")
        lines.extend(
            [
                "",
                f"## {target_rows.get(dataset_id, {}).get('name', dataset_id)}",
                "",
                f"Reference: **{reference_kind}**. Seed: `379`. Blind stratified pairs: `24`.",
                "",
                "| Model | Evaluated | GED MAE | GED RMSE | Norm. GED MAE | Similarity MSE | Spearman | NDCG@5 | Projected |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for model in run.get("models", []):
            lines.append(
                f"| {model.get('id')} | {model.get('evaluated_samples', 0)} | "
                f"{_number(model.get('mae_ged'))} | {_number(model.get('rmse_ged'))} | "
                f"{_number(model.get('mae_normalized_ged'))} | "
                f"{_number(model.get('mse_similarity'), 6)} | "
                f"{_number(model.get('spearman_ged'))} | {_number(model.get('ndcg_at_k'))} | "
                f"{model.get('projected_samples', 0)} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- AIDS700nef and LINUX support exact-GED error claims.",
            "- IMDBMulti and PTC errors are measured against published approximate upper-bound references, not exact GED.",
            "- MUTAG, PROTEINS, and ENZYMES have structural-proxy targets only; exact-GED accuracy is not identifiable.",
            "- IMDBMulti similarity targets saturate near zero for large distances, so inverse-log GED estimates are numerically sensitive.",
            "- Projected SEGMN inputs indicate deterministic truncation required by its quadratic assignment graph and must be reported.",
            "- These 24-pair, one-seed results are a rigorous local audit but remain provisional until the three-seed, 50-pair matrix is rerun.",
            "",
        ]
    )
    OUTPUT_PATH.write_text("\n".join(lines))
    print(OUTPUT_PATH.relative_to(ROOT))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _number(value: object, digits: int = 3) -> str:
    return "-" if not isinstance(value, (int, float)) else f"{float(value):.{digits}f}"


if __name__ == "__main__":
    main()
