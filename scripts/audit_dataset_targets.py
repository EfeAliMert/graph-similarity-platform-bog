from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
import statistics
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from graph_similarity_platform.data import (  # noqa: E402
    list_original_datasets,
    list_original_graphs,
    load_ground_truth_distances,
)
from universal_dataset import ensure_training_distances  # noqa: E402


DATASET_IDS = (
    "aids700nef",
    "linux",
    "imdbmulti",
    "ptc",
    "mutag",
    "proteins",
    "enzymes",
)


def main() -> None:
    catalog = {row["id"]: row for row in list_original_datasets()}
    rows = [audit_dataset(dataset_id, catalog[dataset_id]) for dataset_id in DATASET_IDS]
    payload = {
        "schema_version": 1,
        "canonical_pair_rule": (
            "GED is treated as symmetric. Directional records are collapsed by "
            "unordered graph id and the minimum finite non-negative value is used. "
            "For approximate solvers this is the tighter valid upper bound."
        ),
        "datasets": rows,
    }
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    json_path = reports / "dataset_target_audit.json"
    markdown_path = reports / "dataset_target_audit.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    markdown_path.write_text(render_markdown(payload))
    print(json.dumps({"json": display(json_path), "markdown": display(markdown_path), "datasets": len(rows)}))


def audit_dataset(dataset_id: str, info: dict[str, Any]) -> dict[str, Any]:
    graphs = list_original_graphs(dataset_id)
    graph_ids = {
        int(graph["id"])
        for graph in graphs["graphs"]
        if str(graph.get("id", "")).isdigit()
    }
    row: dict[str, Any] = {
        "dataset_id": dataset_id,
        "name": info["name"],
        "graphs": len(graph_ids),
        "train_graphs": len(graphs["train"]),
        "test_graphs": len(graphs["test"]),
        "reference_kind": info.get("ground_truth_kind"),
        "reference_exact": bool(info.get("ground_truth_exact")),
        "reference_source": info.get("ground_truth_source"),
        "benchmark_available": bool(info.get("ground_truth_benchmark")),
    }
    ged_paths = [
        ROOT / path
        for path in info.get("ground_truth_paths", [])
        if "_ged_" in str(path).lower() or Path(path).stem.lower() == "ged"
    ]
    if not ged_paths:
        distances, metadata = ensure_training_distances(dataset_id)
        row.update(
            {
                "status": "proxy_only",
                "training_target_pairs": len(distances),
                "training_target_kind": metadata.get("target_kind", "structural_proxy"),
                "training_target_source": metadata.get("target_source"),
                "scientific_claim": (
                    "The checkpoint can be evaluated only for fidelity to the declared "
                    "structural proxy; no exact or approximate GED benchmark claim is valid."
                ),
            }
        )
        return row

    raw = read_raw_distances(ged_paths[0])
    finite = {
        (left, right): value
        for (left, right), value in raw.items()
        if math.isfinite(value) and value >= 0
    }
    seen: set[tuple[int, int]] = set()
    conflicts: list[float] = []
    unordered_pairs = 0
    for (left, right), value in finite.items():
        key = tuple(sorted((left, right)))
        if key in seen:
            continue
        seen.add(key)
        unordered_pairs += 1
        reverse = finite.get((right, left))
        if reverse is not None and not math.isclose(value, reverse, abs_tol=1e-12):
            conflicts.append(abs(value - reverse))

    canonical = load_ground_truth_distances(dataset_id, task="ged")
    canonical_conflicts = sum(
        not math.isclose(value, canonical.get((right, left), value), abs_tol=1e-12)
        for (left, right), value in canonical.items()
    )
    values = sorted(finite.values())
    expected_unordered = (
        len(graphs["train"]) * (len(graphs["train"]) + 1) // 2
        + len(graphs["train"]) * len(graphs["test"])
    )
    row.update(
        {
            "status": "exact_verified" if row["reference_exact"] else "approximate_verified",
            "raw_records": len(raw),
            "raw_nonfinite_or_negative": len(raw) - len(finite),
            "raw_unknown_graph_ids": sum(
                left not in graph_ids or right not in graph_ids
                for left, right in raw
            ),
            "raw_bad_diagonal": sum(
                left == right and not math.isclose(value, 0.0, abs_tol=1e-12)
                for (left, right), value in finite.items()
            ),
            "unordered_pairs": unordered_pairs,
            "expected_unordered_pairs": expected_unordered,
            "unordered_coverage": (
                unordered_pairs / expected_unordered if expected_unordered else None
            ),
            "raw_direction_conflicts": len(conflicts),
            "raw_direction_conflict_mean_abs": (
                statistics.fmean(conflicts) if conflicts else 0.0
            ),
            "raw_direction_conflict_max_abs": max(conflicts, default=0.0),
            "canonical_records": len(canonical),
            "canonical_direction_conflicts": canonical_conflicts,
            "distance_min": values[0] if values else None,
            "distance_median": statistics.median(values) if values else None,
            "distance_p95": percentile(values, 0.95),
            "distance_max": values[-1] if values else None,
            "scientific_claim": (
                "Valid exact-GED reference."
                if row["reference_exact"]
                else (
                    "Valid published approximate benchmark reference; values are "
                    "upper bounds and must not be reported as exact GED."
                )
            ),
        }
    )
    return row


def read_raw_distances(path: Path) -> dict[tuple[int, int], float]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        values = {}
        for key, value in payload.items():
            left, right = str(key).split(",", maxsplit=1)
            values[(int(left), int(right))] = float(value)
        return values
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return {
        (int(left), int(right)): float(value)
        for (left, right), value in payload.items()
    }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    return values[round((len(values) - 1) * fraction)]


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Dataset Target Audit",
        "",
        payload["canonical_pair_rule"],
        "",
        "| Dataset | Graphs | Reference | Raw conflicts | Coverage | Canonical conflicts | Status |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in payload["datasets"]:
        coverage = row.get("unordered_coverage")
        lines.append(
            "| {name} | {graphs} | {kind} | {raw} | {coverage} | {canonical} | {status} |".format(
                name=row["name"],
                graphs=row["graphs"],
                kind=row["reference_kind"],
                raw=row.get("raw_direction_conflicts", "n/a"),
                coverage=f"{coverage:.3f}" if isinstance(coverage, float) else "n/a",
                canonical=row.get("canonical_direction_conflicts", "n/a"),
                status=row["status"],
            )
        )
    lines.extend(["", "## Interpretation", ""])
    for row in payload["datasets"]:
        lines.append(f"- **{row['name']}:** {row['scientific_claim']}")
    lines.append("")
    return "\n".join(lines)


def display(path: Path) -> str:
    return str(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
