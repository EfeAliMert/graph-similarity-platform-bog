from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.universal_dataset import split_leakage_comparison  # noqa: E402


REPORT_JSON = ROOT / "reports" / "grouped_split_study.json"
REPORT_MARKDOWN = ROOT / "reports" / "grouped_split_study.md"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare pair-disjoint and subject-disjoint validation leakage."
    )
    parser.add_argument("--seed", type=int, default=379)
    args = parser.parse_args()
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "synthetic_grouped_subjects": synthetic_study(args.seed),
    }
    REPORT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True))
    REPORT_MARKDOWN.write_text(render_markdown(payload))
    print(REPORT_JSON.relative_to(ROOT))
    print(REPORT_MARKDOWN.relative_to(ROOT))
    return 0


def synthetic_study(seed: int) -> dict[str, Any]:
    graphs = [
        {"id": graph_id, "nodes": list(range(8)), "subject_id": graph_id // 2}
        for graph_id in range(12)
    ]
    distances = {
        (left, right): float(abs(left - right) + 1)
        for left in range(12)
        for right in range(left + 1, 12)
    }
    comparison = split_leakage_comparison(graphs, distances, validation_count=16, seed=seed)
    comparison["note"] = (
        "Twelve graphs from six subjects. Pair-disjoint validation can still "
        "share a subject/graph identity; subject-disjoint splits graphs first."
    )
    return comparison


def render_markdown(payload: dict[str, Any]) -> str:
    synthetic = payload["synthetic_grouped_subjects"]
    return "\n".join(
        [
            "# Grouped Split Study",
            "",
            "Pair-disjoint validation keeps `(A,B)` and `(B,A)` on the same side of "
            "the split. It can still share individual graphs or subjects. "
            "Subject-disjoint splits graph identities first, then builds pairs.",
            "",
            "## Synthetic subjects",
            "",
            synthetic["note"],
            "",
            "| Strategy | Train graphs | Val graphs | Graph overlap | Pair overlap |",
            "|---|---:|---:|---:|---:|",
            _row(synthetic["pair_disjoint"]),
            _row(synthetic["subject_disjoint"]),
            "",
        ]
    )


def _row(row: dict[str, Any]) -> str:
    return (
        f"| {row['strategy']} | {row['training_graphs']} | "
        f"{row['validation_graphs']} | {row['graph_overlap']} | {row['pair_overlap']} |"
    )


if __name__ == "__main__":
    raise SystemExit(main())
