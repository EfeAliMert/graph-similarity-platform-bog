from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SIMGNN_ROOT = ROOT / "Models&Datasets" / "SimGNN-v_00001"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SIMGNN_ROOT / "src"))

from graph_similarity_platform.adapters.simgnn_predict import (  # noqa: E402
    build_args,
    transfer_pair,
)
from graph_similarity_platform.data import (  # noqa: E402
    load_original_pair,
    pair_ground_truth,
)
from graph_similarity_platform.graph_utils import graph_from_payload  # noqa: E402
from simgnn import SimGNNTrainer  # noqa: E402
from utils import process_pair  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare SimGNN checkpoints on identical held-out graph pairs."
    )
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--dataset", default="aids700nef")
    parser.add_argument("--left")
    parser.add_argument("--right")
    args = parser.parse_args()

    dataset_root = SIMGNN_ROOT / "original_datasets" / args.dataset
    training_graphs = dataset_root / "train"
    testing_graphs = dataset_root / "test"
    results = [
        evaluate_checkpoint(
            Path(checkpoint),
            training_graphs,
            testing_graphs,
            args.dataset,
            args.left,
            args.right,
        )
        for checkpoint in args.checkpoints
    ]
    print(json.dumps({"dataset": args.dataset, "results": results}, indent=2))


def evaluate_checkpoint(
    checkpoint: Path,
    training_graphs: Path,
    testing_graphs: Path,
    dataset_id: str,
    left_member: str | None,
    right_member: str | None,
) -> dict:
    resolved_checkpoint = checkpoint if checkpoint.is_absolute() else ROOT / checkpoint
    trainer_args = build_args(
        f"{training_graphs}/",
        f"{testing_graphs}/",
        str(resolved_checkpoint),
    )
    trainer = SimGNNTrainer(trainer_args)
    trainer.load()
    trainer.model.eval()

    rows = []
    with torch.no_grad():
        for graph_pair_path in sorted(testing_graphs.glob("*.json")):
            data = process_pair(str(graph_pair_path))
            prediction = float(trainer.model(trainer.transfer_to_torch(data)).view(-1).item())
            rows.append(pair_metrics(data, prediction))

    zero_rows = [row for row in rows if row["exact_ged"] == 0.0]
    nonzero_rows = [row for row in rows if row["exact_ged"] > 0.0]
    result = {
        "checkpoint": str(resolved_checkpoint.relative_to(ROOT)),
        "test_pairs": len(rows),
        "similarity_mse": mean(row["similarity_squared_error"] for row in rows),
        "ged_mae": mean(row["ged_absolute_error"] for row in rows),
        "ged_rmse": math.sqrt(mean(row["ged_squared_error"] for row in rows)),
        "exact_zero_pairs": len(zero_rows),
        "exact_zero_mean_similarity": mean(row["predicted_similarity"] for row in zero_rows),
        "exact_zero_ged_mae": mean(row["ged_absolute_error"] for row in zero_rows),
        "nonzero_ged_mae": mean(row["ged_absolute_error"] for row in nonzero_rows),
    }
    if left_member and right_member:
        result["selected_pair"] = evaluate_selected_pair(
            trainer,
            dataset_id,
            left_member,
            right_member,
        )
    return result


def pair_metrics(data: dict, prediction: float) -> dict[str, float]:
    average_size = 0.5 * (len(data["labels_1"]) + len(data["labels_2"]))
    exact_ged = float(data["ged"])
    exact_similarity = math.exp(-exact_ged / max(average_size, 1.0))
    predicted_ged = -math.log(max(prediction, 1e-12)) * average_size
    return {
        "exact_ged": exact_ged,
        "predicted_similarity": prediction,
        "similarity_squared_error": (prediction - exact_similarity) ** 2,
        "ged_absolute_error": abs(predicted_ged - exact_ged),
        "ged_squared_error": (predicted_ged - exact_ged) ** 2,
    }


def evaluate_selected_pair(
    trainer: SimGNNTrainer,
    dataset_id: str,
    left_member: str,
    right_member: str,
) -> dict:
    pair = load_original_pair(dataset_id, left_member, right_member)
    left = graph_from_payload(pair["left"], name="Graph A")
    right = graph_from_payload(pair["right"], name="Graph B")
    data = {
        "graph_1": [list(edge) for edge in left.edges],
        "graph_2": [list(edge) for edge in right.edges],
        "labels_1": [str(label) for label in left.labels],
        "labels_2": [str(label) for label in right.labels],
    }
    with torch.no_grad():
        similarity = float(trainer.model(transfer_pair(trainer, data)).view(-1).item())
    average_size = 0.5 * (left.node_count + right.node_count)
    predicted_ged = -math.log(max(similarity, 1e-12)) * average_size
    ground_truth = pair_ground_truth(
        dataset_id,
        left_member,
        right_member,
        left.node_count,
        right.node_count,
    )
    return {
        "left": left_member,
        "right": right_member,
        "exact_ged": (ground_truth or {}).get("distance"),
        "predicted_similarity": similarity,
        "predicted_ged": predicted_ged,
    }


def mean(values) -> float | None:
    values = list(values)
    return float(np.mean(values)) if values else None


if __name__ == "__main__":
    main()
