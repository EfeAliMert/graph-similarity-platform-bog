from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SIMGNN_ROOT = ROOT / "Models&Datasets" / "SimGNN-v_00001"
SIMGNN_SRC = SIMGNN_ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SIMGNN_SRC))
sys.path.insert(0, str(ROOT / "scripts"))

from param_parser import parameter_parser  # noqa: E402
from simgnn import SimGNN  # noqa: E402
from checkpoint_provenance import load_verified_hpo  # noqa: E402
from prepare_simgnn_original_dataset import (  # noqa: E402
    dataset_spec,
    load_graphs,
)
from universal_dataset import ensure_training_distances  # noqa: E402
from graph_similarity_platform.adapters.simgnn_utils import (  # noqa: E402
    checkpoint_hyperparameters,
    edge_index,
    normalize_graph_labels,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--training-graphs")
    parser.add_argument("--validation-graphs")
    parser.add_argument("--testing-graphs")
    args = parser.parse_args()

    payload = json.loads(Path(args.payload).read_text())
    hpo, hpo_status = load_verified_hpo(Path(args.checkpoint), ROOT)
    best_trial = hpo.get("best_trial") or {}
    hyperparameters = best_trial.get("config") or hpo.get("hyperparameters") or {}
    state_dict = torch.load(Path(args.checkpoint), map_location="cpu")
    if not isinstance(state_dict, dict):
        raise ValueError("SimGNN checkpoint does not contain a model state dictionary.")
    hyperparameters = checkpoint_hyperparameters(state_dict, hyperparameters)
    trainer_args = build_args(
        args.training_graphs or "",
        args.validation_graphs,
        args.testing_graphs or "",
        args.checkpoint,
        hyperparameters,
    )
    labels = dataset_labels(args.dataset)
    expected_labels = int(state_dict["convolution_1.lin.weight"].shape[1])
    if len(labels) != expected_labels:
        raise ValueError(
            f"Dataset label vocabulary has {len(labels)} entries, but the "
            f"checkpoint expects {expected_labels}."
        )
    model = SimGNN(trainer_args, len(labels))
    model.load_state_dict(state_dict)
    model.eval()
    trainer = SimpleNamespace(
        model=model,
        global_labels={label: index for index, label in enumerate(labels)},
        number_of_labels=len(labels),
    )

    data = {
        "graph_1": payload["graph_1"],
        "graph_2": payload["graph_2"],
        "labels_1": normalize_graph_labels(payload["labels_1"]),
        "labels_2": normalize_graph_labels(payload["labels_2"]),
        "ged": float(payload.get("ged", 0.0)),
    }
    tensor_data = transfer_pair(trainer, data)
    with torch.no_grad():
        similarity = float(trainer.model(tensor_data).view(-1).item())
    if not math.isfinite(similarity) or not 0.0 < similarity <= 1.0:
        raise ValueError(f"SimGNN returned an invalid similarity score: {similarity!r}")
    normalized_ged = max(0.0, -math.log(max(similarity, 1e-12)))
    graph_size = 0.5 * (len(data["labels_1"]) + len(data["labels_2"]))
    predicted_ged = normalized_ged * graph_size
    _, target = ensure_training_distances(args.dataset)
    print(
        json.dumps(
            {
                "score": similarity,
                "distance": 1.0 - similarity,
                "predicted_normalized_ged": normalized_ged,
                "predicted_ged": predicted_ged,
                "score_semantics": "exp(-normalized GED)",
                "architecture_class": "simgnn.SimGNN",
                "unknown_labels": sorted(unknown_labels(trainer, data)),
                "target": target,
                "seed": hpo.get("seed"),
                "pair_split": hpo.get("pair_split"),
                "hyperparameters": hyperparameters or None,
                "hpo": {
                    "study_id": hpo.get("study_id"),
                    "completed_trials": hpo.get("completed_trials"),
                    "validation_mse": best_trial.get("validation_mse"),
                    "test_set_used_for_selection": hpo.get("test_set_used_for_selection"),
                } if hpo else None,
                "hpo_metadata_status": hpo_status,
            }
        )
    )


def dataset_labels(dataset_id: str) -> list[str]:
    config = dataset_spec(dataset_id)
    graphs = load_graphs(config["archive"], config["format"])
    return sorted(
        {
            str(label)
            for graph in graphs.values()
            for label in graph["labels"]
        }
    )


def build_args(
    training_graphs: str,
    validation_graphs: str | None,
    testing_graphs: str,
    checkpoint: str,
    hyperparameters: dict | None = None,
):
    original_argv = sys.argv
    sys.argv = [
        "simgnn_predict",
        "--training-graphs",
        str(training_graphs),
        "--validation-graphs",
        str(validation_graphs or ""),
        "--testing-graphs",
        str(testing_graphs),
        "--load-path",
        str(checkpoint),
    ]
    bindings = {
        "batch_size": "--batch-size",
        "learning_rate": "--learning-rate",
        "weight_decay": "--weight-decay",
        "dropout": "--dropout",
        "filters_1": "--filters-1",
        "filters_2": "--filters-2",
        "filters_3": "--filters-3",
        "tensor_neurons": "--tensor-neurons",
        "bottle_neck_neurons": "--bottle-neck-neurons",
        "bins": "--bins",
    }
    for key, flag in bindings.items():
        if hyperparameters and key in hyperparameters:
            sys.argv.extend([flag, str(hyperparameters[key])])
    if hyperparameters and hyperparameters.get("histogram"):
        sys.argv.append("--histogram")
    try:
        return parameter_parser()
    finally:
        sys.argv = original_argv


def transfer_pair(trainer, data: dict) -> dict:
    edges_1 = data["graph_1"] + [[target, source] for source, target in data["graph_1"]]
    edges_2 = data["graph_2"] + [[target, source] for source, target in data["graph_2"]]
    return {
        "edge_index_1": edge_index(edges_1),
        "edge_index_2": edge_index(edges_2),
        "features_1": labels_to_features(trainer, data["labels_1"]),
        "features_2": labels_to_features(trainer, data["labels_2"]),
    }


def labels_to_features(trainer, labels: list[str]) -> torch.FloatTensor:
    rows = []
    for label in labels:
        row = [0.0] * trainer.number_of_labels
        if label in trainer.global_labels:
            row[trainer.global_labels[label]] = 1.0
        rows.append(row)
    return torch.FloatTensor(np.array(rows, dtype=np.float32))


def unknown_labels(trainer, data: dict) -> set[str]:
    labels = set(data["labels_1"]).union(data["labels_2"])
    return {label for label in labels if label not in trainer.global_labels}


if __name__ == "__main__":
    main()
