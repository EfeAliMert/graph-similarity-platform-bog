from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import torch
from torch_geometric.data import Batch
from torch_geometric.datasets import GEDDataset
from torch_geometric.transforms import OneHotDegree
from torch_geometric.utils import degree


ROOT = Path(__file__).resolve().parents[2]
GFM_MODEL_ROOT = ROOT / "Models&Datasets" / "GFM-code" / "model"
sys.path.insert(0, str(GFM_MODEL_ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from models import GMS  # noqa: E402
from checkpoint_provenance import load_verified_hpo  # noqa: E402
from universal_dataset import dataset_spec  # noqa: E402
from universal_pyg import graph_by_relative_path, load_pyg_records  # noqa: E402

DATASETS = {
    "aids700nef": "AIDS700nef",
    "imdbmulti": "IMDBMulti",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--left-graph", required=True)
    parser.add_argument("--right-graph", required=True)
    args = parser.parse_args()
    hpo, hpo_status = load_verified_hpo(Path(args.checkpoint), ROOT)
    best_trial = hpo.get("best_trial") or {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    dataset_name = str(dataset_spec(args.dataset)["name"])
    checkpoint_dataset_id = str(checkpoint.get("dataset_id", ""))
    if checkpoint_dataset_id and checkpoint_dataset_id != args.dataset:
        raise ValueError(
            f"Checkpoint dataset {checkpoint_dataset_id!r} does not match {args.dataset!r}."
        )
    universal = bool(checkpoint.get("universal_dataset"))
    if universal:
        records = load_pyg_records(
            args.dataset,
            feature_mode=str(checkpoint.get("feature_mode", "degree")),
            max_degree=int(checkpoint.get("max_degree", 32)),
            canonical_order=bool(checkpoint.get("canonical_node_order", False)),
        )
        left = graph_by_relative_path(records, args.left_graph)
        right = graph_by_relative_path(records, args.right_graph)
        fallback_class_count = int(left.x.size(-1))
    else:
        if args.dataset not in DATASETS:
            raise ValueError("This legacy checkpoint has no generic dataset binding.")
        legacy_name = DATASETS[args.dataset]
        dataset_root = ROOT / "Models&Datasets" / "GFM-code" / legacy_name
        train = GEDDataset(root=str(dataset_root), name=legacy_name, train=True)
        test = GEDDataset(root=str(dataset_root), name=legacy_name, train=False)
        attach_degree_features(train, test)
        left = graph_by_path(args.left_graph, train, test, dataset_root, legacy_name)
        right = graph_by_path(args.right_graph, train, test, dataset_root, legacy_name)
        fallback_class_count = int(train[0].x.size(-1))

    torch.manual_seed(int(checkpoint.get("seed", 379)))
    model = GMS(
        number_class=int(checkpoint.get("number_class", fallback_class_count))
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with torch.no_grad():
        score = float(model({"g1": Batch.from_data_list([left]).to(device), "g2": Batch.from_data_list([right]).to(device)}).reshape(-1).item())
    if not math.isfinite(score) or not 0.0 < score <= 1.0:
        raise ValueError(f"Graph Fusion returned an invalid similarity score: {score!r}")

    normalized_ged = max(0.0, -math.log(max(score, 1e-12)))
    graph_size = 0.5 * (int(left.num_nodes) + int(right.num_nodes))
    print(
        json.dumps(
            {
                "score": score,
                "distance": 1.0 - score,
                "predicted_normalized_ged": normalized_ged,
                "predicted_ged": normalized_ged * graph_size,
                "score_semantics": "exp(-normalized GED)",
                "architecture_class": "models.GMS",
                "training_steps": checkpoint.get("steps"),
                "training_batch_size": checkpoint.get("batch_size"),
                "dataset": dataset_name,
                "target": checkpoint.get("target"),
                "seed": checkpoint.get("seed"),
                "pair_split": checkpoint.get("pair_split"),
                "hyperparameters": checkpoint.get("hyperparameters") or best_trial.get("config"),
                "hpo": {
                    "study_id": hpo.get("study_id"),
                    "completed_trials": hpo.get("completed_trials"),
                    "validation_mse": best_trial.get("validation_mse"),
                    "test_set_used_for_selection": hpo.get("test_set_used_for_selection"),
                } if hpo else None,
                "hpo_metadata_status": hpo_status,
                "left_graph": args.left_graph,
                "right_graph": args.right_graph,
            }
        )
    )


def graph_by_path(path: str, train: GEDDataset, test: GEDDataset, dataset_root: Path, dataset_name: str):
    parts = Path(path).parts
    if len(parts) < 3:
        raise ValueError(f"Unsupported graph path: {path}")
    split = parts[-2]
    graph_id = int(Path(parts[-1]).stem)
    dataset = train if split == "train" else test
    raw_dir = dataset_root / "raw" / dataset_name / split
    graph_ids = sorted(int(Path(filename).stem) for filename in glob.glob(str(raw_dir / "*.gexf")))
    return dataset[graph_ids.index(graph_id)]


def attach_degree_features(train: GEDDataset, test: GEDDataset) -> None:
    if train[0].x is not None:
        return
    max_degree = 0
    for graph in train + test:
        if graph.edge_index.size(1) > 0:
            max_degree = max(max_degree, int(degree(graph.edge_index[0]).max().item()))
    transform = OneHotDegree(max_degree, cat=False)
    train.transform = transform
    test.transform = transform


if __name__ == "__main__":
    main()
