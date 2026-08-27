from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_adj, to_dense_batch


ROOT = Path(__file__).resolve().parents[2]
SEGMN_ROOT = ROOT / "Models&Datasets" / "SEGMN-main"
sys.path.insert(0, str(SEGMN_ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from segmn_universal import (  # noqa: E402
    build_args as build_universal_args,
    record_to_segmn,
    transform_pair as transform_universal_pair,
)
from checkpoint_provenance import load_verified_hpo  # noqa: E402
from universal_dataset import dataset_spec, load_graph_records  # noqa: E402

DATASETS = {
    "aids700nef": "AIDS700nef",
    "linux": "LINUX",
}


def adapter_dataset(argv: list[str]) -> str:
    for index, value in enumerate(argv):
        if value == "--dataset" and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith("--dataset="):
            return value.split("=", 1)[1]
    return "aids700nef"


# The original SEGMN files parse argv at import time. Keep their parser on the
# expected dataset arguments and parse this adapter's arguments separately below.
ORIGINAL_ARGV = sys.argv[:]
BOOTSTRAP_DATASET = DATASETS.get(adapter_dataset(ORIGINAL_ARGV), "AIDS700nef")
sys.argv = ["segmn_predict", "--dataset", BOOTSTRAP_DATASET]
from parser1 import parsed_args  # noqa: E402
from new_geddata import GEDDataset  # noqa: E402
from model.SEGMN import SEGMNNet  # noqa: E402

sys.argv = ORIGINAL_ARGV


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
    checkpoint_payload = torch.load(args.checkpoint, map_location=device)
    universal = (
        isinstance(checkpoint_payload, dict)
        and bool(checkpoint_payload.get("universal_dataset"))
    )
    dataset_name = str(dataset_spec(args.dataset)["name"])
    target_metadata = None
    input_projection = None
    projection_metrics = None
    if universal:
        checkpoint_dataset = str(checkpoint_payload.get("dataset_id", ""))
        if checkpoint_dataset and checkpoint_dataset != args.dataset:
            raise ValueError(
                f"Checkpoint dataset {checkpoint_dataset!r} does not match {args.dataset!r}."
            )
        defaults = build_universal_args(device)
        model_args = SimpleNamespace(
            **{
                **vars(defaults),
                **checkpoint_payload["model_args"],
                "device": device,
                "device_count": torch.cuda.device_count(),
            }
        )
        records = load_graph_records(args.dataset)
        left_record = universal_record_by_path(records, args.left_graph)
        right_record = universal_record_by_path(records, args.right_graph)
        left = record_to_segmn(left_record, model_args)
        right = record_to_segmn(right_record, model_args)
        data = transform_universal_pair(left, right, model_args)
        state_dict = checkpoint_payload["state_dict"]
        graph_size = 0.5 * (
            len(left_record["nodes"]) + len(right_record["nodes"])
        )
        target_metadata = checkpoint_payload.get("target")
        input_projection = checkpoint_payload.get("input_projection")
        projection_metrics = {
            "projection_applied": bool(
                len(left_record["nodes"]) > int(model_args.n_max_nodes)
                or len(right_record["nodes"]) > int(model_args.n_max_nodes)
                or len(left_record["edges"]) > int(model_args.n_max_edges)
                or len(right_record["edges"]) > int(model_args.n_max_edges)
            ),
            "left_original_nodes": len(left_record["nodes"]),
            "right_original_nodes": len(right_record["nodes"]),
            "left_used_nodes": int(left.num_nodes),
            "right_used_nodes": int(right.num_nodes),
            "left_original_edges": len(left_record["edges"]),
            "right_original_edges": len(right_record["edges"]),
            "left_used_edges": int(left.numedges.item()),
            "right_used_edges": int(right.numedges.item()),
        }
    else:
        if args.dataset not in DATASETS:
            raise ValueError("This legacy checkpoint has no generic dataset binding.")
        legacy_name = DATASETS[args.dataset]
        dataset_root = ROOT / "Models&Datasets" / "datasets" / legacy_name
        train = GEDDataset(str(dataset_root), legacy_name, train=True)
        test = GEDDataset(str(dataset_root), legacy_name, train=False)
        model_args = parsed_args
        model_args.device = device
        model_args.device_count = torch.cuda.device_count()
        model_args.node_feature_size = train.num_features
        model_args.n_max_nodes = max(g.num_nodes for g in train + test)
        model_args.n_max_edges = max(g.numedges for g in train + test)
        model_args.n_max_l = max(g.l for g in train + test)
        left = graph_by_path(args.left_graph, train, test, dataset_root, legacy_name)
        right = graph_by_path(args.right_graph, train, test, dataset_root, legacy_name)
        data = transform_pair(left, right, train.norm_ged, model_args)
        state_dict = checkpoint_payload
        graph_size = 0.5 * (int(left.num_nodes) + int(right.num_nodes))

    model = SEGMNNet(model_args).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        score = float(model(data).reshape(-1).item())
    if not math.isfinite(score) or not 0.0 < score <= 1.0:
        raise ValueError(f"SEGMN returned an invalid similarity score: {score!r}")
    normalized_ged = max(0.0, -math.log(max(score, 1e-12)))
    print(
        json.dumps(
            {
                "score": score,
                "distance": 1.0 - score,
                "predicted_normalized_ged": normalized_ged,
                "predicted_ged": normalized_ged * graph_size,
                "score_semantics": "exp(-normalized GED)",
                "architecture_class": "model.SEGMN.SEGMNNet",
                "dataset": dataset_name,
                "target": target_metadata,
                "seed": (
                    checkpoint_payload.get("seed")
                    if isinstance(checkpoint_payload, dict)
                    else None
                ),
                "pair_split": (
                    checkpoint_payload.get("pair_split")
                    if isinstance(checkpoint_payload, dict)
                    else None
                ),
                "input_projection": input_projection,
                "compatibility_fixes": (
                    checkpoint_payload.get("compatibility_fixes")
                    if isinstance(checkpoint_payload, dict)
                    else None
                ),
                **(projection_metrics or {"projection_applied": False}),
                "hyperparameters": (
                    checkpoint_payload.get("hyperparameters")
                    if isinstance(checkpoint_payload, dict)
                    else None
                ) or best_trial.get("config"),
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


def universal_record_by_path(records: list[dict], path: str) -> dict:
    member = Path(path)
    split = member.parts[-2]
    graph_id = int(member.stem)
    for record in records:
        if int(record["id"]) == graph_id and record["split"] == split:
            return record
    raise ValueError(f"Graph was not found in the registered dataset: {path}")


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


def transform_pair(left, right, norm_ged: torch.Tensor, args) -> dict:
    left_batch = Batch.from_data_list([left])
    right_batch = Batch.from_data_list([right])
    ass_x, ass_edge_index = assignment_graph(left, right)
    batch_assignment = torch.zeros(ass_x.shape[0], dtype=torch.long)

    return {
        "g0": graph_block(left_batch, args),
        "g1": graph_block(right_batch, args),
        "target": torch.exp(-norm_ged[left.i, right.i]).view(-1).float().to(args.device),
        "ass_x": to_dense_batch(torch.tensor(ass_x, dtype=torch.float), batch=batch_assignment, max_num_nodes=int(args.n_max_nodes) ** 2)[0].to(args.device),
        "ass_x_mask": to_dense_batch(torch.tensor(ass_x, dtype=torch.float), batch=batch_assignment, max_num_nodes=int(args.n_max_nodes) ** 2)[1].to(args.device),
        "ass_edge_index": to_dense_adj(torch.tensor(ass_edge_index, dtype=torch.long), batch=batch_assignment, max_num_nodes=int(args.n_max_nodes) ** 2).to(args.device),
    }


def graph_block(batch: Batch, args) -> dict:
    batch_nodes = batch.batch
    edge_batch = torch.zeros(int(batch.numedges.reshape(-1)[0].item()), dtype=torch.long)
    x_dense = to_dense_batch(batch.x, batch=batch_nodes, max_num_nodes=int(args.n_max_nodes))
    x1_dense = to_dense_batch(batch.x1, batch=edge_batch, max_num_nodes=int(args.n_max_edges.item()))
    return {
        "adj": to_dense_adj(batch.edge_index, batch=batch_nodes, max_num_nodes=int(args.n_max_nodes)).to(args.device),
        "adj1": to_dense_adj(batch.edgeindex1, batch=edge_batch, max_num_nodes=int(args.n_max_edges.item())).to(args.device),
        "x": x_dense[0].to(args.device),
        "x1": x1_dense[0].to(args.device),
        "f": batch.f.view(1, int(args.n_max_l.item()), args.D).to(args.device),
        "h": batch.h.view(1, int(args.n_max_nodes), int(args.n_max_edges.item())).to(args.device),
        "l": batch.l.reshape(-1).to(args.device),
        "numedges": batch.numedges.reshape(-1).to(args.device),
        "edgeindex1": batch.edgeindex1.to(args.device),
        "mask": x_dense[1].to(args.device),
        "mask_x1": x1_dense[1].to(args.device),
    }


def assignment_graph(left, right) -> tuple[np.ndarray, np.ndarray]:
    num_nodes_left = int(left.num_nodes)
    num_nodes_right = int(right.num_nodes)
    ass_x = np.array(list(product(range(num_nodes_left), range(num_nodes_right))), dtype=np.int64)
    neighbors_left = neighbors(left.edge_index.cpu().numpy())
    neighbors_right = neighbors(right.edge_index.cpu().numpy())
    edges = []
    for left_node in range(num_nodes_left):
        for right_node in range(num_nodes_right):
            source = left_node * num_nodes_right + right_node
            for n_left, n_right in product(neighbors_left.get(left_node, []), neighbors_right.get(right_node, [])):
                target = int(n_left) * num_nodes_right + int(n_right)
                edges.append((source, target))
    if not edges:
        return ass_x, np.empty((2, 0), dtype=np.int64)
    return ass_x, np.array(edges, dtype=np.int64).T


def neighbors(edge_index: np.ndarray) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    if edge_index.size == 0:
        return result
    for source, target in edge_index.T:
        result.setdefault(int(source), []).append(int(target))
    return result


if __name__ == "__main__":
    main()
