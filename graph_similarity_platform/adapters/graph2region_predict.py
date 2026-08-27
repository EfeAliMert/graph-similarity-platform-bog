from __future__ import annotations

import argparse
import glob
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from torch_geometric.data import Batch
from torch_geometric.datasets import GEDDataset


ROOT = Path(__file__).resolve().parents[2]
G2R_ROOT = ROOT / "Models&Datasets" / "Graph2Region-main"
sys.path.insert(0, str(G2R_ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from models import G2R  # noqa: E402
from checkpoint_provenance import load_verified_hpo  # noqa: E402
from universal_dataset import dataset_spec  # noqa: E402
from universal_pyg import graph_by_relative_path, load_pyg_records  # noqa: E402

DATASETS = {
    "aids700nef": "AIDS700nef",
    "linux": "LINUX",
    "imdbmulti": "IMDBMulti",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--left-graph", required=True)
    parser.add_argument("--right-graph", required=True)
    parser.add_argument(
        "--disable-compatibility-correction",
        action="store_true",
        help=(
            "Run inference with the original GED-volume equation and random "
            "positional indices instead of the local compatibility corrections."
        ),
    )
    args = parser.parse_args()
    hpo, hpo_status = load_verified_hpo(Path(args.checkpoint), ROOT)
    best_trial = hpo.get("best_trial") or {}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint_payload = torch.load(args.checkpoint, map_location=device)
    universal = (
        isinstance(checkpoint_payload, dict)
        and bool(checkpoint_payload.get("universal_dataset"))
    )
    dataset_name = str(dataset_spec(args.dataset)["name"])
    target_metadata = None
    validation_protocol = None
    legacy_hyperparameters = None
    checkpoint_metrics = None
    if universal:
        checkpoint_dataset = str(checkpoint_payload.get("dataset_id", ""))
        if checkpoint_dataset and checkpoint_dataset != args.dataset:
            raise ValueError(
                f"Checkpoint dataset {checkpoint_dataset!r} does not match {args.dataset!r}."
            )
        model_args = SimpleNamespace(**checkpoint_payload["model_args"])
        records = load_pyg_records(
            args.dataset,
            feature_mode=str(checkpoint_payload.get("feature_mode", "constant")),
            canonical_order=bool(
                checkpoint_payload.get("canonical_node_order", False)
            ),
        )
        left = graph_by_relative_path(records, args.left_graph)
        right = graph_by_relative_path(records, args.right_graph)
        state_dict = checkpoint_payload["state_dict"]
        target_metadata = checkpoint_payload.get("target")
    else:
        if args.dataset not in DATASETS:
            raise ValueError("This legacy checkpoint has no generic dataset binding.")
        legacy_name = DATASETS[args.dataset]
        model_args = load_args(Path(args.checkpoint))
        configured_dataset = str(getattr(model_args, "dataset_name", ""))
        expected_dataset = {
            "AIDS700nef": "aids",
            "LINUX": "linux",
            "IMDBMulti": "imdb",
        }[legacy_name]
        if configured_dataset and configured_dataset.lower() != expected_dataset:
            raise ValueError(
                f"Checkpoint config dataset {configured_dataset!r} does not match {expected_dataset!r}."
            )
        dataset_root = G2R_ROOT / "GED" / legacy_name
        train = GEDDataset(root=str(dataset_root), name=legacy_name, train=True)
        test = GEDDataset(root=str(dataset_root), name=legacy_name, train=False)
        model_args.input_dim = train[0].x.size(-1) if train[0].x is not None else 1
        left = graph_by_path(args.left_graph, train, test, dataset_root, legacy_name)
        right = graph_by_path(args.right_graph, train, test, dataset_root, legacy_name)
        state_dict = checkpoint_payload
        target_metadata = {
            "dataset_id": args.dataset,
            "target_source": "exact benchmark GED",
            "target_semantics": "exp(-GED / average graph size)",
            "exact": True,
        }
        validation_protocol = (
            f"original repository graph holdout with validation fraction "
            f"{float(getattr(model_args, 'val_pct', 0.2)):.2f}; unordered pair-overlap hash not recorded"
        )
        legacy_hyperparameters = {
            key: getattr(model_args, key)
            for key in (
                "batch_size",
                "epochs",
                "hidden_dim",
                "lr",
                "num_layers",
                "num_perms",
                "output_dim",
                "val_pct",
            )
            if hasattr(model_args, key)
        }
        checkpoint_metrics = load_result_metrics(Path(args.checkpoint).parent / "results")
    seed = model_args.seed[0] if isinstance(model_args.seed, list) else model_args.seed
    checkpoint_geometry = bool(getattr(model_args, "corrected_ged_geometry", False))
    checkpoint_positional = bool(
        getattr(model_args, "deterministic_positional_encoding", False)
    )
    if args.disable_compatibility_correction:
        model_args.corrected_ged_geometry = False
        model_args.deterministic_positional_encoding = False
    random.seed(seed)
    torch.manual_seed(seed)

    model = G2R(model_args).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        score = predict(model, left, right, device, model_args)
    if not math.isfinite(score) or not 0.0 < score <= 1.0:
        raise ValueError(f"Graph2Region returned an invalid similarity score: {score!r}")

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
                "architecture_class": "models.G2R",
                "dataset": dataset_name,
                "target": target_metadata,
                "seed": checkpoint_payload.get("seed") if universal else int(seed),
                "pair_split": (
                    checkpoint_payload.get("pair_split")
                    if universal
                    else None
                ),
                "validation_protocol": validation_protocol,
                "hyperparameters": (
                    checkpoint_payload.get("hyperparameters")
                    if universal
                    else None
                ) or best_trial.get("config") or legacy_hyperparameters,
                "checkpoint_metrics": checkpoint_metrics,
                "compatibility_fixes": (
                    checkpoint_payload.get("compatibility_fixes")
                    if universal
                    else None
                ),
                "compatibility_correction_applied": bool(
                    getattr(model_args, "corrected_ged_geometry", False)
                    or getattr(model_args, "deterministic_positional_encoding", False)
                ),
                "compatibility_checkpoint_flags": {
                    "corrected_ged_geometry": checkpoint_geometry,
                    "deterministic_positional_encoding": checkpoint_positional,
                },
                "compatibility_inference_flags": {
                    "corrected_ged_geometry": bool(
                        getattr(model_args, "corrected_ged_geometry", False)
                    ),
                    "deterministic_positional_encoding": bool(
                        getattr(model_args, "deterministic_positional_encoding", False)
                    ),
                },
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


def load_args(checkpoint: Path) -> SimpleNamespace:
    config_path = checkpoint.parent / "config.yaml"
    config = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    defaults = {
        "dataset_name": "aids",
        "experiment": "ged",
        "task": "ged",
        "norm_ged": "norm",
        "norm_mcs": "norm",
        "num_layers": 2,
        "hidden_dim": 16,
        "layer_type": "GIN",
        "skip_connection": "identity",
        "dropout": 0.0,
        "output_dim": 8,
        "alpha_type": "learnable",
        "num_perms": 2,
        "length_pe": 2,
        "max_num_nodes": 90,
        "norm": "layer",
        "act": "ReLU",
        "score_rep": False,
        "num_tasks": 1,
    }
    defaults.update(config)
    return SimpleNamespace(**defaults)


def load_result_metrics(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    metrics: dict[str, float] = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.partition(" ")
        if not separator:
            continue
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics or None


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


def predict(model: G2R, left, right, device: torch.device, model_args: SimpleNamespace) -> float:
    if left.x is None:
        left.x = torch.tensor(left.num_nodes * [1.0]).unsqueeze(-1)
    if right.x is None:
        right.x = torch.tensor(right.num_nodes * [1.0]).unsqueeze(-1)
    g1 = Batch.from_data_list([left]).to(device)
    g2 = Batch.from_data_list([right]).to(device)
    regions_g1, pe_g1 = model(g1.x, g1.edge_index)
    regions_g2, pe_g2 = model(g2.x, g2.edge_index)
    r_g1 = model.union(regions_g1, pe_g1, g1.batch)
    r_g2 = model.union(regions_g2, pe_g2, g2.batch)
    diffs = r_g1 + r_g2 - 2 * model.intersection(r_g1, r_g2)
    r_g1 = torch.mean(r_g1, 0)
    r_g2 = torch.mean(r_g2, 0)
    score = model.predict_norm_ged(r_g1, r_g2).reshape(-1)
    if bool(model_args.score_rep):
        size_sum = float(left.num_nodes + right.num_nodes)
        score_rep = 2 * model.score_fc(diffs.permute(1, 0, -1).flatten(1)) / size_sum
        score = model.gamma_1 * score + model.beta_1 * score_rep.reshape(-1)
    score = score.item()
    return float(score)


if __name__ == "__main__":
    main()
