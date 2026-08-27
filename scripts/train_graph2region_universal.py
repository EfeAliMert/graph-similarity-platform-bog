from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from universal_dataset import (
    build_pair_split,
    build_subject_disjoint_pair_split,
    canonical_pair_key,
    dataset_spec,
    distance_for,
    ensure_training_distances,
    spearman_correlation,
)
from universal_pyg import load_pyg_records


ROOT = Path(__file__).resolve().parents[1]
G2R_ROOT = ROOT / "Models&Datasets" / "Graph2Region-main"
sys.path.insert(0, str(G2R_ROOT))

from models import G2R  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-pairs", type=int, default=256)
    parser.add_argument("--validation-interval", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--identity-probability", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=379)
    parser.add_argument("--split-seed", type=int)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    spec = dataset_spec(args.dataset)
    feature_mode = str(spec.get("feature_mode") or "constant")
    records = load_pyg_records(
        args.dataset,
        feature_mode=feature_mode,
        canonical_order=True,
    )
    train = [record["data"] for record in records if record["split"] == "train"]
    distances, target_metadata = ensure_training_distances(args.dataset)
    if len(train) < 2:
        raise ValueError("At least two training graphs are required.")

    model_args = build_model_args(
        args.dataset,
        max(int(graph.num_nodes) for graph in train),
        args.seed,
        input_dim=int(train[0].x.size(-1)),
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = G2R(model_args).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0001,
    )
    split_builder = (
        build_subject_disjoint_pair_split
        if spec.get("split_strategy") == "subject_disjoint"
        else build_pair_split
    )
    split = split_builder(
        train,
        distances,
        args.validation_pairs,
        args.split_seed if args.split_seed is not None else args.seed + 1,
    )
    validation_pairs = split["validation_pairs"]
    validation_keys = split["validation_keys"]
    zero_training_pairs = [
        (left, right)
        for left, right in split["training_pairs"]
        if distance_for(
            distances,
            int(train[left].graph_id),
            int(train[right].graph_id),
        ) == 0
    ]
    identity_training_pairs = [(index, index) for index in range(len(train))]
    pair_rng = random.Random(args.seed)

    best_mse = float("inf")
    best_state = None
    completed_steps = 0
    for step in range(max(1, args.steps)):
        model.train()
        pairs = sample_finite_pairs(
            train,
            distances,
            max(1, args.batch_size),
            pair_rng,
            excluded_keys=validation_keys,
            preferred_pairs=zero_training_pairs,
            identity_pairs=identity_training_pairs,
            identity_probability=args.identity_probability,
            allowed_pairs=split["training_pairs"],
        )
        prediction, targets = score_pairs(model, train, pairs, distances, device)
        loss = F.mse_loss(prediction, targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        completed_steps = step + 1

        if (
            completed_steps % max(1, args.validation_interval) == 0
            or completed_steps == max(1, args.steps)
        ):
            validation_mse, validation_spearman, validation_mae = evaluate_metrics(
                model,
                train,
                validation_pairs,
                distances,
                device,
            )
            print(
                f"step={completed_steps} train_mse={loss.item():.6f} "
                f"validation_mse={validation_mse:.6f} "
                f"validation_spearman={validation_spearman:.6f} "
                f"validation_mae={validation_mae:.6f}"
            )
            if validation_mse < best_mse:
                best_mse = validation_mse
                best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    checkpoint = ROOT / args.checkpoint
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "universal_dataset": True,
            "dataset_id": args.dataset,
            "dataset": spec["name"],
            "feature_mode": feature_mode,
            "canonical_node_order": True,
            "model_args": vars(model_args),
            "target": target_metadata,
            "seed": args.seed,
            "steps": completed_steps,
            "batch_size": args.batch_size,
            "hyperparameters": {
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
            },
            "best_validation_mse": best_mse,
            "pair_split": split["metadata"],
            "sampling": {
                "strategy": "finite-pair sampling with identity and exact-zero anchors",
                "exact_zero_training_pairs": len(zero_training_pairs),
                "identity_training_pairs": len(identity_training_pairs),
                "identity_probability": args.identity_probability,
            },
            "compatibility_fixes": {
                "corrected_ged_geometry": True,
                "deterministic_positional_encoding": True,
            },
        },
        checkpoint,
    )
    print(f"saved={checkpoint.relative_to(ROOT)}")


def build_model_args(
    dataset_id: str,
    max_nodes: int,
    seed: int,
    input_dim: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_name=dataset_id,
        experiment="ged",
        task="ged",
        input_dim=input_dim,
        num_layers=2,
        hidden_dim=16,
        layer_type="GIN",
        skip_connection="identity",
        dropout=0.0,
        output_dim=8,
        alpha_type="learnable",
        num_perms=2,
        length_pe=2,
        max_num_nodes=max(90, max_nodes + 1),
        norm="layer",
        act="ReLU",
        score_rep=False,
        num_tasks=1,
        seed=seed,
        corrected_ged_geometry=True,
        deterministic_positional_encoding=True,
    )


def sample_finite_pairs(
    dataset: list,
    distances: dict,
    count: int,
    rng: random.Random,
    excluded_keys: set[tuple[int, int]] | None = None,
    preferred_pairs: list[tuple[int, int]] | None = None,
    preferred_probability: float = 0.25,
    identity_pairs: list[tuple[int, int]] | None = None,
    identity_probability: float = 0.20,
    allowed_pairs: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    excluded_keys = excluded_keys or set()
    preferred_pairs = preferred_pairs or []
    identity_pairs = identity_pairs or []
    pairs: list[tuple[int, int]] = []
    attempts = 0
    available_pairs = list(allowed_pairs or [])
    while len(pairs) < count and attempts < count * 200:
        if identity_pairs and rng.random() < identity_probability:
            pairs.append(rng.choice(identity_pairs))
            attempts += 1
            continue
        if preferred_pairs and rng.random() < preferred_probability:
            pairs.append(rng.choice(preferred_pairs))
            attempts += 1
            continue
        if available_pairs:
            left, right = rng.choice(available_pairs)
        else:
            left = rng.randrange(len(dataset))
            right = rng.randrange(len(dataset))
        attempts += 1
        pair_key = canonical_pair_key(
            int(dataset[left].graph_id),
            int(dataset[right].graph_id),
        )
        if pair_key in excluded_keys:
            continue
        if distance_for(
            distances,
            int(dataset[left].graph_id),
            int(dataset[right].graph_id),
        ) is not None:
            pairs.append((left, right))
    if not pairs:
        raise ValueError("No finite GED pairs were available for training.")
    while len(pairs) < count:
        pairs.append(pairs[len(pairs) % len(pairs)])
    return pairs


def score_pairs(
    model: G2R,
    dataset: list,
    pairs: list[tuple[int, int]],
    distances: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    left_graphs = [dataset[left] for left, _ in pairs]
    right_graphs = [dataset[right] for _, right in pairs]
    g1 = Batch.from_data_list(left_graphs).to(device)
    g2 = Batch.from_data_list(right_graphs).to(device)
    regions_g1, pe_g1 = model(g1.x, g1.edge_index)
    regions_g2, pe_g2 = model(g2.x, g2.edge_index)
    # G2R returns [scale, graph, feature]. Average only the multi-scale axis so
    # each graph pair keeps its own region vector and prediction.
    r_g1 = model.union(regions_g1, pe_g1, g1.batch).mean(0)
    r_g2 = model.union(regions_g2, pe_g2, g2.batch).mean(0)
    prediction = model.predict_norm_ged(r_g1, r_g2).reshape(-1)
    targets = []
    for left, right in pairs:
        left_graph, right_graph = dataset[left], dataset[right]
        ged = distance_for(
            distances,
            int(left_graph.graph_id),
            int(right_graph.graph_id),
        )
        denominator = max(
            0.5 * (int(left_graph.num_nodes) + int(right_graph.num_nodes)),
            1.0,
        )
        targets.append(torch.exp(torch.tensor(-float(ged) / denominator)))
    return prediction, torch.stack(targets).float().to(device)


@torch.no_grad()
def evaluate(
    model: G2R,
    dataset: list,
    pairs: list[tuple[int, int]],
    distances: dict,
    device: torch.device,
) -> float:
    return evaluate_metrics(model, dataset, pairs, distances, device)[0]


@torch.no_grad()
def evaluate_metrics(
    model: G2R,
    dataset: list,
    pairs: list[tuple[int, int]],
    distances: dict,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    prediction, targets = score_pairs(model, dataset, pairs, distances, device)
    prediction_values = prediction.detach().cpu().tolist()
    target_values = targets.detach().cpu().tolist()
    return (
        float(F.mse_loss(prediction, targets).item()),
        spearman_correlation(prediction_values, target_values),
        float(F.l1_loss(prediction, targets).item()),
    )


if __name__ == "__main__":
    main()
