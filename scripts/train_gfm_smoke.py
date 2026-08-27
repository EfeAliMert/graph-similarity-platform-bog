from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch


ROOT = Path(__file__).resolve().parents[1]
GFM_MODEL_ROOT = ROOT / "Models&Datasets" / "GFM-code" / "model"
sys.path.insert(0, str(GFM_MODEL_ROOT))

from models import GMS  # noqa: E402
from universal_dataset import (  # noqa: E402
    build_pair_split,
    build_subject_disjoint_pair_split,
    dataset_spec,
    distance_for,
    ensure_training_distances,
    spearman_correlation,
)
from universal_pyg import DEFAULT_MAX_DEGREE, load_pyg_records  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-pairs", type=int, default=1024)
    parser.add_argument("--validation-interval", type=int, default=50)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--identity-probability", type=float, default=0.20)
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--seed", type=int, default=379)
    parser.add_argument("--split-seed", type=int)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = dataset_spec(args.dataset)
    feature_mode = str(dataset.get("feature_mode") or "degree")
    records = load_pyg_records(
        args.dataset,
        feature_mode=feature_mode,
        max_degree=DEFAULT_MAX_DEGREE,
        canonical_order=True,
    )
    train = [record["data"] for record in records if record["split"] == "train"]
    distances, target_metadata = ensure_training_distances(args.dataset)
    if len(train) < 2:
        raise ValueError("At least two training graphs are required.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    number_class = int(train[0].x.size(-1))
    model = GMS(number_class=number_class).to(device)
    if args.initial_checkpoint:
        initial_path = ROOT / args.initial_checkpoint
        initial_payload = torch.load(initial_path, map_location=device)
        initial_dataset = str(initial_payload.get("dataset_id", ""))
        if initial_dataset and initial_dataset != args.dataset:
            raise ValueError(
                f"Initial checkpoint dataset {initial_dataset!r} does not match "
                f"{args.dataset!r}."
            )
        model.load_state_dict(initial_payload["state_dict"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0001,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_norm_ged = normalized_ged_matrix(train, distances)
    pools = build_pair_pools(train_norm_ged, generator)
    split_builder = (
        build_subject_disjoint_pair_split
        if dataset.get("split_strategy") == "subject_disjoint"
        else build_pair_split
    )
    split = split_builder(
        train,
        distances,
        args.validation_pairs,
        args.split_seed if args.split_seed is not None else args.seed + 1,
    )
    validation_pairs = torch.tensor(split["validation_pairs"], dtype=torch.long)
    training_pairs = set(split["training_pairs"])
    training_pairs.update((right, left) for left, right in split["training_pairs"])
    train_pools = [
        pairs[
            torch.tensor(
                [
                    (int(left), int(right)) in training_pairs
                    for left, right in pairs.tolist()
                ],
                dtype=torch.bool,
            )
        ]
        for pairs in pools
    ]
    exact_zero_training_pairs = sum(
        float(train_norm_ged[left, right]) == 0.0
        for left, right in split["training_pairs"]
    )

    best_validation_mse = float("inf")
    best_validation_norm_ged_mae = float("inf")
    best_state = None
    bad_checks = 0
    completed_steps = 0
    checkpoint = ROOT / args.checkpoint
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for step in range(args.steps):
        model.train()
        pairs = sample_pairs(train_pools, args.batch_size, generator)
        pairs = inject_identity_pairs(
            pairs,
            dataset_size=len(train),
            probability=args.identity_probability,
            generator=generator,
        )
        left_ids, right_ids = pairs[:, 0], pairs[:, 1]
        batch = make_batch(train, left_ids, right_ids, device)
        targets = torch.exp(-train_norm_ged[left_ids, right_ids]).float().to(device)
        optimizer.zero_grad()
        prediction = model(batch).reshape(-1)
        loss = F.mse_loss(prediction, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        completed_steps = step + 1

        should_validate = (
            completed_steps % max(1, args.validation_interval) == 0
            or completed_steps == args.steps
        )
        if should_validate:
            validation_mse, validation_norm_ged_mae, validation_spearman = evaluate(
                model,
                train,
                train_norm_ged,
                validation_pairs,
                device,
                args.batch_size,
            )
            print(
                f"step={completed_steps} train_mse={loss.item():.6f} "
                f"validation_mse={validation_mse:.6f} "
                f"validation_norm_ged_mae={validation_norm_ged_mae:.6f} "
                f"validation_spearman={validation_spearman:.6f}"
            )
            if validation_mse < best_validation_mse:
                best_validation_mse = validation_mse
                best_validation_norm_ged_mae = validation_norm_ged_mae
                best_state = copy.deepcopy(model.state_dict())
                bad_checks = 0
                save_checkpoint(
                    checkpoint,
                    best_state,
                    args.dataset,
                    dataset["name"],
                    number_class,
                    target_metadata,
                    args,
                    completed_steps,
                    len(validation_pairs),
                    best_validation_mse,
                    best_validation_norm_ged_mae,
                    split["metadata"],
                    exact_zero_training_pairs,
                    len(train),
                )
            else:
                bad_checks += 1
                if bad_checks >= max(1, args.patience):
                    print(f"early_stopping_step={completed_steps}")
                    break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)

    save_checkpoint(
        checkpoint,
        model.state_dict(),
        args.dataset,
        dataset["name"],
        number_class,
        target_metadata,
        args,
        completed_steps,
        len(validation_pairs),
        best_validation_mse,
        best_validation_norm_ged_mae,
        split["metadata"],
        exact_zero_training_pairs,
        len(train),
    )
    print(f"saved={checkpoint.relative_to(ROOT)}")


def save_checkpoint(
    checkpoint: Path,
    state_dict: dict,
    dataset_id: str,
    dataset_name: str,
    number_class: int,
    target_metadata: dict,
    args: argparse.Namespace,
    completed_steps: int,
    validation_pair_count: int,
    best_validation_mse: float,
    best_validation_norm_ged_mae: float,
    pair_split: dict,
    exact_zero_training_pairs: int,
    training_graph_count: int,
) -> None:
    torch.save(
        {
            "state_dict": state_dict,
            "number_class": number_class,
            "dataset_id": dataset_id,
            "dataset": dataset_name,
            "universal_dataset": True,
            "feature_mode": dataset_spec(dataset_id).get("feature_mode") or "degree",
            "canonical_node_order": True,
            "max_degree": DEFAULT_MAX_DEGREE,
            "target": target_metadata,
            "seed": args.seed,
            "steps": completed_steps,
            "initial_checkpoint": args.initial_checkpoint,
            "batch_size": args.batch_size,
            "hyperparameters": {
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
            },
            "sampling": {
                "strategy": "balanced normalized-GED strata with identity anchors",
                "exact_zero_training_pairs": exact_zero_training_pairs,
                "identity_training_pairs": training_graph_count,
                "identity_probability": args.identity_probability,
            },
            "validation_pairs": validation_pair_count,
            "best_validation_mse": best_validation_mse,
            "best_validation_norm_ged_mae": best_validation_norm_ged_mae,
            "pair_split": pair_split,
        },
        checkpoint,
    )


def normalized_ged_matrix(dataset: list, distances: dict) -> torch.Tensor:
    matrix = torch.full((len(dataset), len(dataset)), float("inf"))
    for left_index, left in enumerate(dataset):
        for right_index, right in enumerate(dataset):
            distance = distance_for(distances, int(left.graph_id), int(right.graph_id))
            if distance is None:
                continue
            denominator = max(0.5 * (int(left.num_nodes) + int(right.num_nodes)), 1.0)
            matrix[left_index, right_index] = float(distance) / denominator
    if not torch.isfinite(matrix).any():
        raise ValueError("No finite training GED pairs were found.")
    return matrix


def pair_bin(norm_ged: torch.Tensor) -> torch.Tensor:
    boundaries = torch.tensor(
        [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
        dtype=norm_ged.dtype,
    )
    return torch.bucketize(norm_ged, boundaries, right=False)


def build_pair_pools(
    norm_ged: torch.Tensor,
    generator: torch.Generator,
) -> list[torch.Tensor]:
    indices = torch.cartesian_prod(
        torch.arange(norm_ged.size(0)),
        torch.arange(norm_ged.size(1)),
    )
    flattened = norm_ged.reshape(-1)
    finite = torch.isfinite(flattened)
    indices = indices[finite]
    bins = pair_bin(flattened[finite])
    non_identity = indices[:, 0] != indices[:, 1]
    indices = indices[non_identity]
    bins = bins[non_identity]
    pools = []
    for bin_index in range(8):
        pool = indices[bins == bin_index]
        if len(pool):
            pool = pool[torch.randperm(len(pool), generator=generator)]
            pools.append(pool)
    return pools


def sample_pairs(
    pools: list[torch.Tensor],
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    active = [pool for pool in pools if len(pool)]
    if not active:
        raise ValueError("No finite GED pairs were available for training.")
    choices = []
    per_pool = max(1, count // len(active))
    for pool in active:
        take = min(per_pool, len(pool))
        indices = torch.randint(len(pool), (take,), generator=generator)
        choices.append(pool[indices])
    while sum(len(item) for item in choices) < count:
        pool = active[len(choices) % len(active)]
        index = torch.randint(len(pool), (1,), generator=generator)
        choices.append(pool[index])
    pairs = torch.cat(choices, dim=0)[:count]
    return pairs[torch.randperm(len(pairs), generator=generator)]


def inject_identity_pairs(
    pairs: torch.Tensor,
    dataset_size: int,
    probability: float,
    generator: torch.Generator,
) -> torch.Tensor:
    identity_count = min(
        len(pairs),
        max(0, round(len(pairs) * max(0.0, min(float(probability), 1.0)))),
    )
    if identity_count == 0:
        return pairs
    graph_ids = torch.randint(
        dataset_size,
        (identity_count,),
        generator=generator,
    )
    anchored = pairs.clone()
    anchored[:identity_count] = torch.stack((graph_ids, graph_ids), dim=1)
    return anchored[torch.randperm(len(anchored), generator=generator)]


def make_batch(
    dataset: list,
    left_ids: torch.Tensor,
    right_ids: torch.Tensor,
    device: torch.device,
) -> dict[str, Batch]:
    return {
        "g1": Batch.from_data_list([dataset[int(index)] for index in left_ids]).to(device),
        "g2": Batch.from_data_list([dataset[int(index)] for index in right_ids]).to(device),
    }


@torch.no_grad()
def evaluate(
    model: GMS,
    dataset: list,
    norm_ged: torch.Tensor,
    pairs: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float, float]:
    model.eval()
    squared_errors = []
    norm_ged_errors = []
    prediction_values = []
    target_values = []
    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start : start + batch_size]
        left_ids, right_ids = batch_pairs[:, 0], batch_pairs[:, 1]
        targets = torch.exp(-norm_ged[left_ids, right_ids]).float().to(device)
        predictions = model(make_batch(dataset, left_ids, right_ids, device)).reshape(-1)
        prediction_values.extend(predictions.detach().cpu().tolist())
        target_values.extend(targets.detach().cpu().tolist())
        squared_errors.append((predictions - targets).pow(2).cpu())
        predicted_norm_ged = -torch.log(predictions.clamp(min=1e-8, max=1.0))
        target_norm_ged = -torch.log(targets.clamp(min=1e-8))
        norm_ged_errors.append((predicted_norm_ged - target_norm_ged).abs().cpu())
    return (
        torch.cat(squared_errors).mean().item(),
        torch.cat(norm_ged_errors).mean().item(),
        spearman_correlation(prediction_values, target_values),
    )


if __name__ == "__main__":
    main()
