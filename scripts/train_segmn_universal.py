from __future__ import annotations

import argparse
import copy
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from segmn_universal import AIDS_ATOM_TYPES, build_args, record_to_segmn, transform_pair
from universal_dataset import (
    build_pair_split,
    build_subject_disjoint_pair_split,
    canonical_pair_key,
    dataset_spec,
    distance_for,
    ensure_training_distances,
    load_graph_records,
    spearman_correlation,
)


ROOT = Path(__file__).resolve().parents[1]
SEGMN_ROOT = ROOT / "Models&Datasets" / "SEGMN-main"
sys.path.insert(0, str(SEGMN_ROOT))

from model.SEGMN import SEGMNNet  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--validation-pairs", type=int, default=128)
    parser.add_argument("--validation-interval", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--identity-probability", type=float, default=0.20)
    parser.add_argument("--node-cap", type=int, default=16)
    parser.add_argument("--edge-cap", type=int, default=32)
    parser.add_argument("--seed", type=int, default=379)
    parser.add_argument("--split-seed", type=int)
    cli = parser.parse_args()

    random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_records = [
        record for record in load_graph_records(cli.dataset)
        if record["split"] == "train"
    ]
    spec = dataset_spec(cli.dataset)
    feature_size = (
        len(raw_records[0]["node_features"][0])
        if raw_records and raw_records[0].get("node_features")
        else None
    )
    label_vocabulary = AIDS_ATOM_TYPES if cli.dataset == "aids700nef" else None
    architecture_profile = {
        "aids700nef": "aids-original",
        "linux": "linux-original",
    }.get(cli.dataset, "compact")
    max_degree = {
        "aids700nef": 6,
        "linux": 7,
    }.get(cli.dataset, 8)
    original_profile = architecture_profile != "compact"
    observed_node_cap = max(len(record["nodes"]) for record in raw_records)
    observed_edge_cap = max(len(record["edges"]) for record in raw_records)
    node_cap = observed_node_cap if original_profile else max(cli.node_cap, observed_node_cap)
    edge_cap = observed_edge_cap if original_profile else max(cli.edge_cap, observed_edge_cap)
    model_args = build_args(
        device,
        node_cap=max(4, min(node_cap, 24)),
        edge_cap=max(4, min(edge_cap, 48)),
        line_edge_cap={"aids700nef": 28, "linux": 30}.get(cli.dataset, 128),
        max_degree=max_degree,
        feature_size=feature_size,
        label_vocabulary=label_vocabulary,
        edge_feature_mode="sum" if label_vocabulary else "concat",
        architecture_profile=architecture_profile,
        canonical_node_order=True,
    )
    records = [
        {
            **record,
            "data": record_to_segmn(record, model_args),
        }
        for record in raw_records
    ]
    distances, target_metadata = ensure_training_distances(cli.dataset)
    split_builder = (
        build_subject_disjoint_pair_split
        if spec.get("split_strategy") == "subject_disjoint"
        else build_pair_split
    )
    split = split_builder(
        records,
        distances,
        cli.validation_pairs,
        cli.split_seed if cli.split_seed is not None else cli.seed + 1,
    )
    validation_pairs = split["validation_pairs"]
    validation_keys = split["validation_keys"]
    zero_training_pairs = [
        (left, right)
        for left, right in split["training_pairs"]
        if distance_for(
            distances,
            int(records[left]["id"]),
            int(records[right]["id"]),
        ) == 0
    ]
    identity_training_pairs = [
        (index, index) for index in range(len(records))
    ]
    rng = random.Random(cli.seed)

    model = SEGMNNet(model_args).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cli.learning_rate,
        weight_decay=0.0001,
    )
    best_mse = float("inf")
    best_state = None
    completed_steps = 0
    for step in range(max(1, cli.steps)):
        model.train()
        training_batch = sample_pairs(
            records,
            distances,
            max(1, cli.batch_size),
            rng,
            excluded_keys=validation_keys,
            preferred_pairs=zero_training_pairs,
            identity_pairs=identity_training_pairs,
            identity_probability=cli.identity_probability,
            allowed_pairs=split["training_pairs"],
        )
        optimizer.zero_grad()
        batch_loss = 0.0
        for left_index, right_index in training_batch:
            target = target_for(
                records[left_index],
                records[right_index],
                distances,
            )
            payload = transform_pair(
                records[left_index]["data"],
                records[right_index]["data"],
                model_args,
                target=target,
            )
            prediction = model(payload).reshape(-1)
            loss = F.mse_loss(prediction, payload["target"])
            (loss / len(training_batch)).backward()
            batch_loss += float(loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        completed_steps = step + 1

        if (
            completed_steps % max(1, cli.validation_interval) == 0
            or completed_steps == max(1, cli.steps)
        ):
            validation_mse, validation_spearman, validation_mae = evaluate_metrics(
                model,
                records,
                validation_pairs,
                distances,
                model_args,
            )
            print(
                f"step={completed_steps} train_mse={batch_loss / len(training_batch):.6f} "
                f"validation_mse={validation_mse:.6f} "
                f"validation_spearman={validation_spearman:.6f} "
                f"validation_mae={validation_mae:.6f}"
            )
            if validation_mse < best_mse:
                best_mse = validation_mse
                best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    checkpoint = ROOT / cli.checkpoint
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    serializable_args = {
        key: value
        for key, value in vars(model_args).items()
        if key not in {"device", "device_count"}
    }
    torch.save(
        {
            "state_dict": best_state,
            "universal_dataset": True,
            "dataset_id": cli.dataset,
            "dataset": dataset_spec(cli.dataset)["name"],
            "model_args": serializable_args,
            "target": target_metadata,
            "seed": cli.seed,
            "steps": completed_steps,
            "batch_size": max(1, cli.batch_size),
            "hyperparameters": {
                "learning_rate": cli.learning_rate,
                "node_cap": int(model_args.n_max_nodes),
                "edge_cap": int(model_args.n_max_edges),
                "batch_size": max(1, cli.batch_size),
            },
            "best_validation_mse": best_mse,
            "pair_split": split["metadata"],
            "sampling": {
                "strategy": "finite-pair sampling with identity and exact-zero anchors",
                "exact_zero_training_pairs": len(zero_training_pairs),
                "identity_training_pairs": len(identity_training_pairs),
                "identity_probability": cli.identity_probability,
            },
            "input_projection": {
                "method": (
                    "label-aware pynauty canonical labeling followed by "
                    "a deterministic leading induced subgraph"
                ),
                "node_cap": int(model_args.n_max_nodes),
                "edge_cap": int(model_args.n_max_edges),
                "node_features": (
                    "atom-type one-hot plus degree one-hot"
                    if label_vocabulary
                    else "continuous features or degree one-hot"
                ),
                "edge_features": str(model_args.edge_feature_mode),
            },
        },
        checkpoint,
    )
    print(f"saved={checkpoint.relative_to(ROOT)}")


def sample_pairs(
    records: list[dict],
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
    pairs = []
    attempts = 0
    available_pairs = list(allowed_pairs or [])
    while len(pairs) < max(1, count) and attempts < max(1, count) * 200:
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
            left = rng.randrange(len(records))
            right = rng.randrange(len(records))
        attempts += 1
        pair_key = canonical_pair_key(
            int(records[left]["id"]),
            int(records[right]["id"]),
        )
        if pair_key in excluded_keys:
            continue
        if distance_for(
            distances,
            int(records[left]["id"]),
            int(records[right]["id"]),
        ) is not None:
            pairs.append((left, right))
    if not pairs:
        raise ValueError("No finite GED pairs were available for training.")
    return pairs


def target_for(left: dict, right: dict, distances: dict) -> float:
    ged = distance_for(distances, int(left["id"]), int(right["id"]))
    denominator = max(
        0.5 * (len(left["nodes"]) + len(right["nodes"])),
        1.0,
    )
    return math.exp(-float(ged) / denominator)


@torch.no_grad()
def evaluate(
    model: SEGMNNet,
    records: list[dict],
    pairs: list[tuple[int, int]],
    distances: dict,
    model_args,
) -> float:
    return evaluate_metrics(model, records, pairs, distances, model_args)[0]


@torch.no_grad()
def evaluate_metrics(
    model: SEGMNNet,
    records: list[dict],
    pairs: list[tuple[int, int]],
    distances: dict,
    model_args,
) -> tuple[float, float, float]:
    model.eval()
    predictions = []
    targets = []
    for left_index, right_index in pairs:
        target = target_for(
            records[left_index],
            records[right_index],
            distances,
        )
        payload = transform_pair(
            records[left_index]["data"],
            records[right_index]["data"],
            model_args,
            target=target,
        )
        prediction = model(payload).reshape(-1)
        predictions.append(float(prediction.item()))
        targets.append(float(payload["target"].item()))
    errors = [prediction - target for prediction, target in zip(predictions, targets)]
    return (
        sum(error * error for error in errors) / len(errors),
        spearman_correlation(predictions, targets),
        sum(abs(error) for error in errors) / len(errors),
    )


if __name__ == "__main__":
    main()
