from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphsim_calibration import (
    calibration_mse,
    fit_isotonic_calibration,
)
from universal_dataset import (
    dataset_spec,
    ensure_training_distances,
    graph_disjoint_split_metadata,
    load_graph_records,
    serialize_basic_gexf,
    spearman_correlation,
)


GRAPHSIM_ROOT = ROOT / "Models&Datasets" / "GraphSim-master"
SIAMESE_ROOT = GRAPHSIM_ROOT / "model" / "Siamese"
SOURCE_ROOT = GRAPHSIM_ROOT / "src"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-pairs", type=int, default=128)
    parser.add_argument("--validation-interval", type=int, default=50)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--zero-fraction", type=float, default=0.125)
    parser.add_argument("--identity-fraction", type=float, default=0.125)
    parser.add_argument("--seed", type=int, default=379)
    parser.add_argument("--split-seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_seed = args.seed if args.split_seed is None else args.split_seed
    dataset = dataset_spec(args.dataset)
    records = materialize_dataset(args.dataset)
    clear_dataset_caches(args.dataset)
    distances, target_metadata = ensure_training_distances(args.dataset)
    symmetric_distances = OrderedDict()
    for (left, right), distance in distances.items():
        symmetric_distances[(left, right)] = distance
        symmetric_distances[(right, left)] = distance
    for cached_matrix in (
        GRAPHSIM_ROOT / "save" / "ds_mat"
    ).glob(f"{args.dataset}_*.pickle"):
        cached_matrix.unlink()
    os.environ["GRAPHSIM_DATASET"] = args.dataset
    os.environ["GRAPHSIM_UNIVERSAL"] = "1"
    if dataset.get("feature_mode") == "continuous":
        os.environ["GRAPHSIM_NODE_FEATURES"] = "1"
    else:
        os.environ.pop("GRAPHSIM_NODE_FEATURES", None)
    os.environ["GRAPHSIM_MAX_NODES"] = str(
        max(int(len(record["nodes"])) for record in records)
    )
    os.environ["GRAPHSIM_BATCH_SIZE"] = str(max(1, min(args.batch_size, 16)))
    os.environ.pop("GRAPHSIM_INFERENCE_ONLY", None)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, str(SIAMESE_ROOT))
    sys.path.insert(0, str(SOURCE_ROOT))

    from config import FLAGS
    from data_siamese import SiameseModelData
    from dist_sim_calculator import DistSimCalculator
    from models_factory import create_model
    from tf_compat import tf
    from utils import save

    FLAGS.learning_rate = max(1e-6, min(float(args.learning_rate), 0.1))
    if dataset.get("split_strategy") == "subject_disjoint":
        FLAGS.valid_percentage = 0.20
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    steps = max(1, min(args.steps, 2000))

    data = SiameseModelData(args.dataset)
    calculator = DistSimCalculator(args.dataset, FLAGS.ds_metric, FLAGS.ds_algo)
    calculator.gidpair_ds_map = symmetric_distances
    save(calculator.sfn, calculator.gidpair_ds_map)
    model = create_model(FLAGS.model, data.input_dim(), data, calculator)
    training_balance = rebalance_training_triples(
        model,
        args.seed,
        zero_fraction=args.zero_fraction,
        identity_fraction=args.identity_fraction,
    )
    target = Path(args.checkpoint)
    target.parent.mkdir(parents=True, exist_ok=True)
    validation_pairs = build_validation_pairs(
        data,
        calculator,
        model,
        args.validation_pairs,
        split_seed,
        reference_graphs=(data.val_gs if dataset.get("split_strategy") == "subject_disjoint" else None),
    )
    calibration_graphs, calibration_audit_graphs = split_calibration_graphs(
        data.val_gs,
        split_seed,
    )
    calibration_pairs = build_validation_pairs(
        data,
        calculator,
        model,
        args.validation_pairs,
        split_seed + 1000,
        validation_graphs=calibration_graphs,
        reference_graphs=calibration_graphs,
    )
    calibration_audit_pairs = build_validation_pairs(
        data,
        calculator,
        model,
        max(32, args.validation_pairs // 2),
        split_seed + 2000,
        validation_graphs=calibration_audit_graphs,
        reference_graphs=calibration_audit_graphs,
    )

    losses = []
    validation_history = []
    best_validation_mse = float("inf")
    bad_checks = 0
    completed_steps = 0
    config = tf.ConfigProto(device_count={"GPU": 0})
    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(model.vars)
        for step in range(steps):
            feed = model.get_feed_dict_for_train(data, False)
            _, loss = session.run(
                [model.opt_op, model.train_loss],
                feed_dict=feed,
            )
            losses.append(float(loss))
            completed_steps = step + 1
            should_validate = (
                completed_steps % max(1, args.validation_interval) == 0
                or completed_steps == steps
            )
            if should_validate:
                validation_mse, validation_spearman, validation_mae = evaluate_validation_metrics(
                    session,
                    model,
                    validation_pairs,
                )
                validation_history.append((completed_steps, validation_mse))
                print(
                    f"step={completed_steps} train_loss={losses[-1]:.6f} "
                    f"validation_mse={validation_mse:.6f} "
                    f"validation_spearman={validation_spearman:.6f} "
                    f"validation_mae={validation_mae:.6f}"
                )
                if validation_mse < best_validation_mse:
                    best_validation_mse = validation_mse
                    bad_checks = 0
                    saver.save(session, str(target))
                else:
                    bad_checks += 1
                    if bad_checks >= max(1, args.patience):
                        print(f"early_stopping_step={completed_steps}")
                        break
        if not validation_history:
            saver.save(session, str(target))
        saver.restore(session, str(target))
        calibration_raw, calibration_targets = predict_pairs(
            session,
            model,
            calibration_pairs,
        )
        output_calibration = fit_isotonic_calibration(
            calibration_raw,
            calibration_targets,
            protocol="post-training validation-only calibration",
            fit_split="calibration validation graphs vs training graphs",
            audit_split="disjoint calibration-audit validation graphs vs training graphs",
            fit_graph_ids=graph_ids(calibration_graphs),
            audit_graph_ids=graph_ids(calibration_audit_graphs),
            fit_audit_graph_overlap=0,
            test_graphs_used=False,
            seed=split_seed + 1000,
        )
        audit_raw, audit_targets = predict_pairs(
            session,
            model,
            calibration_audit_pairs,
        )
        output_calibration.update(
            calibration_mse(
                audit_raw,
                audit_targets,
                output_calibration,
            )
        )
        output_calibration["accepted_by_audit"] = bool(
            output_calibration["audit_mse_calibrated"]
            <= output_calibration["audit_mse_raw"]
        )

    metadata = {
                "dataset_id": args.dataset,
                "dataset": dataset["name"],
                "steps": completed_steps,
                "checkpoint": str(target),
                "initial_loss": losses[0],
                "final_loss": losses[-1],
                "validation_pairs": len(validation_pairs),
                "best_validation_mse": best_validation_mse,
                "validation_history": validation_history,
                "tensorflow": tf.__version__,
                "compatibility_runtime": "tensorflow.compat.v1",
                "universal_dataset": True,
                "feature_mode": (
                    "categorical_node_label"
                    if dataset.get("feature_mode") == "continuous"
                    else "constant_1"
                ),
                "hyperparameters": {
                    "learning_rate": float(FLAGS.learning_rate),
                    "batch_size": max(1, min(args.batch_size, 16)),
                },
                "training_sampling": training_balance,
                "max_nodes": int(os.environ["GRAPHSIM_MAX_NODES"]),
                "target": target_metadata,
                "seed": args.seed,
                "split_seed": split_seed,
                "validation_protocol": "disjoint validation-graph split",
                "output_calibration": output_calibration,
                "pair_split": graph_disjoint_split_metadata(
                    records,
                    FLAGS.valid_percentage,
                ),
            }
    Path(str(target) + ".meta.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata))


def rebalance_training_triples(
    model,
    seed: int,
    zero_fraction: float = 0.125,
    identity_fraction: float = 0.125,
) -> dict:
    """Balance identity and distinct exact-zero pairs in GraphSim training."""
    sampler = getattr(model, "train_triples", None)
    triples = list(getattr(sampler, "li", []) or [])
    if not triples:
        return {"strategy": "original GraphSim sampler", "exact_zero_pairs": 0}

    zero_pairs = []
    identity_pairs = []
    for triple in triples:
        left, right, target = triple
        left_id = graphsim_graph_id(left)
        right_id = graphsim_graph_id(right)
        if left_id == right_id:
            if math.isclose(float(target), 1.0, rel_tol=0.0, abs_tol=1e-9):
                identity_pairs.append(triple)
            continue
        if math.isclose(float(target), 1.0, rel_tol=0.0, abs_tol=1e-9):
            zero_pairs.append(triple)
    zero_target_count = (
        max(len(zero_pairs), round(len(triples) * zero_fraction))
        if zero_pairs
        else 0
    )
    identity_target_count = (
        max(len(identity_pairs), round(len(triples) * identity_fraction))
        if identity_pairs
        else 0
    )
    rng = random.Random(seed + 30)
    additions = [
        rng.choice(zero_pairs)
        for _ in range(zero_target_count - len(zero_pairs))
    ]
    if identity_pairs:
        additions.extend(
            rng.choice(identity_pairs)
            for _ in range(identity_target_count - len(identity_pairs))
        )
    sampler.li.extend(additions)
    sampler.idx = 0
    sampler._shuffle()
    return {
        "strategy": "original GraphSim triples with identity and distinct exact-zero balancing",
        "original_pairs": len(triples),
        "exact_zero_source_pairs": len(zero_pairs),
        "exact_zero_pairs_after_oversampling": zero_target_count,
        "zero_fraction": zero_fraction,
        "identity_source_pairs": len(identity_pairs),
        "identity_pairs_after_oversampling": identity_target_count,
        "identity_fraction": identity_fraction,
    }


def graphsim_graph_id(graph) -> int | None:
    """Return the benchmark graph id in both inductive and transductive modes."""
    nxgraph = getattr(graph, "nxgraph", None)
    if nxgraph is not None:
        graph_metadata = getattr(nxgraph, "graph", {})
        if "gid" in graph_metadata:
            return int(graph_metadata["gid"])
    global_id = getattr(graph, "global_id", None)
    return None if global_id is None else int(global_id)


def clear_dataset_caches(dataset_id: str) -> None:
    for cache_root in (
        GRAPHSIM_ROOT / "save" / "GenericGEXFData",
        GRAPHSIM_ROOT / "save" / "SiameseModelData",
    ):
        if not cache_root.exists():
            continue
        for path in cache_root.glob(f"*{dataset_id}*"):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def materialize_dataset(dataset_id: str) -> list[dict]:
    records = load_graph_records(dataset_id)
    dataset_root = GRAPHSIM_ROOT / "data" / dataset_id
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    for split in ("train", "test"):
        (dataset_root / split).mkdir(parents=True, exist_ok=True)
    for record in records:
        path = dataset_root / record["split"] / f"{record['id']}.gexf"
        path.write_text(serialize_basic_gexf(record))
    return records


def norm_ged_bin(value: float) -> int:
    if value <= 0.0:
        return 0
    for index, upper in enumerate((0.25, 0.5, 0.75, 1.0, 1.5, 2.0), start=1):
        if value <= upper:
            return index
    return 7


def build_validation_pairs(
    data,
    calculator,
    model,
    count: int,
    seed: int,
    validation_graphs=None,
    reference_graphs=None,
):
    rng = random.Random(seed + 1)
    buckets = defaultdict(list)
    for left in validation_graphs if validation_graphs is not None else data.val_gs:
        for right in reference_graphs if reference_graphs is not None else data.train_gs:
            if left is right:
                continue
            _, norm_ged = calculator.calculate_dist_sim(
                left.nxgraph,
                right.nxgraph,
                return_neg1=True,
            )
            if norm_ged < 0:
                continue
            target = float(model.ds_kernel.dist_to_sim_np(norm_ged))
            buckets[norm_ged_bin(float(norm_ged))].append((left, right, target))
    for pairs in buckets.values():
        rng.shuffle(pairs)

    selected = []
    active = [index for index in range(8) if buckets[index]]
    positions = {index: 0 for index in active}
    count = min(max(1, count), sum(len(buckets[index]) for index in active))
    while len(selected) < count and active:
        progressed = False
        for index in list(active):
            position = positions[index]
            if position < len(buckets[index]):
                selected.append(buckets[index][position])
                positions[index] += 1
                progressed = True
            else:
                active.remove(index)
            if len(selected) >= count:
                break
        if not progressed:
            break
    rng.shuffle(selected)
    return selected


def evaluate_validation(session, model, validation_pairs) -> float:
    return evaluate_validation_metrics(session, model, validation_pairs)[0]


def evaluate_validation_metrics(session, model, validation_pairs) -> tuple[float, float, float]:
    raw_scores, targets = predict_pairs(session, model, validation_pairs)
    errors = np.asarray(raw_scores) - np.asarray(targets)
    return (
        float(np.mean(errors ** 2)),
        spearman_correlation(raw_scores, targets),
        float(np.mean(np.abs(errors))),
    )


def predict_pairs(session, model, pairs) -> tuple[list[float], list[float]]:
    raw_scores = []
    targets = []
    for left, right, target in pairs:
        feed = model.get_feed_dict_for_val_test(left, right, target, False)
        raw_scores.append(
            float(session.run(model.pred_sim_without_act(), feed_dict=feed))
        )
        targets.append(float(target))
    return raw_scores, targets


def split_calibration_graphs(graphs, seed: int):
    shuffled = list(graphs)
    random.Random(seed + 3000).shuffle(shuffled)
    if len(shuffled) < 2:
        raise ValueError(
            "GraphSim calibration requires at least two validation graphs."
        )
    audit_count = max(1, int(round(len(shuffled) * 0.25)))
    audit_count = min(audit_count, len(shuffled) - 1)
    return shuffled[audit_count:], shuffled[:audit_count]


def graph_ids(graphs) -> list[int]:
    return sorted(int(graph.nxgraph.graph["gid"]) for graph in graphs)


if __name__ == "__main__":
    main()
