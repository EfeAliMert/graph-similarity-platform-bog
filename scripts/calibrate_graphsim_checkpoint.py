from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphsim_calibration import (  # noqa: E402
    calibration_mse,
    fit_isotonic_calibration,
)
from train_graphsim_compat import (  # noqa: E402
    GRAPHSIM_ROOT,
    SIAMESE_ROOT,
    SOURCE_ROOT,
    build_validation_pairs,
    graph_ids,
    predict_pairs,
    split_calibration_graphs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a GraphSim output calibrator using only training-disjoint "
            "validation graphs."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--fit-pairs", type=int, default=256)
    parser.add_argument("--audit-pairs", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else GRAPHSIM_ROOT / "checkpoints" / args.dataset / "graphsim.ckpt"
    )
    metadata_path = Path(str(checkpoint) + ".meta.json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Checkpoint metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("dataset_id") not in (None, args.dataset):
        raise ValueError(
            f"Checkpoint dataset {metadata.get('dataset_id')!r} does not match "
            f"{args.dataset!r}."
        )

    os.environ["GRAPHSIM_DATASET"] = args.dataset
    os.environ["GRAPHSIM_INFERENCE_ONLY"] = "1"
    os.environ["GRAPHSIM_UNIVERSAL"] = "1"
    os.environ["GRAPHSIM_MAX_NODES"] = str(metadata.get("max_nodes", 90))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, str(SIAMESE_ROOT))
    sys.path.insert(0, str(SOURCE_ROOT))

    from config import FLAGS
    from data_siamese import SiameseModelData
    from dist_sim_calculator import DistSimCalculator
    from models_factory import create_model
    from tf_compat import tf

    seed = int(metadata.get("seed", 379))
    data = SiameseModelData(args.dataset)
    calculator = DistSimCalculator(args.dataset, FLAGS.ds_metric, FLAGS.ds_algo)
    model = create_model(FLAGS.model, data.input_dim(), data, calculator)
    fit_graphs, audit_graphs = split_calibration_graphs(data.val_gs, seed)
    fit_pairs = build_validation_pairs(
        data,
        calculator,
        model,
        max(2, args.fit_pairs),
        seed + 1000,
        validation_graphs=fit_graphs,
    )
    audit_pairs = build_validation_pairs(
        data,
        calculator,
        model,
        max(1, args.audit_pairs),
        seed + 2000,
        validation_graphs=audit_graphs,
    )

    config = tf.ConfigProto(device_count={"GPU": 0})
    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        model.load(session, str(checkpoint))
        fit_raw, fit_targets = predict_pairs(session, model, fit_pairs)
        calibration = fit_isotonic_calibration(
            fit_raw,
            fit_targets,
            protocol="post-training validation-only calibration",
            fit_split="calibration validation graphs vs training graphs",
            audit_split=(
                "disjoint calibration-audit validation graphs vs training graphs"
            ),
            fit_graph_ids=graph_ids(fit_graphs),
            audit_graph_ids=graph_ids(audit_graphs),
            fit_audit_graph_overlap=0,
            test_graphs_used=False,
            seed=seed + 1000,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        audit_raw, audit_targets = predict_pairs(
            session,
            model,
            audit_pairs,
        )
        calibration.update(
            calibration_mse(audit_raw, audit_targets, calibration)
        )
        calibration["accepted_by_audit"] = bool(
            calibration["audit_mse_calibrated"]
            <= calibration["audit_mse_raw"]
        )

    metadata["output_calibration"] = calibration
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2))
    temporary.replace(metadata_path)
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "checkpoint": str(checkpoint),
                "fit_pairs": calibration["fit_pair_count"],
                "audit_pairs": calibration["audit_pair_count"],
                "fit_mse_raw": calibration["fit_mse_raw"],
                "fit_mse_calibrated": calibration["fit_mse_calibrated"],
                "audit_mse_raw": calibration["audit_mse_raw"],
                "audit_mse_calibrated": calibration[
                    "audit_mse_calibrated"
                ],
                "test_graphs_used": calibration["test_graphs_used"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
