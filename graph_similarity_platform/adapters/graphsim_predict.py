from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GRAPHSIM_ROOT = ROOT / "Models&Datasets" / "GraphSim-master"
SIAMESE_ROOT = GRAPHSIM_ROOT / "model" / "Siamese"
SOURCE_ROOT = GRAPHSIM_ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from graphsim_calibration import (  # noqa: E402
    apply_isotonic_calibration,
    calibration_position,
    validate_calibration,
)
from checkpoint_provenance import load_verified_hpo  # noqa: E402
from universal_dataset import ensure_training_distances  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--dataset",
        required=True,
    )
    parser.add_argument("--left-graph", required=True)
    parser.add_argument("--right-graph", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = Path(str(args.checkpoint) + ".meta.json")
    metadata = (
        json.loads(metadata_path.read_text())
        if metadata_path.exists()
        else {}
    )
    hpo, hpo_status = load_verified_hpo(Path(args.checkpoint), ROOT)
    best_trial = hpo.get("best_trial") or {}
    if not metadata.get("target"):
        _, target_metadata = ensure_training_distances(args.dataset)
        metadata["target"] = target_metadata
    os.environ["GRAPHSIM_DATASET"] = args.dataset
    os.environ["GRAPHSIM_INFERENCE_ONLY"] = "1"
    if metadata.get("universal_dataset"):
        if metadata.get("dataset_id") not in (None, args.dataset):
            raise ValueError(
                f"Checkpoint dataset {metadata.get('dataset_id')!r} "
                f"does not match {args.dataset!r}."
            )
        os.environ["GRAPHSIM_UNIVERSAL"] = "1"
        if metadata.get("feature_mode") == "categorical_node_label":
            os.environ["GRAPHSIM_NODE_FEATURES"] = "1"
        else:
            os.environ.pop("GRAPHSIM_NODE_FEATURES", None)
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

    data = SiameseModelData(args.dataset)
    calculator = DistSimCalculator(args.dataset, FLAGS.ds_metric, FLAGS.ds_algo)
    model = create_model(FLAGS.model, data.input_dim(), data, calculator)
    left = graph_by_path(args.left_graph, data)
    right = graph_by_path(args.right_graph, data)

    config = tf.ConfigProto(device_count={"GPU": 0})
    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        model.load(session, args.checkpoint)
        feed = model.get_feed_dict_for_val_test(left, right, 0.0, False)
        raw_score = float(session.run(model.pred_sim_without_act(), feed_dict=feed))

    output_calibration = metadata.get("output_calibration")
    validate_calibration(output_calibration)
    audit_mse_raw = output_calibration.get("audit_mse_raw")
    audit_mse_calibrated = output_calibration.get("audit_mse_calibrated")
    calibration_accepted = bool(output_calibration.get("accepted_by_audit", not (
        isinstance(audit_mse_raw, (int, float))
        and isinstance(audit_mse_calibrated, (int, float))
        and float(audit_mse_calibrated) > float(audit_mse_raw)
    )))
    if calibration_accepted:
        score = apply_isotonic_calibration(raw_score, output_calibration)
        prediction_source = "validation_isotonic_calibration"
    else:
        if not 0.0 < raw_score <= 1.0:
            raise ValueError(
                "GraphSim calibration failed its independent audit and the raw "
                "output is outside (0, 1], so no valid GED inversion is available."
            )
        score = raw_score
        prediction_source = "native_output_calibration_rejected_by_audit"
    normalized_ged = max(0.0, -math.log(score) / float(FLAGS.scale))
    graph_size = 0.5 * (
        left.nxgraph.number_of_nodes() + right.nxgraph.number_of_nodes()
    )
    print(
        json.dumps(
            {
                "score": score,
                "raw_score": raw_score,
                "distance": 1.0 - score,
                "predicted_normalized_ged": normalized_ged,
                "predicted_ged": normalized_ged * graph_size,
                "ged_prediction_available": True,
                "ged_prediction_source": prediction_source,
                "calibration_applied": calibration_accepted,
                "calibration_rejected_by_audit": not calibration_accepted,
                "calibration_method": output_calibration["method"],
                "calibration_position": calibration_position(
                    raw_score,
                    output_calibration,
                ),
                "calibration_fit_pairs": output_calibration.get("fit_pair_count"),
                "calibration_audit_pairs": output_calibration.get(
                    "audit_pair_count"
                ),
                "calibration_fit_mse_raw": output_calibration.get("fit_mse_raw"),
                "calibration_fit_mse": output_calibration.get(
                    "fit_mse_calibrated"
                ),
                "calibration_audit_mse_raw": output_calibration.get(
                    "audit_mse_raw"
                ),
                "calibration_audit_mse": output_calibration.get(
                    "audit_mse_calibrated"
                ),
                "calibration_test_graphs_used": output_calibration.get(
                    "test_graphs_used"
                ),
                "raw_out_of_similarity_domain": not 0.0 < raw_score <= 1.0,
                "score_semantics": f"exp(-{float(FLAGS.scale):g} * normalized GED)",
                "score_kernel_scale": float(FLAGS.scale),
                "architecture_class": type(model).__name__,
                "dataset": args.dataset,
                "left_graph": args.left_graph,
                "right_graph": args.right_graph,
                "tensorflow": tf.__version__,
                "compatibility_runtime": "tensorflow.compat.v1",
                "target": metadata.get("target"),
                "seed": metadata.get("seed"),
                "pair_split": metadata.get("pair_split"),
                "validation_protocol": metadata.get("validation_protocol"),
                "hyperparameters": metadata.get("hyperparameters") or best_trial.get("config"),
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


def graph_by_path(path: str, data):
    member = Path(path)
    if len(member.parts) < 2:
        raise ValueError(f"Unsupported graph path: {path}")
    split = member.parts[-2]
    graph_id = int(member.stem)
    candidates = data.test_gs if split == "test" else data.train_gs + data.val_gs
    for graph in candidates:
        if int(graph.nxgraph.graph["gid"]) == graph_id:
            return graph
    raise ValueError(f"Graph {graph_id} was not found in the {split} split.")


if __name__ == "__main__":
    main()
