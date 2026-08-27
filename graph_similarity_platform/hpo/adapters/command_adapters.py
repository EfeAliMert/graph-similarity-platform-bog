from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .base import (
    ModelHPOAdapter,
    ParameterCapability,
    batch_choices,
    node_cap_choices,
)
from ..types import DatasetProfile, TrialContext


ROOT = Path(__file__).resolve().parents[3]
PYG_PYTHON = ROOT / ".venvs" / "gnn-pyg" / "bin" / "python"
GRAPHSIM_PYTHON = ROOT / ".venvs" / "graphsim" / "bin" / "python"
SIMGNN_ROOT = ROOT / "Models&Datasets" / "SimGNN-v_00001"


class SimGNNHPOAdapter(ModelHPOAdapter):
    model_id = "simgnn"
    display_name = "SimGNN"
    search_space_version = "simgnn-v3"
    resource_name = "epochs"

    def suggest(self, trial: Any, profile: DatasetProfile) -> dict[str, Any]:
        small_data = profile.train_graph_count < 400
        max_nodes = profile.node_count.maximum or 1
        filters_1 = trial.suggest_categorical(
            "filters_1", [32, 64, 128] if small_data else [64, 128, 192]
        )
        filters_2 = max(16, filters_1 // 2)
        filters_3 = max(8, filters_2 // 2)
        histogram = trial.suggest_categorical("histogram", [False, True])
        config = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "dropout": trial.suggest_float("dropout", 0.1, 0.6, step=0.1),
            "batch_size": trial.suggest_categorical(
                "batch_size", batch_choices(profile, (8, 16, 32, 64, 128))
            ),
            "filters_1": filters_1,
            "filters_2": filters_2,
            "filters_3": filters_3,
            "tensor_neurons": trial.suggest_categorical(
                "tensor_neurons", [8, 16, 32]
            ),
            "bottle_neck_neurons": trial.suggest_categorical(
                "bottle_neck_neurons", [8, 16, 32]
            ),
            "histogram": histogram,
        }
        if histogram:
            config["bins"] = trial.suggest_categorical(
                "bins", [8, 16, 32] if max_nodes < 100 else [16, 32]
            )
        else:
            config["bins"] = 16
        return config

    def default_config(self, profile: DatasetProfile) -> dict[str, Any]:
        return {
            "learning_rate": 0.001,
            "weight_decay": 0.0005,
            "dropout": 0.5,
            "batch_size": min(32, max(batch_choices(profile, (8, 16, 32, 64, 128)))),
            "filters_1": 128,
            "filters_2": 64,
            "filters_3": 32,
            "tensor_neurons": 16,
            "bottle_neck_neurons": 16,
            "histogram": False,
            "bins": 16,
        }

    def command(self, context: TrialContext, config: Mapping[str, Any]) -> tuple[list[str], Path]:
        prepared = (
            ROOT
            / "training_logs"
            / "hpo"
            / "prepared"
            / "simgnn"
            / f"{context.profile.fingerprint[:12]}_split{context.split_seed}"
            / context.dataset_id
        )
        command = [
            str(PYG_PYTHON),
            "src/main.py",
            "--training-graphs",
            f"{prepared / 'train'}/",
            "--validation-graphs",
            f"{prepared / 'validation'}/",
            "--testing-graphs",
            f"{prepared / 'test'}/",
            "--epochs",
            str(context.resource),
            "--batch-size",
            str(config["batch_size"]),
            "--learning-rate",
            str(config["learning_rate"]),
            "--weight-decay",
            str(config["weight_decay"]),
            "--dropout",
            str(config["dropout"]),
            "--filters-1",
            str(config["filters_1"]),
            "--filters-2",
            str(config["filters_2"]),
            "--filters-3",
            str(config["filters_3"]),
            "--tensor-neurons",
            str(config["tensor_neurons"]),
            "--bottle-neck-neurons",
            str(config["bottle_neck_neurons"]),
            "--bins",
            str(config["bins"]),
            "--seed",
            str(context.seed),
            "--save-path",
            str(context.checkpoint),
            "--skip-test",
        ]
        if config.get("histogram"):
            command.append("--histogram")
        return command, SIMGNN_ROOT

    def capabilities(self) -> tuple[ParameterCapability, ...]:
        return (
            ParameterCapability("learning_rate", "exposed", "--learning-rate", "Adam learning rate."),
            ParameterCapability("weight_decay", "exposed", "--weight-decay", "Adam L2 penalty."),
            ParameterCapability("dropout", "exposed", "--dropout", "Dropout after the first two GCN layers."),
            ParameterCapability("filters_1..3", "exposed", "--filters-1/2/3", "Three real GCN widths."),
            ParameterCapability("tensor_neurons", "exposed", "--tensor-neurons", "Neural tensor layer width."),
            ParameterCapability("bottle_neck_neurons", "exposed", "--bottle-neck-neurons", "Final hidden width."),
            ParameterCapability("histogram/bins", "exposed", "--histogram/--bins", "Optional node-similarity histogram."),
            ParameterCapability("scheduler", "requires_code_change", "none", "The local trainer has no scheduler hook."),
        )


class GraphSimHPOAdapter(ModelHPOAdapter):
    model_id = "multiscale-set"
    display_name = "Multi-Scale Convolutional Set Matching"
    search_space_version = "graphsim-v2"
    checkpoint_filename = "graphsim.ckpt"

    def suggest(self, trial: Any, profile: DatasetProfile) -> dict[str, Any]:
        zero_max = 0.30 if (profile.zero_target_fraction or 0) >= 0.05 else 0.15
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
            "batch_size": trial.suggest_categorical(
                "batch_size", batch_choices(profile, (4, 8, 16), maximum=16)
            ),
            "patience": trial.suggest_int("patience", 3, 10),
            "zero_fraction": trial.suggest_float("zero_fraction", 0.05, zero_max, step=0.05),
            "identity_fraction": trial.suggest_float("identity_fraction", 0.05, 0.25, step=0.05),
        }

    def default_config(self, profile: DatasetProfile) -> dict[str, Any]:
        return {
            "learning_rate": 0.001,
            "batch_size": min(8, max(batch_choices(profile, (4, 8, 16), maximum=16))),
            "patience": 6,
            "zero_fraction": 0.125,
            "identity_fraction": 0.125,
        }

    def command(self, context: TrialContext, config: Mapping[str, Any]) -> tuple[list[str], Path]:
        return [
            str(GRAPHSIM_PYTHON),
            "scripts/train_graphsim_compat.py",
            "--dataset", context.dataset_id,
            "--checkpoint", str(context.checkpoint),
            "--steps", str(context.resource),
            "--batch-size", str(config["batch_size"]),
            "--learning-rate", str(config["learning_rate"]),
            "--patience", str(config["patience"]),
            "--zero-fraction", str(config["zero_fraction"]),
            "--identity-fraction", str(config["identity_fraction"]),
            "--seed", str(context.seed),
            "--split-seed", str(context.split_seed),
        ], ROOT

    def capabilities(self) -> tuple[ParameterCapability, ...]:
        return (
            ParameterCapability("learning_rate", "exposed", "--learning-rate", "TensorFlow optimizer learning rate."),
            ParameterCapability("batch_size", "exposed", "--batch-size", "Bounded to 16 by the compatibility runner."),
            ParameterCapability("patience", "exposed", "--patience", "Validation early-stopping checks."),
            ParameterCapability("zero_fraction", "exposed", "--zero-fraction", "Distinct GED-zero sampling fraction."),
            ParameterCapability("identity_fraction", "exposed", "--identity-fraction", "Self-pair sampling fraction."),
            ParameterCapability("GCN/CNN dimensions", "internal_not_exposed", "GraphSim config.py", "Requires a versioned model-construction change before HPO."),
        )


class SEGMNHPOAdapter(ModelHPOAdapter):
    model_id = "segmn"
    display_name = "SEGMN"
    search_space_version = "segmn-v2"

    def suggest(self, trial: Any, profile: DatasetProfile) -> dict[str, Any]:
        node_cap = trial.suggest_categorical("node_cap", node_cap_choices(profile))
        return {
            "learning_rate": trial.suggest_float("learning_rate", 5e-5, 2e-3, log=True),
            "batch_size": trial.suggest_categorical(
                "batch_size", batch_choices(profile, (1, 2, 4, 8), maximum=8)
            ),
            "identity_probability": trial.suggest_float(
                "identity_probability", 0.1, 0.3, step=0.05
            ),
            "node_cap": node_cap,
            "edge_cap": max(16, min(int(profile.edge_count.q75 or node_cap * 2), node_cap * 4)),
        }

    def default_config(self, profile: DatasetProfile) -> dict[str, Any]:
        node_cap = min(max(16, int(profile.node_count.q75 or 16)), int(profile.node_count.maximum or 16))
        return {
            "learning_rate": 0.0005,
            "batch_size": min(4, max(batch_choices(profile, (1, 2, 4, 8), maximum=8))),
            "identity_probability": 0.2,
            "node_cap": node_cap,
            "edge_cap": max(16, min(int(profile.edge_count.q75 or node_cap * 2), node_cap * 4)),
        }

    def command(self, context: TrialContext, config: Mapping[str, Any]) -> tuple[list[str], Path]:
        return [
            str(PYG_PYTHON), "scripts/train_segmn_universal.py",
            "--dataset", context.dataset_id,
            "--checkpoint", str(context.checkpoint),
            "--steps", str(context.resource),
            "--batch-size", str(config["batch_size"]),
            "--learning-rate", str(config["learning_rate"]),
            "--identity-probability", str(config["identity_probability"]),
            "--node-cap", str(config["node_cap"]),
            "--edge-cap", str(config["edge_cap"]),
            "--validation-pairs", "128",
            "--validation-interval", str(max(10, min(50, context.resource // 4))),
            "--seed", str(context.seed),
            "--split-seed", str(context.split_seed),
        ], ROOT

    def capabilities(self) -> tuple[ParameterCapability, ...]:
        return (
            ParameterCapability("learning_rate", "exposed", "--learning-rate", "Adam learning rate."),
            ParameterCapability("batch_size", "exposed", "--batch-size", "Pair accumulation count."),
            ParameterCapability("identity_probability", "exposed", "--identity-probability", "Self-pair sampling probability."),
            ParameterCapability("node_cap", "exposed", "--node-cap", "Deterministic assignment-graph memory bound."),
            ParameterCapability("edge_cap", "exposed", "--edge-cap", "Deterministic edge memory bound."),
            ParameterCapability("embedding dimensions", "internal_not_exposed", "build_model_args()", "Requires repository inspection and checkpoint schema migration."),
        )


class GraphFusionHPOAdapter(ModelHPOAdapter):
    model_id = "graph-fusion"
    display_name = "Graph Fusion"
    search_space_version = "graph-fusion-v2"

    def suggest(self, trial: Any, profile: DatasetProfile) -> dict[str, Any]:
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
            "batch_size": trial.suggest_categorical(
                "batch_size", batch_choices(profile, (4, 8, 16, 32, 64))
            ),
            "patience": trial.suggest_int("patience", 4, 16),
            "identity_probability": trial.suggest_float(
                "identity_probability", 0.1, 0.3, step=0.05
            ),
        }

    def default_config(self, profile: DatasetProfile) -> dict[str, Any]:
        return {
            "learning_rate": 0.001,
            "batch_size": min(32, max(batch_choices(profile, (4, 8, 16, 32, 64)))),
            "patience": 12,
            "identity_probability": 0.2,
        }

    def command(self, context: TrialContext, config: Mapping[str, Any]) -> tuple[list[str], Path]:
        return [
            str(PYG_PYTHON), "scripts/train_gfm_smoke.py",
            "--dataset", context.dataset_id,
            "--checkpoint", str(context.checkpoint),
            "--steps", str(context.resource),
            "--batch-size", str(config["batch_size"]),
            "--learning-rate", str(config["learning_rate"]),
            "--patience", str(config["patience"]),
            "--identity-probability", str(config["identity_probability"]),
            "--seed", str(context.seed),
            "--split-seed", str(context.split_seed),
        ], ROOT

    def capabilities(self) -> tuple[ParameterCapability, ...]:
        return (
            ParameterCapability("learning_rate", "exposed", "--learning-rate", "Adam learning rate."),
            ParameterCapability("batch_size", "exposed", "--batch-size", "Graph-pair mini-batch size."),
            ParameterCapability("patience", "exposed", "--patience", "Validation early stopping."),
            ParameterCapability("identity_probability", "exposed", "--identity-probability", "Self-pair anchors."),
            ParameterCapability("attention/hidden dimensions", "internal_not_exposed", "GMS constructor", "The local GMS constructor hard-codes these dimensions."),
        )


class Graph2RegionHPOAdapter(ModelHPOAdapter):
    model_id = "graph2region"
    display_name = "Graph2Region"
    search_space_version = "graph2region-v2"

    def suggest(self, trial: Any, profile: DatasetProfile) -> dict[str, Any]:
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
            "batch_size": trial.suggest_categorical(
                "batch_size", batch_choices(profile, (2, 4, 8, 16), maximum=16)
            ),
            "identity_probability": trial.suggest_float(
                "identity_probability", 0.1, 0.3, step=0.05
            ),
        }

    def default_config(self, profile: DatasetProfile) -> dict[str, Any]:
        return {
            "learning_rate": 0.001,
            "batch_size": min(8, max(batch_choices(profile, (2, 4, 8, 16), maximum=16))),
            "identity_probability": 0.2,
        }

    def command(self, context: TrialContext, config: Mapping[str, Any]) -> tuple[list[str], Path]:
        return [
            str(PYG_PYTHON), "scripts/train_graph2region_universal.py",
            "--dataset", context.dataset_id,
            "--checkpoint", str(context.checkpoint),
            "--steps", str(context.resource),
            "--batch-size", str(config["batch_size"]),
            "--learning-rate", str(config["learning_rate"]),
            "--identity-probability", str(config["identity_probability"]),
            "--validation-pairs", "256",
            "--validation-interval", str(max(10, min(50, context.resource // 4))),
            "--seed", str(context.seed),
            "--split-seed", str(context.split_seed),
        ], ROOT

    def capabilities(self) -> tuple[ParameterCapability, ...]:
        return (
            ParameterCapability("learning_rate", "exposed", "--learning-rate", "Adam learning rate."),
            ParameterCapability("batch_size", "exposed", "--batch-size", "Pair mini-batch size."),
            ParameterCapability("identity_probability", "exposed", "--identity-probability", "Self-pair anchors."),
            ParameterCapability("GNN/region dimensions", "internal_not_exposed", "build_model_args()", "Requires a versioned checkpoint schema change before tuning."),
        )
