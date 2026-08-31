from __future__ import annotations

import importlib.util
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from graph_similarity_platform.data import (
    is_uploaded_dataset,
    original_pair_matches_graphs,
)
from graph_similarity_platform.graph_utils import GraphData


BASE_DIR = Path(__file__).resolve().parents[2]
CHECKPOINT_PATTERNS = ("*.pt", "*.pth", "*.ckpt", "*.ckpt.index")
PYG_ENV = ".venvs/gnn-pyg/bin/python"
GRAPHSIM_ENV = ".venvs/graphsim/bin/python"
DIST_NAMES = {
    "torch_geometric": "torch-geometric",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
}
ALL_DATASETS = [
    "aids700nef",
    "linux",
    "imdbmulti",
    "ptc",
    "mutag",
    "proteins",
    "enzymes",
]


MODELS = [
    {
        "id": "simgnn",
        "name": "SimGNN",
        "family": "Checkpoint-backed GNN",
        "paper": "WSDM 2019",
        "accent": "#dc2626",
        "repository_url": "https://github.com/benedekrozemberczki/SimGNN",
        "implementation_origin": "Community PyTorch reference implementation",
        "implementation_note": "Implements SimGNN from the paper; it is not the authors' official TensorFlow repository.",
        "architecture_class": "simgnn.SimGNNTrainer / SimGNN",
        "checkpoint_origin": "Locally trained in this workspace",
        "checkpoint_note": "Dataset-specific local checkpoint selected by validation similarity MSE; target provenance is shown in the execution detail.",
        "official_pretrained": False,
        "score_semantics": "exp(-normalized GED)",
        "input_binding": "Direct graph payload",
        "local_path": "Models&Datasets/SimGNN-v_00001",
        "entrypoint": "src/main.py",
        "python": PYG_ENV,
        "command": f"{PYG_ENV} src/main.py --load-path <checkpoint>",
        "requires": ["torch", "torch_geometric", "scipy", "pandas"],
        "required_files": [],
        "datasets": ALL_DATASETS,
        "runnable_datasets": ALL_DATASETS,
        "environment": "Project venv at .venvs/gnn-pyg with torch 2.1.1 and torch-geometric 2.4.0. The original repo used older versions, so checkpoint training should still be verified.",
        "setup": [
            "cd Models&Datasets/SimGNN-v_00001",
            "../../.venvs/gnn-pyg/bin/python ../../scripts/prepare_simgnn_original_dataset.py --dataset aids700nef --train-pairs 8000 --validation-pairs 1200 --test-pairs 1200 --clean",
            f"../../{PYG_ENV} src/main.py --training-graphs original_datasets/aids700nef/train/ --validation-graphs original_datasets/aids700nef/validation/ --testing-graphs original_datasets/aids700nef/test/ --epochs 25 --batch-size 128 --save-path checkpoints/simgnn_aids700nef.pt",
        ],
        "needs_checkpoint": True,
        "preferred_checkpoint": "checkpoints/simgnn_aids700nef.pt",
        "checkpoint_by_dataset": {
            "aids700nef": "checkpoints/simgnn_aids700nef.pt",
            "linux": "checkpoints/simgnn_linux.pt",
            "imdbmulti": "checkpoints/simgnn_imdbmulti.pt",
            "ptc": "checkpoints/simgnn_ptc.pt",
            "mutag": "checkpoints/simgnn_mutag.pt",
            "proteins": "checkpoints/simgnn_proteins.pt",
            "enzymes": "checkpoints/simgnn_enzymes.pt",
        },
        "adapter": "simgnn",
        "notes": "Repo entrypoint trains/scores SimGNN dataset folders. Single-pair inference needs a trained checkpoint adapter.",
    },
    {
        "id": "multiscale-set",
        "name": "Multi-Scale Convolutional Set Matching",
        "family": "Checkpoint-backed GNN",
        "paper": "AAAI 2020",
        "accent": "#0891b2",
        "repository_url": "https://github.com/yunshengb/GraphSim",
        "implementation_origin": "Authors' public GraphSim repository",
        "implementation_note": "Original TensorFlow 1.x graph with narrow TensorFlow/networkx compatibility patches for the local runtime.",
        "architecture_class": "GraphSim SiameseRegressionModel",
        "checkpoint_origin": "Locally trained in this workspace",
        "checkpoint_note": "Dataset-specific local checkpoint with validation-only isotonic calibration metadata. Calibration is applied only when its independent validation audit improves MSE.",
        "official_pretrained": False,
        "score_semantics": "exp(-0.7 * normalized GED)",
        "input_binding": "Verified original dataset files",
        "local_path": "Models&Datasets/GraphSim-master",
        "entrypoint": "model/Siamese/run.py",
        "python": GRAPHSIM_ENV,
        "command": f"{GRAPHSIM_ENV} model/Siamese/run.py",
        "requires": ["tensorflow", "networkx", "sklearn", "pandas", "klepto"],
        "required_files": [],
        "datasets": ALL_DATASETS,
        "runnable_datasets": ALL_DATASETS,
        "environment": "Dedicated .venvs/graphsim environment with TensorFlow 2.15.1 running the original TensorFlow 1.x GraphSim graph through tensorflow.compat.v1 on Apple Silicon.",
        "setup": [
            "cd Models&Datasets/GraphSim-master/model/Siamese",
            f"GRAPHSIM_DATASET=aids700nef ../../../../{GRAPHSIM_ENV} run.py",
        ],
        "needs_checkpoint": True,
        "preferred_checkpoint": "checkpoints/aids700nef/graphsim.ckpt",
        "checkpoint_by_dataset": {
            "aids700nef": "checkpoints/aids700nef/graphsim.ckpt",
            "linux": "checkpoints/linux/graphsim.ckpt",
            "imdbmulti": "checkpoints/imdbmulti/graphsim.ckpt",
            "ptc": "checkpoints/ptc/graphsim.ckpt",
            "mutag": "checkpoints/mutag/graphsim.ckpt",
            "proteins": "checkpoints/proteins/graphsim.ckpt",
            "enzymes": "checkpoints/enzymes/graphsim.ckpt",
        },
        "adapter": "multiscale-set",
        "notes": "The original GraphSim GCN, multi-scale matching collector, CNN, and dense stack run through a TensorFlow 1.x compatibility layer. Dataset-specific local checkpoints distinguish exact A* GED, approximate GED benchmark upper bounds, and structural proxies.",
    },
    {
        "id": "segmn",
        "name": "SEGMN",
        "family": "Checkpoint-backed GNN",
        "paper": "arXiv 2024",
        "accent": "#be185d",
        "repository_url": "https://github.com/tourist-wwj/SEGMN",
        "implementation_origin": "Authors' public repository",
        "implementation_note": "Uses the public SEGMNNet architecture at the commit pinned by configs/model_sources.json.",
        "architecture_class": "model.SEGMN.SEGMNNet",
        "checkpoint_origin": "Locally trained in this workspace",
        "checkpoint_note": "Dataset-specific local SEGMNNet checkpoint selected by validation MSE; not author-pretrained.",
        "official_pretrained": False,
        "score_semantics": "exp(-normalized GED)",
        "input_binding": "Verified original dataset files",
        "local_path": "Models&Datasets/SEGMN-main",
        "entrypoint": "main.py",
        "python": PYG_ENV,
        "command": f"../../{PYG_ENV} main.py --load_model True --loaded_model_signature <run-folder>",
        "requires": ["torch", "torch_geometric", "numba", "pynauty"],
        "required_files": [],
        "datasets": ALL_DATASETS,
        "runnable_datasets": ALL_DATASETS,
        "environment": "Project venv at .venvs/gnn-pyg with Python 3.9, torch==2.1.1, torch-geometric==2.4.0, numba==0.58.1, and pynauty==2.8.8.1 for canonical graph labeling.",
        "setup": [
            "cd Models&Datasets/SEGMN-main",
            f"../../{PYG_ENV} main.py",
            f"../../{PYG_ENV} main.py --load_model True --loaded_model_signature AIDS700nef_smoke",
            f"../../{PYG_ENV} main.py --load_model True --loaded_model_signature <run-folder>",
        ],
        "needs_checkpoint": True,
        "preferred_checkpoint": "GSTLogs/AIDS700nef_smoke/best_model.pt",
        "checkpoint_by_dataset": {
            "aids700nef": "checkpoints/aids700nef/segmn_aids700nef_best.pt",
            "linux": "checkpoints/linux/segmn_linux_best.pt",
            "imdbmulti": "checkpoints/imdbmulti/segmn_imdbmulti_best.pt",
            "ptc": "checkpoints/ptc/segmn_ptc_best.pt",
            "mutag": "checkpoints/mutag/segmn_mutag_best.pt",
            "proteins": "checkpoints/proteins/segmn_proteins_best.pt",
            "enzymes": "checkpoints/enzymes/segmn_enzymes_best.pt",
        },
        "adapter": "segmn",
        "notes": "The original SEGMNNet architecture is exposed through dataset-specific local checkpoints and a universal GEXF pair adapter.",
    },
    {
        "id": "graph-fusion",
        "name": "Graph Fusion",
        "family": "Checkpoint-backed GNN",
        "paper": "arXiv 2025",
        "accent": "#4d7c0f",
        "repository_url": "https://github.com/LLiRarry/GFM-code",
        "implementation_origin": "Repository linked by the paper",
        "implementation_note": "Uses the GMS class from the public repository linked directly by arXiv:2502.18291.",
        "architecture_class": "models.GMS",
        "checkpoint_origin": "Locally trained in this workspace",
        "checkpoint_note": "Dataset-specific local GMS checkpoint selected by normalized-GED validation MSE.",
        "official_pretrained": False,
        "score_semantics": "exp(-normalized GED)",
        "input_binding": "Verified original dataset files",
        "local_path": "Models&Datasets/GFM-code",
        "entrypoint": "model/Regression.py",
        "python": PYG_ENV,
        "command": f"../../{PYG_ENV} Regression.py",
        "requires": ["torch", "torch_geometric", "scipy", "tqdm", "pynauty"],
        "required_files": [],
        "datasets": ALL_DATASETS,
        "runnable_datasets": ALL_DATASETS,
        "environment": "Official GFM-code repository cloned locally. The supplied Regression.py targets IMDBMulti; the universal local trainer retains the original GMS architecture for every registered dataset.",
        "setup": [
            "cd Models&Datasets/GFM-code/model",
            f"../../../{PYG_ENV} Regression.py",
        ],
        "needs_checkpoint": True,
        "preferred_checkpoint": "checkpoints/gfm_aids_smoke.pt",
        "checkpoint_by_dataset": {
            "aids700nef": "checkpoints/gfm_aids700nef.pt",
            "linux": "checkpoints/gfm_linux.pt",
            "imdbmulti": "checkpoints/gfm_imdbmulti.pt",
            "ptc": "checkpoints/gfm_ptc.pt",
            "mutag": "checkpoints/gfm_mutag.pt",
            "proteins": "checkpoints/gfm_proteins.pt",
            "enzymes": "checkpoints/gfm_enzymes.pt",
        },
        "adapter": "graph-fusion",
        "notes": "The paper-linked GMS model code is present with dataset-specific local checkpoints and a universal GEXF pair adapter.",
    },
    {
        "id": "graph2region",
        "name": "Graph2Region",
        "family": "Checkpoint-backed GNN",
        "paper": "IEEE TKDE / arXiv 2025",
        "accent": "#c2410c",
        "repository_url": "https://github.com/liuzhouyang/Graph2Region",
        "implementation_origin": "Authors' official repository",
        "implementation_note": "Uses the official G2R region-embedding implementation linked by the paper.",
        "architecture_class": "models.G2R",
        "checkpoint_origin": "Locally trained in this workspace",
        "checkpoint_note": "Dataset-specific local G2R checkpoint selected by validation MSE; not an author-released pretrained checkpoint.",
        "official_pretrained": False,
        "score_semantics": "exp(-normalized GED)",
        "input_binding": "Verified original dataset files",
        "local_path": "Models&Datasets/Graph2Region-main",
        "entrypoint": "run.py",
        "python": PYG_ENV,
        "command": f"../../{PYG_ENV} -m run --dataset_name <dataset_name> --task <ged|mcs>",
        "requires": [
            "torch",
            "torch_geometric",
            "sklearn",
            "scipy",
            "yaml",
            "pynauty",
        ],
        "required_files": ["dataset_g2r.py"],
        "datasets": ALL_DATASETS,
        "runnable_datasets": ALL_DATASETS,
        "environment": "Project venv at .venvs/gnn-pyg with PyTorch/Torch-Geometric packages installed. A local compatibility shim provides dataset_g2r.get_hard_test from dataset.py.",
        "setup": [
            "cd Models&Datasets/Graph2Region-main",
            f"../../{PYG_ENV} run.py --dataset_name aids --experiment ged --task ged",
        ],
        "needs_checkpoint": True,
        "preferred_checkpoint": "exp/ged/g2r_smoke_aids_ged_2026-06-30T2311/g2r_aids_best.pt",
        "checkpoint_by_dataset": {
            "aids700nef": "checkpoints/aids700nef/g2r_aids700nef_best.pt",
            "linux": "checkpoints/linux/g2r_linux_best.pt",
            "imdbmulti": "checkpoints/imdbmulti/g2r_imdbmulti_best.pt",
            "ptc": "checkpoints/ptc/g2r_ptc_best.pt",
            "mutag": "checkpoints/mutag/g2r_mutag_best.pt",
            "proteins": "checkpoints/proteins/g2r_proteins_best.pt",
            "enzymes": "checkpoints/enzymes/g2r_enzymes_best.pt",
        },
        "checkpoint_glob_by_dataset": {
            "aids700nef": "exp/ged/*_aids_ged_*/g2r_aids_best.pt",
            "linux": "exp/ged/*_linux_ged_*/g2r_linux_best.pt",
            "imdbmulti": "exp/ged/*_imdb_ged_*/g2r_imdb_best.pt",
        },
        "adapter": "graph2region",
        "notes": "The authors' G2R architecture is exposed through dataset-specific local checkpoints and a universal GEXF pair adapter.",
    },
]


MODEL_BY_ID = {model["id"]: model for model in MODELS}


def _checkpoint_relative_path(model_id: str, dataset_id: str) -> str:
    paths = {
        "simgnn": f"checkpoints/simgnn_{dataset_id}.pt",
        "multiscale-set": f"checkpoints/{dataset_id}/graphsim.ckpt",
        "segmn": f"checkpoints/{dataset_id}/segmn_{dataset_id}_best.pt",
        "graph-fusion": f"checkpoints/gfm_{dataset_id}.pt",
        "graph2region": f"checkpoints/{dataset_id}/g2r_{dataset_id}_best.pt",
    }
    return paths.get(model_id, "")


def _dataset_runnable(model: dict[str, Any], dataset_id: str | None) -> bool | None:
    if not dataset_id:
        return None
    if is_uploaded_dataset(dataset_id):
        relative_path = _checkpoint_relative_path(model["id"], dataset_id)
        if not relative_path:
            return False
        local_path = BASE_DIR / model["local_path"]
        checkpoint = local_path / relative_path
        checkpoint_exists = checkpoint.exists() or Path(f"{checkpoint}.index").exists()
        if model["id"] != "simgnn":
            return checkpoint_exists
        prepared = local_path / "original_datasets" / dataset_id
        return (
            checkpoint_exists
            and (prepared / "train").exists()
            and (prepared / "test").exists()
        )
    return dataset_id in model.get("runnable_datasets", [])


def _dataset_supported(model: dict[str, Any], dataset_id: str | None) -> bool | None:
    if not dataset_id:
        return None
    if is_uploaded_dataset(dataset_id):
        return bool(_checkpoint_relative_path(model["id"], dataset_id))
    return dataset_id in model.get("datasets", [])


def _supported_datasets(model: dict[str, Any], dataset_id: str | None) -> list[str]:
    datasets = list(model.get("datasets", []))
    if dataset_id and _dataset_supported(model, dataset_id) and dataset_id not in datasets:
        datasets.append(dataset_id)
    return datasets


def _runnable_datasets(model: dict[str, Any], dataset_id: str | None) -> list[str]:
    datasets = list(model.get("runnable_datasets", []))
    if dataset_id and _dataset_runnable(model, dataset_id) and dataset_id not in datasets:
        datasets.append(dataset_id)
    return datasets


def _setup_for_dataset(model: dict[str, Any], dataset_id: str | None) -> list[str]:
    if not dataset_id:
        return model["setup"]
    if model["id"] == "simgnn":
        return [
            (
                f"{PYG_ENV} scripts/prepare_simgnn_original_dataset.py "
                f"--dataset {dataset_id} --clean"
            ),
            "cd Models&Datasets/SimGNN-v_00001",
            (
                f"../../{PYG_ENV} src/main.py "
                f"--training-graphs original_datasets/{dataset_id}/train/ "
                f"--validation-graphs original_datasets/{dataset_id}/validation/ "
                f"--testing-graphs original_datasets/{dataset_id}/test/ "
                f"--save-path checkpoints/simgnn_{dataset_id}.pt"
            ),
        ]
    if model["id"] == "multiscale-set":
        return [
            (
                f"{GRAPHSIM_ENV} scripts/train_graphsim_compat.py "
                f"--dataset {dataset_id} "
                f"--checkpoint Models&Datasets/GraphSim-master/checkpoints/"
                f"{dataset_id}/graphsim.ckpt"
            )
        ]
    if model["id"] == "segmn":
        return [
            (
                f"{PYG_ENV} scripts/train_segmn_universal.py "
                f"--dataset {dataset_id} "
                f"--checkpoint Models&Datasets/SEGMN-main/checkpoints/"
                f"{dataset_id}/segmn_{dataset_id}_best.pt"
            )
        ]
    if model["id"] == "graph-fusion":
        return [
            (
                f"{PYG_ENV} scripts/train_gfm_smoke.py "
                f"--dataset {dataset_id} "
                f"--checkpoint Models&Datasets/GFM-code/checkpoints/"
                f"gfm_{dataset_id}.pt"
            )
        ]
    if model["id"] == "graph2region":
        return [
            (
                f"{PYG_ENV} scripts/train_graph2region_universal.py "
                f"--dataset {dataset_id} "
                f"--checkpoint Models&Datasets/Graph2Region-main/checkpoints/"
                f"{dataset_id}/g2r_{dataset_id}_best.pt"
            )
        ]
    return model["setup"]


def _checkpoint_note(model: dict[str, Any], dataset_id: str | None) -> str | None:
    if dataset_id and is_uploaded_dataset(dataset_id):
        return (
            "Dataset-specific checkpoint trained locally on the uploaded GEXF graphs. "
            "Supplied GED is used when it covers train/train and test/train pairs; "
            "otherwise training uses the explicitly reported structural GED proxy."
        )
    return model.get("checkpoint_note")


def run_models(
    left: GraphData,
    right: GraphData,
    model_ids: list[str],
    dataset_id: str | None = None,
    meta: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    results = []
    runtime_meta = dict(meta or {})
    input_matches_pair = original_pair_matches_graphs(
        dataset_id,
        runtime_meta.get("left_graph"),
        runtime_meta.get("right_graph"),
        left,
        right,
    )
    runtime_meta["input_matches_dataset_pair"] = input_matches_pair
    for model_id in model_ids:
        model = MODEL_BY_ID.get(model_id)
        if model is None:
            continue

        started = time.perf_counter()
        status = inspect_model(model, dataset_id=dataset_id)
        status.setdefault("dataset_runnable", _dataset_runnable(model, dataset_id))
        adapter_result = _run_adapter(model, status, left, right, dataset_id, runtime_meta)
        if adapter_result:
            status = {**status, **adapter_result}
        latency_ms = (time.perf_counter() - started) * 1000.0
        local_path = BASE_DIR / model["local_path"] if model.get("local_path") else None
        checkpoint = (
            _preferred_checkpoint(local_path, model, dataset_id)
            if local_path is not None and status.get("dataset_supported") is not False
            else None
        )
        graph_size = max(0.5 * (left.node_count + right.node_count), 1.0)
        canonical_similarity = _canonical_similarity(status, graph_size)
        adapter_metrics = status.get("adapter_metrics") or {}
        runtime_architecture_class = adapter_metrics.get(
            "architecture_class"
        )
        score_semantics = (
            adapter_metrics.get("score_semantics")
            or model.get("score_semantics")
        )
        score_transformation = {
            "raw_model_output": adapter_metrics.get(
                "raw_score", status.get("score")
            ),
            "native_score": status.get("score"),
            "native_semantics": score_semantics,
            "calibration_applied": adapter_metrics.get(
                "calibration_applied", False
            ),
            "predicted_normalized_ged": adapter_metrics.get(
                "predicted_normalized_ged"
            ),
            "predicted_ged": adapter_metrics.get("predicted_ged"),
            "average_graph_size": graph_size,
            "canonical_similarity": canonical_similarity,
            "canonical_semantics": "exp(-predicted GED / average graph size)",
        }
        results.append(
            {
                "id": model["id"],
                "name": model["name"],
                "family": model["family"],
                "paper": model["paper"],
                "accent": model["accent"],
                "latency_ms": round(latency_ms, 3),
                "score": status.get("score"),
                "model_score": status.get("score"),
                "canonical_similarity": canonical_similarity,
                "comparable_similarity": canonical_similarity,
                "score_transformation": score_transformation,
                "distance": status.get("distance"),
                "status": status["status"],
                "status_label": status["status_label"],
                "detail": status["detail"],
                "command": model["command"],
                "local_path": model["local_path"] or "not found",
                "entrypoint": model["entrypoint"] or "not found",
                "python": model.get("python") or "not configured",
                "requirements": model["requires"],
                "environment": model["environment"],
                "repository_url": model.get("repository_url"),
                "implementation_origin": model.get("implementation_origin"),
                "implementation_note": model.get("implementation_note"),
                "architecture_class": model.get("architecture_class"),
                "runtime_architecture_class": runtime_architecture_class,
                "architecture_loaded": bool(
                    status.get("status") == "executed" and runtime_architecture_class
                ),
                "checkpoint_origin": model.get("checkpoint_origin"),
                "checkpoint_note": _checkpoint_note(model, dataset_id),
                "official_pretrained": bool(model.get("official_pretrained")),
                "selected_checkpoint": (
                    str(checkpoint.relative_to(BASE_DIR)) if checkpoint is not None else None
                ),
                "checkpoint_loaded": bool(
                    status.get("status") == "executed" and checkpoint is not None
                ),
                "score_semantics": score_semantics,
                "input_binding": model.get("input_binding"),
                "input_matches_dataset_pair": input_matches_pair,
                "setup": _setup_for_dataset(model, dataset_id),
                "supported_datasets": _supported_datasets(model, dataset_id),
                "runnable_datasets": _runnable_datasets(model, dataset_id),
                "dataset_supported": status["dataset_supported"],
                "dataset_runnable": status.get("dataset_runnable"),
                "missing_requirements": status["missing_requirements"],
                "missing_runtime": status["missing_runtime"],
                "missing_files": status["missing_files"],
                "checkpoints": status["checkpoints"],
                "adapter_metrics": status.get("adapter_metrics", {}),
                "input_summary": {
                    "left_nodes": left.node_count,
                    "left_edges": left.edge_count,
                    "right_nodes": right.node_count,
                    "right_edges": right.edge_count,
                },
            }
        )
    return results


def _canonical_similarity(status: dict[str, Any], graph_size: float) -> float | None:
    if status.get("status") != "executed":
        return None
    metrics = status.get("adapter_metrics") or {}
    predicted_ged = metrics.get("predicted_ged")
    if (
        isinstance(predicted_ged, (int, float))
        and math.isfinite(float(predicted_ged))
        and float(predicted_ged) >= 0.0
    ):
        return math.exp(-float(predicted_ged) / max(float(graph_size), 1.0))
    return None


def inspect_model(model: dict[str, Any], dataset_id: str | None = None) -> dict[str, Any]:
    local_path = BASE_DIR / model["local_path"] if model["local_path"] else None
    entrypoint = local_path / model["entrypoint"] if local_path and model["entrypoint"] else None
    python_path = _resolve_python(model.get("python", ""))
    missing_runtime = model.get("python") and python_path is None
    missing_requirements = [] if missing_runtime else [
        requirement for requirement in model["requires"] if not _module_available(requirement, python_path)
    ]
    missing_files = _missing_required_files(local_path, model.get("required_files", []))
    checkpoints = _find_checkpoints(local_path) if local_path else []
    dataset_supported = _dataset_supported(model, dataset_id)
    dataset_runnable = _dataset_runnable(model, dataset_id)

    if local_path is None or not local_path.exists():
        return {
            "status": "missing",
            "status_label": "Code missing",
            "detail": model["notes"],
            "missing_requirements": missing_requirements,
            "missing_runtime": bool(missing_runtime),
            "missing_files": missing_files,
            "checkpoints": checkpoints,
            "dataset_supported": dataset_supported,
            "dataset_runnable": dataset_runnable,
        }

    if entrypoint is None or not entrypoint.exists():
        return {
            "status": "missing",
            "status_label": "Entrypoint missing",
            "detail": f"Local folder exists, but {model['entrypoint']} was not found.",
            "missing_requirements": missing_requirements,
            "missing_runtime": bool(missing_runtime),
            "missing_files": missing_files,
            "checkpoints": checkpoints,
            "dataset_supported": dataset_supported,
            "dataset_runnable": dataset_runnable,
        }

    if dataset_supported is False:
        return {
            "status": "dataset_not_supported",
            "status_label": "Dataset mismatch",
            "detail": f"Selected dataset is not listed for this paper architecture. {model['notes']}",
            "missing_requirements": missing_requirements,
            "missing_runtime": bool(missing_runtime),
            "missing_files": missing_files,
            "checkpoints": checkpoints,
            "dataset_supported": dataset_supported,
            "dataset_runnable": dataset_runnable,
        }

    if missing_runtime:
        return {
            "status": "setup_required",
            "status_label": "Runtime missing",
            "detail": f"Required Python runtime was not found: {model.get('python')}. {model['notes']}",
            "missing_requirements": missing_requirements,
            "missing_runtime": True,
            "missing_files": missing_files,
            "checkpoints": checkpoints,
            "dataset_supported": dataset_supported,
            "dataset_runnable": dataset_runnable,
        }

    if missing_requirements:
        missing = ", ".join(missing_requirements)
        return {
            "status": "setup_required",
            "status_label": "Setup required",
            "detail": f"Model not executed. Missing Python packages: {missing}. {model['notes']}",
            "missing_requirements": missing_requirements,
            "missing_runtime": False,
            "missing_files": missing_files,
            "checkpoints": checkpoints,
            "dataset_supported": dataset_supported,
            "dataset_runnable": dataset_runnable,
        }

    if missing_files:
        missing = ", ".join(missing_files)
        return {
            "status": "repo_incomplete",
            "status_label": "Repo incomplete",
            "detail": f"Dependencies are installed, but the local repository is missing required file(s): {missing}. {model['notes']}",
            "missing_requirements": missing_requirements,
            "missing_runtime": False,
            "missing_files": missing_files,
            "checkpoints": checkpoints,
            "dataset_supported": dataset_supported,
            "dataset_runnable": dataset_runnable,
        }

    if model["needs_checkpoint"] and not checkpoints:
        return {
            "status": "checkpoint_required",
            "status_label": "Checkpoint required",
            "detail": f"Model code is present, but no .pt/.pth/.ckpt checkpoint was found. {model['notes']}",
            "missing_requirements": missing_requirements,
            "missing_runtime": False,
            "missing_files": missing_files,
            "checkpoints": checkpoints,
            "dataset_supported": dataset_supported,
        }

    if model.get("adapter") and dataset_supported is not False and dataset_runnable is False:
        runnable = ", ".join(model.get("runnable_datasets", [])) or "none"
        return {
            "status": "checkpoint_required",
            "status_label": "Dataset checkpoint missing",
            "detail": (
                "Model code and dependencies are present, but this local installation does not have a "
                f"checkpoint-backed adapter for the selected dataset yet. Runnable local datasets: {runnable}. {model['notes']}"
            ),
            "missing_requirements": missing_requirements,
            "missing_runtime": False,
            "missing_files": missing_files,
            "checkpoints": checkpoints,
            "dataset_supported": dataset_supported,
            "dataset_runnable": dataset_runnable,
        }

    return {
        "status": "adapter_required",
        "status_label": "Adapter required",
        "detail": f"Dependencies are present. Next step is wiring this original entrypoint to the selected graph pair without fabricating a score. {model['notes']}",
        "missing_requirements": missing_requirements,
        "missing_runtime": False,
        "missing_files": missing_files,
        "checkpoints": checkpoints,
        "dataset_supported": dataset_supported,
        "dataset_runnable": dataset_runnable,
    }


def _module_available(requirement: str, python_path: Path | None = None) -> bool:
    module_name = {
        "sklearn": "sklearn",
        "torch_geometric": "torch_geometric",
        "yaml": "yaml",
        "tensorflow": "tensorflow",
    }.get(requirement, requirement)
    if python_path is not None:
        return _package_installed(str(python_path), DIST_NAMES.get(requirement, requirement))
    return importlib.util.find_spec(module_name) is not None


@lru_cache(maxsize=None)
def _package_installed(python_path: str, dist_name: str) -> bool:
    code = (
        "from importlib import metadata\n"
        "import sys\n"
        "try:\n"
        f"    metadata.version({dist_name!r})\n"
        "except metadata.PackageNotFoundError:\n"
        "    sys.exit(1)\n"
    )
    result = subprocess.run(
        [python_path, "-c", code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=5,
        check=False,
    )
    return result.returncode == 0


def _resolve_python(python_value: str) -> Path | None:
    if not python_value:
        return None
    candidate = Path(python_value)
    if not candidate.is_absolute() and ("/" in python_value or "\\" in python_value):
        candidate = BASE_DIR / candidate
    if candidate.exists():
        return candidate
    executable = shutil.which(python_value)
    return Path(executable) if executable else None


def _missing_required_files(local_path: Path | None, required_files: list[str]) -> list[str]:
    if local_path is None:
        return required_files
    return [path for path in required_files if not (local_path / path).exists()]


def _find_checkpoints(local_path: Path | None) -> list[str]:
    if local_path is None or not local_path.exists():
        return []
    checkpoints = []
    for pattern in CHECKPOINT_PATTERNS:
        try:
            for path in local_path.rglob(pattern):
                if _looks_like_checkpoint(path):
                    checkpoints.append(str(path.relative_to(BASE_DIR)))
        except (FileNotFoundError, OSError):
            # Legacy training code replaces temporary folders while status is polled.
            continue
    return sorted(checkpoints)[:12]


def _looks_like_checkpoint(path: Path) -> bool:
    parts = set(path.parts)
    name = path.name.lower()
    if "processed" in parts or "raw" in parts:
        return False
    if name in {"result.pt"}:
        return False
    return any(token in name for token in ("best", "checkpoint", "ckpt", "epoch", "model", "simgnn", "gfm"))


def _preferred_checkpoint(local_path: Path, model: dict[str, Any], dataset_id: str | None = None) -> Path | None:
    if dataset_id and is_uploaded_dataset(dataset_id):
        preferred = _checkpoint_relative_path(model["id"], dataset_id)
    else:
        preferred = model.get("checkpoint_by_dataset", {}).get(
            dataset_id or "",
            model.get("preferred_checkpoint", ""),
        )
    matches: list[Path] = []
    if preferred:
        direct = local_path / preferred
        if direct.exists() or Path(f"{direct}.index").exists():
            return direct
        try:
            matches.extend(path for path in local_path.rglob(preferred) if path.is_file())
        except (FileNotFoundError, OSError):
            pass
    checkpoint_glob = model.get("checkpoint_glob_by_dataset", {}).get(dataset_id or "")
    if checkpoint_glob:
        matches.extend(path for path in local_path.glob(checkpoint_glob) if path.is_file())
    if not matches:
        return None
    return max(matches, key=_checkpoint_mtime)


def _checkpoint_mtime(path: Path) -> float:
    if path.exists():
        return path.stat().st_mtime
    index_path = Path(f"{path}.index")
    return index_path.stat().st_mtime if index_path.exists() else 0.0


def _run_adapter(
    model: dict[str, Any],
    status: dict[str, Any],
    left: GraphData,
    right: GraphData,
    dataset_id: str | None,
    meta: dict[str, Any],
) -> dict[str, Any] | None:
    adapter = model.get("adapter")
    if (
        adapter != "simgnn"
        and status.get("status") == "adapter_required"
        and meta.get("input_matches_dataset_pair") is False
    ):
        return {
            "status": "input_mismatch",
            "status_label": "Reload dataset pair",
            "detail": (
                "This adapter reads the selected original dataset files. The JSON editors no longer "
                "match those files, so execution was stopped instead of scoring a different pair. "
                "Reload the selected pair, or use SimGNN for direct JSON-payload inference."
            ),
        }
    if adapter == "simgnn":
        return _run_simgnn_adapter(model, status, left, right, dataset_id)
    if adapter == "multiscale-set":
        return _run_multiscale_set_adapter(model, status, dataset_id, meta)
    if adapter == "segmn":
        return _run_segmn_adapter(model, status, dataset_id, meta)
    if adapter == "graph-fusion":
        return _run_graph_fusion_adapter(model, status, dataset_id, meta)
    if adapter == "graph2region":
        return _run_graph2region_adapter(model, status, dataset_id, meta)
    return None


def _run_simgnn_adapter(
    model: dict[str, Any],
    status: dict[str, Any],
    left: GraphData,
    right: GraphData,
    dataset_id: str | None,
) -> dict[str, Any] | None:
    if dataset_id not in ALL_DATASETS and not is_uploaded_dataset(dataset_id):
        return None
    if status["status"] != "adapter_required":
        return None
    local_path = BASE_DIR / model["local_path"]
    checkpoint = _preferred_checkpoint(local_path, model, dataset_id)
    python_path = _resolve_python(model["python"])
    if python_path is None or checkpoint is None:
        return None

    payload = {
        "graph_1": [list(edge) for edge in left.edges],
        "graph_2": [list(edge) for edge in right.edges],
        "labels_1": left.labels,
        "labels_2": right.labels,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(payload, handle)
        payload_path = Path(handle.name)

    try:
        command = [
            str(python_path),
            str(BASE_DIR / "graph_similarity_platform" / "adapters" / "simgnn_predict.py"),
            "--payload",
            str(payload_path),
            "--checkpoint",
            str(checkpoint),
            "--dataset",
            str(dataset_id),
        ]
        metrics, failure = _execute_json_adapter(command, BASE_DIR, 30, "SimGNN")
        if failure is not None:
            return failure
        assert metrics is not None
        score = float(metrics["score"])
        predicted_ged = float(metrics["predicted_ged"])
        detail = (
            "The SimGNN paper architecture executed with a locally trained checkpoint on the selected graph pair. "
            f"Predicted GED {predicted_ged:.4f}; normalized GED {metrics['predicted_normalized_ged']:.4f}."
        )
        detail += _target_note(metrics)
        if metrics.get("unknown_labels"):
            detail += f" Unknown labels were zero-vector encoded: {', '.join(metrics['unknown_labels'])}."
        return {
            "status": "executed",
            "status_label": "Executed",
            "detail": detail,
            "score": score,
            "distance": float(metrics["distance"]),
            "adapter_metrics": metrics,
        }
    except Exception as exc:
        return {
            "status": "adapter_failed",
            "status_label": "Adapter failed",
            "detail": f"SimGNN adapter failed: {type(exc).__name__}: {exc}",
        }
    finally:
        payload_path.unlink(missing_ok=True)


def _execute_json_adapter(
    command: list[str],
    cwd: Path,
    timeout: int,
    label: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        completed = subprocess.run(
            _resource_profile_command(command),
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, {
            "status": "adapter_failed",
            "status_label": "Adapter failed",
            "detail": f"{label} adapter timed out after {timeout} seconds.",
        }

    output_lines = completed.stdout.strip().splitlines()
    metrics = None
    if output_lines:
        try:
            candidate = json.loads(output_lines[-1])
            if isinstance(candidate, dict):
                metrics = candidate
        except (json.JSONDecodeError, TypeError):
            metrics = None

    if completed.returncode != 0 and metrics is None:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no process output"
        return None, {
            "status": "adapter_failed",
            "status_label": "Adapter failed",
            "detail": f"{label} adapter failed: {detail}",
        }
    if not output_lines:
        return None, {
            "status": "adapter_failed",
            "status_label": "Adapter failed",
            "detail": f"{label} adapter returned no JSON result.",
        }
    if metrics is None:
        return None, {
            "status": "adapter_failed",
            "status_label": "Adapter failed",
            "detail": f"{label} adapter returned invalid JSON.",
        }
    metrics.update(_resource_metrics(completed.stderr))
    return metrics, None


def _resource_profile_command(command: list[str]) -> list[str]:
    time_binary = Path("/usr/bin/time")
    if not time_binary.exists():
        return command
    if sys.platform == "darwin":
        return [str(time_binary), "-l", *command]
    if sys.platform.startswith("linux"):
        return [str(time_binary), "-v", *command]
    return command


def _resource_metrics(stderr: str) -> dict[str, int]:
    if not isinstance(stderr, str):
        return {}
    mac_match = re.search(r"^\s*(\d+)\s+maximum resident set size", stderr, re.MULTILINE)
    if mac_match:
        return {"peak_rss_bytes": int(mac_match.group(1))}
    linux_match = re.search(
        r"Maximum resident set size \(kbytes\):\s*(\d+)",
        stderr,
    )
    if linux_match:
        return {"peak_rss_bytes": int(linux_match.group(1)) * 1024}
    return {}


def _run_multiscale_set_adapter(
    model: dict[str, Any],
    status: dict[str, Any],
    dataset_id: str | None,
    meta: dict[str, Any],
) -> dict[str, Any] | None:
    if dataset_id not in ALL_DATASETS and not is_uploaded_dataset(dataset_id):
        return None
    if status["status"] != "adapter_required":
        return None

    local_path = BASE_DIR / model["local_path"]
    checkpoint = _preferred_checkpoint(local_path, model, dataset_id)
    python_path = _resolve_python(model["python"])
    left_graph = meta.get("left_graph")
    right_graph = meta.get("right_graph")
    if python_path is None or checkpoint is None or not left_graph or not right_graph:
        return None

    command = [
        str(python_path),
        str(BASE_DIR / "graph_similarity_platform" / "adapters" / "graphsim_predict.py"),
        "--checkpoint",
        str(checkpoint),
        "--dataset",
        dataset_id,
        "--left-graph",
        str(left_graph),
        "--right-graph",
        str(right_graph),
    ]
    metrics, error = _execute_json_adapter(
        command,
        cwd=local_path / "model" / "Siamese",
        timeout=90,
        label="Multi-Scale Set Matching",
    )
    if error:
        return error
    assert metrics is not None
    detail = (
        "The authors' GraphSim architecture executed through the TensorFlow compatibility runtime "
        f"with a locally trained checkpoint on the selected {metrics['dataset']} graph pair."
    )
    detail += (
        f" Predicted GED {metrics['predicted_ged']:.4f}; "
        f"normalized GED {metrics['predicted_normalized_ged']:.4f}."
    )
    if metrics.get("calibration_applied"):
        detail += (
            f" Raw regression output {metrics['raw_score']:.4f} was mapped by "
            "a validation-only isotonic calibrator before GED inversion."
        )
        if metrics.get("calibration_test_graphs_used") is False:
            detail += " No test graphs or test GED labels were used to fit it."
    elif metrics.get("calibration_rejected_by_audit"):
        detail += (
            " The validation-only calibrator worsened its independent audit MSE "
            "and was rejected; the valid native raw output was used for GED inversion."
        )
    detail += _target_note(metrics)
    return {
        "status": "executed",
        "status_label": "Executed",
        "detail": detail,
        "score": float(metrics["score"]),
        "distance": float(metrics["distance"]),
        "adapter_metrics": metrics,
    }


def _run_segmn_adapter(
    model: dict[str, Any],
    status: dict[str, Any],
    dataset_id: str | None,
    meta: dict[str, Any],
) -> dict[str, Any] | None:
    if dataset_id not in ALL_DATASETS and not is_uploaded_dataset(dataset_id):
        return None
    if status["status"] != "adapter_required":
        return None

    local_path = BASE_DIR / model["local_path"]
    checkpoint = _preferred_checkpoint(local_path, model, dataset_id)
    python_path = _resolve_python(model["python"])
    left_graph = meta.get("left_graph")
    right_graph = meta.get("right_graph")
    if python_path is None or checkpoint is None or not left_graph or not right_graph:
        return None

    command = [
        str(python_path),
        str(BASE_DIR / "graph_similarity_platform" / "adapters" / "segmn_predict.py"),
        "--checkpoint",
        str(checkpoint),
        "--dataset",
        dataset_id,
        "--left-graph",
        str(left_graph),
        "--right-graph",
        str(right_graph),
    ]
    metrics, error = _execute_json_adapter(
        command,
        cwd=local_path,
        timeout=60,
        label="SEGMN",
    )
    if error:
        return error
    assert metrics is not None
    score = float(metrics["score"])
    detail = (
        f"The SEGMN paper architecture executed with a locally trained checkpoint on the selected original {metrics['dataset']} graph pair. "
        f"Predicted GED {metrics['predicted_ged']:.4f}; normalized GED {metrics['predicted_normalized_ged']:.4f}."
    )
    detail += _target_note(metrics)
    return {
        "status": "executed",
        "status_label": "Executed",
        "detail": detail,
        "score": score,
        "distance": float(metrics["distance"]),
        "adapter_metrics": metrics,
    }


def _run_graph_fusion_adapter(
    model: dict[str, Any],
    status: dict[str, Any],
    dataset_id: str | None,
    meta: dict[str, Any],
) -> dict[str, Any] | None:
    if dataset_id not in ALL_DATASETS and not is_uploaded_dataset(dataset_id):
        return None
    if status["status"] != "adapter_required":
        return None

    local_path = BASE_DIR / model["local_path"]
    checkpoint = _preferred_checkpoint(local_path, model, dataset_id)
    python_path = _resolve_python(model["python"])
    left_graph = meta.get("left_graph")
    right_graph = meta.get("right_graph")
    if python_path is None or checkpoint is None or not left_graph or not right_graph:
        return None

    command = [
        str(python_path),
        str(BASE_DIR / "graph_similarity_platform" / "adapters" / "graph_fusion_predict.py"),
        "--checkpoint",
        str(checkpoint),
        "--dataset",
        dataset_id,
        "--left-graph",
        str(left_graph),
        "--right-graph",
        str(right_graph),
    ]
    metrics, error = _execute_json_adapter(
        command,
        cwd=local_path / "model",
        timeout=60,
        label="Graph Fusion",
    )
    if error:
        return error
    assert metrics is not None
    score = float(metrics["score"])
    detail = (
        f"The paper-linked Graph Fusion GMS architecture executed with a locally trained checkpoint on the selected original {metrics['dataset']} graph pair. "
        f"Predicted GED {metrics['predicted_ged']:.4f}; normalized GED {metrics['predicted_normalized_ged']:.4f}."
    )
    detail += _target_note(metrics)
    return {
        "status": "executed",
        "status_label": "Executed",
        "detail": detail,
        "score": score,
        "distance": float(metrics["distance"]),
        "adapter_metrics": metrics,
    }


def _run_graph2region_adapter(
    model: dict[str, Any],
    status: dict[str, Any],
    dataset_id: str | None,
    meta: dict[str, Any],
) -> dict[str, Any] | None:
    if dataset_id not in ALL_DATASETS and not is_uploaded_dataset(dataset_id):
        return None
    if status["status"] != "adapter_required":
        return None

    local_path = BASE_DIR / model["local_path"]
    checkpoint = _preferred_checkpoint(local_path, model, dataset_id)
    python_path = _resolve_python(model["python"])
    left_graph = meta.get("left_graph")
    right_graph = meta.get("right_graph")
    if python_path is None or checkpoint is None or not left_graph or not right_graph:
        return None

    command = [
        str(python_path),
        str(BASE_DIR / "graph_similarity_platform" / "adapters" / "graph2region_predict.py"),
        "--checkpoint",
        str(checkpoint),
        "--dataset",
        dataset_id,
        "--left-graph",
        str(left_graph),
        "--right-graph",
        str(right_graph),
    ]
    if meta.get("disable_compatibility_correction"):
        command.append("--disable-compatibility-correction")
    metrics, error = _execute_json_adapter(
        command,
        cwd=local_path,
        timeout=60,
        label="Graph2Region",
    )
    if error:
        return error
    assert metrics is not None
    score = float(metrics["score"])
    detail = (
        f"The official Graph2Region architecture executed with a locally trained checkpoint on the selected original {metrics['dataset']} graph pair. "
        f"Predicted GED {metrics['predicted_ged']:.4f}; normalized GED {metrics['predicted_normalized_ged']:.4f}."
    )
    detail += _target_note(metrics)
    return {
        "status": "executed",
        "status_label": "Executed",
        "detail": detail,
        "score": score,
        "distance": float(metrics["distance"]),
        "adapter_metrics": metrics,
    }


def _target_note(metrics: dict[str, Any]) -> str:
    target = metrics.get("target") or {}
    source = target.get("target_source")
    target_kind = target.get("target_kind") or metrics.get("reference_kind")
    if not source:
        target_note = ""
    elif target.get("exact") or target_kind == "exact":
        target_note = " Training target: exact benchmark GED."
    elif target_kind == "approximate_benchmark":
        target_note = (
            " Training target: published approximate GED benchmark; the label is "
            "the minimum Beam/Hungarian/VJ upper bound, not exact GED."
        )
    else:
        target_note = " Training target: derived structural GED proxy (not exact GED ground truth)."
    split = metrics.get("pair_split") or {}
    overlap = split.get("pair_overlap_count", split.get("pair_overlap"))
    if overlap == 0:
        split_note = " Validation split metadata: verified zero pair overlap."
    elif metrics.get("validation_protocol"):
        split_note = f" Validation protocol: {metrics['validation_protocol']}."
    else:
        split_note = (
            " Validation split metadata is absent in this checkpoint; retrain it "
            "before using the score in a research accuracy claim."
        )
    projection_note = ""
    if metrics.get("projection_applied"):
        projection = metrics.get("input_projection") or {}
        original_nodes = projection.get("original_nodes")
        used_nodes = projection.get("used_nodes")
        if original_nodes and used_nodes:
            projection_note = (
                f" Input projection: {original_nodes} original nodes were reduced "
                f"deterministically to {used_nodes} nodes for this checkpoint."
            )
        else:
            projection_note = " A deterministic checkpoint input projection was applied."
    return target_note + split_note + projection_note
