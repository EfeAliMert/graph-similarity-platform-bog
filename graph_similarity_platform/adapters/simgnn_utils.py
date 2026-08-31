from __future__ import annotations

import numpy as np
import torch


def checkpoint_hyperparameters(
    state_dict: dict,
    recorded: dict | None = None,
) -> dict:
    """Recover architecture dimensions from SimGNN checkpoint tensors."""
    required = (
        "convolution_1.lin.weight",
        "convolution_2.lin.weight",
        "convolution_3.lin.weight",
        "tensor_network.bias",
        "fully_connected_first.weight",
    )
    missing = [name for name in required if name not in state_dict]
    if missing:
        raise ValueError(
            "SimGNN checkpoint is missing architecture tensor(s): "
            + ", ".join(missing)
        )
    values = dict(recorded or {})
    values.update(
        {
            "filters_1": int(state_dict["convolution_1.lin.weight"].shape[0]),
            "filters_2": int(state_dict["convolution_2.lin.weight"].shape[0]),
            "filters_3": int(state_dict["convolution_3.lin.weight"].shape[0]),
            "tensor_neurons": int(state_dict["tensor_network.bias"].shape[0]),
            "bottle_neck_neurons": int(
                state_dict["fully_connected_first.weight"].shape[0]
            ),
        }
    )
    feature_count = int(state_dict["fully_connected_first.weight"].shape[1])
    bins = feature_count - values["tensor_neurons"]
    values["histogram"] = bins > 0
    if bins > 0:
        values["bins"] = bins
    return values


def normalize_graph_labels(labels: list) -> list[str]:
    normalized = [str(label) for label in labels]
    if (
        normalized
        and len(set(normalized)) == len(normalized)
        and all(is_number(label) for label in normalized)
    ):
        return ["0"] * len(normalized)
    return normalized


def is_number(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def edge_index(edges: list[list[int]]) -> torch.LongTensor:
    values = np.asarray(edges, dtype=np.int64).reshape(-1, 2).T
    return torch.from_numpy(values).long()
