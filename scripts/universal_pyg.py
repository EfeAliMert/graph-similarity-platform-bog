from __future__ import annotations

from pathlib import Path
from typing import Any

import pynauty
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from universal_dataset import load_graph_records


DEFAULT_MAX_DEGREE = 32


def canonical_node_order(record: dict[str, Any]) -> list[Any]:
    """Return a feature-aware canonical node order for an attributed graph."""
    nodes = list(record["nodes"])
    if len(nodes) < 2:
        return nodes
    positions = {node_id: index for index, node_id in enumerate(nodes)}
    adjacency = {index: [] for index in range(len(nodes))}
    for source, target in record["edges"]:
        if source not in positions or target not in positions or source == target:
            continue
        left = positions[source]
        right = positions[target]
        adjacency[left].append(right)
        adjacency[right].append(left)
    adjacency = {
        index: sorted(set(neighbors))
        for index, neighbors in adjacency.items()
    }

    raw_features = record.get("node_features")
    labels = list(record.get("labels") or [])
    color_groups: dict[str, set[int]] = {}
    for index in range(len(nodes)):
        if raw_features is not None:
            token = repr(tuple(float(value) for value in raw_features[index]))
        elif index < len(labels):
            token = str(labels[index])
        else:
            token = "__unattributed__"
        color_groups.setdefault(token, set()).add(index)

    graph = pynauty.Graph(
        number_of_vertices=len(nodes),
        directed=False,
        adjacency_dict=adjacency,
        vertex_coloring=[color_groups[token] for token in sorted(color_groups)],
    )
    return [nodes[index] for index in pynauty.canon_label(graph)]


def load_pyg_records(
    dataset_id: str,
    feature_mode: str = "degree",
    max_degree: int = DEFAULT_MAX_DEGREE,
    canonical_order: bool = False,
) -> list[dict[str, Any]]:
    return [
        {
            **record,
            "data": record_to_data(
                record,
                feature_mode=feature_mode,
                max_degree=max_degree,
                canonical_order=canonical_order,
            ),
        }
        for record in load_graph_records(dataset_id)
    ]


def record_to_data(
    record: dict[str, Any],
    feature_mode: str = "degree",
    max_degree: int = DEFAULT_MAX_DEGREE,
    canonical_order: bool = False,
) -> Data:
    ordered_nodes = (
        canonical_node_order(record)
        if canonical_order
        else list(record["nodes"])
    )
    node_map = {node_id: index for index, node_id in enumerate(ordered_nodes)}
    undirected_edges: set[tuple[int, int]] = set()
    for source, target in record["edges"]:
        if source not in node_map or target not in node_map or source == target:
            continue
        left, right = node_map[source], node_map[target]
        undirected_edges.add(tuple(sorted((left, right))))
    directed_edges = [
        directed
        for left, right in sorted(undirected_edges)
        for directed in ((left, right), (right, left))
    ]
    edge_index = (
        torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
        if directed_edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    node_count = len(node_map)
    if feature_mode == "continuous":
        features = record.get("node_features")
        if not features or len(features) != node_count:
            raise ValueError("Continuous node features are missing from the graph record.")
        feature_by_node = {
            node_id: features[index]
            for index, node_id in enumerate(record["nodes"])
        }
        x = torch.tensor(
            [feature_by_node[node_id] for node_id in ordered_nodes],
            dtype=torch.float,
        )
    elif feature_mode == "constant":
        x = torch.ones((node_count, 1), dtype=torch.float)
    elif feature_mode == "degree":
        degree = torch.bincount(edge_index[0], minlength=node_count)
        degree = degree.clamp(max=max_degree)
        x = F.one_hot(degree, num_classes=max_degree + 1).float()
    else:
        raise ValueError(f"Unsupported feature mode: {feature_mode}")
    return Data(
        x=x,
        edge_index=edge_index,
        num_nodes=node_count,
        graph_id=int(record["id"]),
        graph_path=str(Path(record["member"])),
    )


def graph_by_relative_path(
    records: list[dict[str, Any]],
    graph_path: str,
) -> Data:
    parts = Path(graph_path).parts
    if len(parts) < 2:
        raise ValueError(f"Unsupported graph path: {graph_path}")
    split = parts[-2]
    graph_id = int(Path(parts[-1]).stem)
    for record in records:
        if int(record["id"]) == graph_id and record["split"] == split:
            return record["data"]
    raise ValueError(f"Graph was not found in the registered dataset: {graph_path}")
