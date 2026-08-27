from __future__ import annotations

from itertools import product
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_adj, to_dense_batch


def dense_feature_batch(
    x: torch.Tensor,
    batch: torch.Tensor,
    max_num_nodes: int,
    batch_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad node/edge features even when a graph has zero nodes or zero edges.

    PyG ``to_dense_batch`` calls ``batch.max()`` and crashes on empty tensors,
    which happens for isolated-node graphs and after node-cap truncation.
    """
    if x.numel() == 0 or batch.numel() == 0:
        feature_dim = int(x.size(-1)) if x.dim() >= 2 else 0
        return (
            x.new_zeros((batch_size, max_num_nodes, feature_dim)),
            torch.zeros((batch_size, max_num_nodes), dtype=torch.bool, device=x.device),
        )
    return to_dense_batch(
        x,
        batch=batch,
        max_num_nodes=max_num_nodes,
        batch_size=batch_size,
    )

from universal_pyg import canonical_node_order


AIDS_ATOM_TYPES = [
    "O", "S", "C", "N", "Cl", "Br", "B", "Si", "Hg", "I", "Bi", "P", "F",
    "Cu", "Ho", "Pd", "Ru", "Pt", "Sn", "Li", "Ga", "Tb", "As", "Co", "Pb",
    "Sb", "Se", "Ni", "Te",
]


def build_args(
    device: torch.device,
    node_cap: int = 16,
    edge_cap: int = 32,
    line_edge_cap: int = 128,
    max_degree: int = 8,
    feature_size: int | None = None,
    label_vocabulary: list[str] | None = None,
    edge_feature_mode: str = "concat",
    architecture_profile: str = "compact",
    canonical_node_order: bool = False,
) -> SimpleNamespace:
    if label_vocabulary:
        feature_size = len(label_vocabulary) + max_degree + 1
    feature_size = int(feature_size or (max_degree + 1))
    if edge_feature_mode not in {"concat", "sum"}:
        raise ValueError(f"Unsupported SEGMN edge feature mode: {edge_feature_mode}")
    if architecture_profile not in {"compact", "aids-original", "linux-original"}:
        raise ValueError(f"Unsupported SEGMN architecture profile: {architecture_profile}")
    original_profile = architecture_profile in {"aids-original", "linux-original"}
    return SimpleNamespace(
        dataset="Universal",
        device=device,
        device_count=torch.cuda.device_count(),
        node_feature_size=feature_size,
        n_max_nodes=node_cap,
        n_max_edges=edge_cap,
        m_max_edges=line_edge_cap,
        n_max_l=line_edge_cap,
        degree=max_degree,
        D=feature_size,
        x_size=feature_size if edge_feature_mode == "sum" else feature_size * 2,
        label_vocabulary=list(label_vocabulary or []),
        edge_feature_mode=edge_feature_mode,
        architecture_profile=architecture_profile,
        canonical_node_order=bool(canonical_node_order),
        embedding_size=128 if original_profile else 32,
        embedding_size_attention=96 if original_profile else 32,
        graph_transformer_active=True,
        encoder_ffn_size=128 if original_profile else 64,
        share_qk=True,
        msa_bias=True,
        encoder_mask=False,
        interaction_mask=False,
        align_mask=False,
        cnn_mask=False,
        n_heads=4 if original_profile else 2,
        channel_align=True,
        n_channel_transformer_heads=4 if original_profile else 2,
        channel_ffn_size=128 if original_profile else 64,
        conv_channels_0=32 if original_profile else 16,
        conv_channels_1=64 if original_profile else 32,
        conv_channels_2=1,
        conv_channels_3=256 if original_profile else 64,
        conv_l_relu_slope=0.33,
        conv_dropout=0.1,
        pooling_res=12,
        dropout=0.1,
        GNN="GCN",
    )


def record_to_segmn(record: dict[str, Any], args: SimpleNamespace) -> Data:
    ordered_nodes = (
        canonical_node_order(record)
        if bool(getattr(args, "canonical_node_order", False))
        else list(record["nodes"])
    )
    selected_nodes = ordered_nodes[: int(args.n_max_nodes)]
    node_map = {node_id: index for index, node_id in enumerate(selected_nodes)}
    undirected_edge_set: set[tuple[int, int]] = set()
    for source, target in record["edges"]:
        if source not in node_map or target not in node_map or source == target:
            continue
        edge = tuple(sorted((node_map[source], node_map[target])))
        undirected_edge_set.add(edge)
    undirected_edges = sorted(undirected_edge_set)[: int(args.n_max_edges)]

    directed_edges = [
        directed
        for source, target in undirected_edges
        for directed in ((source, target), (target, source))
    ]
    edge_index = (
        torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
        if directed_edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    node_count = len(selected_nodes)
    degrees = torch.bincount(edge_index[0], minlength=node_count)
    raw_features = record.get("node_features")
    if raw_features is not None:
        selected_features = [raw_features[record["nodes"].index(node)] for node in selected_nodes]
        x = torch.tensor(selected_features, dtype=torch.float)
        if x.size(-1) != int(args.D):
            raise ValueError("SEGMN continuous node feature dimension does not match model args.")
    elif getattr(args, "label_vocabulary", None):
        vocabulary = {
            str(label): index
            for index, label in enumerate(args.label_vocabulary)
        }
        labels = list(record.get("labels") or [])
        selected_labels = [labels[record["nodes"].index(node)] for node in selected_nodes]
        unknown = sorted({str(label) for label in selected_labels if str(label) not in vocabulary})
        if unknown:
            raise ValueError(f"SEGMN encountered unknown node labels: {', '.join(unknown)}")
        label_indices = torch.tensor(
            [vocabulary[str(label)] for label in selected_labels],
            dtype=torch.long,
        )
        label_features = F.one_hot(
            label_indices,
            num_classes=len(vocabulary),
        ).float()
        degree_features = F.one_hot(
            degrees.clamp(max=int(args.degree)),
            num_classes=int(args.degree) + 1,
        ).float()
        x = torch.cat((label_features, degree_features), dim=-1)
    else:
        x = F.one_hot(
            degrees.clamp(max=int(args.degree)),
            num_classes=int(args.D),
        ).float()

    edge_count = len(undirected_edges)
    edge_feature_mode = getattr(args, "edge_feature_mode", "concat")
    x1 = (
        torch.stack(
            [
                x[source] + x[target]
                if edge_feature_mode == "sum"
                else torch.cat((x[source], x[target]))
                for source, target in undirected_edges
            ]
        )
        if undirected_edges
        else torch.zeros((0, int(args.x_size)), dtype=torch.float)
    )
    h = torch.zeros((int(args.n_max_nodes), int(args.n_max_edges)))
    for edge_id, (source, target) in enumerate(undirected_edges):
        normalization = 1.0 / max(
            float((degrees[source] + 1) * (degrees[target] + 1)) ** 0.5,
            1.0,
        )
        h[source, edge_id] = normalization
        h[target, edge_id] = normalization

    line_edges: list[tuple[int, int]] = []
    shared_features: list[torch.Tensor] = []
    for left in range(edge_count):
        for right in range(left + 1, edge_count):
            shared = set(undirected_edges[left]) & set(undirected_edges[right])
            if not shared:
                continue
            line_edges.append((left, right))
            shared_features.append(x[next(iter(shared))])
            if len(line_edges) >= int(args.m_max_edges):
                break
        if len(line_edges) >= int(args.m_max_edges):
            break
    directed_line_edges = [
        directed
        for source, target in line_edges
        for directed in ((source, target), (target, source))
    ]
    edgeindex1 = (
        torch.tensor(directed_line_edges, dtype=torch.long).t().contiguous()
        if directed_line_edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    f = torch.zeros((int(args.m_max_edges), int(args.D)))
    if shared_features:
        f[: len(shared_features)] = torch.stack(shared_features)
    return Data(
        x=x,
        x1=x1,
        edge_index=edge_index,
        edgeindex1=edgeindex1,
        h=h,
        f=f,
        l=torch.tensor(len(line_edges), dtype=torch.long),
        numedges=torch.tensor(edge_count, dtype=torch.long),
        num_nodes=node_count,
        graph_id=int(record["id"]),
        original_num_nodes=len(record["nodes"]),
        original_num_edges=len(record["edges"]),
    )
def transform_pair(
    left: Data,
    right: Data,
    args: SimpleNamespace,
    target: float = 0.0,
) -> dict[str, Any]:
    left_batch = Batch.from_data_list([left])
    right_batch = Batch.from_data_list([right])
    ass_x, ass_edge_index = assignment_graph(left, right)
    assignment_batch = torch.zeros(ass_x.shape[0], dtype=torch.long)
    assignment_count = int(args.n_max_nodes) ** 2
    ass_x_tensor = torch.tensor(ass_x, dtype=torch.float)
    ass_edge_tensor = torch.tensor(ass_edge_index, dtype=torch.long)
    assignment_dense = dense_feature_batch(
        ass_x_tensor,
        assignment_batch,
        assignment_count,
    )
    return {
        "g0": graph_block(left_batch, args),
        "g1": graph_block(right_batch, args),
        "target": torch.tensor([target], dtype=torch.float, device=args.device),
        "ass_x": assignment_dense[0].to(args.device),
        "ass_x_mask": assignment_dense[1].to(args.device),
        "ass_edge_index": to_dense_adj(
            ass_edge_tensor,
            batch=assignment_batch,
            max_num_nodes=assignment_count,
        ).to(args.device),
    }


def graph_block(batch: Batch, args: SimpleNamespace) -> dict[str, torch.Tensor]:
    node_batch = batch.batch
    edge_count = int(batch.numedges.reshape(-1)[0].item())
    edge_batch = torch.zeros(edge_count, dtype=torch.long)
    x_dense = dense_feature_batch(
        batch.x,
        node_batch,
        int(args.n_max_nodes),
    )
    x1_dense = dense_feature_batch(
        batch.x1,
        edge_batch,
        int(args.n_max_edges),
    )
    return {
        "adj": to_dense_adj(
            batch.edge_index,
            batch=node_batch,
            max_num_nodes=int(args.n_max_nodes),
        ).to(args.device),
        "adj1": to_dense_adj(
            batch.edgeindex1,
            batch=edge_batch,
            max_num_nodes=int(args.n_max_edges),
        ).to(args.device),
        "x": x_dense[0].to(args.device),
        "x1": x1_dense[0].to(args.device),
        "f": batch.f.view(1, int(args.m_max_edges), int(args.D)).to(args.device),
        "h": batch.h.view(
            1,
            int(args.n_max_nodes),
            int(args.n_max_edges),
        ).to(args.device),
        "l": batch.l.reshape(-1).to(args.device),
        "numedges": batch.numedges.reshape(-1).to(args.device),
        "edgeindex1": batch.edgeindex1.to(args.device),
        "mask": x_dense[1].to(args.device),
        "mask_x1": x1_dense[1].to(args.device),
    }


def assignment_graph(left: Data, right: Data) -> tuple[np.ndarray, np.ndarray]:
    num_nodes_left = int(left.num_nodes)
    num_nodes_right = int(right.num_nodes)
    ass_x = np.array(
        list(product(range(num_nodes_left), range(num_nodes_right))),
        dtype=np.int64,
    )
    neighbors_left = neighbors(left.edge_index.cpu().numpy())
    neighbors_right = neighbors(right.edge_index.cpu().numpy())
    edges = []
    for left_node in range(num_nodes_left):
        for right_node in range(num_nodes_right):
            source = left_node * num_nodes_right + right_node
            for n_left, n_right in product(
                neighbors_left.get(left_node, []),
                neighbors_right.get(right_node, []),
            ):
                target = int(n_left) * num_nodes_right + int(n_right)
                edges.append((source, target))
    if not edges:
        return ass_x, np.empty((2, 0), dtype=np.int64)
    return ass_x, np.array(edges, dtype=np.int64).T


def neighbors(edge_index: np.ndarray) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    if edge_index.size == 0:
        return result
    for source, target in edge_index.T:
        result.setdefault(int(source), []).append(int(target))
    return result
