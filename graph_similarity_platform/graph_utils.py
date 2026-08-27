from __future__ import annotations

import hashlib
import math
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


class GraphInputError(ValueError):
    pass


def graph_from_payload(payload: Any, name: str = "Graph") -> "GraphData":
    if isinstance(payload, str):
        raise GraphInputError(f"{name} must be parsed JSON, not a raw string.")
    if isinstance(payload, list):
        payload = {"edges": payload}
    if not isinstance(payload, dict):
        raise GraphInputError(f"{name} must be a JSON object or an edge-list array.")

    edges = payload.get("edges", payload.get("graph", payload.get("edge_list")))
    labels = payload.get("labels", payload.get("node_labels", payload.get("label")))
    nodes = payload.get("nodes")
    if edges is None:
        raise GraphInputError(f"{name} is missing an edges array.")

    return GraphData.from_parts(edges=edges, labels=labels, nodes=nodes, name=name)


@dataclass
class GraphData:
    name: str
    nodes: list[int]
    edges: list[tuple[int, int]]
    labels: list[str]
    original_nodes: list[Any]
    adj: dict[int, set[int]]

    @classmethod
    def from_parts(
        cls,
        edges: Iterable[Iterable[Any]],
        labels: Any = None,
        nodes: Iterable[Any] | None = None,
        name: str = "Graph",
    ) -> "GraphData":
        normalized_edges: list[tuple[Any, Any]] = []
        raw_nodes: set[Any] = set()

        if nodes is not None:
            raw_nodes.update(_normalize_node_id(node) for node in nodes)

        if labels is not None and isinstance(labels, list):
            raw_nodes.update(range(len(labels)))

        for edge in edges:
            if not isinstance(edge, (list, tuple)) or len(edge) < 2:
                raise GraphInputError(f"{name} has an invalid edge: {edge!r}")
            source = _normalize_node_id(edge[0])
            target = _normalize_node_id(edge[1])
            raw_nodes.update([source, target])
            if source != target:
                normalized_edges.append((source, target))

        ordered_original_nodes = sorted(raw_nodes, key=lambda item: (str(type(item)), str(item)))
        node_map = {node: idx for idx, node in enumerate(ordered_original_nodes)}
        dense_edges = sorted(
            {
                tuple(sorted((node_map[source], node_map[target])))
                for source, target in normalized_edges
                if source in node_map and target in node_map and source != target
            }
        )

        dense_labels = ["0"] * len(ordered_original_nodes)
        if isinstance(labels, list):
            for index, label in enumerate(labels):
                normalized_index = _normalize_node_id(index)
                if normalized_index in node_map:
                    dense_labels[node_map[normalized_index]] = str(label)
        elif isinstance(labels, dict):
            for key, label in labels.items():
                normalized_key = _normalize_node_id(key)
                if normalized_key in node_map:
                    dense_labels[node_map[normalized_key]] = str(label)

        adjacency = {node: set() for node in range(len(ordered_original_nodes))}
        for source, target in dense_edges:
            adjacency[source].add(target)
            adjacency[target].add(source)

        return cls(
            name=name,
            nodes=list(range(len(ordered_original_nodes))),
            edges=dense_edges,
            labels=dense_labels,
            original_nodes=ordered_original_nodes,
            adj=adjacency,
        )

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def density(self) -> float:
        if self.node_count < 2:
            return 0.0
        return (2.0 * self.edge_count) / (self.node_count * (self.node_count - 1))

    @property
    def degrees(self) -> list[int]:
        return [len(self.adj[node]) for node in self.nodes]

    @property
    def degree_counter(self) -> Counter:
        return Counter(self.degrees)

    @property
    def label_counter(self) -> Counter:
        return Counter(self.labels)

    def connected_components(self) -> list[list[int]]:
        seen: set[int] = set()
        components: list[list[int]] = []
        for node in self.nodes:
            if node in seen:
                continue
            queue = deque([node])
            seen.add(node)
            component = []
            while queue:
                current = queue.popleft()
                component.append(current)
                for neighbor in self.adj[current]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        queue.append(neighbor)
            components.append(component)
        return components

    def local_clustering(self) -> list[float]:
        values = []
        for node in self.nodes:
            neighbors = list(self.adj[node])
            degree = len(neighbors)
            if degree < 2:
                values.append(0.0)
                continue
            links = 0
            for i, source in enumerate(neighbors):
                for target in neighbors[i + 1 :]:
                    if target in self.adj[source]:
                        links += 1
            values.append(links / (degree * (degree - 1) / 2.0))
        return values

    def triangle_count(self) -> int:
        total = 0
        for source, target in self.edges:
            total += len(self.adj[source].intersection(self.adj[target]))
        return total // 3

    def adjacency_matrix(self) -> np.ndarray:
        matrix = np.zeros((self.node_count, self.node_count), dtype=float)
        for source, target in self.edges:
            matrix[source, target] = 1.0
            matrix[target, source] = 1.0
        return matrix

    def spectral_signature(self, size: int = 12) -> np.ndarray:
        if self.node_count == 0:
            return np.zeros(size, dtype=float)
        adjacency = self.adjacency_matrix()
        degrees = adjacency.sum(axis=1)
        inv_sqrt = np.zeros_like(degrees)
        nonzero = degrees > 0
        inv_sqrt[nonzero] = 1.0 / np.sqrt(degrees[nonzero])
        normalized_adjacency = adjacency * inv_sqrt[:, None] * inv_sqrt[None, :]
        laplacian = np.eye(self.node_count) - normalized_adjacency
        eigenvalues = np.linalg.eigvalsh(laplacian)
        eigenvalues = np.sort(np.real(eigenvalues))
        if eigenvalues.size >= size:
            return eigenvalues[:size]
        return np.pad(eigenvalues, (0, size - eigenvalues.size))

    def node_features(self) -> np.ndarray:
        if self.node_count == 0:
            return np.zeros((0, 6), dtype=float)
        degrees = np.array(self.degrees, dtype=float)
        max_degree = max(float(degrees.max()), 1.0)
        clustering = np.array(self.local_clustering(), dtype=float)
        label_values = np.array([stable_label_value(label) for label in self.labels], dtype=float)
        label_values = label_values / 997.0
        triangle_touch = np.zeros(self.node_count, dtype=float)
        for source, target in self.edges:
            common = len(self.adj[source].intersection(self.adj[target]))
            triangle_touch[source] += common
            triangle_touch[target] += common
        max_triangle_touch = max(float(triangle_touch.max()), 1.0)
        return np.column_stack(
            [
                degrees / max_degree,
                degrees / max(float(self.node_count - 1), 1.0),
                clustering,
                label_values,
                triangle_touch / max_triangle_touch,
                np.ones(self.node_count, dtype=float),
            ]
        )

    def edge_features(self) -> np.ndarray:
        if not self.edges:
            return np.zeros((0, 6), dtype=float)
        degrees = self.degrees
        clustering = self.local_clustering()
        rows = []
        max_degree = max(max(degrees), 1)
        for source, target in self.edges:
            common = len(self.adj[source].intersection(self.adj[target]))
            rows.append(
                [
                    (degrees[source] + degrees[target]) / (2.0 * max_degree),
                    abs(degrees[source] - degrees[target]) / max_degree,
                    common / max(max_degree, 1),
                    1.0 if self.labels[source] == self.labels[target] else 0.0,
                    (clustering[source] + clustering[target]) / 2.0,
                    self.density,
                ]
            )
        return np.array(rows, dtype=float)

    def summary(self) -> dict[str, Any]:
        components = self.connected_components()
        degrees = self.degrees
        clustering = self.local_clustering()
        return {
            "nodes": self.node_count,
            "edges": self.edge_count,
            "density": round(self.density, 4),
            "components": len(components),
            "component_sizes": sorted((len(component) for component in components), reverse=True),
            "avg_degree": round(sum(degrees) / max(len(degrees), 1), 3),
            "max_degree": max(degrees, default=0),
            "avg_clustering": round(sum(clustering) / max(len(clustering), 1), 4),
            "triangles": self.triangle_count(),
            "labels": dict(self.label_counter.most_common(8)),
        }

    def to_preview(self) -> dict[str, Any]:
        return {
            "nodes": [{"id": node, "label": self.labels[node]} for node in self.nodes[:80]],
            "edges": [{"source": source, "target": target} for source, target in self.edges[:160]],
        }


def stable_label_value(label: str) -> int:
    digest = hashlib.sha1(str(label).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 997


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    norm = float(np.linalg.norm(left) * np.linalg.norm(right))
    if norm == 0.0:
        return 1.0 if np.linalg.norm(left) == np.linalg.norm(right) else 0.0
    return clamp(float(np.dot(left, right) / norm))


def counter_cosine(left: Counter, right: Counter) -> float:
    keys = set(left).union(right)
    if not keys:
        return 1.0
    left_vector = np.array([left.get(key, 0.0) for key in keys], dtype=float)
    right_vector = np.array([right.get(key, 0.0) for key in keys], dtype=float)
    return cosine_similarity(left_vector, right_vector)


def histogram_similarity(left: Iterable[float], right: Iterable[float], bins: int = 12) -> float:
    left_array = np.asarray(list(left), dtype=float)
    right_array = np.asarray(list(right), dtype=float)
    if left_array.size == 0 and right_array.size == 0:
        return 1.0
    if left_array.size == 0 or right_array.size == 0:
        return 0.0
    span_min = float(min(left_array.min(), right_array.min()))
    span_max = float(max(left_array.max(), right_array.max(), span_min + 1e-9))
    left_hist, _ = np.histogram(left_array, bins=bins, range=(span_min, span_max))
    right_hist, _ = np.histogram(right_array, bins=bins, range=(span_min, span_max))
    return cosine_similarity(left_hist, right_hist)


def set_matching_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 and right.size == 0:
        return 1.0
    if left.size == 0 or right.size == 0:
        return 0.0
    left_norm = _row_normalize(left)
    right_norm = _row_normalize(right)
    similarity_matrix = np.clip(left_norm @ right_norm.T, 0.0, 1.0)
    left_to_right = float(np.mean(np.max(similarity_matrix, axis=1)))
    right_to_left = float(np.mean(np.max(similarity_matrix, axis=0)))
    cardinality_penalty = math.exp(-abs(left.shape[0] - right.shape[0]) / max(left.shape[0], right.shape[0], 1))
    return clamp(((left_to_right + right_to_left) / 2.0) * cardinality_penalty)


def scale_similarity(left: float, right: float) -> float:
    denominator = max(abs(left), abs(right), 1e-9)
    return clamp(1.0 - abs(left - right) / denominator)


def clamp(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _normalize_node_id(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lstrip("-").isdigit():
            return int(stripped)
        return stripped
    return value
