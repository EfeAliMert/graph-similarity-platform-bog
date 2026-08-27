from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from statistics import fmean, median, pstdev
from typing import Any, Iterable, Mapping
from xml.etree import ElementTree as ET

from scripts.universal_dataset import (
    canonical_pair_key,
    dataset_spec,
    distance_for,
    ensure_training_distances,
    load_graph_records,
)

from .fingerprint import dataset_fingerprint
from .types import DatasetProfile, DistributionStats


PROFILE_VERSION = "dataset-profile-v1"
ROOT = Path(__file__).resolve().parents[2]
PROFILE_CACHE = ROOT / "training_logs" / "hpo" / "dataset_profiles"


class DatasetProfiler:
    def __init__(self, cache_dir: Path = PROFILE_CACHE):
        self.cache_dir = cache_dir

    def profile(
        self,
        dataset_id: str,
        preprocessing: Mapping[str, Any] | None = None,
        refresh: bool = False,
    ) -> DatasetProfile:
        preprocessing = dict(preprocessing or {})
        fingerprint, _identity = dataset_fingerprint(dataset_id, preprocessing)
        cache_path = self.cache_dir / f"{dataset_id}__{fingerprint}.json"
        if cache_path.is_file() and not refresh:
            return _profile_from_dict(json.loads(cache_path.read_text()))

        profile = self._compute(dataset_id, fingerprint, preprocessing)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json_write(cache_path, profile.to_dict())
        return profile

    def _compute(
        self,
        dataset_id: str,
        fingerprint: str,
        preprocessing: Mapping[str, Any],
    ) -> DatasetProfile:
        spec = dataset_spec(dataset_id)
        records = load_graph_records(dataset_id)
        if not records:
            raise ValueError(f"Dataset {dataset_id!r} contains no readable GEXF graphs.")

        node_counts = [len(record["nodes"]) for record in records]
        edge_counts = [len(record["edges"]) for record in records]
        densities = [
            (2.0 * edges / (nodes * (nodes - 1))) if nodes > 1 else 0.0
            for nodes, edges in zip(node_counts, edge_counts)
        ]
        degree_values: list[float] = []
        component_counts: list[float] = []
        node_labels: set[str] = set()
        edge_labels: set[str] = set()
        for record in records:
            degrees = Counter()
            adjacency = {str(node): set() for node in record["nodes"]}
            for source, target in record["edges"]:
                source_text, target_text = str(source), str(target)
                degrees[source_text] += 1
                degrees[target_text] += 1
                adjacency.setdefault(source_text, set()).add(target_text)
                adjacency.setdefault(target_text, set()).add(source_text)
            degree_values.extend(float(degrees[str(node)]) for node in record["nodes"])
            component_counts.append(float(_component_count(adjacency)))
            node_labels.update(str(label) for label in record.get("labels", []))
            edge_labels.update(_edge_labels(record.get("content", b"")))

        distances, target = ensure_training_distances(dataset_id)
        graph_by_id = {int(record["id"]): record for record in records}
        ged_values: list[float] = []
        normalized_values: list[float] = []
        seen: set[tuple[int, int]] = set()
        for (left_id, right_id), raw_distance in distances.items():
            key = canonical_pair_key(int(left_id), int(right_id))
            if key in seen or key[0] not in graph_by_id or key[1] not in graph_by_id:
                continue
            distance = float(raw_distance)
            if not math.isfinite(distance) or distance < 0:
                continue
            seen.add(key)
            denominator = max(
                0.5
                * (
                    len(graph_by_id[key[0]]["nodes"])
                    + len(graph_by_id[key[1]]["nodes"])
                ),
                1.0,
            )
            ged_values.append(distance)
            normalized_values.append(distance / denominator)

        target_variance = _variance(normalized_values)
        zero_fraction = (
            sum(value == 0 for value in ged_values) / len(ged_values)
            if ged_values
            else None
        )
        return DatasetProfile(
            dataset_id=dataset_id,
            dataset_name=str(spec.get("name") or dataset_id),
            fingerprint=fingerprint,
            profile_version=PROFILE_VERSION,
            target_kind=str(target.get("target_kind") or spec.get("target_kind")),
            target_source=target.get("target_source") or spec.get("target_source"),
            target_exact=bool(target.get("exact", spec.get("target_exact", False))),
            split_strategy=str(
                spec.get("split_strategy") or "canonical unordered pair holdout"
            ),
            graph_count=len(records),
            train_graph_count=sum(record.get("split") == "train" for record in records),
            test_graph_count=sum(record.get("split") == "test" for record in records),
            node_count=_distribution(node_counts),
            edge_count=_distribution(edge_counts),
            density=_distribution(densities),
            degree=_distribution(degree_values),
            connected_components=_distribution(component_counts),
            node_label_cardinality=len(node_labels),
            edge_label_cardinality=len(edge_labels),
            node_features_available=any("node_features" in record for record in records),
            edge_labels_available=bool(edge_labels),
            ged=_distribution(ged_values),
            normalized_ged=_distribution(normalized_values),
            target_variance=target_variance,
            zero_target_fraction=zero_fraction,
            preprocessing=preprocessing,
        )


def _distribution(values: Iterable[float | int]) -> DistributionStats:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return DistributionStats(0, None, None, None, None, None, None, None)
    return DistributionStats(
        count=len(finite),
        minimum=finite[0],
        q25=_quantile(finite, 0.25),
        median=float(median(finite)),
        mean=float(fmean(finite)),
        q75=_quantile(finite, 0.75),
        maximum=finite[-1],
        standard_deviation=float(pstdev(finite)) if len(finite) > 1 else 0.0,
    )


def _quantile(values: list[float], probability: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _variance(values: list[float]) -> float | None:
    if not values:
        return None
    mean = fmean(values)
    return fmean((value - mean) ** 2 for value in values)


def _component_count(adjacency: dict[str, set[str]]) -> int:
    unseen = set(adjacency)
    components = 0
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            node = stack.pop()
            neighbors = adjacency.get(node, set()).intersection(unseen)
            unseen.difference_update(neighbors)
            stack.extend(neighbors)
    return components


def _edge_labels(content: bytes) -> set[str]:
    if not content:
        return set()
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return set()
    labels: set[str] = set()
    for edge in root.findall(".//{*}edge"):
        label = edge.attrib.get("label")
        if label not in (None, ""):
            labels.add(str(label))
        for value in edge.findall(".//{*}attvalue"):
            raw = value.attrib.get("value")
            if raw not in (None, ""):
                labels.add(str(raw))
    return labels


def _profile_from_dict(payload: dict[str, Any]) -> DatasetProfile:
    for key in (
        "node_count",
        "edge_count",
        "density",
        "degree",
        "connected_components",
        "ged",
        "normalized_ged",
    ):
        payload[key] = DistributionStats(**payload[key])
    return DatasetProfile(**payload)


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)
