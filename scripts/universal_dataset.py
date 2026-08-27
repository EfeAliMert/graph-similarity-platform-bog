from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import pickle
import random
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_ROOT = ROOT / "Models&Datasets" / "drive-download-20260630T100606Z-3-001"
UPLOADED_ROOT = ROOT / "Models&Datasets" / "uploaded_datasets"
DERIVED_ROOT = ROOT / "Models&Datasets" / "derived_training"
STATIC_DATASETS = {
    "aids700nef": {
        "name": "AIDS700nef",
        "archive": ORIGINAL_ROOT / "AIDS700nef.zip",
        "format": "zip",
        "ged": ORIGINAL_ROOT / "aids700nef_ged_astar_gidpair_dist_map.pickle",
        "target_exact": True,
        "target_kind": "exact",
        "target_source": "exact A* GED benchmark",
        "target_semantics": "GED distance; each trainer applies its documented native target transform",
    },
    "linux": {
        "name": "LINUX",
        "archive": ORIGINAL_ROOT / "LINUX.tar.gz",
        "format": "tar",
        "ged": ORIGINAL_ROOT / "linux_ged_astar_gidpair_dist_map.pickle",
        "target_exact": True,
        "target_kind": "exact",
        "target_source": "exact A* GED benchmark",
        "target_semantics": "GED distance; each trainer applies its documented native target transform",
    },
    "imdbmulti": {
        "name": "IMDBMulti",
        "archive": ORIGINAL_ROOT / "IMDBMulti.zip",
        "format": "zip",
        "ged": ORIGINAL_ROOT / "imdbmulti_ged_astar_gidpair_dist_map.pickle",
        "target_exact": False,
        "target_kind": "approximate_benchmark",
        "target_source": (
            "approximate GED benchmark: minimum upper bound from Beam, "
            "Hungarian, and VJ"
        ),
        "target_semantics": "approximate GED upper bound; each trainer applies its documented native target transform",
    },
    "ptc": {
        "name": "PTC",
        "archive": ORIGINAL_ROOT / "PTC.zip",
        "format": "zip",
        "ged": ORIGINAL_ROOT / "ptc_ged_astar_gidpair_dist_map.pickle",
        "target_exact": False,
        "target_kind": "approximate_benchmark",
        "target_source": (
            "approximate GED benchmark: minimum upper bound from Beam, "
            "Hungarian, and VJ"
        ),
        "target_semantics": "approximate GED upper bound; each trainer applies its documented native target transform",
    },
    "mutag": {
        "name": "MUTAG",
        "archive": ORIGINAL_ROOT / "MUTAG.zip",
        "format": "zip",
        "ged": None,
        "target_exact": False,
        "target_kind": "structural_proxy",
        "target_semantics": "structural proxy distance; not a GED benchmark label",
    },
    "proteins": {
        "name": "PROTEINS",
        "archive": ORIGINAL_ROOT / "PROTEINS.zip",
        "format": "zip",
        "ged": None,
        "target_exact": False,
        "target_kind": "structural_proxy",
        "target_semantics": "structural proxy distance; not a GED benchmark label",
    },
    "enzymes": {
        "name": "ENZYMES",
        "archive": ORIGINAL_ROOT / "ENZYMES.zip",
        "format": "zip",
        "ged": None,
        "target_exact": False,
        "target_kind": "structural_proxy",
        "target_semantics": "structural proxy distance; not a GED benchmark label",
    },
}


def dataset_spec(dataset_id: str) -> dict[str, Any]:
    if dataset_id in STATIC_DATASETS:
        return {
            **STATIC_DATASETS[dataset_id],
            "id": dataset_id,
            "uploaded": False,
            "training_ready": True,
        }
    manifest_path = UPLOADED_ROOT / dataset_id / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Unknown dataset: {dataset_id}")
    manifest = json.loads(manifest_path.read_text())
    ground_truth = manifest.get("ground_truth") or []
    ged_name = next((name for name in ground_truth if "ged" in Path(name).stem.lower()), None)
    return {
        "id": dataset_id,
        "name": manifest.get("name", dataset_id),
        "archive": manifest_path.parent / manifest.get("archive", "dataset.zip"),
        "format": manifest.get("format", "zip"),
        "ged": manifest_path.parent / ged_name if ged_name else None,
        "uploaded": True,
        "training_ready": bool(manifest.get("training_ready")),
        "target_exact": bool(manifest.get("target_exact", False)),
        "target_kind": manifest.get(
            "target_kind",
            "exact" if manifest.get("target_exact", False) else "structural_proxy",
        ),
        "target_source": manifest.get("target_source"),
        "target_semantics": manifest.get("target_semantics"),
        "split_strategy": manifest.get("split_strategy"),
        "split_seed": manifest.get("split_seed"),
        "feature_mode": manifest.get("feature_mode", "degree"),
        "node_feature_dimension": manifest.get("node_feature_dimension"),
    }


def load_graph_records(dataset_id: str) -> list[dict[str, Any]]:
    spec = dataset_spec(dataset_id)
    records = []
    for member, content in _archive_gexf(spec["archive"], spec["format"]):
        graph_id = int(Path(member).stem)
        split = "train" if "/train/" in f"/{member}" else "test"
        parsed = parse_gexf(content.decode("utf-8", errors="replace"))
        records.append(
            {
                "id": graph_id,
                "split": split,
                "member": member,
                "content": content,
                **parsed,
            }
        )
    return sorted(records, key=lambda item: (item["split"] == "test", item["id"]))


def ensure_training_distances(dataset_id: str) -> tuple[dict[tuple[int, int], float], dict[str, Any]]:
    spec = dataset_spec(dataset_id)
    registered_training_target = (
        spec["ged"] is not None
        and Path(spec["ged"]).exists()
        and (not spec["uploaded"] or spec["training_ready"])
    )
    if registered_training_target:
        distances = load_distance_file(Path(spec["ged"]))
        return distances, {
            "dataset_id": dataset_id,
            "target_source": spec.get("target_source") or "registered GED target",
            "target_semantics": spec.get("target_semantics"),
            "target_kind": spec.get("target_kind", "structural_proxy"),
            "exact": bool(spec.get("target_exact", False)),
            "path": display_path(Path(spec["ged"])),
            "pairs": len(distances),
        }

    output_dir = DERIVED_ROOT / dataset_id
    distance_path = output_dir / "structural_proxy_ged.pickle"
    metadata_path = output_dir / "metadata.json"
    if distance_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        metadata.setdefault("target_kind", "structural_proxy")
        metadata["exact"] = False
        return load_distance_file(distance_path), metadata

    records = load_graph_records(dataset_id)
    descriptors = {record["id"]: graph_descriptor(record) for record in records}
    distances: dict[tuple[int, int], float] = {}
    ids = sorted(descriptors)
    for left_index, left_id in enumerate(ids):
        for right_id in ids[left_index:]:
            distances[(left_id, right_id)] = structural_proxy_ged(
                descriptors[left_id],
                descriptors[right_id],
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    with distance_path.open("wb") as handle:
        pickle.dump(distances, handle, protocol=pickle.HIGHEST_PROTOCOL)
    metadata = {
        "dataset_id": dataset_id,
        "target_source": "derived structural GED proxy",
        "target_kind": "structural_proxy",
        "target_semantics": "structural proxy distance; not a GED benchmark label",
        "exact": False,
        "method": (
            "absolute node and edge count gaps plus half-L1 node-label and "
            "degree-histogram differences"
        ),
        "path": display_path(distance_path),
        "pairs": len(distances),
        "graphs": len(records),
    }
    if spec["uploaded"] and spec["ged"] is not None:
        metadata["reason"] = (
            "The uploaded GED file did not cover both train/train and test/train "
            "pairs, so it was retained for available-pair evaluation but was not "
            "mixed into the full training target."
        )
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return distances, metadata


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_distance_file(path: Path) -> dict[tuple[int, int], float]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        values = {}
        for key, distance in payload.items():
            left, right = str(key).split(",", maxsplit=1)
            values[(int(left), int(right))] = float(distance)
        return canonicalize_symmetric_distances(values)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    values = {
        (int(left), int(right)): float(distance)
        for (left, right), distance in payload.items()
    }
    return canonicalize_symmetric_distances(values)


def canonicalize_symmetric_distances(
    values: dict[tuple[int, int], float],
) -> dict[tuple[int, int], float]:
    """Use one symmetric target per unordered pair.

    The minimum of directional approximate-GED upper bounds is the tighter
    valid upper bound; exact symmetric labels are unaffected.
    """
    canonical: dict[tuple[int, int], float] = {}
    for (left, right), raw_distance in values.items():
        distance = float(raw_distance)
        if not math.isfinite(distance) or distance < 0:
            continue
        key = canonical_pair_key(int(left), int(right))
        previous = canonical.get(key)
        if previous is None or distance < previous:
            canonical[key] = distance
    symmetric: dict[tuple[int, int], float] = {}
    for (left, right), distance in canonical.items():
        symmetric[(left, right)] = distance
        if left != right:
            symmetric[(right, left)] = distance
    return symmetric


def materialize_raw_dataset(
    dataset_id: str,
    target_root: Path,
    dataset_name: str | None = None,
    clean: bool = False,
) -> dict[str, Any]:
    records = load_graph_records(dataset_id)
    distances, target_metadata = ensure_training_distances(dataset_id)
    name = dataset_name or dataset_spec(dataset_id)["name"]
    raw_root = target_root / "raw" / name
    if clean and target_root.exists():
        shutil.rmtree(target_root)
    (raw_root / "train").mkdir(parents=True, exist_ok=True)
    (raw_root / "test").mkdir(parents=True, exist_ok=True)
    for record in records:
        (raw_root / record["split"] / f"{record['id']}.gexf").write_bytes(record["content"])
    with (raw_root / "ged.pickle").open("wb") as handle:
        pickle.dump(distances, handle, protocol=pickle.HIGHEST_PROTOCOL)
    (raw_root / "target_metadata.json").write_text(json.dumps(target_metadata, indent=2))
    return {
        "dataset_id": dataset_id,
        "name": name,
        "target_root": str(target_root),
        "train_graphs": sum(record["split"] == "train" for record in records),
        "test_graphs": sum(record["split"] == "test" for record in records),
        "target": target_metadata,
    }


def distance_for(
    distances: dict[tuple[int, int], float],
    left_id: int,
    right_id: int,
) -> float | None:
    value = distances.get((left_id, right_id))
    if value is None:
        value = distances.get((right_id, left_id))
    return None if value is None else float(value)


def canonical_pair_key(left_id: int, right_id: int) -> tuple[int, int]:
    return (left_id, right_id) if left_id <= right_id else (right_id, left_id)


def spearman_correlation(left: list[float], right: list[float]) -> float:
    """Return tie-aware Spearman rank correlation without SciPy."""
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    covariance = sum(
        (left_rank - left_mean) * (right_rank - right_mean)
        for left_rank, right_rank in zip(left_ranks, right_ranks)
    )
    left_scale = math.sqrt(sum((rank - left_mean) ** 2 for rank in left_ranks))
    right_scale = math.sqrt(sum((rank - right_mean) ** 2 for rank in right_ranks))
    denominator = left_scale * right_scale
    return float(covariance / denominator) if denominator > 0 else 0.0


def _average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: (float(item[1]), item[0]))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and float(ordered[end][1]) == float(ordered[cursor][1]):
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[ordered[position][0]] = average_rank
        cursor = end
    return ranks


def graph_disjoint_split_metadata(
    records: list[dict[str, Any]],
    validation_fraction: float,
) -> dict[str, Any]:
    train_graph_ids = [
        int(record["id"])
        for record in records
        if record.get("split") == "train"
    ]
    cutoff = int(len(train_graph_ids) * (1.0 - float(validation_fraction)))
    training_ids = train_graph_ids[:cutoff]
    validation_ids = train_graph_ids[cutoff:]
    identity = {
        "training_graph_ids": training_ids,
        "validation_graph_ids": validation_ids,
    }
    split_sha256 = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "strategy": "validation graphs are disjoint from training graphs",
        "training_graphs": len(training_ids),
        "validation_graphs": len(validation_ids),
        "pair_overlap": 0,
        "split_sha256": split_sha256,
        **identity,
    }


def build_pair_split(
    dataset: list,
    distances: dict[tuple[int, int], float],
    validation_count: int,
    seed: int,
) -> dict[str, Any]:
    """Create a deterministic, stratified pair holdout with no train/val overlap.

    Validation is capped per target bin so a rare target (notably exact GED
    zero) cannot be consumed entirely by the holdout. This keeps checkpoint
    selection representative while preserving examples of every learnable
    target stratum in training.
    """
    candidates: list[tuple[int, int, float]] = []
    for left_index, left in enumerate(dataset):
        for right_index in range(left_index + 1, len(dataset)):
            right = dataset[right_index]
            left_id = _graph_id(left)
            right_id = _graph_id(right)
            ged = distance_for(distances, left_id, right_id)
            if ged is None or not math.isfinite(ged):
                continue
            denominator = max(0.5 * (_graph_size(left) + _graph_size(right)), 1.0)
            candidates.append((left_index, right_index, float(ged) / denominator))

    if len(candidates) < 2:
        raise ValueError(
            "At least two finite, non-identity graph pairs are required for a leakage-free split."
        )

    rng = random.Random(seed)
    buckets: dict[int, list[tuple[int, int, float]]] = {}
    for pair in candidates:
        buckets.setdefault(_normalized_ged_bin(pair[2]), []).append(pair)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    maximum_holdout = max(1, min(len(candidates) - 1, len(candidates) // 5))
    requested = min(max(1, int(validation_count)), maximum_holdout)
    validation: list[tuple[int, int]] = []
    validation_caps = {
        bin_index: (
            min(
                len(bucket) - 1,
                max(1, math.ceil(len(bucket) * 0.20)),
            )
            if len(bucket) > 1
            else 0
        )
        for bin_index, bucket in buckets.items()
    }
    active_bins = [
        bin_index
        for bin_index in sorted(buckets)
        if validation_caps[bin_index] > 0
    ]
    positions = {bin_index: 0 for bin_index in active_bins}
    while len(validation) < requested and active_bins:
        progressed = False
        for bin_index in list(active_bins):
            position = positions[bin_index]
            bucket = buckets[bin_index]
            if position >= validation_caps[bin_index]:
                active_bins.remove(bin_index)
                continue
            left_index, right_index, _ = bucket[position]
            positions[bin_index] += 1
            validation.append((left_index, right_index))
            progressed = True
            if len(validation) >= requested:
                break
        if not progressed:
            break

    # A tiny dataset can contain only singleton strata. In that unavoidable
    # case, retain the global train/validation split even though one stratum
    # cannot be represented in both partitions.
    if not validation:
        fallback_bin = max(buckets, key=lambda index: (len(buckets[index]), -index))
        left_index, right_index, _ = buckets[fallback_bin][0]
        validation.append((left_index, right_index))

    validation_keys = {
        canonical_pair_key(_graph_id(dataset[left]), _graph_id(dataset[right]))
        for left, right in validation
    }
    training = [
        (left, right)
        for left, right, _ in candidates
        if canonical_pair_key(_graph_id(dataset[left]), _graph_id(dataset[right]))
        not in validation_keys
    ]
    if not training:
        raise ValueError("Pair holdout consumed every finite training pair.")

    split_hash = hashlib.sha256(
        json.dumps(
            {
                "seed": seed,
                "validation": sorted([list(key) for key in validation_keys]),
            },
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    validation_bin_counts = Counter(
        _normalized_ged_bin(normalized_ged)
        for left, right, normalized_ged in candidates
        if canonical_pair_key(_graph_id(dataset[left]), _graph_id(dataset[right]))
        in validation_keys
    )
    candidate_bin_counts = Counter(
        _normalized_ged_bin(normalized_ged)
        for _, _, normalized_ged in candidates
    )
    bin_distribution = {
        str(bin_index): {
            "candidates": int(candidate_bin_counts[bin_index]),
            "training": int(
                candidate_bin_counts[bin_index] - validation_bin_counts[bin_index]
            ),
            "validation": int(validation_bin_counts[bin_index]),
        }
        for bin_index in sorted(candidate_bin_counts)
    }
    return {
        "training_pairs": training,
        "validation_pairs": validation,
        "validation_keys": validation_keys,
        "metadata": {
            "strategy": (
                "deterministic normalized-GED-stratified pair holdout with "
                "rare-stratum training reserve"
            ),
            "seed": seed,
            "candidate_pairs": len(candidates),
            "training_pairs": len(training),
            "validation_pairs": len(validation),
            "pair_overlap": 0,
            "split_sha256": split_hash,
            "validation_cap_per_stratum": 0.20,
            "normalized_ged_bin_distribution": bin_distribution,
        },
    }


def build_subject_disjoint_pair_split(
    dataset: list,
    distances: dict[tuple[int, int], float],
    validation_count: int,
    seed: int,
    validation_fraction: float = 0.20,
) -> dict[str, Any]:
    """Split graph identities before constructing pairs to prevent subject leakage."""
    if len(dataset) < 4:
        raise ValueError("A subject-disjoint split requires at least four graphs.")
    indices = list(range(len(dataset)))
    validation_graph_count = max(
        2,
        len(indices) - int(len(indices) * (1.0 - validation_fraction)),
    )
    validation_graph_count = min(validation_graph_count, len(indices) - 2)
    validation_indices = set(indices[-validation_graph_count:])
    training_indices = set(indices[:-validation_graph_count])

    def finite_pairs(allowed: set[int]) -> list[tuple[int, int, float]]:
        pairs = []
        ordered = sorted(allowed)
        for offset, left in enumerate(ordered):
            for right in ordered[offset + 1 :]:
                ged = distance_for(
                    distances,
                    _graph_id(dataset[left]),
                    _graph_id(dataset[right]),
                )
                if ged is None or not math.isfinite(ged):
                    continue
                denominator = max(
                    0.5 * (_graph_size(dataset[left]) + _graph_size(dataset[right])),
                    1.0,
                )
                pairs.append((left, right, float(ged) / denominator))
        return pairs

    training_candidates = finite_pairs(training_indices)
    validation_candidates = finite_pairs(validation_indices)
    if not training_candidates or not validation_candidates:
        raise ValueError("Subject-disjoint train/validation pairs could not be constructed.")

    rng = random.Random(seed + 1)
    buckets: dict[int, list[tuple[int, int, float]]] = {}
    for pair in validation_candidates:
        buckets.setdefault(_normalized_ged_bin(pair[2]), []).append(pair)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    requested = min(max(1, int(validation_count)), len(validation_candidates))
    validation: list[tuple[int, int]] = []
    active_bins = sorted(buckets)
    positions = {bin_index: 0 for bin_index in active_bins}
    while len(validation) < requested and active_bins:
        progressed = False
        for bin_index in list(active_bins):
            position = positions[bin_index]
            bucket = buckets[bin_index]
            if position >= len(bucket):
                active_bins.remove(bin_index)
                continue
            left, right, _ = bucket[position]
            positions[bin_index] += 1
            validation.append((left, right))
            progressed = True
            if len(validation) >= requested:
                break
        if not progressed:
            break

    training = [(left, right) for left, right, _ in training_candidates]
    training_graph_ids = sorted(_graph_id(dataset[index]) for index in training_indices)
    validation_graph_ids = sorted(_graph_id(dataset[index]) for index in validation_indices)
    split_identity = {
        "seed": seed,
        "training_graph_ids": training_graph_ids,
        "validation_graph_ids": validation_graph_ids,
    }
    return {
        "training_pairs": training,
        "validation_pairs": validation,
        "validation_keys": {
            canonical_pair_key(_graph_id(dataset[left]), _graph_id(dataset[right]))
            for left, right in validation
        },
        "metadata": {
            "strategy": "subject-disjoint graph split before pair generation",
            "seed": seed,
            "training_graphs": len(training_graph_ids),
            "validation_graphs": len(validation_graph_ids),
            "training_pairs": len(training),
            "validation_pairs": len(validation),
            "graph_overlap": 0,
            "pair_overlap": 0,
            "training_graph_ids": training_graph_ids,
            "validation_graph_ids": validation_graph_ids,
            "split_sha256": hashlib.sha256(
                json.dumps(split_identity, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        },
    }


def split_leakage_comparison(
    graphs: list,
    distances: dict[tuple[int, int], float],
    validation_count: int,
    seed: int = 379,
) -> dict[str, Any]:
    pair_split = build_pair_split(graphs, distances, validation_count, seed)
    subject_split = build_subject_disjoint_pair_split(
        graphs,
        distances,
        validation_count,
        seed,
    )
    return {
        "seed": seed,
        "pair_disjoint": _split_overlap_row(graphs, pair_split, "pair-disjoint"),
        "subject_disjoint": _split_overlap_row(
            graphs,
            subject_split,
            "subject-disjoint",
        ),
    }


def _split_overlap_row(graphs: list, split: dict[str, Any], label: str) -> dict[str, Any]:
    training_graphs = {
        _graph_id(graphs[left])
        for left, right in split["training_pairs"]
    } | {
        _graph_id(graphs[right])
        for left, right in split["training_pairs"]
    }
    validation_graphs = {
        _graph_id(graphs[left])
        for left, right in split["validation_pairs"]
    } | {
        _graph_id(graphs[right])
        for left, right in split["validation_pairs"]
    }
    metadata = split.get("metadata") or {}
    return {
        "strategy": label,
        "training_pairs": len(split["training_pairs"]),
        "validation_pairs": len(split["validation_pairs"]),
        "training_graphs": len(training_graphs),
        "validation_graphs": len(validation_graphs),
        "graph_overlap": len(training_graphs & validation_graphs),
        "pair_overlap": int(metadata.get("pair_overlap") or 0),
        "split_sha256": metadata.get("split_sha256"),
    }


def _graph_id(graph: Any) -> int:
    if isinstance(graph, dict):
        value = graph.get("id")
    else:
        value = getattr(graph, "graph_id", None)
    if value is None:
        raise ValueError("Training graph is missing a stable graph id.")
    return int(value)


def _graph_size(graph: Any) -> int:
    if isinstance(graph, dict):
        if isinstance(graph.get("nodes"), list):
            return len(graph["nodes"])
        graph = graph.get("data", graph)
    value = getattr(graph, "num_nodes", None)
    if value is None:
        raise ValueError("Training graph is missing its node count.")
    return int(value)


def _normalized_ged_bin(value: float) -> int:
    for index, upper in enumerate((0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)):
        if value <= upper:
            return index
    return 7


def graph_descriptor(record: dict[str, Any]) -> dict[str, Any]:
    degree_histogram = Counter()
    degrees = Counter()
    for source, target in record["edges"]:
        degrees[source] += 1
        degrees[target] += 1
    for node_id in record["nodes"]:
        degree_histogram[degrees[node_id]] += 1
    labels = [str(label) for label in record["labels"]]
    if labels and len(set(labels)) == len(labels) and all(_is_number(label) for label in labels):
        labels = ["0"] * len(labels)
    return {
        "nodes": len(record["nodes"]),
        "edges": len(record["edges"]),
        "labels": Counter(labels),
        "degrees": degree_histogram,
    }


def structural_proxy_ged(left: dict[str, Any], right: dict[str, Any]) -> float:
    node_gap = abs(left["nodes"] - right["nodes"])
    edge_gap = abs(left["edges"] - right["edges"])
    label_gap = _counter_l1(left["labels"], right["labels"]) / 2.0
    degree_gap = _counter_l1(left["degrees"], right["degrees"]) / 2.0
    return float(node_gap + edge_gap + label_gap + degree_gap)


def parse_gexf(xml_text: str) -> dict[str, Any]:
    root = ET.fromstring(xml_text)
    attribute_titles = {
        attribute.attrib.get("id", ""): attribute.attrib.get("title", "")
        for attributes in root.findall(".//{*}attributes")
        if attributes.attrib.get("class") == "node"
        for attribute in attributes.findall("./{*}attribute")
    }
    nodes = root.findall(".//{*}node")
    node_ids = [node.attrib.get("id", str(index)) for index, node in enumerate(nodes)]
    node_attributes = {
        node.attrib.get("id", str(index)): {
            attribute_titles.get(value.attrib.get("for", ""), value.attrib.get("for", "")): value.attrib.get("value", "")
            for value in node.findall(".//{*}attvalue")
        }
        for index, node in enumerate(nodes)
    }
    labels_by_id = {}
    for index, node in enumerate(nodes):
        node_id = node.attrib.get("id", str(index))
        labels_by_id[node_id] = str(
            node_attributes[node_id].get("type") or _node_label(node)
        )
    feature_names = sorted(
        {
            name
            for values in node_attributes.values()
            for name in values
            if name.startswith("feature_") and name.removeprefix("feature_").isdigit()
        },
        key=lambda name: int(name.removeprefix("feature_")),
    )
    node_features = None
    if feature_names and all(
        all(name in node_attributes[node_id] for name in feature_names)
        for node_id in node_ids
    ):
        node_features = [
            [float(node_attributes[node_id][name]) for name in feature_names]
            for node_id in node_ids
        ]
    edges = []
    for edge in root.findall(".//{*}edge"):
        source = edge.attrib.get("source")
        target = edge.attrib.get("target")
        if source is not None and target is not None and source != target:
            edges.append((source, target))
    result = {
        "nodes": node_ids,
        "edges": edges,
        "labels": [labels_by_id[node_id] for node_id in node_ids],
    }
    if node_features is not None:
        result["node_features"] = node_features
        result["node_feature_names"] = feature_names
    return result


def serialize_basic_gexf(record: dict[str, Any]) -> str:
    root = ET.Element(
        "gexf",
        xmlns="http://www.gexf.net/1.2draft",
        version="1.2",
    )
    graph = ET.SubElement(root, "graph", mode="static", defaultedgetype="undirected")
    attributes = ET.SubElement(graph, "attributes", {"class": "node"})
    ET.SubElement(attributes, "attribute", id="type", title="type", type="string")
    feature_names = list(record.get("node_feature_names") or [])
    for feature_name in feature_names:
        ET.SubElement(
            attributes,
            "attribute",
            id=feature_name,
            title=feature_name,
            type="double",
        )
    nodes = ET.SubElement(graph, "nodes")
    labels = list(record.get("labels") or [])
    node_features = record.get("node_features")
    for index, node_id in enumerate(record["nodes"]):
        node = ET.SubElement(nodes, "node", id=str(node_id), label=str(node_id))
        values = ET.SubElement(node, "attvalues")
        label = labels[index] if index < len(labels) else "0"
        ET.SubElement(values, "attvalue", {"for": "type", "value": str(label)})
        if node_features is not None and index < len(node_features):
            for feature_name, value in zip(feature_names, node_features[index]):
                ET.SubElement(
                    values,
                    "attvalue",
                    {"for": feature_name, "value": f"{float(value):.10g}"},
                )
    edges = ET.SubElement(graph, "edges")
    for index, (source, target) in enumerate(record["edges"]):
        ET.SubElement(
            edges,
            "edge",
            id=str(index),
            source=str(source),
            target=str(target),
        )
    return ET.tostring(root, encoding="unicode")


def _node_label(node: ET.Element) -> str:
    for attvalue in node.findall(".//{*}attvalue"):
        value = attvalue.attrib.get("value")
        if value not in (None, ""):
            return str(value)
    return str(node.attrib.get("label") or "0")


def _counter_l1(left: Counter, right: Counter) -> int:
    return sum(abs(left[key] - right[key]) for key in set(left) | set(right))


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _archive_gexf(
    archive_path: Path,
    archive_format: str,
) -> list[tuple[str, bytes]]:
    if archive_format == "zip":
        with zipfile.ZipFile(archive_path) as archive:
            return [
                (name, archive.read(name))
                for name in archive.namelist()
                if name.lower().endswith(".gexf")
            ]
    if archive_format == "tar":
        records = []
        with tarfile.open(archive_path) as archive:
            for member in archive.getmembers():
                if not member.isfile() or not member.name.lower().endswith(".gexf"):
                    continue
                extracted = archive.extractfile(member)
                if extracted is not None:
                    records.append((member.name, extracted.read()))
        return records
    raise ValueError(f"Unsupported archive format: {archive_format}")
