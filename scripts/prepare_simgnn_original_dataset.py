from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import pickle
import random
import shutil
import tarfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

try:
    from .universal_dataset import (
        dataset_spec as universal_dataset_spec,
        ensure_training_distances,
    )
except ImportError:
    from universal_dataset import (
        dataset_spec as universal_dataset_spec,
        ensure_training_distances,
    )

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "Models&Datasets" / "SimGNN-v_00001" / "original_datasets"
NORM_GED_BOUNDS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
NORM_GED_BIN_NAMES = (
    "zero",
    "(0,0.25]",
    "(0.25,0.5]",
    "(0.5,0.75]",
    "(0.75,1.0]",
    "(1.0,1.5]",
    "(1.5,2.0]",
    ">2.0",
)
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="aids700nef")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Parent directory for the dataset-specific train/validation/test folders.",
    )
    parser.add_argument("--train-pairs", type=int, default=2400)
    parser.add_argument("--validation-pairs", type=int, default=600)
    parser.add_argument("--test-pairs", type=int, default=700)
    parser.add_argument("--seed", type=int, default=379)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    config = dataset_spec(args.dataset)
    archive_path = config["archive"]
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = (ROOT / output_root).resolve()
    output_dir = output_root / args.dataset
    train_dir = output_dir / "train"
    validation_dir = output_dir / "validation"
    test_dir = output_dir / "test"

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    clear_json_pairs(train_dir)
    clear_json_pairs(validation_dir)
    clear_json_pairs(test_dir)

    graphs = load_graphs(archive_path, config["format"])
    ged, target_metadata = ensure_training_distances(args.dataset)
    splits = graph_splits(archive_path, config["format"])
    train_ids = sorted(graph_id for graph_id, split in splits.items() if split == "train")
    test_ids = sorted(graph_id for graph_id, split in splits.items() if split == "test")
    subject_disjoint = config.get("split_strategy") == "subject_disjoint"
    if subject_disjoint:
        validation_graph_count = max(2, len(train_ids) - int(len(train_ids) * 0.80))
        validation_graph_count = min(validation_graph_count, len(train_ids) - 2)
        validation_ids = train_ids[-validation_graph_count:]
        train_ids = train_ids[:-validation_graph_count]
        train_candidates = build_candidates(train_ids, train_ids, ged, graphs)
        validation_candidates = build_candidates(
            validation_ids,
            validation_ids,
            ged,
            graphs,
        )
        validation_count = min(max(1, args.validation_pairs), len(validation_candidates))
        validation_pairs = balanced_sample(
            validation_candidates,
            validation_count,
            random.Random(args.seed + 2),
        )
        validation_keys = set()
    else:
        validation_ids = []
        train_candidates = build_candidates(train_ids, train_ids, ged, graphs)
        validation_count = min(
            max(1, args.validation_pairs),
            max(1, len(train_candidates) // 5),
            max(1, len(train_candidates) - 1),
        )
        validation_candidates = validation_pool_with_training_reserve(
            train_candidates,
            random.Random(args.seed + 10),
        )
        validation_pairs = balanced_sample(
            validation_candidates,
            validation_count,
            random.Random(args.seed + 1),
        )
        validation_keys = (
            {pair_key(left, right) for left, right, _, _ in validation_pairs}
            if len(validation_pairs) < len(train_candidates)
            else set()
        )
    train_pairs = balanced_sample(
        train_candidates,
        args.train_pairs,
        random.Random(args.seed),
        excluded=validation_keys,
    )
    train_pairs = oversample_exact_zero_pairs(
        train_pairs,
        train_candidates,
        args.train_pairs,
        random.Random(args.seed + 20),
        excluded=validation_keys,
    )
    train_pairs = inject_identity_pairs(
        train_pairs,
        train_ids,
        args.train_pairs,
        random.Random(args.seed + 21),
    )
    if args.test_pairs > 0:
        test_candidates = build_candidates(
            test_ids,
            test_ids if subject_disjoint else train_ids,
            ged,
            graphs,
        )
        test_pairs = balanced_sample(
            test_candidates,
            args.test_pairs,
            random.Random(args.seed + 2),
        )
    else:
        test_pairs = []

    write_pairs(train_dir, train_pairs, graphs)
    write_pairs(validation_dir, validation_pairs, graphs)
    write_pairs(test_dir, test_pairs, graphs)
    manifest = {
        "dataset": args.dataset,
        "archive": str(archive_path.relative_to(ROOT)),
        "target": target_metadata,
        "seed": args.seed,
        "sampling": (
            "left-graph coverage plus round-robin normalized-GED strata; "
            "training-reserved exact-zero pairs and identity anchors with "
            "deterministic oversampling"
        ),
        "pair_split": (
            {
                "strategy": "subject-disjoint graph split before pair generation",
                "training_graph_ids": train_ids,
                "validation_graph_ids": validation_ids,
                "test_graph_ids": test_ids,
                "graph_overlap": 0,
                "pair_overlap": 0,
                "split_sha256": hashlib.sha256(
                    json.dumps(
                        {
                            "training": train_ids,
                            "validation": validation_ids,
                            "test": test_ids,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            }
            if subject_disjoint
            else {
                "strategy": "canonical unordered pair holdout",
                "pair_overlap": 0,
                "split_sha256": hashlib.sha256(
                    json.dumps(
                        sorted([list(key) for key in validation_keys]),
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            }
        ),
        "train": pair_stats(train_pairs),
        "validation": pair_stats(validation_pairs),
        "test": pair_stats(test_pairs),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def dataset_spec(dataset_id: str) -> dict:
    spec = universal_dataset_spec(dataset_id)
    return spec


def load_ged(path: Path) -> dict[tuple[int, int], float]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        values = {}
        for key, distance in payload.items():
            left, right = str(key).split(",", maxsplit=1)
            values[(int(left), int(right))] = float(distance)
        return values
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return {
        (int(left), int(right)): float(distance)
        for (left, right), distance in payload.items()
    }


def load_graphs(archive_path: Path, archive_format: str) -> dict[int, dict]:
    graphs = {}
    with open_archive(archive_path, archive_format) as archive:
        for name in graph_members(archive, archive_format):
            if not name.endswith(".gexf"):
                continue
            graph_id = int(Path(name).stem)
            graphs[graph_id] = parse_gexf(read_member(archive, archive_format, name))
    return graphs


def graph_splits(archive_path: Path, archive_format: str) -> dict[int, str]:
    splits = {}
    with open_archive(archive_path, archive_format) as archive:
        for name in graph_members(archive, archive_format):
            if not name.endswith(".gexf"):
                continue
            split = "train" if "/train/" in name else "test"
            splits[int(Path(name).stem)] = split
    return splits


def open_archive(archive_path: Path, archive_format: str):
    if archive_format == "zip":
        return zipfile.ZipFile(archive_path)
    if archive_format == "tar":
        return tarfile.open(archive_path)
    raise ValueError(f"Unsupported archive format: {archive_format}")


def graph_members(archive, archive_format: str) -> list[str]:
    if archive_format == "zip":
        return archive.namelist()
    return [member.name for member in archive.getmembers() if member.isfile()]


def read_member(archive, archive_format: str, name: str) -> str:
    if archive_format == "zip":
        return archive.read(name).decode("utf-8", errors="replace")
    extracted = archive.extractfile(name)
    if extracted is None:
        raise FileNotFoundError(name)
    return extracted.read().decode("utf-8", errors="replace")


def parse_gexf(xml_text: str) -> dict:
    root = ET.fromstring(xml_text)
    node_elements = root.findall(".//{*}node")
    raw_ids = [node.attrib.get("id", str(index)) for index, node in enumerate(node_elements)]
    ordered_ids = sorted(raw_ids, key=lambda value: int(value) if str(value).isdigit() else str(value))
    node_map = {node_id: index for index, node_id in enumerate(ordered_ids)}

    labels_by_id = {}
    for index, node in enumerate(node_elements):
        node_id = node.attrib.get("id", str(index))
        labels_by_id[node_id] = node_label(node)

    labels = [labels_by_id[node_id] for node_id in ordered_ids]
    if labels and len(set(labels)) == len(labels) and all(is_number(label) for label in labels):
        labels = ["0"] * len(labels)
    edges = []
    for edge in root.findall(".//{*}edge"):
        source = edge.attrib.get("source")
        target = edge.attrib.get("target")
        if source in node_map and target in node_map and source != target:
            edges.append([node_map[source], node_map[target]])
    return {"edges": edges, "labels": labels}


def node_label(node: ET.Element) -> str:
    for attvalue in node.findall(".//{*}attvalue"):
        value = attvalue.attrib.get("value")
        if value:
            return str(value)
    return str(node.attrib.get("label") or node.attrib.get("id") or "0")


def is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def ged_value(ged: dict, left: int, right: int) -> float | None:
    value = ged.get((left, right))
    if value is None:
        value = ged.get((right, left))
    return None if value is None else float(value)


def normalized_ged(value: float, left: int, right: int, graphs: dict[int, dict]) -> float:
    denominator = 0.5 * (len(graphs[left]["labels"]) + len(graphs[right]["labels"]))
    return value / max(denominator, 1.0)


def norm_ged_bin(value: float) -> int:
    if value <= NORM_GED_BOUNDS[0]:
        return 0
    for index, upper in enumerate(NORM_GED_BOUNDS[1:], start=1):
        if value <= upper:
            return index
    return len(NORM_GED_BIN_NAMES) - 1


def build_candidates(
    left_ids: list[int],
    right_ids: list[int],
    ged: dict,
    graphs: dict[int, dict],
) -> list[tuple[int, int, float, float]]:
    candidates = []
    seen: set[tuple[int, int]] = set()
    for left in left_ids:
        for right in right_ids:
            if left == right:
                continue
            key = pair_key(left, right)
            if key in seen:
                continue
            value = ged_value(ged, left, right)
            if value is not None:
                seen.add(key)
                candidates.append((left, right, value, normalized_ged(value, left, right, graphs)))
    return candidates


def balanced_sample(
    candidates: list[tuple[int, int, float, float]],
    count: int,
    rng: random.Random,
    excluded: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int, float, float]]:
    excluded = excluded or set()
    available = [
        pair
        for pair in candidates
        if pair_key(pair[0], pair[1]) not in excluded
    ]
    if count <= 0:
        return []
    count = min(count, len(available))

    by_left: dict[int, list[tuple[int, int, float, float]]] = defaultdict(list)
    by_bin: dict[int, list[tuple[int, int, float, float]]] = defaultdict(list)
    for pair in available:
        by_left[pair[0]].append(pair)
        by_bin[norm_ged_bin(pair[3])].append(pair)
    for pairs in by_left.values():
        rng.shuffle(pairs)
    for pairs in by_bin.values():
        rng.shuffle(pairs)

    selected = []
    selected_keys: set[tuple[int, int]] = set()
    selected_bins: Counter[int] = Counter()

    left_ids = list(by_left)
    rng.shuffle(left_ids)
    for left in left_ids:
        if len(selected) >= count:
            break
        choices = by_left[left]
        pair = min(
            choices,
            key=lambda item: (
                selected_bins[norm_ged_bin(item[3])],
                rng.random(),
            ),
        )
        key = pair_key(pair[0], pair[1])
        if key not in selected_keys:
            selected.append(pair)
            selected_keys.add(key)
            selected_bins[norm_ged_bin(pair[3])] += 1

    active_bins = [index for index in range(len(NORM_GED_BIN_NAMES)) if by_bin[index]]
    positions = {index: 0 for index in active_bins}
    while len(selected) < count and active_bins:
        progressed = False
        for index in list(active_bins):
            pairs = by_bin[index]
            position = positions[index]
            while position < len(pairs):
                pair = pairs[position]
                position += 1
                key = pair_key(pair[0], pair[1])
                if key not in selected_keys:
                    selected.append(pair)
                    selected_keys.add(key)
                    selected_bins[index] += 1
                    progressed = True
                    break
            positions[index] = position
            if position >= len(pairs):
                active_bins.remove(index)
            if len(selected) >= count:
                break
        if not progressed:
            break
    rng.shuffle(selected)
    return selected


def validation_pool_with_training_reserve(
    candidates: list[tuple[int, int, float, float]],
    rng: random.Random,
) -> list[tuple[int, int, float, float]]:
    """Keep examples from every available GED stratum for leakage-free training."""
    by_bin: dict[int, list[tuple[int, int, float, float]]] = defaultdict(list)
    for pair in candidates:
        by_bin[norm_ged_bin(pair[3])].append(pair)

    reserved: set[tuple[int, int]] = set()
    for pairs in by_bin.values():
        shuffled = list(pairs)
        rng.shuffle(shuffled)
        reserve_count = 1 if len(shuffled) == 1 else max(1, len(shuffled) // 2)
        reserved.update(pair_key(pair[0], pair[1]) for pair in shuffled[:reserve_count])
    return [
        pair
        for pair in candidates
        if pair_key(pair[0], pair[1]) not in reserved
    ]


def oversample_exact_zero_pairs(
    sampled: list[tuple[int, int, float, float]],
    candidates: list[tuple[int, int, float, float]],
    count: int,
    rng: random.Random,
    excluded: set[tuple[int, int]] | None = None,
    minimum_fraction: float = 0.025,
) -> list[tuple[int, int, float, float]]:
    """Expose rare non-identity GED=0 pairs without leaking validation pairs."""
    excluded = excluded or set()
    zeros = [
        pair
        for pair in candidates
        if pair[2] == 0
        and pair_key(pair[0], pair[1]) not in excluded
    ]
    if not zeros or count <= 0:
        return sampled[:count]

    target_zero_count = min(count, max(1, round(count * minimum_fraction)))
    zero_samples = [rng.choice(zeros) for _ in range(target_zero_count)]
    nonzero_samples = [pair for pair in sampled if pair[2] != 0]
    rng.shuffle(nonzero_samples)
    combined = zero_samples + nonzero_samples[: max(0, count - target_zero_count)]
    refill_pool = nonzero_samples or zeros
    while len(combined) < count:
        combined.append(rng.choice(refill_pool))
    rng.shuffle(combined)
    return combined


def inject_identity_pairs(
    sampled: list[tuple[int, int, float, float]],
    graph_ids: list[int],
    count: int,
    rng: random.Random,
    fraction: float = 0.10,
) -> list[tuple[int, int, float, float]]:
    """Add diverse self-pairs so similarity one is an explicit training anchor."""
    if not graph_ids or count <= 0:
        return sampled[:count]
    target_identity_count = min(count, max(1, round(count * fraction)))
    identities = [(graph_id, graph_id, 0.0, 0.0) for graph_id in graph_ids]
    identity_samples = [rng.choice(identities) for _ in range(target_identity_count)]
    nonidentity_samples = [pair for pair in sampled if pair[0] != pair[1]]
    rng.shuffle(nonidentity_samples)
    combined = identity_samples + nonidentity_samples[: count - target_identity_count]
    refill_pool = nonidentity_samples or identities
    while len(combined) < count:
        combined.append(rng.choice(refill_pool))
    rng.shuffle(combined)
    return combined


def pair_key(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left <= right else (right, left)


def pair_stats(pairs: list[tuple[int, int, float, float]]) -> dict:
    bins = Counter(norm_ged_bin(pair[3]) for pair in pairs)
    values = sorted(pair[3] for pair in pairs)

    def percentile(fraction: float) -> float | None:
        if not values:
            return None
        index = round((len(values) - 1) * fraction)
        return round(values[index], 6)

    return {
        "pairs": len(pairs),
        "unique_left_graphs": len({pair[0] for pair in pairs}),
        "unique_right_graphs": len({pair[1] for pair in pairs}),
        "exact_zero_pairs": sum(pair[2] == 0 for pair in pairs),
        "identity_pairs": sum(pair[0] == pair[1] for pair in pairs),
        "distinct_exact_zero_pairs": sum(
            pair[2] == 0 and pair[0] != pair[1] for pair in pairs
        ),
        "normalized_ged_bins": {
            name: bins[index] for index, name in enumerate(NORM_GED_BIN_NAMES)
        },
        "normalized_ged_quantiles": {
            "min": percentile(0.0),
            "p25": percentile(0.25),
            "median": percentile(0.5),
            "p75": percentile(0.75),
            "max": percentile(1.0),
        },
    }


def clear_json_pairs(output_dir: Path) -> None:
    for path in output_dir.glob("*.json"):
        path.unlink()


def write_pairs(
    output_dir: Path,
    pairs: list[tuple[int, int, float, float]],
    graphs: dict[int, dict],
) -> None:
    for index, (left, right, value, norm_value) in enumerate(pairs):
        payload = {
            "graph_1": graphs[left]["edges"],
            "graph_2": graphs[right]["edges"],
            "labels_1": graphs[left]["labels"],
            "labels_2": graphs[right]["labels"],
            "ged": value,
            "normalized_ged": norm_value,
            "gid_1": left,
            "gid_2": right,
        }
        (output_dir / f"{index}.json").write_text(json.dumps(payload))


if __name__ == "__main__":
    main()
