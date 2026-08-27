from __future__ import annotations

import csv
from datetime import datetime, timezone
import io
import json
import math
import pickle
import re
import shutil
import tarfile
import tempfile
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


BASE_DIR = Path(__file__).resolve().parent.parent
SIMGNN_DATASET_DIR = BASE_DIR / "Models&Datasets" / "SimGNN-v_00001" / "dataset"
ORIGINAL_DATASET_DIR = BASE_DIR / "Models&Datasets" / "drive-download-20260630T100606Z-3-001"
UPLOADED_DATASET_DIR = BASE_DIR / "Models&Datasets" / "uploaded_datasets"
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_GRAPH_BYTES = 20 * 1024 * 1024
MAX_ARCHIVE_TOTAL_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_GRAPHS = 20_000

ORIGINAL_DATASETS = [
    {
        "id": "aids700nef",
        "name": "AIDS700nef",
        "domain": "Molecular graphs",
        "archive": "AIDS700nef.zip",
        "format": "zip",
        "tasks": ["GED", "MCS"],
        "ground_truth": [
            "aids700nef_ged_astar_gidpair_dist_map.pickle",
            "aids700nef_mcs_mccreesh2017_gidpair_dist_map.pickle",
        ],
        "result_archive": "aids700nef_result.tar.gz",
        "target_exact": True,
        "target_kind": "exact",
        "target_source": "exact A* GED benchmark",
        "papers": ["SimGNN", "AAAI 2020", "SEGMN", "Graph2Region"],
    },
    {
        "id": "linux",
        "name": "LINUX",
        "domain": "Program dependency/function graphs",
        "archive": "LINUX.tar.gz",
        "format": "tar",
        "tasks": ["GED", "MCS"],
        "ground_truth": [
            "linux_ged_astar_gidpair_dist_map.pickle",
            "linux_mcs_mccreesh2017_gidpair_dist_map.pickle",
        ],
        "result_archive": "linux_result.tar.gz",
        "target_exact": True,
        "target_kind": "exact",
        "target_source": "exact A* GED benchmark",
        "papers": ["SimGNN", "AAAI 2020", "SEGMN", "Graph2Region"],
    },
    {
        "id": "imdbmulti",
        "name": "IMDBMulti",
        "domain": "Ego-network movie graphs",
        "archive": "IMDBMulti.zip",
        "format": "zip",
        "tasks": ["GED", "MCS"],
        "ground_truth": [
            "imdbmulti_ged_astar_gidpair_dist_map.pickle",
            "imdbmulti_mcs_mccreesh2017_gidpair_dist_map.pickle",
        ],
        "result_archive": "imdbmulti_result.tar.gz",
        "target_exact": False,
        "target_kind": "approximate_benchmark",
        "target_source": (
            "approximate GED benchmark: minimum upper bound from Beam, "
            "Hungarian, and VJ"
        ),
        "papers": ["SimGNN", "AAAI 2020", "SEGMN", "Graph2Region"],
    },
    {
        "id": "ptc",
        "name": "PTC",
        "domain": "Chemical compound graphs",
        "archive": "PTC.zip",
        "format": "zip",
        "tasks": ["GED", "MCS"],
        "ground_truth": [
            "ptc_ged_astar_gidpair_dist_map.pickle",
            "ptc_mcs_mccreesh2017_gidpair_dist_map.pickle",
        ],
        "result_archive": "ptc_result.tar.gz",
        "target_exact": False,
        "target_kind": "approximate_benchmark",
        "target_source": (
            "approximate GED benchmark: minimum upper bound from Beam, "
            "Hungarian, and VJ"
        ),
        "papers": ["AAAI 2020", "SEGMN", "Graph2Region"],
    },
    {
        "id": "mutag",
        "name": "MUTAG",
        "domain": "TU Dortmund molecular graph classification",
        "archive": "MUTAG.zip",
        "format": "zip",
        "tasks": ["Preview", "Classification labels"],
        "ground_truth": [],
        "result_archive": "",
        "target_exact": False,
        "target_kind": "structural_proxy",
        "target_source": "derived structural proxy; no GED benchmark labels",
        "papers": ["TUDataset"],
    },
    {
        "id": "proteins",
        "name": "PROTEINS",
        "domain": "TU Dortmund protein graph classification",
        "archive": "PROTEINS.zip",
        "format": "zip",
        "tasks": ["Preview", "Classification labels"],
        "ground_truth": [],
        "result_archive": "",
        "target_exact": False,
        "target_kind": "structural_proxy",
        "target_source": "derived structural proxy; no GED benchmark labels",
        "papers": ["TUDataset"],
    },
    {
        "id": "enzymes",
        "name": "ENZYMES",
        "domain": "TU Dortmund enzyme graph classification",
        "archive": "ENZYMES.zip",
        "format": "zip",
        "tasks": ["Preview", "Classification labels"],
        "ground_truth": [],
        "result_archive": "",
        "target_exact": False,
        "target_kind": "structural_proxy",
        "target_source": "derived structural proxy; no GED benchmark labels",
        "papers": ["TUDataset"],
    },
]


class DatasetUploadError(ValueError):
    pass


def save_uploaded_dataset(
    archive_file: Any,
    name: str,
    dataset_id: str | None = None,
    domain: str | None = None,
    ground_truth_file: Any | None = None,
) -> dict[str, Any]:
    display_name = str(name or "").strip()
    if not display_name:
        raise DatasetUploadError("Dataset name is required.")
    clean_id = _dataset_slug(dataset_id or display_name)
    if any(dataset["id"] == clean_id for dataset in _all_datasets()):
        raise DatasetUploadError(f"Dataset id already exists: {clean_id}")
    if archive_file is None or not getattr(archive_file, "filename", ""):
        raise DatasetUploadError("A ZIP, TAR, TAR.GZ, or TGZ graph archive is required.")

    archive_format = _archive_format_from_name(archive_file.filename)
    UPLOADED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    target_dir = UPLOADED_DATASET_DIR / clean_id
    with tempfile.TemporaryDirectory(prefix=f".{clean_id}-", dir=UPLOADED_DATASET_DIR) as temporary:
        staging_dir = Path(temporary)
        source_path = staging_dir / f"source.{_archive_extension(archive_format)}"
        archive_file.save(source_path)
        if source_path.stat().st_size > MAX_UPLOAD_BYTES:
            raise DatasetUploadError("Uploaded archive exceeds the 512 MB limit.")

        graphs, embedded_ground_truth = _read_uploaded_archive(source_path, archive_format)
        normalized_graphs, graph_name_map = _normalize_uploaded_graphs(graphs, clean_id)
        ground_truth_payload = _read_upload_file(ground_truth_file)
        if ground_truth_payload is None:
            ground_truth_payload = embedded_ground_truth
        ground_truth = (
            _parse_uploaded_ground_truth(
                ground_truth_payload[0],
                ground_truth_payload[1],
                graph_name_map,
            )
            if ground_truth_payload
            else {}
        )

        normalized_archive = staging_dir / "dataset.zip"
        with zipfile.ZipFile(normalized_archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for graph in normalized_graphs:
                archive.writestr(graph["member"], graph["content"])

        ground_truth_files: list[str] = []
        if ground_truth:
            ground_truth_path = staging_dir / "ged.json"
            ground_truth_path.write_text(
                json.dumps(
                    {
                        f"{left},{right}": value
                        for (left, right), value in sorted(ground_truth.items())
                    },
                    indent=2,
                )
            )
            ground_truth_files.append("ged.json")

        train_graphs = sum(graph["split"] == "train" for graph in normalized_graphs)
        test_graphs = sum(graph["split"] == "test" for graph in normalized_graphs)
        train_ids = {graph["id"] for graph in normalized_graphs if graph["split"] == "train"}
        test_ids = {graph["id"] for graph in normalized_graphs if graph["split"] == "test"}
        train_pairs = sum(left in train_ids and right in train_ids for left, right in ground_truth)
        test_pairs = sum(
            (left in test_ids and right in train_ids) or (right in test_ids and left in train_ids)
            for left, right in ground_truth
        )
        training_ready = train_graphs >= 2 and test_graphs >= 1 and train_pairs > 0 and test_pairs > 0
        manifest = {
            "id": clean_id,
            "name": display_name[:120],
            "domain": str(domain or "Uploaded graph collection").strip()[:160],
            "archive": "dataset.zip",
            "format": "zip",
            "tasks": ["GED", "Preview"] if ground_truth else ["Preview"],
            "ground_truth": ground_truth_files,
            "result_archive": "",
            "papers": ["User upload"],
            "source": "uploaded",
            "storage_dir": _display_path(target_dir),
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "original_filename": Path(archive_file.filename).name,
            "graph_name_map": {
                str(graph["id"]): graph["source_member"] for graph in normalized_graphs
            },
            "ged_pairs": len(ground_truth),
            "target_exact": False,
            "target_kind": "unverified_ged" if ground_truth else "structural_proxy",
            "target_source": (
                "user-provided GED reference; exactness not independently verified"
                if ground_truth
                else "derived structural proxy; no GED benchmark labels"
            ),
            "target_semantics": (
                "GED-like distance supplied by the user"
                if ground_truth
                else "structural proxy distance; not a GED benchmark label"
            ),
            "training_ready": training_ready,
            "training_pair_counts": {"train": train_pairs, "test": test_pairs},
        }
        (staging_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        shutil.move(str(staging_dir), str(target_dir))

    _cached_ground_truth.cache_clear()
    return next(dataset for dataset in list_original_datasets() if dataset["id"] == clean_id)


def dataset_storage_spec(dataset_id: str) -> dict[str, Any]:
    dataset = _dataset_by_id(dataset_id)
    archive_path = _archive_path(dataset)
    ground_truth_paths = [_dataset_file_path(dataset, path) for path in dataset["ground_truth"]]
    return {
        "id": dataset["id"],
        "name": dataset["name"],
        "archive_path": str(archive_path),
        "format": dataset["format"],
        "ground_truth_paths": [str(path) for path in ground_truth_paths if path.exists()],
        "uploaded": dataset.get("source") == "uploaded",
        "training_ready": bool(dataset.get("training_ready", True)),
    }


def is_uploaded_dataset(dataset_id: str | None) -> bool:
    if not dataset_id:
        return False
    try:
        return _dataset_by_id(dataset_id).get("source") == "uploaded"
    except ValueError:
        return False


def uploaded_dataset_trainable(dataset_id: str | None) -> bool:
    if not is_uploaded_dataset(dataset_id):
        return False
    return bool(_dataset_by_id(str(dataset_id)).get("training_ready"))


def list_samples(limit: int = 100) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    if not SIMGNN_DATASET_DIR.exists():
        return samples

    for split in ["train", "test"]:
        split_dir = SIMGNN_DATASET_DIR / split
        if not split_dir.exists():
            continue
        for path in sorted(split_dir.glob("*.json"), key=lambda item: int(item.stem) if item.stem.isdigit() else item.stem):
            if len(samples) >= limit:
                return samples
            try:
                payload = json.loads(path.read_text())
                samples.append(
                    {
                        "id": path.stem,
                        "split": split,
                        "name": f"{split}/{path.stem}.json",
                        "nodes_1": len(payload.get("labels_1", [])),
                        "nodes_2": len(payload.get("labels_2", [])),
                        "edges_1": len(payload.get("graph_1", [])),
                        "edges_2": len(payload.get("graph_2", [])),
                        "ged": payload.get("ged"),
                    }
                )
            except (OSError, json.JSONDecodeError):
                continue
    return samples


def list_original_datasets() -> list[dict[str, Any]]:
    datasets = []
    for dataset in _all_datasets():
        archive_path = _archive_path(dataset)
        split_counts = _archive_split_counts(archive_path, dataset["format"])
        ground_truth = [
            _display_path(_dataset_file_path(dataset, path))
            for path in dataset["ground_truth"]
            if _dataset_file_path(dataset, path).exists()
        ]
        result_archive = dataset.get("result_archive") or ""
        result_path = _dataset_file_path(dataset, result_archive) if result_archive else None
        datasets.append(
            {
                **dataset,
                "archive_path": _display_path(archive_path),
                "available": archive_path.exists(),
                "train_graphs": split_counts.get("train", 0),
                "test_graphs": split_counts.get("test", 0),
                "graph_count": sum(split_counts.values()),
                "ground_truth_available": len(ground_truth),
                "ground_truth_exact": bool(dataset.get("target_exact", False)),
                "ground_truth_kind": dataset.get("target_kind", "structural_proxy"),
                "ground_truth_source": dataset.get("target_source"),
                "ground_truth_benchmark": bool(
                    ground_truth
                    and dataset.get("target_kind") in {"exact", "approximate_benchmark"}
                ),
                "ground_truth_paths": ground_truth,
                "result_archive_available": bool(result_path and result_path.exists()),
                "uploaded": dataset.get("source") == "uploaded",
                "training_ready": bool(dataset.get("training_ready", True)),
            }
        )
    return datasets


def load_original_dataset(dataset_id: str) -> dict[str, Any]:
    dataset = _dataset_by_id(dataset_id)
    archive_path = _archive_path(dataset)
    if not archive_path.exists():
        raise FileNotFoundError(f"Dataset archive not found: {dataset['archive']}")

    graph_members = _archive_graph_members(archive_path, dataset["format"])
    train_members = [member for member in graph_members if "/train/" in f"/{member}"]
    test_members = [member for member in graph_members if "/test/" in f"/{member}"]

    left_member = _choose_member(train_members or graph_members, preferred_index=0)
    right_member = _choose_member(test_members or graph_members, preferred_index=1)
    if left_member is None or right_member is None:
        raise FileNotFoundError(f"No GEXF graphs found in {dataset['archive']}")

    return load_original_pair(dataset_id, left_member, right_member)


def list_original_graphs(dataset_id: str) -> dict[str, Any]:
    dataset = _dataset_by_id(dataset_id)
    archive_path = _archive_path(dataset)
    if not archive_path.exists():
        raise FileNotFoundError(f"Dataset archive not found: {dataset['archive']}")

    members = _archive_graph_members(archive_path, dataset["format"])
    graphs = [_graph_option(member) for member in members]
    dataset_info = next(item for item in list_original_datasets() if item["id"] == dataset["id"])
    return {
        "dataset": dataset_info,
        "graphs": graphs,
        "train": [graph for graph in graphs if graph["split"] == "train"],
        "test": [graph for graph in graphs if graph["split"] == "test"],
    }


def load_original_graph_collection(dataset_id: str, members: list[str] | None = None) -> list[dict[str, Any]]:
    dataset = _dataset_by_id(dataset_id)
    archive_path = _archive_path(dataset)
    if not archive_path.exists():
        raise FileNotFoundError(f"Dataset archive not found: {dataset['archive']}")

    graph_members = _archive_graph_members(archive_path, dataset["format"])
    selected_members = graph_members if members is None else members
    available = set(graph_members)
    missing = [member for member in selected_members if member not in available]
    if missing:
        raise FileNotFoundError(f"Graph not found in {dataset['archive']}: {missing[0]}")

    texts = _read_archive_texts(archive_path, dataset["format"], selected_members)
    return [
        {
            **_graph_option(member),
            "graph": _parse_gexf(texts[member]),
        }
        for member in selected_members
    ]


def load_ground_truth_distances(dataset_id: str, task: str = "ged") -> dict[tuple[int, int], float]:
    dataset = _dataset_by_id(dataset_id)
    task_path = next((path for path in dataset["ground_truth"] if f"_{task.lower()}_" in path.lower()), None)
    if task_path is None and task.lower() == "ged":
        task_path = next((path for path in dataset["ground_truth"] if Path(path).stem.lower() == "ged"), None)
    if task_path is None:
        raise FileNotFoundError(f"No {task.upper()} ground-truth file registered for {dataset['name']}.")
    return _cached_ground_truth(str(_dataset_file_path(dataset, task_path)))


def ground_truth_is_exact(dataset_id: str) -> bool:
    """Return whether a registered target map is declared as exact ground truth."""
    return bool(_dataset_by_id(dataset_id).get("target_exact", False))


def ground_truth_kind(dataset_id: str) -> str:
    """Return exact, approximate_benchmark, or structural_proxy provenance."""
    dataset = _dataset_by_id(dataset_id)
    if dataset.get("target_kind"):
        return str(dataset["target_kind"])
    return "exact" if dataset.get("target_exact", False) else "structural_proxy"


def pair_ground_truth(
    dataset_id: str | None,
    left_member: str | None,
    right_member: str | None,
    left_nodes: int,
    right_nodes: int,
    task: str = "ged",
) -> dict[str, Any] | None:
    if not dataset_id or not left_member or not right_member:
        return None
    try:
        left_id = int(Path(left_member).stem)
        right_id = int(Path(right_member).stem)
    except ValueError:
        return None

    try:
        distances = load_ground_truth_distances(dataset_id, task=task)
    except FileNotFoundError:
        return None
    distance = distances.get((left_id, right_id))
    if distance is None:
        distance = distances.get((right_id, left_id))
    if distance is None:
        return None

    graph_size = max(0.5 * (left_nodes + right_nodes), 1.0)
    normalized = float(distance) / graph_size
    similarity = math.exp(-normalized)
    dataset = _dataset_by_id(dataset_id)
    exact = bool(dataset.get("target_exact", False))
    kind = ground_truth_kind(dataset_id)
    return {
        "task": task.upper(),
        "distance": float(distance),
        "normalized_distance": normalized,
        "similarity": similarity,
        "left_graph": left_member,
        "right_graph": right_member,
        "exact": exact,
        "reference_kind": kind,
        "source": str(dataset.get("target_source") or "registered GED target map"),
    }


def load_original_pair(dataset_id: str, left_member: str, right_member: str) -> dict[str, Any]:
    dataset = _dataset_by_id(dataset_id)
    archive_path = _archive_path(dataset)
    if not archive_path.exists():
        raise FileNotFoundError(f"Dataset archive not found: {dataset['archive']}")

    graph_members = set(_archive_graph_members(archive_path, dataset["format"]))
    if left_member not in graph_members:
        raise FileNotFoundError(f"Graph A not found in {dataset['archive']}: {left_member}")
    if right_member not in graph_members:
        raise FileNotFoundError(f"Graph B not found in {dataset['archive']}: {right_member}")

    left_xml = _read_archive_text(archive_path, dataset["format"], left_member)
    right_xml = _read_archive_text(archive_path, dataset["format"], right_member)
    left_graph = _parse_gexf(left_xml)
    right_graph = _parse_gexf(right_xml)

    datasets = list_original_datasets()
    dataset_info = next(item for item in datasets if item["id"] == dataset["id"])
    return {
        "left": left_graph,
        "right": right_graph,
        "dataset": dataset_info,
        "meta": {
            "dataset_id": dataset["id"],
            "dataset": dataset["name"],
            "source": _display_path(archive_path),
            "left_graph": left_member,
            "right_graph": right_member,
            "tasks": dataset["tasks"],
            "ground_truth": dataset_info["ground_truth_paths"],
        },
    }


def original_pair_matches_graphs(
    dataset_id: str | None,
    left_member: str | None,
    right_member: str | None,
    left_graph: Any,
    right_graph: Any,
) -> bool | None:
    if not dataset_id or not left_member or not right_member:
        return None

    from .graph_utils import graph_from_payload

    try:
        pair = load_original_pair(dataset_id, left_member, right_member)
        expected_left = graph_from_payload(pair["left"], name="Expected Graph A")
        expected_right = graph_from_payload(pair["right"], name="Expected Graph B")
    except (FileNotFoundError, ValueError, TypeError):
        return False

    return _graph_signature(left_graph) == _graph_signature(expected_left) and _graph_signature(
        right_graph
    ) == _graph_signature(expected_right)


def _graph_signature(graph: Any) -> tuple[Any, ...]:
    return (
        int(graph.node_count),
        tuple(graph.edges),
        tuple(str(label) for label in graph.labels),
    )


def _graph_option(member: str) -> dict[str, Any]:
    split = "train" if "/train/" in f"/{member}" else "test" if "/test/" in f"/{member}" else "graph"
    graph_id = Path(member).stem
    return {
        "id": graph_id,
        "split": split,
        "member": member,
        "label": f"{split}/{graph_id}.gexf" if split in {"train", "test"} else Path(member).name,
    }


def load_sample(split: str, sample_id: str) -> dict[str, Any]:
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test.")
    clean_id = Path(sample_id).stem
    path = SIMGNN_DATASET_DIR / split / f"{clean_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Sample not found: {split}/{clean_id}.json")
    payload = json.loads(path.read_text())
    return {
        "left": {"edges": payload.get("graph_1", []), "labels": payload.get("labels_1", [])},
        "right": {"edges": payload.get("graph_2", []), "labels": payload.get("labels_2", [])},
        "meta": {
            "split": split,
            "id": clean_id,
            "ged": payload.get("ged"),
            "source": str(path.relative_to(BASE_DIR)),
        },
    }


def sample_pair() -> dict[str, Any]:
    try:
        return load_original_dataset("aids700nef")
    except (FileNotFoundError, ValueError, ET.ParseError):
        pass

    try:
        return load_sample("train", "0")
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return {
            "left": {
                "edges": [[0, 1], [1, 2], [2, 0], [2, 3]],
                "labels": ["C", "C", "N", "O"],
            },
            "right": {
                "edges": [[0, 1], [1, 2], [2, 3], [3, 0], [1, 3]],
                "labels": ["C", "N", "N", "O"],
            },
            "meta": {"source": "built-in fallback"},
        }


def local_archives() -> list[dict[str, str]]:
    candidates = [
        ("SimGNN", BASE_DIR / "Models&Datasets" / "SimGNN-v_00001"),
        ("GraphSim / Multi-Scale", BASE_DIR / "Models&Datasets" / "GraphSim-master"),
        ("SEGMN", BASE_DIR / "Models&Datasets" / "SEGMN-main"),
        ("Graph2Region", BASE_DIR / "Models&Datasets" / "Graph2Region-main"),
        ("GED/MCS Dataset Archives", BASE_DIR / "Models&Datasets" / "drive-download-20260630T100606Z-3-001"),
        ("Downloaded Real Datasets", BASE_DIR / "Models&Datasets" / "downloaded_real"),
        ("Uploaded Datasets", UPLOADED_DATASET_DIR),
    ]
    archives = []
    for name, path in candidates:
        archives.append(
            {
                "name": name,
                "path": str(path.relative_to(BASE_DIR)),
                "available": path.exists(),
            }
        )
    return archives


def _all_datasets() -> list[dict[str, Any]]:
    datasets = list(ORIGINAL_DATASETS)
    if not UPLOADED_DATASET_DIR.exists():
        return datasets
    for manifest_path in sorted(UPLOADED_DATASET_DIR.glob("*/manifest.json")):
        try:
            payload = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("id") != manifest_path.parent.name:
            continue
        if not (manifest_path.parent / str(payload.get("archive", ""))).exists():
            continue
        datasets.append(payload)
    return datasets


def _dataset_file_path(dataset: dict[str, Any], relative_path: str) -> Path:
    if dataset.get("source") == "uploaded":
        return UPLOADED_DATASET_DIR / dataset["id"] / relative_path
    return ORIGINAL_DATASET_DIR / relative_path


def _archive_path(dataset: dict[str, Any]) -> Path:
    return _dataset_file_path(dataset, dataset["archive"])


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def _dataset_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")
    if not slug:
        raise DatasetUploadError("Dataset id must contain a letter or number.")
    if len(slug) > 64:
        slug = slug[:64].rstrip("-")
    return slug


def _archive_format_from_name(filename: str) -> str:
    lowered = str(filename).lower()
    if lowered.endswith(".zip"):
        return "zip"
    if lowered.endswith((".tar", ".tar.gz", ".tgz")):
        return "tar"
    raise DatasetUploadError("Dataset archive must be ZIP, TAR, TAR.GZ, or TGZ.")


def _archive_extension(archive_format: str) -> str:
    return "zip" if archive_format == "zip" else "tar.gz"


def _safe_archive_member(name: str) -> bool:
    normalized = str(name).replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    return bool(parts) and not normalized.startswith("/") and ".." not in parts


def _read_uploaded_archive(
    archive_path: Path,
    archive_format: str,
) -> tuple[list[dict[str, Any]], tuple[bytes, str] | None]:
    graphs: list[dict[str, Any]] = []
    embedded_ground_truth: tuple[bytes, str] | None = None
    total_bytes = 0

    def validate_file(name: str, size: int) -> bool:
        nonlocal total_bytes
        if not _safe_archive_member(name):
            raise DatasetUploadError(f"Unsafe archive member path: {name}")
        total_bytes += int(size)
        if total_bytes > MAX_ARCHIVE_TOTAL_BYTES:
            raise DatasetUploadError("Archive expands beyond the 512 MB validation limit.")
        lowered = name.lower()
        is_graph = lowered.endswith(".gexf")
        is_ground_truth = (
            Path(lowered).suffix in {".csv", ".json"}
            and "ged" in Path(lowered).stem
        )
        if is_graph and size > MAX_ARCHIVE_GRAPH_BYTES:
            raise DatasetUploadError(f"Graph file exceeds 20 MB: {name}")
        return is_graph or is_ground_truth

    def add_file(name: str, content: bytes) -> None:
        nonlocal embedded_ground_truth, total_bytes
        lowered = name.lower()
        if lowered.endswith(".gexf"):
            if len(graphs) >= MAX_ARCHIVE_GRAPHS:
                raise DatasetUploadError("Archive contains more than 20,000 graphs.")
            try:
                parsed = _parse_gexf(content.decode("utf-8", errors="replace"))
            except ET.ParseError as exc:
                raise DatasetUploadError(f"Invalid GEXF file {name}: {exc}") from exc
            if not parsed["nodes"]:
                raise DatasetUploadError(f"GEXF graph has no nodes: {name}")
            graphs.append({"source_member": name, "content": content})
        elif (
            embedded_ground_truth is None
            and Path(lowered).suffix in {".csv", ".json"}
            and "ged" in Path(lowered).stem
        ):
            embedded_ground_truth = (content, Path(name).name)

    try:
        if archive_format == "zip":
            with zipfile.ZipFile(archive_path) as archive:
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    if validate_file(info.filename, info.file_size):
                        add_file(info.filename, archive.read(info.filename))
        else:
            with tarfile.open(archive_path) as archive:
                for member in archive.getmembers():
                    if not member.isfile():
                        continue
                    if not validate_file(member.name, member.size):
                        continue
                    extracted = archive.extractfile(member)
                    if extracted is not None:
                        add_file(member.name, extracted.read())
    except (zipfile.BadZipFile, tarfile.TarError, OSError) as exc:
        raise DatasetUploadError(f"Dataset archive could not be read: {exc}") from exc

    if len(graphs) < 2:
        raise DatasetUploadError("Dataset archive must contain at least two GEXF graphs.")
    return graphs, embedded_ground_truth


def _normalize_uploaded_graphs(
    graphs: list[dict[str, Any]],
    dataset_id: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    ordered = sorted(graphs, key=lambda graph: graph["source_member"].lower())
    stems = [Path(graph["source_member"]).stem for graph in ordered]
    if len(set(stems)) != len(stems):
        raise DatasetUploadError("GEXF file stems must be unique across the archive.")

    numeric_stems = all(stem.isdigit() for stem in stems)
    assigned_ids = [int(stem) for stem in stems] if numeric_stems else list(range(len(stems)))
    marked_splits = [_member_split(graph["source_member"]) for graph in ordered]
    has_train = "train" in marked_splits
    has_test = "test" in marked_splits
    if not (has_train and has_test):
        cutoff = max(1, min(len(ordered) - 1, round(len(ordered) * 0.8)))
        marked_splits = ["train" if index < cutoff else "test" for index in range(len(ordered))]
    else:
        marked_splits = [split if split in {"train", "test"} else "train" for split in marked_splits]

    aliases: dict[str, int] = {}
    normalized = []
    for graph, graph_id, split in zip(ordered, assigned_ids, marked_splits):
        source_member = graph["source_member"]
        aliases[str(graph_id)] = graph_id
        aliases[Path(source_member).stem] = graph_id
        aliases[Path(source_member).name] = graph_id
        aliases[source_member.replace("\\", "/")] = graph_id
        normalized.append(
            {
                "id": graph_id,
                "split": split,
                "source_member": source_member,
                "member": f"{dataset_id}/{split}/{graph_id}.gexf",
                "content": graph["content"],
            }
        )
    return normalized, aliases


def _member_split(member_name: str) -> str | None:
    parts = {part.lower() for part in str(member_name).replace("\\", "/").split("/")}
    if "train" in parts:
        return "train"
    if "test" in parts:
        return "test"
    return None


def _read_upload_file(upload: Any | None) -> tuple[bytes, str] | None:
    if upload is None or not getattr(upload, "filename", ""):
        return None
    payload = upload.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise DatasetUploadError("Ground-truth file exceeds the 512 MB limit.")
    return payload, Path(upload.filename).name


def _parse_uploaded_ground_truth(
    payload: bytes,
    filename: str,
    aliases: dict[str, int],
) -> dict[tuple[int, int], float]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        rows = _ground_truth_csv_rows(payload)
    elif suffix == ".json":
        rows = _ground_truth_json_rows(payload)
    else:
        raise DatasetUploadError("GED ground truth must be CSV or JSON.")

    values: dict[tuple[int, int], float] = {}
    for left_raw, right_raw, distance_raw in rows:
        left = _resolve_graph_alias(left_raw, aliases)
        right = _resolve_graph_alias(right_raw, aliases)
        try:
            distance = float(distance_raw)
        except (TypeError, ValueError) as exc:
            raise DatasetUploadError(f"Invalid GED value: {distance_raw}") from exc
        if not math.isfinite(distance) or distance < 0:
            raise DatasetUploadError(f"GED values must be finite and non-negative: {distance_raw}")
        values[(left, right)] = distance
    if not values:
        raise DatasetUploadError("GED file contains no usable graph pairs.")
    return values


def _ground_truth_csv_rows(payload: bytes) -> list[tuple[Any, Any, Any]]:
    text = payload.decode("utf-8-sig", errors="strict")
    rows = [row for row in csv.reader(io.StringIO(text)) if row and any(cell.strip() for cell in row)]
    if not rows:
        return []
    header = [cell.strip().lower() for cell in rows[0]]
    aliases = {
        "left": {"left", "graph_a", "graph1", "left_id"},
        "right": {"right", "graph_b", "graph2", "right_id"},
        "ged": {"ged", "distance", "edit_distance"},
    }
    indices = {}
    for key, names in aliases.items():
        indices[key] = next((index for index, value in enumerate(header) if value in names), None)
    if all(index is not None for index in indices.values()):
        data_rows = rows[1:]
        return [
            (row[indices["left"]], row[indices["right"]], row[indices["ged"]])
            for row in data_rows
            if len(row) > max(indices.values())
        ]
    return [(row[0], row[1], row[2]) for row in rows if len(row) >= 3]


def _ground_truth_json_rows(payload: bytes) -> list[tuple[Any, Any, Any]]:
    raw = json.loads(payload.decode("utf-8-sig", errors="strict"))
    if isinstance(raw, dict):
        rows = []
        for key, value in raw.items():
            parts = str(key).split(",", maxsplit=1)
            if len(parts) != 2:
                raise DatasetUploadError("JSON GED object keys must use 'left,right'.")
            rows.append((parts[0], parts[1], value))
        return rows
    if isinstance(raw, list):
        rows = []
        for item in raw:
            if isinstance(item, dict):
                rows.append(
                    (
                        item.get("left", item.get("graph_a")),
                        item.get("right", item.get("graph_b")),
                        item.get("ged", item.get("distance")),
                    )
                )
            elif isinstance(item, list) and len(item) >= 3:
                rows.append((item[0], item[1], item[2]))
            else:
                raise DatasetUploadError("JSON GED rows must be objects or [left, right, ged] arrays.")
        return rows
    raise DatasetUploadError("JSON GED data must be an object or array.")


def _resolve_graph_alias(value: Any, aliases: dict[str, int]) -> int:
    text = str(value).strip().replace("\\", "/")
    candidates = (text, Path(text).name, Path(text).stem)
    for candidate in candidates:
        if candidate in aliases:
            return aliases[candidate]
    raise DatasetUploadError(f"GED references a graph not present in the archive: {value}")


def _dataset_by_id(dataset_id: str) -> dict[str, Any]:
    for dataset in _all_datasets():
        if dataset["id"] == dataset_id:
            return dataset
    raise ValueError(f"Unknown original dataset: {dataset_id}")


@lru_cache(maxsize=8)
def _cached_ground_truth(path: str) -> dict[tuple[int, int], float]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {source.name}")
    if source.suffix.lower() == ".json":
        raw_json = json.loads(source.read_text())
        values = {}
        for key, distance in raw_json.items():
            left, right = str(key).split(",", maxsplit=1)
            values[(int(left), int(right))] = float(distance)
        return _canonicalize_symmetric_distances(values)
    with source.open("rb") as handle:
        raw = pickle.load(handle)
    values = {
        (int(left), int(right)): float(distance)
        for (left, right), distance in raw.items()
    }
    return _canonicalize_symmetric_distances(values)


def _canonicalize_symmetric_distances(
    values: dict[tuple[int, int], float],
) -> dict[tuple[int, int], float]:
    """Collapse directional GED records to one symmetric unordered-pair target.

    Approximate GED solvers return upper bounds. When both directions are
    present, their minimum is the tighter valid upper bound. Exact symmetric
    maps are unchanged by the same rule.
    """
    canonical: dict[tuple[int, int], float] = {}
    for (left, right), raw_distance in values.items():
        distance = float(raw_distance)
        if not math.isfinite(distance) or distance < 0:
            continue
        key = (left, right) if left <= right else (right, left)
        previous = canonical.get(key)
        if previous is None or distance < previous:
            canonical[key] = distance
    symmetric: dict[tuple[int, int], float] = {}
    for (left, right), distance in canonical.items():
        symmetric[(left, right)] = distance
        if left != right:
            symmetric[(right, left)] = distance
    return symmetric


def _archive_graph_members(archive_path: Path, archive_format: str) -> list[str]:
    if not archive_path.exists():
        return []
    if archive_format == "zip":
        with zipfile.ZipFile(archive_path) as archive:
            names = [name for name in archive.namelist() if name.endswith(".gexf")]
    elif archive_format == "tar":
        with tarfile.open(archive_path) as archive:
            names = [member.name for member in archive.getmembers() if member.isfile() and member.name.endswith(".gexf")]
    else:
        names = []
    return sorted(names, key=_graph_member_sort_key)


def _archive_split_counts(archive_path: Path, archive_format: str) -> dict[str, int]:
    counts = {"train": 0, "test": 0}
    for member in _archive_graph_members(archive_path, archive_format):
        if "/train/" in f"/{member}":
            counts["train"] += 1
        elif "/test/" in f"/{member}":
            counts["test"] += 1
    return counts


def _read_archive_text(archive_path: Path, archive_format: str, member_name: str) -> str:
    if archive_format == "zip":
        with zipfile.ZipFile(archive_path) as archive:
            return archive.read(member_name).decode("utf-8", errors="replace")
    if archive_format == "tar":
        with tarfile.open(archive_path) as archive:
            member = archive.getmember(member_name)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise FileNotFoundError(f"Could not read {member_name}")
            return extracted.read().decode("utf-8", errors="replace")
    raise ValueError(f"Unsupported archive format: {archive_format}")


def _read_archive_texts(archive_path: Path, archive_format: str, member_names: list[str]) -> dict[str, str]:
    if archive_format == "zip":
        with zipfile.ZipFile(archive_path) as archive:
            return {
                member_name: archive.read(member_name).decode("utf-8", errors="replace")
                for member_name in member_names
            }
    if archive_format == "tar":
        with tarfile.open(archive_path) as archive:
            texts = {}
            for member_name in member_names:
                member = archive.getmember(member_name)
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise FileNotFoundError(f"Could not read {member_name}")
                texts[member_name] = extracted.read().decode("utf-8", errors="replace")
            return texts
    raise ValueError(f"Unsupported archive format: {archive_format}")


def _parse_gexf(xml_text: str) -> dict[str, Any]:
    root = ET.fromstring(xml_text)
    node_elements = root.findall(".//{*}node")
    edge_elements = root.findall(".//{*}edge")
    nodes = [node.attrib.get("id", str(index)) for index, node in enumerate(node_elements)]
    labels = {}
    for index, node in enumerate(node_elements):
        node_id = node.attrib.get("id", str(index))
        labels[node_id] = _node_label(node)

    edges = []
    for edge in edge_elements:
        source = edge.attrib.get("source")
        target = edge.attrib.get("target")
        if source is not None and target is not None:
            edges.append([source, target])

    return {"nodes": nodes, "edges": edges, "labels": labels}


def _node_label(node: ET.Element) -> str:
    for attvalue in node.findall(".//{*}attvalue"):
        value = attvalue.attrib.get("value")
        if value not in (None, ""):
            return str(value)
    return str(node.attrib.get("label") or node.attrib.get("id") or "0")


def _choose_member(members: list[str], preferred_index: int = 0) -> str | None:
    if not members:
        return None
    index = min(preferred_index, len(members) - 1)
    return members[index]


def _graph_member_sort_key(name: str) -> tuple[str, int, str]:
    stem = Path(name).stem
    number = int(stem) if stem.isdigit() else 10**9
    split_rank = 0 if "/train/" in f"/{name}" else 1
    return (name.split("/")[0], split_rank, number, name)
