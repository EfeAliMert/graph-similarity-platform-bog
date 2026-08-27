from __future__ import annotations

import argparse
import json
import math
import pickle
import zipfile
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import torch
from torch_geometric.datasets import GEDDataset, TUDataset


ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_ROOT = ROOT / "Models&Datasets" / "downloaded_real"
APP_DATASET_DIR = ROOT / "Models&Datasets" / "drive-download-20260630T100606Z-3-001"

GED_DATASETS: list[str] = []
TU_DATASETS = ["MUTAG", "PROTEINS", "ENZYMES"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download real graph datasets and export app-readable GEXF archives.")
    parser.add_argument("--ged", nargs="*", default=GED_DATASETS, help="PyG GEDDataset names to download/export.")
    parser.add_argument("--tu", nargs="*", default=TU_DATASETS, help="PyG TUDataset names to download/export.")
    parser.add_argument("--force", action="store_true", help="Re-export archives even if output files exist.")
    args = parser.parse_args()

    APP_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"ged": [], "tu": []}
    for name in args.ged:
        try:
            manifest["ged"].append(export_ged_dataset(name, force=args.force))
        except Exception as exc:
            manifest["ged"].append({"name": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    for name in args.tu:
        try:
            manifest["tu"].append(export_tu_dataset(name, force=args.force))
        except Exception as exc:
            manifest["tu"].append({"name": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})

    manifest_path = DOWNLOAD_ROOT / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def export_ged_dataset(name: str, force: bool = False) -> dict:
    dataset_id = name.lower()
    archive_path = APP_DATASET_DIR / f"{name}.zip"
    ged_path = APP_DATASET_DIR / f"{dataset_id}_ged_astar_gidpair_dist_map.pickle"
    if archive_path.exists() and ged_path.exists() and not force:
        return {"name": name, "archive": str(archive_path.relative_to(ROOT)), "ged": str(ged_path.relative_to(ROOT)), "status": "exists"}

    root = DOWNLOAD_ROOT / "pyg_ged"
    train = GEDDataset(root=str(root), name=name, train=True)
    test = GEDDataset(root=str(root), name=name, train=False)
    graph_records = [("train", data) for data in train] + [("test", data) for data in test]

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for split, data in graph_records:
            graph_id = int(data.i.item()) if hasattr(data, "i") else len(archive.namelist())
            member = f"{name}/{split}/{graph_id}.gexf"
            archive.writestr(member, data_to_gexf(data, graph_id=graph_id))

    graph_ids = [int(data.i.item()) for _, data in graph_records if hasattr(data, "i")]
    distances = {}
    for left in graph_ids:
        for right in graph_ids:
            value = _ged_value(train.ged, left, right)
            if value is not None:
                distances[(left, right)] = value
    with ged_path.open("wb") as handle:
        pickle.dump(distances, handle)

    return {
        "name": name,
        "archive": str(archive_path.relative_to(ROOT)),
        "ged": str(ged_path.relative_to(ROOT)),
        "graphs": len(graph_records),
        "ged_pairs": len(distances),
        "status": "downloaded",
    }


def export_tu_dataset(
    name: str,
    force: bool = False,
    app_dataset_dir: Path = APP_DATASET_DIR,
) -> dict:
    dataset_id = _tu_dataset_id(name)
    app_dataset_dir.mkdir(parents=True, exist_ok=True)
    archive_path = app_dataset_dir / f"{name}.zip"
    labels_path = app_dataset_dir / f"{dataset_id}_graph_labels.json"
    if archive_path.exists() and labels_path.exists() and not force:
        return {"name": name, "archive": str(archive_path.relative_to(ROOT)), "labels": str(labels_path.relative_to(ROOT)), "status": "exists"}

    dataset = TUDataset(root=str(DOWNLOAD_ROOT / "tu"), name=name)
    split_at = max(1, int(round(len(dataset) * 0.8)))
    labels = {}
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, data in enumerate(dataset):
            split = "train" if index < split_at else "test"
            member = f"{name}/{split}/{index}.gexf"
            graph_label = int(data.y.view(-1)[0].item()) if getattr(data, "y", None) is not None else None
            labels[member] = graph_label
            archive.writestr(member, data_to_gexf(data, graph_id=index, graph_label=graph_label))
    labels_path.write_text(json.dumps(labels, indent=2))
    return {
        "name": name,
        "archive": str(archive_path.relative_to(ROOT)),
        "labels": str(labels_path.relative_to(ROOT)),
        "graphs": len(dataset),
        "train_graphs": split_at,
        "test_graphs": len(dataset) - split_at,
        "status": "downloaded",
    }


def data_to_gexf(data, graph_id: int, graph_label: int | None = None) -> str:
    gexf = ET.Element("gexf", {"xmlns": "http://www.gexf.net/1.2draft", "version": "1.2"})
    graph_attrs = {"mode": "static", "defaultedgetype": "undirected", "id": str(graph_id)}
    if graph_label is not None:
        graph_attrs["label"] = str(graph_label)
    graph = ET.SubElement(gexf, "graph", graph_attrs)
    nodes_el = ET.SubElement(graph, "nodes")
    edges_el = ET.SubElement(graph, "edges")

    node_count = int(getattr(data, "num_nodes", 0) or 0)
    labels = node_labels(data, node_count)
    for node_id in range(node_count):
        node_el = ET.SubElement(nodes_el, "node", {"id": str(node_id), "label": str(labels[node_id])})
        attvalues = ET.SubElement(node_el, "attvalues")
        ET.SubElement(attvalues, "attvalue", {"for": "label", "value": str(labels[node_id])})

    for edge_index, (source, target) in enumerate(unique_edges(data.edge_index if hasattr(data, "edge_index") else None)):
        ET.SubElement(edges_el, "edge", {"id": str(edge_index), "source": str(source), "target": str(target)})
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(gexf, encoding="unicode")


def node_labels(data, node_count: int) -> list[str]:
    x = getattr(data, "x", None)
    if x is not None and x.numel() > 0:
        if x.dim() == 1:
            return [str(int(value.item()) if float(value.item()).is_integer() else round(float(value.item()), 4)) for value in x[:node_count]]
        if x.size(-1) > 1:
            return [f"x{int(row.argmax().item())}" for row in x[:node_count]]
        return [str(int(row.view(-1)[0].item()) if float(row.view(-1)[0].item()).is_integer() else round(float(row.view(-1)[0].item()), 4)) for row in x[:node_count]]
    return degree_labels(getattr(data, "edge_index", None), node_count)


def degree_labels(edge_index, node_count: int) -> list[str]:
    degrees = [0] * node_count
    for source, target in unique_edges(edge_index):
        if 0 <= source < node_count:
            degrees[source] += 1
        if 0 <= target < node_count:
            degrees[target] += 1
    return [f"d{degree}" for degree in degrees]


def unique_edges(edge_index) -> list[tuple[int, int]]:
    if edge_index is None or edge_index.numel() == 0:
        return []
    edges = set()
    for source, target in edge_index.t().tolist():
        source_i = int(source)
        target_i = int(target)
        if source_i == target_i:
            continue
        edges.add(tuple(sorted((source_i, target_i))))
    return sorted(edges)


def _ged_value(matrix: torch.Tensor, left: int, right: int) -> float | None:
    try:
        value = float(matrix[left, right].item())
    except (IndexError, RuntimeError):
        return None
    if math.isnan(value) or math.isinf(value) or value < 0:
        return None
    return value


def _tu_dataset_id(name: str) -> str:
    return name.lower().replace("-", "_")


if __name__ == "__main__":
    main()
