from __future__ import annotations

import argparse
import pickle
import shutil
import tarfile
import zipfile
from pathlib import Path

import networkx as nx
import torch
from torch_geometric.datasets import GEDDataset


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = ROOT / "Models&Datasets" / "drive-download-20260630T100606Z-3-001"
DATASETS = {
    "aids700nef": {
        "name": "AIDS700nef",
        "archive": "AIDS700nef.zip",
        "format": "zip",
        "ged": "aids700nef_ged_astar_gidpair_dist_map.pickle",
        "mcs": "aids700nef_mcs_mccreesh2017_gidpair_dist_map.pickle",
    },
    "linux": {
        "name": "LINUX",
        "archive": "LINUX.tar.gz",
        "format": "tar",
        "ged": "linux_ged_astar_gidpair_dist_map.pickle",
        "mcs": "linux_mcs_mccreesh2017_gidpair_dist_map.pickle",
    },
    "imdbmulti": {
        "name": "IMDBMulti",
        "archive": "IMDBMulti.zip",
        "format": "zip",
        "ged": "imdbmulti_ged_astar_gidpair_dist_map.pickle",
        "mcs": "imdbmulti_mcs_mccreesh2017_gidpair_dist_map.pickle",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--root", required=True, help="Target PyG dataset root, e.g. Models&Datasets/Graph2Region-main/GED/LINUX")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--raw-only", action="store_true", help="Only extract raw graphs and GED pickle; do not run PyG processing.")
    parser.add_argument("--extra-mats", action="store_true", help="Write Graph2Region MCS/Bunke matrix files.")
    args = parser.parse_args()

    config = DATASETS[args.dataset]
    target_root = (ROOT / args.root).resolve()
    raw_dataset_dir = target_root / "raw" / config["name"]

    if args.clean and target_root.exists():
        shutil.rmtree(target_root)
    (raw_dataset_dir / "train").mkdir(parents=True, exist_ok=True)
    (raw_dataset_dir / "test").mkdir(parents=True, exist_ok=True)

    archive_path = ARCHIVE_DIR / config["archive"]
    extract_gexf_files(archive_path, config["format"], raw_dataset_dir)
    shutil.copy2(ARCHIVE_DIR / config["ged"], raw_dataset_dir / "ged.pickle")

    if not args.raw_only:
        GEDDataset(root=str(target_root), name=config["name"], train=True)
        GEDDataset(root=str(target_root), name=config["name"], train=False)

    if args.extra_mats:
        if args.raw_only:
            raise ValueError("--extra-mats requires PyG processing; remove --raw-only.")
        write_extra_mats(target_root, config)

    print(
        {
            "dataset": args.dataset,
            "root": str(target_root.relative_to(ROOT)),
            "name": config["name"],
            "train_graphs": len(list((raw_dataset_dir / "train").glob("*.gexf"))),
            "test_graphs": len(list((raw_dataset_dir / "test").glob("*.gexf"))),
            "raw_only": args.raw_only,
            "extra_mats": args.extra_mats,
        }
    )


def extract_gexf_files(archive_path: Path, archive_format: str, raw_dataset_dir: Path) -> None:
    if archive_format == "zip":
        with zipfile.ZipFile(archive_path) as archive:
            for name in archive.namelist():
                if name.endswith(".gexf"):
                    write_graph_member(raw_dataset_dir, name, archive.read(name))
        return

    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith(".gexf"):
                continue
            extracted = archive.extractfile(member)
            if extracted is not None:
                write_graph_member(raw_dataset_dir, member.name, extracted.read())


def write_graph_member(raw_dataset_dir: Path, member_name: str, content: bytes) -> None:
    split = "train" if "/train/" in f"/{member_name}" else "test"
    graph_name = Path(member_name).name
    (raw_dataset_dir / split / graph_name).write_bytes(content)


def write_extra_mats(target_root: Path, config: dict[str, str]) -> None:
    name = config["name"]
    raw_dataset_dir = target_root / "raw" / name
    processed_dir = target_root / "processed"
    train_ids = graph_ids(raw_dataset_dir / "train")
    test_ids = graph_ids(raw_dataset_dir / "test")
    all_ids = train_ids + test_ids
    assoc = {graph_id: index for index, graph_id in enumerate(all_ids)}
    node_counts = torch.tensor([num_nodes(raw_dataset_dir, graph_id) for graph_id in all_ids], dtype=torch.float)

    ged = pickle.load(open(ARCHIVE_DIR / config["ged"], "rb"))
    ged_mat = distance_matrix(ged, assoc)
    avg_nodes = 0.5 * (node_counts.view(-1, 1) + node_counts.view(1, -1))
    bunke_ged = torch.exp(-torch.nan_to_num(ged_mat / avg_nodes, posinf=1e9))
    torch.save(bunke_ged, processed_dir / f"{name}_bunke_ged_with_edges_similarity_mat.pt")

    mcs = pickle.load(open(ARCHIVE_DIR / config["mcs"], "rb"))
    mcs_mat = distance_matrix(mcs, assoc, diagonal=node_counts)
    norm_mcs = torch.nan_to_num(mcs_mat / avg_nodes)
    bunke_mcs = torch.nan_to_num(mcs_mat / torch.maximum(node_counts.view(-1, 1), node_counts.view(1, -1)))
    graph_union_mcs = torch.nan_to_num(mcs_mat / (node_counts.view(-1, 1) + node_counts.view(1, -1) - mcs_mat))

    torch.save(norm_mcs, processed_dir / f"{name}_norm_mcs_similarity_mat.pt")
    torch.save(bunke_mcs, processed_dir / f"{name}_bunke_mcs_similarity_mat.pt")
    torch.save(graph_union_mcs, processed_dir / f"{name}_graph_union_mcs_similarity_mat.pt")


def graph_ids(path: Path) -> list[int]:
    return sorted(int(item.stem) for item in path.glob("*.gexf"))


def num_nodes(raw_dataset_dir: Path, graph_id: int) -> int:
    candidates = list(raw_dataset_dir.glob(f"*/{graph_id}.gexf"))
    if not candidates:
        raise FileNotFoundError(f"Graph {graph_id}.gexf not found under {raw_dataset_dir}")
    return nx.read_gexf(candidates[0]).number_of_nodes()


def distance_matrix(values: dict, assoc: dict[int, int], diagonal: torch.Tensor | None = None) -> torch.Tensor:
    size = len(assoc)
    mat = torch.zeros((size, size), dtype=torch.float)
    if diagonal is not None:
        mat.fill_(0)
        mat[torch.arange(size), torch.arange(size)] = diagonal
    else:
        mat.fill_(float("inf"))
        mat[torch.arange(size), torch.arange(size)] = 0
    for (left, right), value in values.items():
        if left in assoc and right in assoc:
            i = assoc[left]
            j = assoc[right]
            mat[i, j] = float(value)
            mat[j, i] = float(value)
    return mat


if __name__ == "__main__":
    main()
