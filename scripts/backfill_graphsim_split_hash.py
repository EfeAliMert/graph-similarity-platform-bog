from __future__ import annotations

import hashlib
import json
from pathlib import Path

from universal_dataset import graph_disjoint_split_metadata, load_graph_records


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = ROOT / "Models&Datasets" / "GraphSim-master" / "checkpoints"
DATASETS = (
    "aids700nef",
    "linux",
    "imdbmulti",
    "ptc",
    "mutag",
    "proteins",
    "enzymes",
)


def main() -> int:
    updated = 0
    metadata_by_dataset = {}
    for dataset_id in DATASETS:
        path = CHECKPOINT_ROOT / dataset_id / "graphsim.ckpt.meta.json"
        payload = json.loads(path.read_text())
        records = load_graph_records(dataset_id)
        metadata = graph_disjoint_split_metadata(records, 0.25)
        metadata_by_dataset[dataset_id] = metadata
        payload["pair_split"] = metadata
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        updated += 1
        print(f"{dataset_id}: {payload['pair_split']['split_sha256']}")

    snapshot_root = ROOT / "training_logs" / "research_matrix"
    snapshot_updated = 0
    for path in snapshot_root.glob(
        "*/checkpoints/multiscale-set/*/seed-*/graphsim.ckpt.meta.json"
    ):
        dataset_id = path.parent.parent.name
        metadata = metadata_by_dataset.get(dataset_id)
        if metadata is None:
            continue
        payload = json.loads(path.read_text())
        payload["pair_split"] = metadata
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        run_metadata_path = path.parent / "run_metadata.json"
        if run_metadata_path.exists():
            run_metadata = json.loads(run_metadata_path.read_text())
            run_metadata["pair_split"] = metadata
            run_metadata_path.write_text(
                json.dumps(run_metadata, indent=2, sort_keys=True)
            )
        snapshot_updated += 1

    manifests_updated = 0
    for manifest_path in snapshot_root.glob("*/manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        changed = False
        for job in manifest.get("jobs", []):
            if job.get("model_id") != "multiscale-set":
                continue
            for snapshot in job.get("snapshots", []):
                path = ROOT / snapshot["path"]
                if path.name != "graphsim.ckpt.meta.json" or not path.exists():
                    continue
                snapshot["bytes"] = path.stat().st_size
                snapshot["sha256"] = _sha256(path)
                changed = True
        if changed:
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
            manifests_updated += 1

    print(
        f"active_updated={updated} snapshot_updated={snapshot_updated} "
        f"manifests_updated={manifests_updated}"
    )
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
