from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from universal_dataset import load_graph_records, serialize_basic_gexf  # noqa: E402


DATASET_IDS = (
    "aids700nef",
    "linux",
    "imdbmulti",
    "ptc",
    "mutag",
    "proteins",
    "enzymes",
)
GRAPHSIM_ROOT = ROOT / "Models&Datasets" / "GraphSim-master"
BUILTIN_CACHE_NAMES = {
    "aids700nef": "AIDS700nefData",
    "linux": "LINUXData",
    "imdbmulti": "IMDBMultiData",
    "ptc": "PTCData",
}


class ModelDatasetPreparationError(RuntimeError):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare derived dataset files required by model adapters."
    )
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    selected = tuple(args.dataset) if args.dataset else DATASET_IDS
    unknown = sorted(set(selected) - set(DATASET_IDS))
    if unknown:
        raise ModelDatasetPreparationError(
            "Unknown dataset id(s): " + ", ".join(unknown)
        )
    if not (GRAPHSIM_ROOT / "src" / "data.py").is_file():
        raise ModelDatasetPreparationError(
            "GraphSim source is missing. Run `make models` before `make datasets`."
        )

    for dataset_id in selected:
        records = load_graph_records(dataset_id)
        dataset_root = GRAPHSIM_ROOT / "data" / dataset_id
        issues = materialized_dataset_issues(records, dataset_root)
        if args.verify_only:
            if issues:
                raise ModelDatasetPreparationError(
                    f"GraphSim dataset {dataset_id} is incomplete: {issues[0]}"
                )
            print(f"[verified] GraphSim {dataset_id} adapter data")
            continue
        if issues:
            materialize_graphsim_dataset(records, dataset_root)
            print(f"[prepared] GraphSim {dataset_id}: {len(records)} graphs")
        else:
            print(f"[verified] GraphSim {dataset_id} adapter data")

    if not args.verify_only:
        clear_graphsim_caches(set(selected))


def materialized_dataset_issues(records: list[dict], dataset_root: Path) -> list[str]:
    expected = {
        Path(record["split"]) / f'{int(record["id"])}.gexf'
        for record in records
    }
    if not dataset_root.is_dir():
        return ["dataset directory is missing"]
    actual = {
        path.relative_to(dataset_root)
        for path in dataset_root.glob("*/*.gexf")
        if path.is_file()
    }
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        return [f"missing {missing[0]} ({len(missing)} missing file(s))"]
    if unexpected:
        return [f"unexpected {unexpected[0]} ({len(unexpected)} extra file(s))"]
    empty = sorted(
        relative for relative in expected if (dataset_root / relative).stat().st_size == 0
    )
    if empty:
        return [f"empty graph file: {empty[0]}"]
    return []


def materialize_graphsim_dataset(records: list[dict], dataset_root: Path) -> None:
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    for split in ("train", "test"):
        (dataset_root / split).mkdir(parents=True, exist_ok=True)
    for record in records:
        target = dataset_root / record["split"] / f'{int(record["id"])}.gexf'
        target.write_text(serialize_basic_gexf(record))


def clear_graphsim_caches(dataset_ids: set[str]) -> None:
    save_root = GRAPHSIM_ROOT / "save"
    for generic_cache in ("GenericGEXFData", "SiameseModelData"):
        shutil.rmtree(save_root / generic_cache, ignore_errors=True)
    for dataset_id in dataset_ids:
        cache_name = BUILTIN_CACHE_NAMES.get(dataset_id)
        if cache_name:
            shutil.rmtree(save_root / cache_name, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except ModelDatasetPreparationError as exc:
        print(f"Model dataset preparation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
