from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import uuid

try:
    from scripts.checkpoint_provenance import (
        checkpoint_fingerprint,
        load_verified_hpo,
    )
except ModuleNotFoundError:  # Direct execution adds scripts/, not the project root.
    from checkpoint_provenance import checkpoint_fingerprint, load_verified_hpo


ROOT = Path(__file__).resolve().parents[1]
PYG_PYTHON = ROOT / ".venvs" / "gnn-pyg" / "bin" / "python"
GRAPHSIM_PYTHON = ROOT / ".venvs" / "graphsim" / "bin" / "python"
SIMGNN_ROOT = ROOT / "Models&Datasets" / "SimGNN-v_00001"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run validation-only hyperparameter optimization for a local GNN."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=379)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--no-promote", action="store_true")
    args = parser.parse_args()

    args.trials = max(2, min(args.trials, 20))
    args.budget = max(1, min(args.budget, 200))
    args.batch_size = max(1, min(args.batch_size, 256))
    if args.model not in search_spaces(args.batch_size):
        raise ValueError(f"Hyperparameter search is not registered for {args.model}.")

    study_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
    study_dir = ROOT / "training_logs" / "hpo" / args.model / args.dataset / study_id
    study_dir.mkdir(parents=True, exist_ok=True)
    active_target = active_checkpoint(args.model, args.dataset)
    incumbent_validation_mse = active_validation_metric(args.model, active_target)
    incumbent_protocol = checkpoint_protocol(args.model, active_target)
    if args.split_seed is None:
        args.split_seed = protocol_split_seed(
            args.model,
            incumbent_protocol,
            args.seed,
        )
    configs = trial_configs(args.model, args.batch_size, args.trials, args.seed)

    dataset_manifest = {}
    if args.model == "simgnn":
        prepare_simgnn_dataset(args.dataset, args.split_seed)
        manifest_path = (
            SIMGNN_ROOT / "original_datasets" / args.dataset / "manifest.json"
        )
        if manifest_path.is_file():
            dataset_manifest = json.loads(manifest_path.read_text())

    trials = []
    for index, config in enumerate(configs):
        trial_dir = study_dir / f"trial_{index:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        candidate = trial_checkpoint(args.model, trial_dir)
        command, cwd = trial_command(args, config, candidate)
        print(f"HPO trial {index + 1}/{len(configs)} config={json.dumps(config, sort_keys=True)}", flush=True)
        return_code, output = run_command(command, cwd)
        metric = read_validation_metric(args.model, candidate, output) if return_code == 0 else None
        trial = {
            "index": index,
            "status": "completed" if metric is not None else "failed",
            "config": config,
            "validation_mse": metric,
            "checkpoint": display_path(candidate),
            "return_code": return_code,
        }
        trials.append(trial)
        (trial_dir / "trial.json").write_text(json.dumps(trial, indent=2, sort_keys=True))
        metric_text = f"{metric:.8f}" if metric is not None else "unavailable"
        print(f"HPO trial {index + 1} validation_mse={metric_text}", flush=True)

    successful = [trial for trial in trials if finite(trial["validation_mse"])]
    if not successful:
        raise RuntimeError("Every hyperparameter trial failed; no checkpoint was promoted.")
    best = min(successful, key=lambda trial: (trial["validation_mse"], trial["index"]))
    candidate_protocol = (
        dataset_manifest
        if args.model == "simgnn"
        else checkpoint_protocol(args.model, ROOT / best["checkpoint"])
    )
    protocol_comparable = protocols_are_comparable(
        args.model,
        incumbent_protocol,
        candidate_protocol,
        active_target.is_file() or args.model == "multiscale-set",
    )
    promoted = []
    improved = protocol_comparable and is_improvement(
        best["validation_mse"], incumbent_validation_mse
    )
    if not args.no_promote and improved:
        promoted = promote_checkpoint(
            args.model,
            ROOT / best["checkpoint"],
            active_target,
            study_dir,
        )
    elif not args.no_promote:
        if not protocol_comparable:
            print(
                "HPO promotion skipped: candidate and incumbent validation "
                "splits are not comparable.",
                flush=True,
            )
        else:
            incumbent_text = (
                f"{incumbent_validation_mse:.8f}"
                if incumbent_validation_mse is not None
                else "unavailable"
            )
            print(
                "HPO promotion skipped: best trial did not improve the active "
                f"validation MSE ({best['validation_mse']:.8f} vs "
                f"{incumbent_text}).",
                flush=True,
            )

    report = {
        "study_id": study_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model,
        "dataset_id": args.dataset,
        "strategy": "deterministic shuffled grid search",
        "objective": "minimum validation similarity MSE",
        "test_set_used_for_selection": False,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "target": candidate_protocol.get("target"),
        "pair_split": candidate_protocol.get("pair_split"),
        "budget": args.budget,
        "requested_trials": args.trials,
        "completed_trials": len(successful),
        "best_trial": best,
        "incumbent_validation_mse": incumbent_validation_mse,
        "validation_protocol_comparable": protocol_comparable,
        "incumbent_pair_split": incumbent_protocol.get("pair_split"),
        "improved_over_incumbent": improved,
        "active_checkpoint": display_path(active_target),
        "promoted": bool(promoted),
        "promoted_files": promoted,
        "active_checkpoint_fingerprint": (
            checkpoint_fingerprint(active_target) if promoted else None
        ),
        "best_trial_fingerprint": checkpoint_fingerprint(ROOT / best["checkpoint"]),
        "trials": trials,
    }
    report_dir = ROOT / "reports" / "hpo"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{args.model}_{args.dataset}_{study_id}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    sidecar = Path(str(active_target) + ".hpo.json")
    if promoted:
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(
        "HPO_RESULT="
        + json.dumps(
            {
                "best_validation_mse": best["validation_mse"],
                "best_config": best["config"],
                "report": display_path(report_path),
                "incumbent_validation_mse": incumbent_validation_mse,
                "improved_over_incumbent": improved,
                "validation_protocol_comparable": protocol_comparable,
                "promoted": bool(promoted),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def search_spaces(batch_size: int) -> dict[str, dict[str, list]]:
    batch_choices = sorted({max(1, batch_size // 2), batch_size, min(256, batch_size * 2)})
    return {
        "simgnn": {
            "learning_rate": [0.0003, 0.0007, 0.001, 0.0015],
            "dropout": [0.25, 0.4, 0.5],
            "weight_decay": [0.00001, 0.0001, 0.0005],
            "batch_size": batch_choices,
        },
        "multiscale-set": {
            "learning_rate": [0.0003, 0.001, 0.003],
            "batch_size": [4, 8, 16],
        },
        "segmn": {
            "learning_rate": [0.0001, 0.0003, 0.0005, 0.001],
            "node_cap": [12, 16, 24],
        },
        "graph-fusion": {
            "learning_rate": [0.0003, 0.0007, 0.001, 0.002],
            "batch_size": batch_choices,
        },
        "graph2region": {
            "learning_rate": [0.0003, 0.0007, 0.001, 0.002],
            "batch_size": [4, 8, 16],
        },
    }


def trial_configs(model_id: str, batch_size: int, count: int, seed: int) -> list[dict]:
    space = search_spaces(batch_size)[model_id]
    keys = list(space)
    configs = [dict(zip(keys, values)) for values in itertools.product(*(space[key] for key in keys))]
    default = default_config(model_id, batch_size)
    configs = [config for config in configs if config != default]
    random.Random(seed).shuffle(configs)
    return ([default] + configs)[:count]


def default_config(model_id: str, batch_size: int) -> dict:
    return {
        "simgnn": {
            "learning_rate": 0.001,
            "dropout": 0.5,
            "weight_decay": 0.0005,
            "batch_size": batch_size,
        },
        "multiscale-set": {"learning_rate": 0.001, "batch_size": min(batch_size, 16)},
        "segmn": {"learning_rate": 0.0005, "node_cap": 16},
        "graph-fusion": {"learning_rate": 0.001, "batch_size": batch_size},
        "graph2region": {"learning_rate": 0.001, "batch_size": min(batch_size, 16)},
    }[model_id]


def active_checkpoint(model_id: str, dataset_id: str) -> Path:
    targets = {
        "simgnn": ROOT / "Models&Datasets" / "SimGNN-v_00001" / "checkpoints" / f"simgnn_{dataset_id}.pt",
        "multiscale-set": ROOT / "Models&Datasets" / "GraphSim-master" / "checkpoints" / dataset_id / "graphsim.ckpt",
        "segmn": ROOT / "Models&Datasets" / "SEGMN-main" / "checkpoints" / dataset_id / f"segmn_{dataset_id}_best.pt",
        "graph-fusion": ROOT / "Models&Datasets" / "GFM-code" / "checkpoints" / f"gfm_{dataset_id}.pt",
        "graph2region": ROOT / "Models&Datasets" / "Graph2Region-main" / "checkpoints" / dataset_id / f"g2r_{dataset_id}_best.pt",
    }
    return targets[model_id]


def trial_checkpoint(model_id: str, trial_dir: Path) -> Path:
    return trial_dir / ("graphsim.ckpt" if model_id == "multiscale-set" else "model.pt")


def trial_command(args, config: dict, checkpoint: Path) -> tuple[list[str], Path]:
    common = ["--dataset", args.dataset, "--checkpoint", str(checkpoint), "--seed", str(args.seed)]
    if args.model == "simgnn":
        command = [
            str(PYG_PYTHON), "src/main.py",
            "--training-graphs", f"original_datasets/{args.dataset}/train/",
            "--validation-graphs", f"original_datasets/{args.dataset}/validation/",
            "--testing-graphs", f"original_datasets/{args.dataset}/test/",
            "--epochs", str(args.budget),
            "--batch-size", str(config["batch_size"]),
            "--learning-rate", str(config["learning_rate"]),
            "--dropout", str(config["dropout"]),
            "--weight-decay", str(config["weight_decay"]),
            "--seed", str(args.seed),
            "--save-path", str(checkpoint),
        ]
        return command, SIMGNN_ROOT
    if args.model == "multiscale-set":
        return [
            str(GRAPHSIM_PYTHON), "scripts/train_graphsim_compat.py", *common,
            "--steps", str(args.budget * 20),
            "--batch-size", str(config["batch_size"]),
            "--learning-rate", str(config["learning_rate"]),
        ], ROOT
    if args.model == "segmn":
        node_cap = int(config["node_cap"])
        return [
            str(PYG_PYTHON), "scripts/train_segmn_universal.py", *common,
            "--steps", str(args.budget * 50),
            "--learning-rate", str(config["learning_rate"]),
            "--batch-size", "4",
            "--validation-pairs", "128",
            "--validation-interval", "25",
            "--node-cap", str(node_cap),
            "--edge-cap", str(min(48, node_cap * 2)),
            "--split-seed", str(args.split_seed),
        ], ROOT
    script = (
        "scripts/train_gfm_smoke.py"
        if args.model == "graph-fusion"
        else "scripts/train_graph2region_universal.py"
    )
    multiplier = 20 if args.model == "graph-fusion" else 50
    return [
        str(PYG_PYTHON), script, *common,
        "--steps", str(args.budget * multiplier),
        "--batch-size", str(config["batch_size"]),
        "--learning-rate", str(config["learning_rate"]),
        "--split-seed", str(args.split_seed),
        *(
            ["--validation-pairs", "256", "--validation-interval", "50"]
            if args.model == "graph2region"
            else []
        ),
    ], ROOT


def prepare_simgnn_dataset(dataset_id: str, seed: int) -> None:
    command = [
        str(PYG_PYTHON), "scripts/prepare_simgnn_original_dataset.py",
        "--dataset", dataset_id,
        "--train-pairs", "8000",
        "--validation-pairs", "1200",
        "--test-pairs", "1200",
        "--seed", str(seed),
        "--clean",
    ]
    return_code, _ = run_command(command, ROOT)
    if return_code != 0:
        raise RuntimeError("SimGNN dataset preparation failed.")


def run_command(command: list[str], cwd: Path) -> tuple[int, str]:
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines = []
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line)
        print(line, end="", flush=True)
    return process.wait(), "".join(lines)


def read_validation_metric(model_id: str, checkpoint: Path, output: str) -> float | None:
    if model_id == "simgnn":
        matches = re.findall(r"Restored best validation checkpoint \(MSE=([0-9.eE+-]+)\)", output)
        return float(matches[-1]) if matches else None
    if model_id == "multiscale-set":
        metadata_path = Path(str(checkpoint) + ".meta.json")
        if not metadata_path.is_file():
            return None
        return numeric(json.loads(metadata_path.read_text()).get("best_validation_mse"))
    command = [
        str(PYG_PYTHON),
        "-c",
        "import json,sys,torch; x=torch.load(sys.argv[1],map_location='cpu'); "
        "print(json.dumps(x.get('best_validation_mse')))",
        str(checkpoint),
    ]
    completed = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return None
    return numeric(json.loads(completed.stdout.strip().splitlines()[-1]))


def active_validation_metric(model_id: str, checkpoint: Path) -> float | None:
    if model_id == "simgnn":
        payload, status = load_verified_hpo(checkpoint, ROOT)
        if status != "verified_checkpoint":
            return None
        return numeric((payload.get("best_trial") or {}).get("validation_mse"))
    if model_id == "multiscale-set":
        metadata = Path(str(checkpoint) + ".meta.json")
        if not metadata.is_file():
            return None
        try:
            payload = json.loads(metadata.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        return numeric(payload.get("best_validation_mse"))
    if not checkpoint.is_file():
        return None
    return read_validation_metric(model_id, checkpoint, "")


def checkpoint_protocol(model_id: str, checkpoint: Path) -> dict:
    if model_id == "simgnn":
        payload, status = load_verified_hpo(checkpoint, ROOT)
        return payload if status == "verified_checkpoint" else {}
    if model_id == "multiscale-set":
        metadata = Path(str(checkpoint) + ".meta.json")
        if not metadata.is_file():
            return {}
        try:
            return json.loads(metadata.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
    if not checkpoint.is_file():
        return {}
    command = [
        str(PYG_PYTHON),
        "-c",
        "import json,sys,torch; x=torch.load(sys.argv[1],map_location='cpu'); "
        "print(json.dumps({k:x.get(k) for k in "
        "['seed','pair_split','target']}))",
        str(checkpoint),
    ]
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return {}
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {}


def protocol_split_seed(model_id: str, protocol: dict, fallback_seed: int) -> int:
    pair_split = protocol.get("pair_split") or {}
    split_seed = pair_split.get("seed")
    if split_seed is not None:
        return int(split_seed)
    if model_id == "simgnn" and protocol.get("split_seed") is not None:
        return int(protocol["split_seed"])
    if model_id == "simgnn" and protocol.get("seed") is not None:
        return int(protocol["seed"])
    return fallback_seed if model_id == "simgnn" else fallback_seed + 1


def protocols_are_comparable(
    model_id: str,
    incumbent: dict,
    candidate: dict,
    incumbent_exists: bool,
) -> bool:
    if not incumbent_exists:
        return True
    incumbent_split = incumbent.get("pair_split") or {}
    candidate_split = candidate.get("pair_split") or {}
    incumbent_hash = incumbent_split.get("split_sha256")
    candidate_hash = candidate_split.get("split_sha256")
    if incumbent_hash and candidate_hash:
        return incumbent_hash == candidate_hash
    if (incumbent_hash or candidate_hash) and model_id != "simgnn":
        return False
    if model_id == "simgnn":
        incumbent_seed = incumbent.get("split_seed", incumbent.get("seed"))
        candidate_seed = candidate.get("split_seed", candidate.get("seed"))
        return incumbent_seed is not None and incumbent_seed == candidate_seed
    return False


def is_improvement(candidate: float | None, incumbent: float | None) -> bool:
    candidate_value = numeric(candidate)
    incumbent_value = numeric(incumbent)
    return candidate_value is not None and (
        incumbent_value is None or candidate_value < incumbent_value
    )


def promote_checkpoint(
    model_id: str,
    candidate: Path,
    target: Path,
    study_dir: Path,
) -> list[str]:
    target.parent.mkdir(parents=True, exist_ok=True)
    backup_dir = study_dir / "previous_active"
    backup_dir.mkdir(parents=True, exist_ok=True)
    if model_id == "multiscale-set":
        candidate_files = sorted(candidate.parent.glob(candidate.name + "*"))
        target_files = sorted(target.parent.glob(target.name + "*"))
        for path in target_files:
            shutil.copy2(path, backup_dir / path.name)
        promoted = []
        for source in candidate_files:
            suffix = source.name[len(candidate.name):]
            destination = Path(str(target) + suffix)
            shutil.copy2(source, destination)
            promoted.append(display_path(destination))
        return promoted
    if target.is_file():
        shutil.copy2(target, backup_dir / target.name)
    shutil.copy2(candidate, target)
    return [display_path(target)]


def numeric(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def finite(value) -> bool:
    return numeric(value) is not None


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
