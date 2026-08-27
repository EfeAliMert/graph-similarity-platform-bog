from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graph_similarity_platform.models.real_models import ALL_DATASETS, MODELS  # noqa: E402
from scripts.checkpoint_provenance import load_verified_hpo  # noqa: E402
from scripts.graphsim_calibration import validate_calibration  # noqa: E402


def build_checkpoint_audit() -> dict:
    rows = []
    for model in MODELS:
        for dataset_id in ALL_DATASETS:
            checkpoint = ROOT / model["local_path"] / model["checkpoint_by_dataset"][dataset_id]
            metadata_path = None
            metadata = {}
            if model["id"] == "simgnn":
                metadata_path = (
                    ROOT
                    / model["local_path"]
                    / "original_datasets"
                    / dataset_id
                    / "manifest.json"
                )
                metadata = _read_json(metadata_path)
            elif model["id"] == "multiscale-set":
                metadata_path = Path(str(checkpoint) + ".meta.json")
                metadata = _read_json(metadata_path)
            elif checkpoint.exists():
                metadata = _read_torch_checkpoint(checkpoint)
            pair_split = metadata.get("pair_split")
            overlap = (
                pair_split.get("pair_overlap_count", pair_split.get("pair_overlap"))
                if isinstance(pair_split, dict)
                else None
            )
            target = metadata.get("target")
            calibration = metadata.get("output_calibration")
            calibration_verified = (
                _graphsim_calibration_verified(calibration)
                if model["id"] == "multiscale-set"
                else None
            )
            hpo_metadata, hpo_status = load_verified_hpo(checkpoint, ROOT)
            row = {
                "model_id": model["id"],
                "model_name": model["name"],
                "dataset_id": dataset_id,
                "checkpoint": _display_path(checkpoint),
                "checkpoint_exists": checkpoint.exists()
                or Path(str(checkpoint) + ".index").exists(),
                "checkpoint_sha256": _checkpoint_digest(checkpoint),
                "metadata_path": _display_path(metadata_path) if metadata_path else None,
                "metadata_exists": metadata_path.exists() if metadata_path else checkpoint.exists(),
                "seed": metadata.get("seed"),
                "target": target,
                "pair_split": pair_split,
                "pair_overlap": overlap,
                "split_sha256": (
                    pair_split.get("split_sha256")
                    if isinstance(pair_split, dict)
                    else None
                ),
                "training_steps": metadata.get("steps"),
                "batch_size": metadata.get("batch_size"),
                "output_calibration": _calibration_summary(calibration),
                "calibration_verified": calibration_verified,
                "hpo_status": hpo_status,
                "hpo_verified": hpo_status == "verified_checkpoint",
                "hpo_best_trial": (
                    _portable_metadata(hpo_metadata.get("best_trial"))
                    if isinstance(hpo_metadata, dict)
                    else None
                ),
                "hpo_final_training": (
                    _portable_metadata(hpo_metadata.get("final_training"))
                    if isinstance(hpo_metadata, dict)
                    else None
                ),
            }
            row["protocol_verified"] = bool(
                row["checkpoint_exists"]
                and isinstance(row["seed"], int)
                and isinstance(target, dict)
                and overlap == 0
                and row["split_sha256"]
                and (
                    calibration_verified
                    if model["id"] == "multiscale-set"
                    else True
                )
            )
            rows.append(row)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total": len(rows),
        "verified": sum(row["protocol_verified"] for row in rows),
        "hpo_verified": sum(row["hpo_verified"] for row in rows),
        "complete": all(row["protocol_verified"] for row in rows),
        "rows": rows,
    }


def write_checkpoint_audit(payload: dict) -> tuple[Path, Path]:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    json_path = reports / "checkpoint_audit.json"
    markdown_path = reports / "checkpoint_audit.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    lines = [
        "# Checkpoint Protocol Audit",
        "",
        f"Verified: {payload['verified']}/{payload['total']}",
        f"HPO-to-checkpoint binding verified: {payload['hpo_verified']}/{payload['total']}",
        "",
        "| Dataset | Model | Exists | Seed | Pair overlap | Calibration | HPO binding | Split hash | Protocol verified |",
        "|---|---|---|---:|---:|---|---|---|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['dataset_id']} | {row['model_name']} | "
            f"{row['checkpoint_exists']} | {row['seed']} | {row['pair_overlap']} | "
            f"{_calibration_label(row['calibration_verified'])} | "
            f"{row['hpo_status']} | "
            f"{row['split_sha256'] or '-'} | {row['protocol_verified']} |"
        )
    markdown_path.write_text("\n".join(lines))
    return json_path, markdown_path


def main() -> int:
    payload = build_checkpoint_audit()
    json_path, markdown_path = write_checkpoint_audit(payload)
    print(f"verified={payload['verified']}/{payload['total']}")
    print(f"hpo_verified={payload['hpo_verified']}/{payload['total']}")
    print(f"json={_display_path(json_path)}")
    print(f"markdown={_display_path(markdown_path)}")
    return 0 if payload["complete"] else 2


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_torch_checkpoint(path: Path) -> dict:
    import torch

    loaded = torch.load(path, map_location="cpu", weights_only=False)
    return loaded if isinstance(loaded, dict) else {}


def _checkpoint_digest(path: Path) -> str | None:
    paths = [path] if path.is_file() else sorted(
        candidate
        for candidate in path.parent.glob(f"{path.name}*")
        if candidate.is_file() and not candidate.name.endswith(".meta.json")
    )
    if not paths:
        return None
    digest = hashlib.sha256()
    for candidate in paths:
        digest.update(candidate.name.encode())
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _graphsim_calibration_verified(calibration: object) -> bool:
    if not isinstance(calibration, dict):
        return False
    try:
        validate_calibration(calibration)
    except (TypeError, ValueError):
        return False
    fit_graph_ids = calibration.get("fit_graph_ids")
    audit_graph_ids = calibration.get("audit_graph_ids")
    graph_partitions_verified = bool(
        isinstance(fit_graph_ids, list)
        and isinstance(audit_graph_ids, list)
        and fit_graph_ids
        and audit_graph_ids
        and set(fit_graph_ids).isdisjoint(audit_graph_ids)
    )
    finite_metrics = (
        calibration.get("fit_mse_calibrated"),
        calibration.get("audit_mse_raw"),
        calibration.get("audit_mse_calibrated"),
    )
    integrity_verified = bool(
        calibration.get("method") == "validation_isotonic_regression"
        and calibration.get("fit_audit_graph_overlap") == 0
        and graph_partitions_verified
        and calibration.get("test_graphs_used") is False
        and _positive_int(calibration.get("fit_pair_count"), minimum=2)
        and _positive_int(calibration.get("audit_pair_count"), minimum=1)
        and all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in finite_metrics
        )
    )
    if not integrity_verified:
        return False
    calibrated_mse = float(calibration["audit_mse_calibrated"])
    raw_mse = float(calibration["audit_mse_raw"])
    accepted = calibration.get("accepted_by_audit")
    if accepted is False:
        return calibrated_mse > raw_mse
    return calibrated_mse <= raw_mse


def _calibration_summary(calibration: object) -> dict | None:
    if not isinstance(calibration, dict):
        return None
    fields = (
        "method",
        "fit_pair_count",
        "audit_pair_count",
        "fit_mse_raw",
        "fit_mse_calibrated",
        "audit_mse_raw",
        "audit_mse_calibrated",
        "accepted_by_audit",
        "fit_audit_graph_overlap",
        "test_graphs_used",
    )
    return {field: calibration.get(field) for field in fields}


def _positive_int(value: object, minimum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _calibration_label(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "verified" if value else "missing/invalid"


def _portable_metadata(value: object) -> object:
    if isinstance(value, dict):
        return {key: _portable_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_metadata(item) for item in value]
    if isinstance(value, str):
        path = Path(value)
        if path.is_absolute():
            return _display_path(path)
    return value


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
