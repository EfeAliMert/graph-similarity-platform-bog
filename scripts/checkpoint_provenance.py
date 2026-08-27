from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def checkpoint_fingerprint(checkpoint: Path) -> str | None:
    """Hash model weights without including mutable metadata sidecars."""
    checkpoint = checkpoint.resolve()
    if checkpoint.is_file():
        files = [checkpoint]
    else:
        files = []
        for suffix in (".data-*", ".index", ".meta"):
            files.extend(checkpoint.parent.glob(checkpoint.name + suffix))
        files = sorted(set(files), key=lambda path: path.name)
    if not files:
        return None

    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.removeprefix(checkpoint.name).encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def load_verified_hpo(
    checkpoint: Path,
    root: Path,
) -> tuple[dict[str, Any], str]:
    """Return HPO metadata only when it identifies the active model weights."""
    checkpoint = checkpoint.resolve()
    sidecar = Path(str(checkpoint) + ".hpo.json")
    if not sidecar.is_file():
        return {}, "not_recorded"
    try:
        payload = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError):
        return {}, "invalid_sidecar"

    recorded_active = _resolve_recorded_path(payload.get("active_checkpoint"), root)
    if recorded_active is not None and recorded_active != checkpoint:
        return {}, "active_path_mismatch"

    active_fingerprint = checkpoint_fingerprint(checkpoint)
    recorded_fingerprint = payload.get("active_checkpoint_fingerprint")
    if recorded_fingerprint:
        if active_fingerprint == recorded_fingerprint:
            return payload, "verified_checkpoint"
        return {}, "stale_checkpoint"

    best_trial = payload.get("best_trial") or {}
    candidate = _resolve_recorded_path(best_trial.get("checkpoint"), root)
    if candidate is None:
        return {}, "unverifiable_sidecar"
    candidate_fingerprint = checkpoint_fingerprint(candidate)
    if active_fingerprint is None or candidate_fingerprint is None:
        return {}, "unverifiable_sidecar"
    if active_fingerprint != candidate_fingerprint:
        return {}, "stale_checkpoint"
    return payload, "verified_checkpoint"


def _resolve_recorded_path(value: Any, root: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()
