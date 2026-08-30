from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "model_sources.json"
INSTALL_MARKER = ".gsp-model-source.json"


class ModelInstallError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install the five pinned graph-similarity model sources."
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Install one model ID. Repeat the option to select several models.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check the local sources without cloning or applying patches.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    config_path = args.config.resolve()
    specs = load_specs(config_path)
    requested = set(args.model)
    if requested:
        known = {spec["id"] for spec in specs}
        unknown = sorted(requested - known)
        if unknown:
            raise ModelInstallError(
                f"Unknown model ID(s): {', '.join(unknown)}. "
                f"Known IDs: {', '.join(sorted(known))}."
            )
        specs = [spec for spec in specs if spec["id"] in requested]

    if not args.verify_only and shutil.which("git") is None:
        raise ModelInstallError("git is required to download model sources.")

    installed = []
    for index, spec in enumerate(specs, start=1):
        prefix = f"[{index}/{len(specs)}] {spec['name']}"
        if args.verify_only:
            verify_model(spec, root)
            print(f"{prefix}: verified")
        else:
            state = install_model(spec, root)
            print(f"{prefix}: {state}")
        installed.append(spec["id"])

    action = "Verified" if args.verify_only else "Installed"
    print(f"{action} {len(installed)}/{len(specs)} model sources.")


def load_specs(config_path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(config_path.read_text())
    except FileNotFoundError as exc:
        raise ModelInstallError(f"Model source manifest not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ModelInstallError(f"Invalid model source manifest: {exc}") from exc
    if payload.get("schema_version") != 1:
        raise ModelInstallError("Unsupported model source manifest schema.")
    specs = payload.get("models")
    if not isinstance(specs, list) or not specs:
        raise ModelInstallError("Model source manifest has no model entries.")
    required = {"id", "name", "repository", "commit", "path", "required_files"}
    seen = set()
    for spec in specs:
        missing = sorted(required - set(spec))
        if missing:
            raise ModelInstallError(
                f"Model entry is missing required keys: {', '.join(missing)}."
            )
        if spec["id"] in seen:
            raise ModelInstallError(f"Duplicate model ID: {spec['id']}.")
        seen.add(spec["id"])
    return specs


def install_model(spec: dict[str, Any], root: Path) -> str:
    target = safe_path(root, spec["path"])
    patch_path = (
        safe_path(root, spec["patch"])
        if spec.get("patch")
        else None
    )
    target.parent.mkdir(parents=True, exist_ok=True)

    if not target.exists():
        clone_model(spec, target)
        state = "downloaded"
    elif is_compatible(spec, target):
        verify_pinned_checkout_when_available(spec, target)
        write_marker(spec, target, patch_path, "preexisting-compatible")
        return "already ready"
    elif not is_standalone_git_checkout(target):
        raise ModelInstallError(
            f"{spec['name']} already exists at {target}, but it is incomplete or "
            "incompatible. Move that folder aside and run `make models` again."
        )
    else:
        verify_pinned_checkout(spec, target)
        state = "found pinned checkout"

    if patch_path is not None:
        if not patch_path.is_file():
            raise ModelInstallError(
                f"Compatibility patch for {spec['name']} is missing: {patch_path}"
            )
        patch_state = apply_patch(target, patch_path)
        state = f"{state}; {patch_state}"

    verify_model(spec, root)
    write_marker(spec, target, patch_path, "pinned-upstream")
    return state


def clone_model(spec: dict[str, Any], target: Path) -> None:
    print(f"Downloading {spec['name']} from {spec['repository']} ...", flush=True)
    result = run(
        ["git", "clone", "--no-checkout", spec["repository"], str(target)]
    )
    if result.returncode != 0:
        raise ModelInstallError(
            f"Could not clone {spec['name']}: {last_error(result)}"
        )
    result = run(
        ["git", "checkout", "--detach", spec["commit"]],
        cwd=target,
    )
    if result.returncode != 0:
        raise ModelInstallError(
            f"Could not checkout {spec['commit']} for {spec['name']}: "
            f"{last_error(result)}"
        )


def apply_patch(target: Path, patch_path: Path) -> str:
    check = run(["git", "apply", "--check", str(patch_path)], cwd=target)
    if check.returncode == 0:
        applied = run(["git", "apply", str(patch_path)], cwd=target)
        if applied.returncode != 0:
            raise ModelInstallError(
                f"Could not apply {patch_path.name}: {last_error(applied)}"
            )
        return f"applied {patch_path.name}"

    reverse = run(
        ["git", "apply", "--reverse", "--check", str(patch_path)],
        cwd=target,
    )
    if reverse.returncode == 0:
        return f"{patch_path.name} already applied"
    raise ModelInstallError(
        f"{patch_path.name} does not apply cleanly to {target}. "
        "The checkout may have local edits or the wrong source revision."
    )


def verify_model(spec: dict[str, Any], root: Path) -> None:
    target = safe_path(root, spec["path"])
    problems = compatibility_problems(spec, target)
    if problems:
        raise ModelInstallError(
            f"{spec['name']} verification failed: {'; '.join(problems)}"
        )
    verify_pinned_checkout_when_available(spec, target)


def is_compatible(spec: dict[str, Any], target: Path) -> bool:
    return not compatibility_problems(spec, target)


def compatibility_problems(
    spec: dict[str, Any], target: Path
) -> list[str]:
    if not target.is_dir():
        return [f"source directory is missing: {target}"]
    problems = []
    for relative in spec.get("required_files", []):
        path = target / relative
        if not path.is_file():
            problems.append(f"missing {relative}")
    for check in spec.get("content_checks", []):
        path = target / check["path"]
        if not path.is_file():
            problems.append(f"missing {check['path']}")
            continue
        try:
            content = path.read_text(errors="replace")
        except OSError as exc:
            problems.append(f"cannot read {check['path']}: {exc}")
            continue
        if check["contains"] not in content:
            problems.append(
                f"{check['path']} lacks compatibility marker "
                f"{check['contains']!r}"
            )
    return problems


def verify_pinned_checkout_when_available(
    spec: dict[str, Any], target: Path
) -> None:
    if is_standalone_git_checkout(target):
        verify_pinned_checkout(spec, target)


def verify_pinned_checkout(spec: dict[str, Any], target: Path) -> None:
    result = run(["git", "rev-parse", "HEAD"], cwd=target)
    actual = result.stdout.strip() if result.returncode == 0 else ""
    if actual != spec["commit"]:
        raise ModelInstallError(
            f"{spec['name']} is at commit {actual or 'unknown'}, expected "
            f"{spec['commit']}. Move the folder aside and run `make models` again."
        )


def is_standalone_git_checkout(path: Path) -> bool:
    return (path / ".git").exists()


def write_marker(
    spec: dict[str, Any],
    target: Path,
    patch_path: Path | None,
    source_state: str,
) -> None:
    payload = {
        "schema_version": 1,
        "model_id": spec["id"],
        "repository": spec["repository"],
        "pinned_commit": spec["commit"],
        "patch": spec.get("patch"),
        "patch_sha256": sha256_file(patch_path) if patch_path else None,
        "source_state": source_state,
        "installed_at": datetime.now(timezone.utc).isoformat(),
    }
    (target / INSTALL_MARKER).write_text(json.dumps(payload, indent=2) + "\n")


def safe_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ModelInstallError(f"Path escapes repository root: {relative}") from exc
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def last_error(result: subprocess.CompletedProcess) -> str:
    return (result.stderr or result.stdout or "unknown git error").strip()


if __name__ == "__main__":
    try:
        main()
    except ModelInstallError as exc:
        print(f"Model setup failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
