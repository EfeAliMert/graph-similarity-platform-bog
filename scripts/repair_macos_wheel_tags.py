from __future__ import annotations

import base64
import csv
import hashlib
from importlib.metadata import PackageNotFoundError, distribution
import io
from pathlib import Path
import platform
import subprocess
import sys


TARGETS = {
    "torch": {
        "binary_glob": "torch/lib/libtorch_cpu.dylib",
        "tag": "cp{python}-none-macosx_11_0_arm64",
    },
    "grpcio": {
        "binary_glob": "grpc/_cython/cygrpc*.so",
        "tag": "cp{python}-cp{python}-macosx_11_0_universal2",
    },
}


def main() -> int:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        print("wheel_tag_repair=not_required")
        return 0

    python_tag = f"{sys.version_info.major}{sys.version_info.minor}"
    repaired = 0
    for package, spec in TARGETS.items():
        try:
            dist = distribution(package)
        except PackageNotFoundError:
            continue
        binaries = [
            Path(dist.locate_file(path))
            for path in (dist.files or [])
            if Path(path).match(spec["binary_glob"])
        ]
        if not binaries:
            raise RuntimeError(f"Could not locate the compiled binary for {package}.")
        architecture = subprocess.check_output(
            ["/usr/bin/file", str(binaries[0])],
            text=True,
        )
        if "arm64" not in architecture:
            raise RuntimeError(
                f"Refusing to retag {package}; its binary does not contain arm64."
            )

        wheel_path = Path(dist._path) / "WHEEL"
        text = wheel_path.read_text()
        expected_tag = spec["tag"].format(python=python_tag)
        tag_lines = [
            line for line in text.splitlines() if line.startswith("Tag:")
        ]
        if tag_lines == [f"Tag: {expected_tag}"]:
            print(f"{package}=already_valid")
            continue
        updated = "\n".join(
            f"Tag: {expected_tag}" if line.startswith("Tag:") else line
            for line in text.splitlines()
        ) + "\n"
        wheel_path.write_text(updated)
        _update_record(Path(dist._path), wheel_path)
        repaired += 1
        print(f"{package}=retagged:{expected_tag}")
    print(f"wheel_tags_repaired={repaired}")
    return 0


def _update_record(dist_info: Path, wheel_path: Path) -> None:
    record_path = dist_info / "RECORD"
    if not record_path.exists():
        return
    rows = list(csv.reader(io.StringIO(record_path.read_text())))
    relative_wheel = f"{dist_info.name}/WHEEL"
    digest = base64.urlsafe_b64encode(
        hashlib.sha256(wheel_path.read_bytes()).digest()
    ).decode().rstrip("=")
    for row in rows:
        if row and row[0] == relative_wheel:
            row[1:] = [f"sha256={digest}", str(wheel_path.stat().st_size)]
            break
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    record_path.write_text(buffer.getvalue())


if __name__ == "__main__":
    raise SystemExit(main())
