from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the remaining local research study: held-out evaluation, retrieval, ablations, and report."
    )
    parser.add_argument("--datasets", default="aids700nef,linux,imdbmulti,ptc")
    parser.add_argument("--pairs", type=int, default=50)
    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    commands = [
        [
            sys.executable,
            "scripts/run_research_matrix.py",
            "--datasets",
            args.datasets,
            "--benchmark-only",
            "--evaluate-existing",
            "--evaluate-pairs",
            str(args.pairs),
            "--execute",
            "--continue-on-error",
        ],
        [
            sys.executable,
            "scripts/run_retrieval_study.py",
            "--prefilter-datasets",
            args.datasets,
            "--rerank-datasets",
            "aids700nef,linux",
            *(["--skip-rerank"] if args.skip_rerank else []),
        ],
        [sys.executable, "scripts/run_grouped_split_study.py"],
    ]
    if not args.skip_ablations:
        commands.insert(
            2,
            [
                sys.executable,
                "scripts/run_adapter_ablations.py",
                "--datasets",
                "aids700nef,linux",
                "--pairs",
                str(args.pairs),
            ],
        )
    commands.append([sys.executable, "scripts/compile_research_report.py"])

    for command in commands:
        print("RUN", " ".join(command))
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode and not args.continue_on_error:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
