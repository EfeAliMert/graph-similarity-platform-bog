from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graph_similarity_platform.research_summary import (  # noqa: E402
    finalize_matrix_artifacts,
    write_research_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    finalized = finalize_matrix_artifacts(manifest)
    reports = write_research_summary(manifest)
    print(f"sidecars={finalized['written']} skipped={finalized['skipped']}")
    print(f"json={reports['json']}")
    print(f"markdown={reports['markdown']}")


if __name__ == "__main__":
    main()
