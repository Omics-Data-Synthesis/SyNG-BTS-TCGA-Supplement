# Stage 08: build the interactive website JSON from SyntheSize metrics.
#
# Incremental by default: configurations present under data/processed/curves/
# overwrite the matching website/plots/**/data.json; bootstrap files for
# cohorts that have not been re-run stay in place. Then rebuilds
# website/manifest.json from every data.json on disk, using sample/marker
# counts from config/datasets.tsv.
#
# grouping=none datasets are skipped (classifiers need a binary label),
# matching stage 06.
#
# Run from the repository root:
#
#   uv run workflow/08_website/run.py
#
# Optional flags:
#   --replace         wipe website/plots/ before rebuilding from the cache
#   --manifest-only   only rebuild website/manifest.json from existing JSON

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

STAGE_DIR = Path(__file__).resolve().parent
REPO = STAGE_DIR.parents[1]
sys.path.insert(0, str(STAGE_DIR))

PLOTS_ROOT = REPO / "website" / "plots"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build website plot JSON and manifest.")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="delete website/plots/ before writing from the SyntheSize cache",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="rebuild website/manifest.json without reading the curves cache",
    )
    args = parser.parse_args()

    if args.replace and args.manifest_only:
        raise SystemExit("choose one of --replace or --manifest-only")

    if args.replace:
        if PLOTS_ROOT.is_dir():
            shutil.rmtree(PLOTS_ROOT)
            print(f"removed {PLOTS_ROOT.relative_to(REPO)}")
        PLOTS_ROOT.mkdir(parents=True)

    if not args.manifest_only:
        import build_data

        build_data.main()

    import generate_manifest

    generate_manifest.main()


if __name__ == "__main__":
    main()
