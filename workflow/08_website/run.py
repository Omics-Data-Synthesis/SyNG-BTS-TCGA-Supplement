# Stage 08: rebuild website JSON from stage 06's numerical archive.
# Default and --replace both replace the complete generated plot tree.
# --manifest-only validates existing JSON with the standard library (Pages CI).

from __future__ import annotations

import argparse
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
        help="compatibility flag; full replacement is now the default",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="rebuild website/manifest.json from validated plot JSON (no scientific dependencies)",
    )
    args = parser.parse_args()

    if args.replace and args.manifest_only:
        raise SystemExit("choose one of --replace or --manifest-only")

    if not args.manifest_only:
        import build_data

        build_data.main()

    import generate_manifest

    generate_manifest.main()


if __name__ == "__main__":
    main()
