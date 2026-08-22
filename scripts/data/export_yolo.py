"""Export raw SoccerNet data into the standard YOLO directory layout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from football_tracking.data.yolo_exporter import export_yolo_dataset
from football_tracking.data.paths import resolve_project_output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("yolo"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-mode", choices=("hardlink", "symlink", "copy"), default="hardlink")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    output_path = resolve_project_output_path(args.output, project_root)
    summary = export_yolo_dataset(
        args.source,
        output_path,
        seed=args.seed,
        image_mode=args.image_mode,
        allowed_output_root=project_root,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

