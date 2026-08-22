"""Validate raw SoccerNet Tracking 2023 annotations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from football_tracking.data.validator import validate_dataset, write_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/reports/data_validation.json"),
    )
    args = parser.parse_args()
    report = validate_dataset(args.source)
    write_report(report, args.report)
    print(
        json.dumps(
            {
                "valid": report["valid"],
                "num_sequences": report.get("num_sequences", 0),
                "num_annotations": report.get("num_annotations", 0),
                "error_count": report.get("error_count", 0),
                "warning_count": report.get("warning_count", 0),
                "report": str(args.report),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

