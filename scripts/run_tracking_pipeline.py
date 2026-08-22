"""Run role tracking, optional ball tracking, and team assignment on a sequence."""

from __future__ import annotations

import argparse
from pathlib import Path

from football_tracking.pipeline import TrackingPipeline
from football_tracking.vision.ball_tracker import BallTemporalTracker, SlicedBallDetector
from football_tracking.vision.role_tracker import RoleTracker
from football_tracking.vision.team_classifier import TeamAssigner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--role-weights", type=Path, required=True)
    parser.add_argument("--ball-weights", type=Path)
    parser.add_argument("--tracker-config", type=Path, default=Path("configs/botsort.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tracking"))
    parser.add_argument("--device", default=None, help="CUDA device such as 0; omit for Ultralytics auto.")
    parser.add_argument("--role-imgsz", type=int, default=1280)
    parser.add_argument("--role-conf", type=float, default=0.25)
    parser.add_argument("--ball-tile-size", type=int, default=640)
    parser.add_argument("--ball-conf", type=float, default=0.10)
    parser.add_argument("--disable-team", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    role_tracker = RoleTracker(
        args.role_weights,
        tracker=args.tracker_config,
        device=args.device,
        imgsz=args.role_imgsz,
        output_conf=args.role_conf,
    )
    ball_detector = (
        SlicedBallDetector(
            args.ball_weights,
            tile_size=args.ball_tile_size,
            confidence=args.ball_conf,
            device=args.device,
        )
        if args.ball_weights
        else None
    )
    pipeline = TrackingPipeline(
        role_tracker,
        ball_detector=ball_detector,
        ball_tracker=BallTemporalTracker(),
        team_assigner=None if args.disable_team else TeamAssigner(),
    )
    jsonl_path, mot_path = pipeline.run_sequence(args.sequence, args.output_dir)
    print(f"tracks.jsonl: {jsonl_path}")
    print(f"track_result.txt: {mot_path}")
    if ball_detector is None:
        print("Ball detector disabled: pass --ball-weights to enable it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
