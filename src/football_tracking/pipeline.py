"""Run the SoccerNet role, ball, and team pipeline on a frame sequence."""

from __future__ import annotations

import configparser
from pathlib import Path
from typing import Iterable

from football_tracking.io.results import ResultWriter
from football_tracking.vision.ball_tracker import BallTemporalTracker, SlicedBallDetector
from football_tracking.vision.role_tracker import RoleTracker
from football_tracking.vision.team_classifier import TeamAssigner


def _read_sequence_meta(sequence_dir: Path) -> tuple[Path, str]:
    parser = configparser.ConfigParser()
    parser.read(sequence_dir / "seqinfo.ini", encoding="utf-8")
    section = parser["Sequence"]
    image_dir = sequence_dir / section.get("imDir", "img1")
    extension = section.get("imExt", ".jpg")
    if not extension.startswith("."):
        extension = "." + extension
    return image_dir, extension


def _frame_paths(sequence_dir: Path) -> list[Path]:
    image_dir, extension = _read_sequence_meta(sequence_dir)
    paths = list(image_dir.glob(f"*{extension}"))
    return sorted(paths, key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem)


class TrackingPipeline:
    """Process one SoccerNet sequence and write streamable result files."""

    def __init__(
        self,
        role_tracker: RoleTracker,
        *,
        ball_detector: SlicedBallDetector | None = None,
        ball_tracker: BallTemporalTracker | None = None,
        team_assigner: TeamAssigner | None = None,
    ) -> None:
        self.role_tracker = role_tracker
        self.ball_detector = ball_detector
        self.ball_tracker = ball_tracker or BallTemporalTracker()
        self.team_assigner = team_assigner

    def run_sequence(self, sequence_dir: Path, output_root: Path) -> tuple[Path, Path]:
        import cv2

        sequence_dir = Path(sequence_dir).resolve()
        frames = _frame_paths(sequence_dir)
        if not frames:
            raise FileNotFoundError(f"No sequence frames found under {sequence_dir}")

        output_dir = Path(output_root) / sequence_dir.name
        writer = ResultWriter(
            output_dir / "tracks.jsonl",
            output_dir / "track_result.txt",
        )
        try:
            for frame_id, frame_path in enumerate(frames, start=1):
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    raise RuntimeError(f"Could not read frame: {frame_path}")

                roles = self.role_tracker.track_frame(frame)
                teams = (
                    self.team_assigner.update(frame, roles, frame_id)
                    if self.team_assigner is not None
                    else {}
                )
                ball = None
                if self.ball_detector is not None:
                    ball_detections = self.ball_detector.detect(frame)
                    ball = self.ball_tracker.update(ball_detections, frame_id)
                writer.write_frame(frame_id, roles, teams, ball)
        finally:
            writer.close()
        return output_dir / "tracks.jsonl", output_dir / "track_result.txt"
