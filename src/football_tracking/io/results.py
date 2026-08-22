"""Write role-aware JSONL and MOT-compatible tracking outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TextIO

from football_tracking.vision.ball_tracker import BallTrack
from football_tracking.vision.role_tracker import RoleDetection
from football_tracking.vision.team_classifier import TeamPrediction


BALL_MOT_TRACK_ID = 1_000_000


class ResultWriter:
    """Stream tracking results without keeping a whole sequence in memory."""

    def __init__(self, jsonl_path: Path, mot_path: Path) -> None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        mot_path.parent.mkdir(parents=True, exist_ok=True)
        self._jsonl: TextIO = jsonl_path.open("w", encoding="utf-8")
        self._mot: TextIO = mot_path.open("w", encoding="utf-8")

    def write_frame(
        self,
        frame_id: int,
        roles: list[RoleDetection],
        teams: dict[int, TeamPrediction],
        ball: BallTrack | None,
    ) -> None:
        for detection in roles:
            team = teams.get(detection.track_id) if detection.track_id is not None else None
            record = {
                "frame_id": frame_id,
                "track_id": detection.track_id,
                "bbox": list(detection.bbox),
                "role_id": detection.role_id,
                "role_name": detection.role_name,
                "role_confidence": detection.confidence,
                "team_id": team.team_id if team else None,
                "team_confidence": team.confidence if team else None,
                "visible": True,
                "predicted": False,
            }
            self._write_json(record)
            if detection.track_id is not None:
                self._write_mot(frame_id, detection.track_id, detection.bbox, detection.confidence)

        if ball is not None and ball.center is not None:
            record = {
                "frame_id": frame_id,
                "track_id": ball.track_id,
                "bbox": list(ball.bbox) if ball.bbox is not None else None,
                "role_id": 0,
                "role_name": "ball",
                "role_confidence": ball.confidence,
                "team_id": 2,
                "team_confidence": 1.0,
                "visible": ball.visible,
                "predicted": ball.predicted,
                "missed_frames": ball.missed_frames,
            }
            self._write_json(record)
            if ball.visible and ball.bbox is not None:
                self._write_mot(frame_id, BALL_MOT_TRACK_ID, ball.bbox, ball.confidence)

    def close(self) -> None:
        self._jsonl.close()
        self._mot.close()

    def _write_json(self, record: dict) -> None:
        self._jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_mot(
        self,
        frame_id: int,
        track_id: int,
        bbox: tuple[int, int, int, int],
        confidence: float,
    ) -> None:
        x1, y1, x2, y2 = bbox
        self._mot.write(
            f"{frame_id},{track_id},{x1},{y1},{max(0, x2 - x1)},{max(0, y2 - y1)},"
            f"{confidence:.6f},-1,-1,-1\n"
        )
