"""Team assignment from player jersey color with temporal smoothing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .role_tracker import RoleDetection


@dataclass(frozen=True)
class TeamPrediction:
    team_id: int | None
    confidence: float | None


class TeamAssigner:
    """Fit two team color centroids during warmup and classify tracks."""

    def __init__(
        self,
        *,
        warmup_frames: int = 60,
        min_samples: int = 10,
        ema_alpha: float = 0.9,
    ) -> None:
        self.warmup_frames = warmup_frames
        self.min_samples = min_samples
        self.ema_alpha = ema_alpha
        self._samples: list[np.ndarray] = []
        self._centroids: np.ndarray | None = None
        self._track_features: dict[int, np.ndarray] = {}

    @property
    def fitted(self) -> bool:
        return self._centroids is not None

    def update(
        self,
        frame: np.ndarray,
        detections: Iterable[RoleDetection],
        frame_index: int,
    ) -> dict[int, TeamPrediction]:
        detections = list(detections)
        if not self.fitted and frame_index <= self.warmup_frames:
            self._samples.extend(
                feature
                for detection in detections
                if detection.role_name == "player"
                for feature in [self._feature(frame, detection.bbox)]
                if feature is not None
            )
            if frame_index == self.warmup_frames and len(self._samples) >= self.min_samples:
                self._fit()

        if not self.fitted:
            return {}

        predictions: dict[int, TeamPrediction] = {}
        players = [item for item in detections if item.role_name == "player"]
        for detection in players:
            if detection.track_id is None:
                continue
            feature = self._feature(frame, detection.bbox)
            if feature is None:
                continue
            feature = self._smooth(detection.track_id, feature)
            team_id, confidence = self._nearest(feature)
            predictions[detection.track_id] = TeamPrediction(team_id, confidence)

        team_positions: dict[int, list[np.ndarray]] = {0: [], 1: []}
        for detection in players:
            prediction = predictions.get(detection.track_id) if detection.track_id is not None else None
            if prediction is not None and prediction.team_id in team_positions:
                team_positions[prediction.team_id].append(np.asarray(self._foot(detection.bbox)))

        for detection in detections:
            if detection.role_name == "referee" and detection.track_id is not None:
                predictions[detection.track_id] = TeamPrediction(2, 1.0)
            elif detection.role_name == "goalkeeper" and detection.track_id is not None:
                prediction = self._goalkeeper_team(detection, team_positions)
                if prediction is not None:
                    predictions[detection.track_id] = prediction
        return predictions

    def _fit(self) -> None:
        samples = np.asarray(self._samples, dtype=np.float32)
        first = 0
        second = int(np.argmax(np.linalg.norm(samples - samples[first], axis=1)))
        centroids = np.asarray([samples[first], samples[second]], dtype=np.float32)
        for _ in range(20):
            distances = np.linalg.norm(samples[:, None, :] - centroids[None, :, :], axis=2)
            labels = distances.argmin(axis=1)
            updated = np.asarray(
                [samples[labels == index].mean(axis=0) if np.any(labels == index) else centroids[index] for index in range(2)]
            )
            if np.allclose(updated, centroids):
                break
            centroids = updated
        order = np.argsort(centroids[:, 0])
        self._centroids = centroids[order]

    def _feature(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        import cv2

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1 + int((y2 - y1) * 0.20) : y1 + int((y2 - y1) * 0.65), x1 + int((x2 - x1) * 0.20) : x1 + int((x2 - x1) * 0.80)]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        grass = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([90, 255, 255]))
        non_grass = cv2.bitwise_not(grass)
        if cv2.countNonZero(non_grass) < crop.shape[0] * crop.shape[1] * 0.1:
            non_grass = np.full(crop.shape[:2], 255, dtype=np.uint8)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        pixels = lab[non_grass > 0]
        if len(pixels) == 0:
            return None
        return np.median(pixels[:, 1:], axis=0).astype(np.float32)

    def _smooth(self, track_id: int, feature: np.ndarray) -> np.ndarray:
        previous = self._track_features.get(track_id)
        if previous is None:
            smoothed = feature
        else:
            smoothed = self.ema_alpha * previous + (1.0 - self.ema_alpha) * feature
        self._track_features[track_id] = smoothed
        return smoothed

    def _nearest(self, feature: np.ndarray) -> tuple[int, float]:
        assert self._centroids is not None
        distances = np.linalg.norm(self._centroids - feature, axis=1)
        margin = abs(float(distances[0] - distances[1])) / max(float(distances.sum()), 1e-6)
        return int(np.argmin(distances)), margin

    @staticmethod
    def _foot(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, float(y2))

    @staticmethod
    def _goalkeeper_team(
        detection: RoleDetection,
        team_positions: dict[int, list[np.ndarray]],
    ) -> TeamPrediction | None:
        available = {team: np.asarray(points).mean(axis=0) for team, points in team_positions.items() if points}
        if not available:
            return None
        position = np.asarray(TeamAssigner._foot(detection.bbox))
        distances = {team: float(np.linalg.norm(position - center)) for team, center in available.items()}
        team = min(distances, key=distances.get)
        return TeamPrediction(int(team), 1.0)
