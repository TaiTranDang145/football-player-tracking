"""Sliced ball detection and a small temporal ball tracker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BallDetection:
    bbox: tuple[int, int, int, int]
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass(frozen=True)
class BallTrack:
    track_id: int
    bbox: tuple[int, int, int, int] | None
    center: tuple[float, float] | None
    confidence: float
    visible: bool
    predicted: bool
    missed_frames: int


class SlicedBallDetector:
    """Run a dedicated ball model on overlapping tiles of a frame."""

    def __init__(
        self,
        weights: str | Path,
        *,
        tile_size: int = 640,
        overlap: float = 0.20,
        confidence: float = 0.10,
        nms_iou: float = 0.10,
        device: str | int | None = None,
    ) -> None:
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if not 0.0 <= overlap < 1.0:
            raise ValueError("overlap must be in [0, 1)")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not 0.0 <= nms_iou <= 1.0:
            raise ValueError("nms_iou must be in [0, 1]")

        self.weights = Path(weights)
        self.tile_size = tile_size
        self.overlap = overlap
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.device = device
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            if not self.weights.is_file():
                raise FileNotFoundError(f"Ball detector weights not found: {self.weights}")
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "Ultralytics is required for ball inference."
                ) from exc
            self._model = YOLO(str(self.weights))
        return self._model

    def detect(self, frame: np.ndarray) -> list[BallDetection]:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError("frame must be a color image as a HxWxC numpy array")

        height, width = frame.shape[:2]
        candidates: list[BallDetection] = []
        for x0 in self._starts(width):
            for y0 in self._starts(height):
                tile = frame[y0 : min(y0 + self.tile_size, height), x0 : min(x0 + self.tile_size, width)]
                kwargs: dict[str, Any] = {
                    "source": tile,
                    "imgsz": self.tile_size,
                    "conf": self.confidence,
                    "verbose": False,
                }
                if self.device is not None:
                    kwargs["device"] = self.device
                result = self.model.predict(**kwargs)[0]
                boxes = getattr(result, "boxes", None)
                if boxes is None or len(boxes) == 0:
                    continue

                xyxy = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)
                names = getattr(result, "names", getattr(self.model, "names", {}))
                for index, box in enumerate(xyxy):
                    if not self._is_ball_class(names, int(classes[index])):
                        continue
                    x1, y1, x2, y2 = box
                    absolute = (
                        max(0, int(round(x1)) + x0),
                        max(0, int(round(y1)) + y0),
                        min(width, int(round(x2)) + x0),
                        min(height, int(round(y2)) + y0),
                    )
                    if absolute[2] > absolute[0] and absolute[3] > absolute[1]:
                        candidates.append(BallDetection(absolute, float(confidences[index])))
        return self._nms(candidates)

    def _starts(self, length: int) -> list[int]:
        if length <= self.tile_size:
            return [0]
        stride = max(1, int(round(self.tile_size * (1.0 - self.overlap))))
        values = list(range(0, length - self.tile_size + 1, stride))
        end = length - self.tile_size
        if values[-1] != end:
            values.append(end)
        return values

    @staticmethod
    def _is_ball_class(names: Any, class_id: int) -> bool:
        if isinstance(names, dict):
            value = str(names.get(class_id, class_id)).lower()
            return "ball" in value or (len(names) == 1 and class_id == 0)
        if isinstance(names, (list, tuple)):
            value = str(names[class_id]).lower() if 0 <= class_id < len(names) else ""
            return "ball" in value or (len(names) == 1 and class_id == 0)
        return class_id == 0

    def _nms(self, detections: list[BallDetection]) -> list[BallDetection]:
        kept: list[BallDetection] = []
        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            if all(self._iou(detection.bbox, previous.bbox) <= self.nms_iou for previous in kept):
                kept.append(detection)
        return kept

    @staticmethod
    def _iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = first
        bx1, by1, bx2, by2 = second
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if intersection == 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        return intersection / max(area_a + area_b - intersection, 1)


class BallTemporalTracker:
    """Keep one ball track and predict through short detection gaps."""

    def __init__(
        self,
        *,
        track_id: int = 1,
        max_missed: int = 10,
        max_jump: float = 250.0,
        velocity_alpha: float = 0.7,
    ) -> None:
        self.track_id = track_id
        self.max_missed = max_missed
        self.max_jump = max_jump
        self.velocity_alpha = velocity_alpha
        self._center: np.ndarray | None = None
        self._velocity = np.zeros(2, dtype=np.float32)
        self._last_frame: int | None = None
        self._missed = 0

    def update(self, detections: list[BallDetection], frame_index: int) -> BallTrack:
        dt = 1 if self._last_frame is None else max(1, frame_index - self._last_frame)
        predicted = None if self._center is None else self._center + self._velocity * dt
        chosen: BallDetection | None = None

        if detections:
            if predicted is None:
                chosen = max(detections, key=lambda item: item.confidence)
            else:
                chosen = min(
                    detections,
                    key=lambda item: float(np.linalg.norm(np.asarray(item.center) - predicted)),
                )
                distance = float(np.linalg.norm(np.asarray(chosen.center) - predicted))
                if distance > self.max_jump:
                    chosen = None

        if chosen is not None:
            current = np.asarray(chosen.center, dtype=np.float32)
            if self._center is not None:
                measured_velocity = (current - self._center) / dt
                self._velocity = (
                    self.velocity_alpha * self._velocity
                    + (1.0 - self.velocity_alpha) * measured_velocity
                )
            self._center = current
            self._last_frame = frame_index
            self._missed = 0
            return BallTrack(self.track_id, chosen.bbox, chosen.center, chosen.confidence, True, False, 0)

        if predicted is not None and self._missed < self.max_missed:
            self._center = predicted
            self._last_frame = frame_index
            self._missed += 1
            center = (float(predicted[0]), float(predicted[1]))
            return BallTrack(self.track_id, None, center, 0.0, False, True, self._missed)

        self._center = None
        self._velocity[:] = 0.0
        self._last_frame = frame_index
        self._missed = 0
        return BallTrack(self.track_id, None, None, 0.0, False, False, 0)
