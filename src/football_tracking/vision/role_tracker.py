"""YOLO role detection with persistent BoT-SORT tracking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROLE_IDS = {
    "player": 1,
    "goalkeeper": 2,
    "referee": 3,
}


@dataclass(frozen=True)
class RoleDetection:
    """One player, goalkeeper, or referee detection in frame coordinates."""

    bbox: tuple[int, int, int, int]
    confidence: float
    role_id: int
    role_name: str
    track_id: int | None


class RoleTracker:
    """Run the SoccerNet role detector and persistent BoT-SORT tracker.

    The model is loaded lazily so importing the data package does not require
    Ultralytics. Call :meth:`track_frame` once per video frame in order.
    """

    def __init__(
        self,
        weights: str | Path,
        *,
        tracker: str | Path = "botsort.yaml",
        device: str | int | None = None,
        imgsz: int = 1280,
        candidate_conf: float = 0.15,
        output_conf: float = 0.25,
        iou: float = 0.60,
    ) -> None:
        if imgsz <= 0:
            raise ValueError("imgsz must be positive")
        if not 0.0 <= candidate_conf <= 1.0:
            raise ValueError("candidate_conf must be in [0, 1]")
        if not 0.0 <= output_conf <= 1.0:
            raise ValueError("output_conf must be in [0, 1]")
        if not 0.0 <= iou <= 1.0:
            raise ValueError("iou must be in [0, 1]")

        self.weights = Path(weights)
        self.tracker = str(tracker)
        self.device = device
        self.imgsz = imgsz
        self.candidate_conf = candidate_conf
        self.output_conf = output_conf
        self.iou = iou
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            if not self.weights.is_file():
                raise FileNotFoundError(f"Role detector weights not found: {self.weights}")
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "Ultralytics is required for role inference. "
                    "Install the training dependencies first."
                ) from exc
            self._model = YOLO(str(self.weights))
        return self._model

    def track_frame(self, frame: np.ndarray) -> list[RoleDetection]:
        """Detect and track roles in one frame.

        Frames must be passed in chronological order. The underlying tracker
        keeps its state through ``persist=True``.
        """

        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError("frame must be a color image as a HxWxC numpy array")

        kwargs: dict[str, Any] = {
            "source": frame,
            "imgsz": self.imgsz,
            "conf": self.candidate_conf,
            "iou": self.iou,
            "tracker": self.tracker,
            "persist": True,
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device

        result = self.model.track(**kwargs)[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        ids_tensor = getattr(boxes, "id", None)
        track_ids = ids_tensor.cpu().numpy().astype(int) if ids_tensor is not None else None
        names = getattr(result, "names", getattr(self.model, "names", {}))

        detections: list[RoleDetection] = []
        for index, box in enumerate(xyxy):
            confidence = float(confidences[index])
            if confidence < self.output_conf:
                continue

            class_id = int(classes[index])
            role_name = self._class_name(names, class_id)
            role_id = ROLE_IDS.get(role_name)
            if role_id is None:
                continue

            x1, y1, x2, y2 = (int(round(float(value))) for value in box)
            track_id = int(track_ids[index]) if track_ids is not None else None
            detections.append(
                RoleDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    role_id=role_id,
                    role_name=role_name,
                    track_id=track_id,
                )
            )
        return detections

    @staticmethod
    def _class_name(names: Any, class_id: int) -> str:
        if isinstance(names, dict):
            value = names.get(class_id, class_id)
        elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            value = names[class_id]
        else:
            value = class_id
        return str(value).strip().lower()
