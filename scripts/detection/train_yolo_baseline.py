"""Train a small Ultralytics YOLO baseline for pipeline verification."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional, Union

from tqdm import tqdm


class TrainingProgress:
    """Display epoch and batch progress through Ultralytics callbacks."""

    def __init__(self) -> None:
        self.epoch_bar: Optional[Any] = None
        self.batch_bar: Optional[Any] = None
        self.batch_completed = 0

    @staticmethod
    def _loss(trainer: Any) -> Optional[float]:
        """Return the current scalar loss when Ultralytics exposes one."""
        value = getattr(trainer, "loss", None)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def on_train_start(self, trainer: Any) -> None:
        total_epochs = int(getattr(trainer, "epochs", 0) or 0)
        self.epoch_bar = tqdm(
            total=total_epochs,
            desc="YOLO training",
            unit="epoch",
            position=0,
            dynamic_ncols=True,
        )

    def on_train_epoch_start(self, trainer: Any) -> None:
        if self.batch_bar is not None:
            self.batch_bar.close()

        train_loader = getattr(trainer, "train_loader", None)
        total_batches = len(train_loader) if train_loader is not None else int(getattr(trainer, "nb", 0) or 0)
        epoch_number = int(getattr(trainer, "epoch", 0)) + 1
        total_epochs = int(getattr(trainer, "epochs", 0) or 0)
        self.batch_completed = 0
        self.batch_bar = tqdm(
            total=total_batches,
            desc="Epoch %d/%d" % (epoch_number, total_epochs),
            unit="batch",
            position=1,
            leave=False,
            dynamic_ncols=True,
        )

    def on_train_batch_end(self, trainer: Any) -> None:
        if self.batch_bar is None:
            return

        batch_number = getattr(trainer, "batch_i", None)
        completed = int(batch_number) + 1 if batch_number is not None else self.batch_completed + 1
        self.batch_bar.update(max(0, completed - self.batch_completed))
        self.batch_completed = completed

        loss = self._loss(trainer)
        if loss is not None:
            self.batch_bar.set_postfix(loss="%.4f" % loss)

    def on_train_epoch_end(self, trainer: Any) -> None:
        if self.epoch_bar is not None:
            loss = self._loss(trainer)
            if loss is not None:
                self.epoch_bar.set_postfix(loss="%.4f" % loss)
            self.epoch_bar.update(1)

    def close(self, _trainer: Any = None) -> None:
        if self.batch_bar is not None:
            self.batch_bar.close()
            self.batch_bar = None
        if self.epoch_bar is not None:
            self.epoch_bar.close()
            self.epoch_bar = None


def _resolve_device(requested: str) -> Union[int, str]:
    """Resolve ``auto`` without forcing GPU usage on CPU-only machines."""
    if requested != "auto":
        return requested

    try:
        import torch
    except ImportError:
        return "cpu"
    return 0 if torch.cuda.is_available() else "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Ultralytics model checkpoint or model YAML (default: yolo11n.pt).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("yolo/dataset.yaml"),
        help="YOLO dataset YAML path.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Smoke-test epoch count; use a value from 5 to 10 (default: 5).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size for the quick baseline (default: 640).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch size; lower this if GPU memory is insufficient (default: 4).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device: auto, cpu, 0, or a CUDA device string (default: auto).",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", type=Path, default=Path("outputs/runs/detection"))
    parser.add_argument("--name", default="yolo_baseline_smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early-stopping patience; it will not shorten a 5-epoch smoke run by default.",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow Ultralytics to reuse the output directory.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not 5 <= args.epochs <= 10:
        raise SystemExit("--epochs must be between 5 and 10 for this baseline.")
    if args.imgsz <= 0 or args.batch <= 0 or args.workers < 0:
        raise SystemExit("--imgsz and --batch must be positive; --workers cannot be negative.")

    data_path = args.data.resolve()
    if not data_path.is_file():
        raise SystemExit("Dataset YAML not found: %s" % data_path)

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit(
            "Ultralytics is not installed. Install it before running this script: "
            "pip install ultralytics"
        ) from error

    model = YOLO(args.model)
    progress = TrainingProgress()
    model.add_callback("on_train_start", progress.on_train_start)
    model.add_callback("on_train_epoch_start", progress.on_train_epoch_start)
    model.add_callback("on_train_batch_end", progress.on_train_batch_end)
    model.add_callback("on_train_epoch_end", progress.on_train_epoch_end)
    model.add_callback("on_train_end", progress.close)
    model.add_callback("teardown", progress.close)

    try:
        results = model.train(
            data=str(data_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=_resolve_device(args.device),
            workers=args.workers,
            project=str(args.project.resolve()),
            name=args.name,
            pretrained=True,
            seed=args.seed,
            patience=args.patience,
            cache=False,
            plots=True,
            save=True,
            val=True,
            exist_ok=args.exist_ok,
            verbose=True,
        )
    finally:
        progress.close()

    print("Training finished. Results: %s" % results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
