"""Train the dedicated one-class ball detector."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--data", type=Path, default=Path("ball_dataset/dataset.yaml"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", type=Path, default=Path("outputs/runs/ball"))
    parser.add_argument("--name", default="ball_yolo11s_640")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.data.is_file():
        raise SystemExit(f"Ball dataset YAML not found: {args.data}")
    if args.epochs <= 0 or args.imgsz <= 0 or args.batch <= 0 or args.workers < 0:
        raise SystemExit("epochs, imgsz, and batch must be positive; workers cannot be negative")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install the training dependencies before training the ball model.") from exc

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        pretrained=True,
        seed=args.seed,
        cache=False,
        plots=True,
        save=True,
        val=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
