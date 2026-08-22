"""Create a sliced-inference training set from the existing YOLO ball labels."""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def _starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    values = list(range(0, length - tile_size + 1, stride))
    end = length - tile_size
    if values[-1] != end:
        values.append(end)
    return values


def _read_labels(path: Path, width: int, height: int) -> list[tuple[int, float, float, float, float]]:
    labels = []
    if not path.is_file():
        return labels
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        class_id, cx, cy, box_w, box_h = (float(value) for value in fields)
        if int(class_id) != 0:
            continue
        x1 = (cx - box_w / 2.0) * width
        y1 = (cy - box_h / 2.0) * height
        x2 = (cx + box_w / 2.0) * width
        y2 = (cy + box_h / 2.0) * height
        labels.append((0, x1, y1, x2, y2))
    return labels


def _tile_labels(
    labels: list[tuple[int, float, float, float, float]],
    x0: int,
    y0: int,
    tile_size: int,
    width: int,
    height: int,
) -> list[str]:
    result = []
    x1_tile, y1_tile = x0, y0
    x2_tile, y2_tile = min(x0 + tile_size, width), min(y0 + tile_size, height)
    actual_w, actual_h = x2_tile - x1_tile, y2_tile - y1_tile
    for class_id, x1, y1, x2, y2 in labels:
        center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if not (x1_tile <= center_x < x2_tile and y1_tile <= center_y < y2_tile):
            continue
        clipped_x1 = max(x1, x1_tile) - x1_tile
        clipped_y1 = max(y1, y1_tile) - y1_tile
        clipped_x2 = min(x2, x2_tile) - x1_tile
        clipped_y2 = min(y2, y2_tile) - y1_tile
        if clipped_x2 <= clipped_x1 or clipped_y2 <= clipped_y1:
            continue
        cx = ((clipped_x1 + clipped_x2) / 2.0) / actual_w
        cy = ((clipped_y1 + clipped_y2) / 2.0) / actual_h
        bw = (clipped_x2 - clipped_x1) / actual_w
        bh = (clipped_y2 - clipped_y1) / actual_h
        result.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return result


def build_split(
    images_dir: Path,
    labels_dir: Path,
    output_root: Path,
    *,
    tile_size: int,
    overlap: float,
    negative_ratio: float,
    rng: random.Random,
) -> int:
    import cv2

    output_images = output_root / "images"
    output_labels = output_root / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    created = 0
    for image_path in sorted(images_dir.glob("*")):
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        labels = _read_labels(labels_dir / f"{image_path.stem}.txt", width, height)
        for x0 in _starts(width, tile_size, overlap):
            for y0 in _starts(height, tile_size, overlap):
                tile_labels = _tile_labels(labels, x0, y0, tile_size, width, height)
                if not tile_labels and rng.random() > negative_ratio:
                    continue
                tile = frame[y0 : min(y0 + tile_size, height), x0 : min(x0 + tile_size, width)]
                name = f"{image_path.stem}_x{x0}_y{y0}"
                if not cv2.imwrite(str(output_images / f"{name}{image_path.suffix}"), tile):
                    raise RuntimeError(f"Could not write tile for {image_path}")
                (output_labels / f"{name}.txt").write_text(
                    "\n".join(tile_labels) + ("\n" if tile_labels else ""),
                    encoding="utf-8",
                )
                created += 1
    return created


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("yolo"))
    parser.add_argument("--output-root", type=Path, default=Path("ball_dataset"))
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--negative-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.tile_size <= 0 or not 0.0 <= args.overlap < 1.0 or not 0.0 <= args.negative_ratio <= 1.0:
        raise SystemExit("tile-size must be positive; overlap and negative-ratio must be in [0, 1]")
    rng = random.Random(args.seed)
    total = 0
    for split in ("train", "val", "test"):
        total += build_split(
            args.dataset_root / "images" / split,
            args.dataset_root / "labels" / split,
            args.output_root / split,
            tile_size=args.tile_size,
            overlap=args.overlap,
            negative_ratio=args.negative_ratio,
            rng=rng,
        )
    (args.output_root / "dataset.yaml").write_text(
        "train: train/images\nval: val/images\ntest: test/images\nnames:\n  0: ball\n",
        encoding="utf-8",
    )
    print(f"Created {total} ball tiles under {args.output_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
