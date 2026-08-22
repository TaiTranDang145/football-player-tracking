"""Convert the raw SoccerNet MOT layout to the standard YOLO layout."""

from __future__ import annotations

import configparser
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .validator import (
    ROLE_NAMES,
    _frame_path,
    _parse_gameinfo,
    _parse_seqinfo,
    discover_sequences,
)


YOLO_CLASSES = {
    0: "ball",
    1: "player",
    2: "goalkeeper",
    3: "referee",
}


def _split_sequences(sequence_infos: Sequence[Mapping], seed: int = 42) -> Dict[str, str]:
    """Split whole game groups, keeping every sequence from a game together."""

    groups: Dict[str, List[str]] = defaultdict(list)
    for info in sequence_infos:
        groups[str(info["game_id"])].append(str(info["sequence_name"]))
    group_names = list(groups)
    random.Random(seed).shuffle(group_names)
    count = len(group_names)
    if count == 0:
        return {}
    if count >= 3:
        counts = [max(1, int(count * ratio)) for ratio in (0.70, 0.15, 0.15)]
        while sum(counts) > count:
            index = max(range(3), key=lambda item: counts[item])
            if counts[index] > 1:
                counts[index] -= 1
            else:
                break
        while sum(counts) < count:
            counts[0] += 1
    else:
        counts = [count, 0, 0]
    result: Dict[str, str] = {}
    cursor = 0
    for split, amount in zip(("train", "val", "test"), counts):
        for game_id in group_names[cursor : cursor + amount]:
            for sequence_name in groups[game_id]:
                result[sequence_name] = split
        cursor += amount
    return result


def _read_annotations(sequence_dir: Path, meta: Mapping, tracklets: Mapping[int, Mapping]) -> Tuple[Dict[int, List[Dict]], int]:
    annotations_by_frame: Dict[int, List[Dict]] = defaultdict(list)
    skipped_invalid = 0
    gt_path = sequence_dir / "gt" / "gt.txt"
    for raw_line in gt_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [value.strip() for value in line.split(",")]
        if len(fields) < 7:
            skipped_invalid += 1
            continue
        try:
            frame_id = int(float(fields[0]))
            track_id = int(float(fields[1]))
            x, y, width, height = (float(fields[index]) for index in range(2, 6))
        except ValueError:
            skipped_invalid += 1
            continue
        if frame_id < 1 or frame_id > int(meta["sequence_length"]):
            skipped_invalid += 1
            continue
        if width <= 0 or height <= 0:
            skipped_invalid += 1
            continue
        tracklet = tracklets.get(track_id)
        if tracklet is None or int(tracklet["role_id"]) not in YOLO_CLASSES:
            continue
        x1 = max(0.0, min(float(meta["image_width"]), x))
        y1 = max(0.0, min(float(meta["image_height"]), y))
        x2 = max(0.0, min(float(meta["image_width"]), x + width))
        y2 = max(0.0, min(float(meta["image_height"]), y + height))
        if x2 <= x1 or y2 <= y1:
            skipped_invalid += 1
            continue
        annotations_by_frame[frame_id].append(
            {
                "role_id": int(tracklet["role_id"]),
                "bbox_xyxy": (x1, y1, x2, y2),
            }
        )
    return annotations_by_frame, skipped_invalid


def _to_yolo_line(annotation: Mapping, image_width: int, image_height: int) -> str:
    x1, y1, x2, y2 = annotation["bbox_xyxy"]
    width = x2 - x1
    height = y2 - y1
    center_x = (x1 + x2) / 2.0 / image_width
    center_y = (y1 + y2) / 2.0 / image_height
    normalized_width = width / image_width
    normalized_height = height / image_height
    return "%d %.8f %.8f %.8f %.8f" % (
        annotation["role_id"],
        center_x,
        center_y,
        normalized_width,
        normalized_height,
    )


def _materialize_image(source: Path, destination: Path, image_mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    if image_mode == "hardlink":
        os.link(str(source), str(destination))
    elif image_mode == "symlink":
        destination.symlink_to(source)
    elif image_mode == "copy":
        shutil.copy2(str(source), str(destination))
    else:
        raise ValueError("image_mode must be hardlink, symlink, or copy")


def _write_dataset_yaml(output_root: Path) -> None:
    (output_root / "dataset.yaml").write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: ball\n"
        "  1: player\n"
        "  2: goalkeeper\n"
        "  3: referee\n",
        encoding="utf-8",
    )


def export_yolo_dataset(
    source_root: Path,
    output_root: Path,
    seed: int = 42,
    image_mode: str = "hardlink",
) -> Dict:
    """Export raw annotations and frames into a YOLO dataset."""

    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    sequence_dirs = discover_sequences(source_root)
    if not sequence_dirs:
        raise ValueError("no sequences found below %s" % source_root)

    loaded = []
    for sequence_dir in sequence_dirs:
        seqinfo_path = sequence_dir / "seqinfo.ini"
        gameinfo_path = sequence_dir / "gameinfo.ini"
        meta, seqinfo_issues = _parse_seqinfo(seqinfo_path)
        game_id, tracklets, gameinfo_issues = _parse_gameinfo(gameinfo_path, sequence_dir.name)
        errors = [issue for issue in seqinfo_issues + gameinfo_issues if issue["severity"] == "error"]
        if errors:
            raise ValueError("cannot export %s: %s" % (sequence_dir.name, errors[0]["message"]))
        loaded.append(
            {
                "sequence_dir": sequence_dir,
                "sequence_name": meta["name"],
                "game_id": game_id,
                "meta": meta,
                "tracklets": tracklets,
            }
        )
    split_by_sequence = _split_sequences(loaded, seed=seed)

    for split in ("train", "val", "test"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    class_counts: Counter = Counter()
    skipped_invalid = 0
    image_counts: Counter = Counter()
    for info in loaded:
        split = split_by_sequence[info["sequence_name"]]
        annotations_by_frame, skipped = _read_annotations(
            info["sequence_dir"], info["meta"], info["tracklets"]
        )
        skipped_invalid += skipped
        for frame_id in range(1, int(info["meta"]["sequence_length"]) + 1):
            source_image = _frame_path(info["sequence_dir"], info["meta"], frame_id)
            if not source_image.exists():
                raise FileNotFoundError("missing frame: %s" % source_image)
            stem = "%s_%06d" % (info["sequence_name"], frame_id)
            output_image = output_root / "images" / split / (stem + info["meta"]["image_extension"])
            output_label = output_root / "labels" / split / (stem + ".txt")
            _materialize_image(source_image, output_image, image_mode)
            annotations = annotations_by_frame.get(frame_id, [])
            lines = [
                _to_yolo_line(annotation, int(info["meta"]["image_width"]), int(info["meta"]["image_height"]))
                for annotation in annotations
            ]
            output_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            image_counts[split] += 1
            for annotation in annotations:
                class_counts[(split, YOLO_CLASSES[annotation["role_id"]])] += 1

    _write_dataset_yaml(output_root)
    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "seed": seed,
        "image_mode": image_mode,
        "num_sequences": len(loaded),
        "num_images_by_split": dict(sorted(image_counts.items())),
        "num_objects_by_split_and_class": {
            "%s/%s" % (split, class_name): count
            for (split, class_name), count in sorted(class_counts.items())
        },
        "skipped_invalid_annotations": skipped_invalid,
        "class_names": YOLO_CLASSES,
    }
    (output_root / "export_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary

