"""Render contiguous one-second YOLO preview clips for visual QA."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from football_tracking.data.paths import resolve_project_output_path


CLASS_NAMES = {
    0: "ball",
    1: "player",
    2: "goalkeeper",
    3: "referee",
}
CLASS_COLORS = {
    0: (255, 180, 0),
    1: (0, 220, 80),
    2: (40, 140, 255),
    3: (255, 70, 70),
}
Clip = Tuple[str, int, int, List[Path]]


def _parse_labels(label_path: Path, image_width: int, image_height: int) -> Tuple[List[Dict], List[str]]:
    labels: List[Dict] = []
    issues: List[str] = []
    if not label_path.exists():
        return labels, ["missing label file"]
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        values = line.split()
        if len(values) != 5:
            issues.append("line %d: expected 5 fields" % line_number)
            continue
        try:
            class_id = int(values[0])
            center_x, center_y, width, height = [float(value) for value in values[1:]]
        except ValueError:
            issues.append("line %d: non-numeric value" % line_number)
            continue
        if class_id not in CLASS_NAMES:
            issues.append("line %d: unknown class %d" % (line_number, class_id))
        if width <= 0 or height <= 0:
            issues.append("line %d: non-positive bbox size" % line_number)
        if not all(0.0 <= value <= 1.0 for value in (center_x, center_y, width, height)):
            issues.append("line %d: value outside [0, 1]" % line_number)
        x1 = (center_x - width / 2.0) * image_width
        y1 = (center_y - height / 2.0) * image_height
        x2 = (center_x + width / 2.0) * image_width
        y2 = (center_y + height / 2.0) * image_height
        epsilon = 1e-6 * max(image_width, image_height)
        if x1 < -epsilon or y1 < -epsilon or x2 > image_width + epsilon or y2 > image_height + epsilon:
            issues.append("line %d: bbox exceeds image boundary" % line_number)
        labels.append({"class_id": class_id, "box": (x1, y1, x2, y2)})
    return labels, issues


def _draw_preview(image_path: Path, label_path: Path, title: str) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    labels, _ = _parse_labels(label_path, image.width, image.height)
    for label in labels:
        class_id = label["class_id"]
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        x1, y1, x2, y2 = label["box"]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=max(2, image.width // 500))
        text = CLASS_NAMES.get(class_id, "unknown")
        text_box = draw.textbbox((x1, max(0, y1 - 14)), text, font=font)
        draw.rectangle(text_box, fill=color)
        draw.text((x1, max(0, y1 - 14)), text, fill=(0, 0, 0), font=font)
    draw.rectangle((0, 0, min(image.width, 700), 28), fill=(0, 0, 0))
    draw.text((8, 7), title, fill=(255, 255, 255), font=font)
    return image


def _sequence_and_frame(path: Path) -> Tuple[str, int]:
    sequence, frame_text = path.stem.rsplit("_", 1)
    return sequence, int(frame_text)


def _contiguous_clip_candidates(image_paths: Sequence[Path], frames_per_clip: int) -> List[Clip]:
    grouped: Dict[str, List[Tuple[int, Path]]] = defaultdict(list)
    for image_path in image_paths:
        try:
            sequence, frame_number = _sequence_and_frame(image_path)
        except (TypeError, ValueError):
            continue
        grouped[sequence].append((frame_number, image_path))

    candidates: List[Clip] = []
    for sequence in sorted(grouped):
        frames = sorted(grouped[sequence], key=lambda item: item[0])
        run: List[Tuple[int, Path]] = []
        runs: List[List[Tuple[int, Path]]] = []
        previous_frame = None
        for frame in frames:
            if previous_frame is None or frame[0] == previous_frame + 1:
                run.append(frame)
            else:
                if run:
                    runs.append(run)
                run = [frame]
            previous_frame = frame[0]
        if run:
            runs.append(run)

        for run in runs:
            if len(run) < frames_per_clip:
                continue
            start = (len(run) - frames_per_clip) // 2
            clip_frames = run[start : start + frames_per_clip]
            candidates.append(
                (
                    sequence,
                    clip_frames[0][0],
                    clip_frames[-1][0],
                    [path for _, path in clip_frames],
                )
            )
    return candidates


def _select_clips(image_paths: Sequence[Path], clips_per_split: int, frames_per_clip: int) -> List[Clip]:
    candidates = _contiguous_clip_candidates(image_paths, frames_per_clip)
    if not candidates:
        return []
    if len(candidates) <= clips_per_split:
        return candidates
    indexes = [round(index * (len(candidates) - 1) / float(clips_per_split - 1)) for index in range(clips_per_split)]
    return [candidates[index] for index in indexes]


def _render_contact_sheet(image_paths: Sequence[Path], output_path: Path, images_root: Path, labels_root: Path) -> None:
    columns = 5
    tile_width, tile_height = 384, 240
    rows = int(math.ceil(len(image_paths) / float(columns)))
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), (25, 25, 25))
    for index, image_path in enumerate(image_paths):
        relative = image_path.relative_to(images_root)
        label_path = labels_root / relative.with_suffix(".txt")
        title = "%s / %s" % (relative.parent.name, relative.stem)
        image = _draw_preview(image_path, label_path, title)
        image.thumbnail((tile_width - 8, tile_height - 8))
        x = (index % columns) * tile_width + (tile_width - image.width) // 2
        y = (index // columns) * tile_height + (tile_height - image.height) // 2
        sheet.paste(image, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def run_visual_qa(
    yolo_root: Path,
    output_root: Path,
    clips_per_split: int = 2,
    frames_per_clip: int = 25,
) -> Dict:
    yolo_root = Path(yolo_root).resolve()
    project_root = Path(__file__).resolve().parents[2]
    output_root = resolve_project_output_path(output_root, project_root)
    report = {
        "yolo_root": str(yolo_root),
        "output_root": str(output_root),
        "fps_assumption": 25,
        "clips_per_split": clips_per_split,
        "frames_per_clip": frames_per_clip,
        "splits": {},
        "valid": True,
    }

    # This visual QA intentionally inspects only train and only sampled clips.
    for split in ("train",):
        images_root = yolo_root / "images" / split
        labels_root = yolo_root / "labels" / split
        image_paths = sorted(images_root.glob("*.jpg"))
        label_paths = sorted(labels_root.glob("*.txt"))
        label_names = {path.stem for path in label_paths}
        selected_clips = _select_clips(image_paths, clips_per_split, frames_per_clip)
        sampled_paths = [path for _, _, _, paths in selected_clips for path in paths]
        class_counts: Counter = Counter()
        issues: List[Dict] = []
        empty_labels = 0

        for image_path in sampled_paths:
            label_path = labels_root / (image_path.stem + ".txt")
            if not label_path.exists():
                issues.append({"image": image_path.name, "issue": "missing label file"})
                continue
            with Image.open(image_path) as image:
                labels, label_issues = _parse_labels(label_path, image.width, image.height)
            if not labels:
                empty_labels += 1
            for label in labels:
                class_counts[CLASS_NAMES.get(label["class_id"], "unknown")] += 1
            for issue in label_issues:
                issues.append({"image": image_path.name, "issue": issue})

        orphan_labels = sorted(label_names - {path.stem for path in image_paths})
        for stem in orphan_labels:
            issues.append({"label": stem + ".txt", "issue": "label has no matching image"})

        preview_paths: List[str] = []
        clip_reports: List[Dict] = []
        for clip_index, (sequence, start_frame, end_frame, clip_paths) in enumerate(selected_clips, start=1):
            preview_path = output_root / ("%s_clip_%02d_preview.jpg" % (split, clip_index))
            _render_contact_sheet(clip_paths, preview_path, images_root, labels_root)
            preview_paths.append(str(preview_path))
            clip_reports.append(
                {
                    "clip": clip_index,
                    "sequence": sequence,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "num_frames": len(clip_paths),
                    "preview": str(preview_path),
                }
            )

        split_report = {
            "num_images": len(image_paths),
            "num_labels": len(label_paths),
            "num_sampled_images": len(sampled_paths),
            "validation_scope": "sampled contiguous clips only",
            "empty_labels_in_sample": empty_labels,
            "class_counts_in_sample": dict(sorted(class_counts.items())),
            "num_issues": len(issues),
            "issues_first_20": issues[:20],
            "clips": clip_reports,
            "previews": preview_paths,
        }
        report["splits"][split] = split_report
        if issues or len(selected_clips) < clips_per_split:
            report["valid"] = False

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "qa_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo-root", type=Path, default=Path("yolo"))
    parser.add_argument("--output", type=Path, default=Path("outputs/visualizations/yolo_qa"))
    parser.add_argument("--clips-per-split", type=int, default=2)
    parser.add_argument("--frames-per-clip", type=int, default=25)
    args = parser.parse_args()
    report = run_visual_qa(args.yolo_root, args.output, args.clips_per_split, args.frames_per_clip)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
