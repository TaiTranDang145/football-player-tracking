"""Validation utilities for the raw SoccerNet Tracking 2023 dataset."""

from __future__ import annotations

import configparser
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple


ROLE_NAMES = {
    0: "ball",
    1: "player",
    2: "goalkeeper",
    3: "referee",
    4: "other",
}

TEAM_NAMES = {
    0: "team_left",
    1: "team_right",
    2: "neutral",
}


def _issue(
    severity: str,
    sequence: str,
    message: str,
    line: Optional[int] = None,
) -> Dict:
    result = {
        "severity": severity,
        "sequence": sequence,
        "message": message,
    }
    if line is not None:
        result["line"] = line
    return result


def _read_key_values(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _parse_seqinfo(path: Path) -> Tuple[Dict, List[Dict]]:
    issues: List[Dict] = []
    parser = configparser.ConfigParser()
    try:
        parser.read(str(path), encoding="utf-8")
        section = parser["Sequence"]
        meta = {
            "name": section.get("name", path.parent.name).strip(),
            "image_dir": section.get("imDir", "img1").strip(),
            "image_extension": section.get("imExt", ".jpg").strip(),
            "frame_rate": section.getfloat("frameRate"),
            "sequence_length": section.getint("seqLength"),
            "image_width": section.getint("imWidth"),
            "image_height": section.getint("imHeight"),
        }
        if not meta["image_extension"].startswith("."):
            meta["image_extension"] = "." + meta["image_extension"]
        if meta["sequence_length"] <= 0 or meta["image_width"] <= 0 or meta["image_height"] <= 0:
            issues.append(_issue("error", path.parent.name, "seqinfo.ini contains non-positive dimensions or length"))
        return meta, issues
    except (configparser.Error, KeyError, ValueError) as exc:
        return {}, [_issue("error", path.parent.name, "cannot parse seqinfo.ini: %s" % exc)]


def _normalize_tracklet(raw_value: str) -> Tuple[int, int]:
    description = re.sub(r"\s+", " ", raw_value.split(";", 1)[0].strip().lower())
    if description == "ball":
        return 0, 2
    if description == "referee":
        return 3, 2
    if description == "other":
        return 4, 2
    if "goalkeeper" in description:
        role_id = 2
    elif "player" in description:
        role_id = 1
    else:
        raise ValueError("unknown role %r" % raw_value)
    if "team left" in description:
        return role_id, 0
    if "team right" in description:
        return role_id, 1
    raise ValueError("team is missing from role %r" % raw_value)


def _parse_gameinfo(path: Path, sequence: str) -> Tuple[str, Dict[int, Dict], List[Dict]]:
    values = _read_key_values(path)
    issues: List[Dict] = []
    game_id = values.get("gameID", "").strip()
    if not game_id:
        issues.append(_issue("error", sequence, "gameinfo.ini is missing gameID"))
    tracklets: Dict[int, Dict] = {}
    pattern = re.compile(r"^trackletID_(\d+)$", re.IGNORECASE)
    for key, raw_value in values.items():
        match = pattern.match(key)
        if not match:
            continue
        track_id = int(match.group(1))
        try:
            role_id, team_id = _normalize_tracklet(raw_value)
            tracklets[track_id] = {
                "track_id": track_id,
                "raw_label": raw_value,
                "role_id": role_id,
                "role_name": ROLE_NAMES[role_id],
                "team_id": team_id,
                "team_name": TEAM_NAMES[team_id],
            }
        except ValueError as exc:
            issues.append(_issue("error", sequence, "trackletID_%s: %s" % (track_id, exc)))
    if not tracklets:
        issues.append(_issue("error", sequence, "gameinfo.ini contains no valid trackletID entries"))
    return game_id, tracklets, issues


def _frame_path(sequence_dir: Path, meta: Mapping, frame_id: int) -> Path:
    return sequence_dir / meta["image_dir"] / ("%06d%s" % (frame_id, meta["image_extension"]))


def _validate_sequence(sequence_dir: Path) -> Dict:
    sequence_name = sequence_dir.name
    issues: List[Dict] = []
    required = {
        "seqinfo.ini": sequence_dir / "seqinfo.ini",
        "gameinfo.ini": sequence_dir / "gameinfo.ini",
        "gt.txt": sequence_dir / "gt" / "gt.txt",
        "image_dir": sequence_dir / "img1",
    }
    for label, path in required.items():
        if not path.exists():
            issues.append(_issue("error", sequence_name, "missing %s: %s" % (label, path)))
    if issues:
        return {
            "sequence_name": sequence_name,
            "game_id": None,
            "issues": issues,
            "num_annotations": 0,
            "class_counts": {},
        }

    meta, seqinfo_issues = _parse_seqinfo(required["seqinfo.ini"])
    issues.extend(seqinfo_issues)
    game_id, tracklets, gameinfo_issues = _parse_gameinfo(required["gameinfo.ini"], sequence_name)
    issues.extend(gameinfo_issues)
    if not meta or not tracklets:
        return {
            "sequence_name": sequence_name,
            "game_id": game_id or None,
            "issues": issues,
            "num_annotations": 0,
            "class_counts": {},
        }

    class_counts: Counter = Counter()
    team_counts: Counter = Counter()
    referenced_frames = set()
    annotation_count = 0
    gt_path = required["gt.txt"]
    for line_number, raw_line in enumerate(gt_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [value.strip() for value in line.split(",")]
        if len(fields) < 7:
            issues.append(_issue("error", sequence_name, "gt.txt row has fewer than 7 fields", line_number))
            continue
        try:
            frame_id = int(float(fields[0]))
            track_id = int(float(fields[1]))
            x, y, width, height = (float(fields[index]) for index in range(2, 6))
            float(fields[6])
        except ValueError:
            issues.append(_issue("error", sequence_name, "gt.txt row contains a non-numeric value", line_number))
            continue
        annotation_count += 1
        referenced_frames.add(frame_id)
        if frame_id < 1 or frame_id > meta["sequence_length"]:
            issues.append(_issue("error", sequence_name, "frame_id outside seqLength", line_number))
        if track_id not in tracklets:
            issues.append(_issue("error", sequence_name, "track_id=%s missing from gameinfo.ini" % track_id, line_number))
            continue
        if width <= 0 or height <= 0:
            issues.append(_issue("error", sequence_name, "bbox width/height must be positive", line_number))
        if x < 0 or y < 0:
            issues.append(_issue("error", sequence_name, "bbox x/y must not be negative", line_number))
        if x + width > meta["image_width"] or y + height > meta["image_height"]:
            issues.append(_issue("warning", sequence_name, "bbox extends beyond image boundary", line_number))
        tracklet = tracklets[track_id]
        class_counts[tracklet["role_name"]] += 1
        team_counts[tracklet["team_name"]] += 1

    missing_frames = []
    for frame_id in sorted(referenced_frames):
        if not _frame_path(sequence_dir, meta, frame_id).exists():
            missing_frames.append(frame_id)
    if missing_frames:
        issues.append(_issue("error", sequence_name, "missing image frames: %s" % missing_frames[:20]))

    return {
        "sequence_name": sequence_name,
        "game_id": game_id,
        "meta": meta,
        "num_annotations": annotation_count,
        "num_referenced_frames": len(referenced_frames),
        "class_counts": dict(sorted(class_counts.items())),
        "team_counts": dict(sorted(team_counts.items())),
        "num_tracklets": len(tracklets),
        "issues": issues,
    }


def discover_sequences(source_root: Path) -> List[Path]:
    """Find sequence folders that contain a seqinfo.ini file."""

    return sorted(path.parent for path in Path(source_root).rglob("seqinfo.ini"))


def validate_dataset(source_root: Path) -> Dict:
    """Validate all sequences below ``source_root`` and return a JSON report."""

    source_root = Path(source_root).resolve()
    if not source_root.exists():
        return {
            "valid": False,
            "source_root": str(source_root),
            "sequences": [],
            "issues": [_issue("error", "", "source root does not exist")],
        }
    sequence_dirs = discover_sequences(source_root)
    if not sequence_dirs:
        return {
            "valid": False,
            "source_root": str(source_root),
            "sequences": [],
            "issues": [_issue("error", "", "no seqinfo.ini files found")],
        }

    sequences = [_validate_sequence(path) for path in sequence_dirs]
    issues = [issue for sequence in sequences for issue in sequence["issues"]]
    class_counts: Counter = Counter()
    team_counts: Counter = Counter()
    for sequence in sequences:
        class_counts.update(sequence.get("class_counts", {}))
        team_counts.update(sequence.get("team_counts", {}))
    error_count = sum(issue["severity"] == "error" for issue in issues)
    warning_count = sum(issue["severity"] == "warning" for issue in issues)
    return {
        "valid": error_count == 0,
        "source_root": str(source_root),
        "num_sequences": len(sequences),
        "num_annotations": sum(sequence.get("num_annotations", 0) for sequence in sequences),
        "class_counts": dict(sorted(class_counts.items())),
        "team_counts": dict(sorted(team_counts.items())),
        "error_count": error_count,
        "warning_count": warning_count,
        "sequences": sequences,
        "issues": issues,
    }


def write_report(report: Mapping, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

