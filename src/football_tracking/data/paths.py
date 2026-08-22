"""Safe project path helpers."""

from __future__ import annotations

from pathlib import Path


def resolve_project_output_path(candidate: Path, project_root: Path) -> Path:
    """Resolve an output path and keep it inside the project root."""

    root = Path(project_root).resolve()
    raw_path = Path(candidate)
    resolved = (root / raw_path if not raw_path.is_absolute() else raw_path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("output path must stay inside project root: %s" % candidate)
    return resolved
