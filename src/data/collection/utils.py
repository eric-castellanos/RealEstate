"""Shared helpers for data collection scripts."""

from __future__ import annotations

from pathlib import Path
import re


def project_root() -> Path:
    """Return repository root from this module location."""
    return Path(__file__).resolve().parents[3]


def ensure_dir(path: Path) -> Path:
    """Create directory when missing and return the same path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(value: str) -> str:
    """Convert label text into a filesystem-safe slug."""
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value
