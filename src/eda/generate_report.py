"""Generate an automated EDA report from processed parquet data."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

PROCESSED_GLOB = "data/processed/*.parquet"
REPORT_PATH = "reports/eda_report.html"
SAMPLE_SIZE = int(os.getenv("EDA_SAMPLE_SIZE", "5000"))
RANDOM_SEED = 42
UNIQUE_RATIO_DROP_THRESHOLD = float(os.getenv("EDA_UNIQUE_RATIO_DROP_THRESHOLD", "0.98"))


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_processed_data(root: Path) -> pd.DataFrame:
    files = sorted(root.glob(PROCESSED_GLOB))
    if not files:
        raise FileNotFoundError(f"No processed parquet files found using pattern: {PROCESSED_GLOB}")

    LOGGER.info("Found %s processed parquet files", len(files))
    frames: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_parquet(path)
        frame["processed_source_file"] = str(path.relative_to(root))
        frames.append(frame)
        LOGGER.info("Loaded %s rows from %s", len(frame), path.relative_to(root))

    combined = pd.concat(frames, ignore_index=True, sort=False)
    LOGGER.info("Combined %s rows from processed data", len(combined))
    return combined


def sample_data(frame: pd.DataFrame, sample_size: int = SAMPLE_SIZE) -> pd.DataFrame:
    if len(frame) <= sample_size:
        LOGGER.info("Dataset has %s rows; using full dataset (no sampling)", len(frame))
        return frame

    sampled = frame.sample(n=sample_size, random_state=RANDOM_SEED)
    LOGGER.info("Sampled %s rows from %s total rows", len(sampled), len(frame))
    return sampled


def make_hashable(value: object) -> object:
    if np.isscalar(value):
        try:
            if pd.isna(value):
                return value
        except TypeError:
            pass
    if isinstance(value, np.ndarray):
        return tuple(make_hashable(item) for item in value.tolist())
    if isinstance(value, (list, tuple, set)):
        return tuple(make_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((make_hashable(k), make_hashable(v)) for k, v in value.items()))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def sanitize_object_columns(frame: pd.DataFrame) -> pd.DataFrame:
    object_cols = frame.select_dtypes(include=["object"]).columns
    if len(object_cols) == 0:
        return frame

    sanitized = frame.copy()
    for col in object_cols:
        sanitized[col] = sanitized[col].map(make_hashable)
    LOGGER.info("Sanitized %s object columns for profiling compatibility", len(object_cols))
    return sanitized


def trim_expensive_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop very high-cardinality object columns to reduce profile memory/time."""
    drop_cols: list[str] = []
    n_rows = max(len(frame), 1)
    for col in frame.select_dtypes(include=["object"]).columns:
        unique_ratio = frame[col].nunique(dropna=False) / n_rows
        if unique_ratio >= UNIQUE_RATIO_DROP_THRESHOLD:
            drop_cols.append(col)

    if drop_cols:
        LOGGER.info(
            "Dropping %s high-cardinality text columns for profiling: %s",
            len(drop_cols),
            ", ".join(sorted(drop_cols)),
        )
        return frame.drop(columns=drop_cols)
    return frame


def run() -> None:
    root = project_root()
    data = load_processed_data(root)
    sampled = sample_data(data)
    sampled = sanitize_object_columns(sampled)
    sampled = trim_expensive_columns(sampled)

    report = ProfileReport(
        sampled,
        title="Real Estate EDA Report",
        minimal=True,
        correlations={"pearson": {"calculate": False}, "spearman": {"calculate": False}, "kendall": {"calculate": False}, "cramers": {"calculate": False}, "phi_k": {"calculate": False}},
        interactions=None,
    )
    out = root / REPORT_PATH
    ensure_dir(out.parent)
    report.to_file(out)
    LOGGER.info("Saved EDA report to %s", out)


if __name__ == "__main__":
    run()
