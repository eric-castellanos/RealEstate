"""Clean and prepare processed parquet data for modeling."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

INPUT_GLOB = "data/processed/*.parquet"
OUTPUT_PATH = "data/processed/cleaned_data.parquet"
MISSING_DROP_THRESHOLD = 0.70
FLOOD_ZONE_DROP_THRESHOLD = 0.80

EXACT_JUNK_COLUMNS = {
    "acs_year",
    "processed_source_file",
    "list_price_min",
    "list_price_max",
    "assessed_value",
    "tax",
    "tax_history",
    "neighborhoods",
    "nearby_schools",
    "text",
}
PREFIX_JUNK_COLUMNS = ("agent_", "broker_", "office_", "builder_")

CORE_FEATURES = ["beds", "full_baths", "sqft", "year_built", "lot_sqft"]
GEO_DEMOGRAPHIC_FEATURES = [
    "latitude",
    "longitude",
    "distance_to_city_center_m",
    "distance_to_nearest_park_m",
    "distance_to_nearest_highway_m",
    "distance_to_coastline_m",
    "population",
    "median_household_income",
    "education_total_25_plus",
    "bachelors_or_higher_25_plus",
    "bachelors_or_higher_share",
    "mortgage_rate_30y",
    "unemployment_rate",
    "cpi",
    "federal_funds_rate",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_processed_data(root: Path) -> pd.DataFrame:
    output_name = Path(OUTPUT_PATH).name
    files = sorted(path for path in root.glob(INPUT_GLOB) if path.name != output_name)
    if not files:
        raise FileNotFoundError(f"No processed parquet files found via pattern: {INPUT_GLOB}")

    LOGGER.info("Found %s input parquet files", len(files))
    frames: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_parquet(path)
        frame["processed_source_file"] = str(path.relative_to(root))
        frames.append(frame)
        LOGGER.info("Loaded %s rows from %s", len(frame), path.relative_to(root))

    combined = pd.concat(frames, ignore_index=True, sort=False)
    LOGGER.info("Combined %s rows and %s columns", len(combined), len(combined.columns))
    return combined


def select_junk_columns(frame: pd.DataFrame) -> list[str]:
    to_drop: list[str] = []
    for col in frame.columns:
        if col in EXACT_JUNK_COLUMNS:
            to_drop.append(col)
            continue
        if any(col.startswith(prefix) for prefix in PREFIX_JUNK_COLUMNS):
            to_drop.append(col)
    return sorted(set(to_drop))


def fill_with_group_and_global_median(
    frame: pd.DataFrame,
    column: str,
    group_columns: tuple[str, ...] = ("state",),
) -> None:
    if column not in frame.columns:
        return

    for group_col in group_columns:
        if group_col in frame.columns and frame[column].isna().any():
            group_median = frame.groupby(group_col)[column].transform("median")
            frame[column] = frame[column].fillna(group_median)

    if frame[column].isna().any():
        global_median = frame[column].median()
        if pd.notna(global_median):
            frame[column] = frame[column].fillna(global_median)


def clean_core_features(frame: pd.DataFrame) -> pd.DataFrame:
    now_year = datetime.now().year

    for col in CORE_FEATURES + ["sold_price"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    if "beds" in frame.columns:
        before = len(frame)
        frame = frame[frame["beds"].isna() | (frame["beds"] > 0)].copy()
        LOGGER.info("Removed %s rows with invalid beds <= 0", before - len(frame))

    if "sold_price" in frame.columns:
        before = len(frame)
        frame = frame[frame["sold_price"].isna() | (frame["sold_price"] > 0)].copy()
        LOGGER.info("Removed %s rows with invalid sold_price <= 0", before - len(frame))

    for positive_col in ("full_baths", "sqft", "lot_sqft"):
        if positive_col in frame.columns:
            frame.loc[frame[positive_col] <= 0, positive_col] = np.nan

    if "year_built" in frame.columns:
        frame.loc[(frame["year_built"] < 1700) | (frame["year_built"] > now_year + 1), "year_built"] = np.nan

    for col in CORE_FEATURES:
        fill_with_group_and_global_median(frame, col)

    return frame


def handle_geo_and_demographic_missingness(frame: pd.DataFrame) -> pd.DataFrame:
    dropped: list[str] = []
    for col in GEO_DEMOGRAPHIC_FEATURES:
        if col not in frame.columns:
            continue
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        median_val = frame[col].median()
        if pd.notna(median_val):
            frame[col] = frame[col].fillna(median_val)
        if frame[col].isna().all():
            dropped.append(col)

    if dropped:
        LOGGER.info(
            "Dropping %s geo/demographic columns with no usable values: %s",
            len(dropped),
            ", ".join(sorted(dropped)),
        )
        frame = frame.drop(columns=dropped)
    return frame


def add_log_features(frame: pd.DataFrame) -> pd.DataFrame:
    feature_map = {
        "sqft": "log_sqft",
        "lot_sqft": "log_lot_sqft",
        "sold_price": "log_price",
    }
    for source_col, target_col in feature_map.items():
        if source_col in frame.columns:
            frame[target_col] = np.log1p(frame[source_col].clip(lower=0))
    return frame


def run() -> None:
    root = project_root()
    data = load_processed_data(root)

    junk_cols = [col for col in select_junk_columns(data) if col in data.columns]
    if junk_cols:
        LOGGER.info("Dropping %s junk columns", len(junk_cols))
        data = data.drop(columns=junk_cols)

    if "flood_zone" in data.columns:
        flood_missing_ratio = data["flood_zone"].isna().mean()
        if flood_missing_ratio > FLOOD_ZONE_DROP_THRESHOLD:
            data = data.drop(columns=["flood_zone"])
            LOGGER.info("Dropped flood_zone (missing ratio %.3f > %.2f)", flood_missing_ratio, FLOOD_ZONE_DROP_THRESHOLD)

    missing_ratio = data.isna().mean()
    high_missing_cols = sorted(missing_ratio[missing_ratio > MISSING_DROP_THRESHOLD].index.tolist())
    if high_missing_cols:
        LOGGER.info(
            "Dropping %s columns with missing ratio > %.2f: %s",
            len(high_missing_cols),
            MISSING_DROP_THRESHOLD,
            ", ".join(high_missing_cols),
        )
        data = data.drop(columns=high_missing_cols)

    data = clean_core_features(data)
    data = handle_geo_and_demographic_missingness(data)

    if "sold_price" in data.columns:
        before = len(data)
        data = data[data["sold_price"].notna()].copy()
        LOGGER.info("Dropped %s rows without sold_price after cleaning", before - len(data))

    data = add_log_features(data)

    modeling_data = data.select_dtypes(include=["number", "bool"]).copy()
    LOGGER.info("Selected %s numeric/bool modeling columns", len(modeling_data.columns))
    LOGGER.info("Final dataset shape: %s rows x %s cols", len(modeling_data), len(modeling_data.columns))

    out = root / OUTPUT_PATH
    ensure_dir(out.parent)
    modeling_data.to_parquet(out, index=False)
    LOGGER.info("Saved cleaned modeling data to %s", out)


if __name__ == "__main__":
    run()
