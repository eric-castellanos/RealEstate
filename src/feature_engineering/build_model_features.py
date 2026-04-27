"""Build model features from cleaned real estate data."""

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

INPUT_PATH = "data/processed/cleaned_data.parquet"
OUTPUT_PATH = "data/processed/model_features.parquet"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    current_year = datetime.now().year

    if "year_built" in frame.columns:
        frame["property_age"] = current_year - frame["year_built"]
        frame.loc[frame["property_age"] < 0, "property_age"] = np.nan

    full_baths = pd.to_numeric(frame.get("full_baths"), errors="coerce")
    if "half_baths" in frame.columns:
        half_baths = pd.to_numeric(frame["half_baths"], errors="coerce").fillna(0)
        frame["total_baths"] = full_baths + 0.5 * half_baths
    else:
        frame["total_baths"] = full_baths

    beds = pd.to_numeric(frame.get("beds"), errors="coerce")
    sqft = pd.to_numeric(frame.get("sqft"), errors="coerce")
    lot_sqft = pd.to_numeric(frame.get("lot_sqft"), errors="coerce")
    sold_price = pd.to_numeric(frame.get("sold_price"), errors="coerce")
    household_income = pd.to_numeric(frame.get("median_household_income"), errors="coerce")

    frame["sqft_per_bedroom"] = np.where((beds > 0) & beds.notna(), sqft / beds, np.nan)
    frame["lot_to_sqft_ratio"] = np.where((sqft > 0) & sqft.notna(), lot_sqft / sqft, np.nan)

    if "distance_to_coastline_m" in frame.columns:
        dist_coast = pd.to_numeric(frame["distance_to_coastline_m"], errors="coerce")
        frame["coastal_5km"] = dist_coast < 5000
    else:
        frame["coastal_5km"] = False

    if "distance_to_city_center_m" in frame.columns:
        dist_city = pd.to_numeric(frame["distance_to_city_center_m"], errors="coerce")
        frame["urban_10km"] = dist_city < 10000
    else:
        frame["urban_10km"] = False

    frame["price_to_income"] = np.where(
        (household_income > 0) & household_income.notna(),
        sold_price / household_income,
        np.nan,
    )

    frame["log_sqft"] = np.log1p(sqft.clip(lower=0))
    frame["log_lot_sqft"] = np.log1p(lot_sqft.clip(lower=0))
    frame["log_price"] = np.log1p(sold_price.clip(lower=0))

    return frame


def run() -> None:
    root = project_root()
    input_path = root / INPUT_PATH
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    data = pd.read_parquet(input_path)
    LOGGER.info("Loaded cleaned data with shape: %s rows x %s cols", len(data), len(data.columns))

    data = add_features(data)
    data = data.replace([np.inf, -np.inf], np.nan)
    LOGGER.info("Replaced inf/-inf values with nulls")

    numeric_cols = data.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        medians = data[numeric_cols].median()
        data[numeric_cols] = data[numeric_cols].fillna(medians)
        LOGGER.info("Filled nulls in %s numeric columns with median values", len(numeric_cols))

    out = root / OUTPUT_PATH
    ensure_dir(out.parent)
    data.to_parquet(out, index=False)
    LOGGER.info("Saved model features to %s", out)


if __name__ == "__main__":
    run()
