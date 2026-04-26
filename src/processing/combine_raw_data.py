"""Combine raw datasets into a property-level processed parquet file.

HomeHarvest files are used as the base row-level dataset.
Census MSA data is joined by metro, and FRED macro series is expanded onto each
property row using an as-of date join.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

HOMEHARVEST_GLOB = "data/raw/homeharvest/*.parquet"
CENSUS_PATH = "data/raw/census/msa_data.parquet"
FRED_PATH = "data/raw/fred/macro_data.parquet"
OUTPUT_PATH = "data/processed/combined_raw_data.parquet"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_metro(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    text = (
        text.replace(" metro area", "")
        .replace(" metropolitan area", "")
        .replace(" msa", "")
    )
    return " ".join(text.split())


def detect_property_date(frame: pd.DataFrame) -> pd.Series:
    """Return best-effort property date from available date columns."""
    candidate_columns = [
        "list_date",
        "last_update_date",
        "pending_date",
        "sold_date",
    ]
    available = [col for col in candidate_columns if col in frame.columns]
    if not available:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")

    parsed = [pd.to_datetime(frame[col], errors="coerce") for col in available]
    date_series = parsed[0]
    for candidate in parsed[1:]:
        date_series = date_series.fillna(candidate)
    return date_series


def load_homeharvest(root: Path) -> pd.DataFrame:
    files = sorted(root.glob(HOMEHARVEST_GLOB))
    if not files:
        raise FileNotFoundError(f"No parquet files found using pattern: {HOMEHARVEST_GLOB}")

    frames: list[pd.DataFrame] = []
    LOGGER.info("Found %s HomeHarvest parquet files", len(files))
    for path in files:
        frame = pd.read_parquet(path)
        frame["source_file"] = str(path.relative_to(root))
        frames.append(frame)
        LOGGER.info("Loaded %s rows from %s", len(frame), path.relative_to(root))

    properties = pd.concat(frames, ignore_index=True, sort=False)
    properties["property_event_date"] = detect_property_date(properties)
    properties["_metro_key"] = properties.get("metro_name", "").map(normalize_metro)
    LOGGER.info("Combined %s property rows", len(properties))
    return properties


def load_census(root: Path) -> pd.DataFrame:
    path = root / CENSUS_PATH
    if not path.exists():
        LOGGER.warning("Census file not found at %s; skipping census join", path)
        return pd.DataFrame()

    census = pd.read_parquet(path)
    if "msa" not in census.columns:
        census = census.reset_index()
    if "msa" not in census.columns:
        LOGGER.warning("Census data does not contain 'msa'; skipping census join")
        return pd.DataFrame()

    census["_metro_key"] = census["msa"].map(normalize_metro)
    # Keep one row per metro key.
    census = census.sort_values("_metro_key").drop_duplicates("_metro_key", keep="first")
    LOGGER.info("Loaded %s census metro rows", len(census))
    return census


def load_fred(root: Path) -> pd.DataFrame:
    path = root / FRED_PATH
    if not path.exists():
        LOGGER.warning("FRED file not found at %s; skipping fred join", path)
        return pd.DataFrame()

    fred = pd.read_parquet(path)
    if "date" not in fred.columns:
        fred = fred.reset_index()
    if "date" not in fred.columns:
        LOGGER.warning("FRED data does not contain 'date'; skipping fred join")
        return pd.DataFrame()

    fred["date"] = pd.to_datetime(fred["date"], errors="coerce")
    fred = fred.dropna(subset=["date"]).sort_values("date")
    fred = fred.ffill()
    LOGGER.info("Loaded %s FRED rows", len(fred))
    return fred


def run() -> None:
    root = project_root()
    combined = load_homeharvest(root)

    census = load_census(root)
    if not census.empty:
        census_cols = [col for col in census.columns if col != "msa"]
        combined = combined.merge(census[census_cols], on="_metro_key", how="left")
        LOGGER.info("Joined census features onto property rows")

    fred = load_fred(root)
    if not fred.empty:
        combined = combined.sort_values("property_event_date")
        known_dates = combined[combined["property_event_date"].notna()].copy()
        missing_dates = combined[combined["property_event_date"].isna()].copy()

        if not known_dates.empty:
            known_dates = pd.merge_asof(
                known_dates,
                fred,
                left_on="property_event_date",
                right_on="date",
                direction="backward",
            )
        if not missing_dates.empty:
            # Fall back to latest available macro snapshot for undated properties.
            latest_fred = fred.iloc[-1]
            for col in fred.columns:
                if col != "date":
                    missing_dates[col] = latest_fred[col]
            missing_dates["date"] = latest_fred["date"]

        combined = pd.concat([known_dates, missing_dates], ignore_index=True, sort=False)
        LOGGER.info("Joined FRED macro features onto property rows")

    if "_metro_key" in combined.columns:
        combined = combined.drop(columns=["_metro_key"])

    out = root / OUTPUT_PATH
    ensure_dir(out.parent)
    combined.to_parquet(out, index=False)
    LOGGER.info("Saved %s total rows to %s", len(combined), out)


if __name__ == "__main__":
    run()
