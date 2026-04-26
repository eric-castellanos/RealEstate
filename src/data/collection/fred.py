"""Fetch key macroeconomic series from FRED and store raw parquet output."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict

import pandas as pd
import requests

try:
    from .utils import ensure_dir, project_root
except ImportError:  # Allows direct execution
    from utils import ensure_dir, project_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
SERIES: Dict[str, str] = {
    "MORTGAGE30US": "mortgage_rate_30y",
    "UNRATE": "unemployment_rate",
    "CPIAUCSL": "cpi",
    "FEDFUNDS": "federal_funds_rate",
}


def output_path() -> Path:
    out_dir = ensure_dir(project_root() / "data" / "raw" / "fred")
    return out_dir / "macro_data.parquet"


def fetch_series(series_id: str, api_key: str) -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }
    response = requests.get(FRED_BASE_URL, params=params, timeout=30)
    response.raise_for_status()

    payload = response.json()
    observations = payload.get("observations", [])
    frame = pd.DataFrame(observations)
    if frame.empty:
        return pd.DataFrame(columns=["date", series_id])

    frame = frame[["date", "value"]].rename(columns={"value": series_id})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame[series_id] = pd.to_numeric(frame[series_id], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    return frame


def run() -> None:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY environment variable is required.")

    LOGGER.info("Fetching %s FRED series", len(SERIES))
    merged = None

    for series_id, column_name in SERIES.items():
        LOGGER.info("Fetching FRED series %s", series_id)
        series_df = fetch_series(series_id=series_id, api_key=api_key).rename(
            columns={series_id: column_name}
        )
        merged = (
            series_df
            if merged is None
            else merged.merge(series_df, on="date", how="outer")
        )

    if merged is None:
        merged = pd.DataFrame(columns=["date"] + list(SERIES.values()))

    merged = merged.sort_values("date").set_index("date")
    out = output_path()
    merged.to_parquet(out)
    LOGGER.info("Saved %s rows to %s", len(merged), out)


if __name__ == "__main__":
    run()
