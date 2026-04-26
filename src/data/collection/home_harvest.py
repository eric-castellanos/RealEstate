"""Collect property listings from HomeHarvest for major US metro areas."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

try:
    from .utils import ensure_dir, project_root, slugify
except ImportError:  # Allows direct execution: python src/data/collection/home_harvest.py
    from utils import ensure_dir, project_root, slugify

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetroArea:
    name: str
    query_location: str
    file_city: str


TOP_25_METRO_AREAS = [
    MetroArea("New York-Newark-Jersey City", "New York, NY", "new_york"),
    MetroArea("Los Angeles-Long Beach-Anaheim", "Los Angeles, CA", "los_angeles"),
    MetroArea("Chicago-Naperville-Elgin", "Chicago, IL", "chicago"),
    MetroArea("Dallas-Fort Worth-Arlington", "Dallas, TX", "dallas"),
    MetroArea("Houston-The Woodlands-Sugar Land", "Houston, TX", "houston"),
    MetroArea("Washington-Arlington-Alexandria", "Washington, DC", "washington"),
    MetroArea("Miami-Fort Lauderdale-Pompano Beach", "Miami, FL", "miami"),
    MetroArea("Philadelphia-Camden-Wilmington", "Philadelphia, PA", "philadelphia"),
    MetroArea("Atlanta-Sandy Springs-Alpharetta", "Atlanta, GA", "atlanta"),
    MetroArea("Phoenix-Mesa-Chandler", "Phoenix, AZ", "phoenix"),
    MetroArea("Boston-Cambridge-Newton", "Boston, MA", "boston"),
    MetroArea("Riverside-San Bernardino-Ontario", "Riverside, CA", "riverside"),
    MetroArea("San Francisco-Oakland-Berkeley", "San Francisco, CA", "san_francisco"),
    MetroArea("Detroit-Warren-Dearborn", "Detroit, MI", "detroit"),
    MetroArea("Seattle-Tacoma-Bellevue", "Seattle, WA", "seattle"),
    MetroArea("Minneapolis-St. Paul-Bloomington", "Minneapolis, MN", "minneapolis"),
    MetroArea("San Diego-Chula Vista-Carlsbad", "San Diego, CA", "san_diego"),
    MetroArea("Tampa-St. Petersburg-Clearwater", "Tampa, FL", "tampa"),
    MetroArea("Denver-Aurora-Lakewood", "Denver, CO", "denver"),
    MetroArea("Baltimore-Columbia-Towson", "Baltimore, MD", "baltimore"),
    MetroArea("Charlotte-Concord-Gastonia", "Charlotte, NC", "charlotte"),
    MetroArea("Orlando-Kissimmee-Sanford", "Orlando, FL", "orlando"),
    MetroArea("San Antonio-New Braunfels", "San Antonio, TX", "san_antonio"),
    MetroArea("Portland-Vancouver-Hillsboro", "Portland, OR", "portland"),
    MetroArea("St. Louis", "St. Louis, MO", "st_louis"),
]


def output_directory() -> Path:
    return ensure_dir(project_root() / "data" / "raw" / "homeharvest")


def to_pandas_frame(df, metro: MetroArea) -> pd.DataFrame:
    if df is None or getattr(df, "empty", False):
        return pd.DataFrame(
            {"metro_name": [metro.name], "query_location": [metro.query_location]}
        )

    frame = df.copy()
    frame["metro_name"] = metro.name
    frame["query_location"] = metro.query_location
    return frame


@lru_cache(maxsize=1)
def get_scrape_property():
    try:
        from homeharvest import scrape_property
        return scrape_property
    except Exception as exc:  # pragma: no cover - environment dependent
        LOGGER.error("HomeHarvest import failed: %s", exc)
        return None


def fetch_with_homeharvest(metro: MetroArea):
    scrape_property = get_scrape_property()
    if scrape_property is None:
        return None

    try:
        return scrape_property(
            location=metro.query_location,
            listing_type="for_sale",
            extra_property_data=True,
        )
    except Exception as exc:  # pragma: no cover - network/provider dependent
        LOGGER.warning("HomeHarvest fetch failed for %s: %s", metro.name, exc)
        return None


def save_metro_properties(metro: MetroArea, out_dir: Path) -> Path:
    file_name = f"properties_{slugify(metro.file_city)}.parquet"
    output_path = out_dir / file_name
    LOGGER.info("Fetching HomeHarvest properties for %s", metro.name)

    raw_df = fetch_with_homeharvest(metro)
    frame = to_pandas_frame(raw_df, metro)
    frame.to_parquet(output_path, index=False)

    LOGGER.info("Saved %s rows to %s", len(frame), output_path)
    return output_path


def run() -> None:
    out_dir = output_directory()
    LOGGER.info("Writing HomeHarvest datasets to %s", out_dir)

    for metro in TOP_25_METRO_AREAS:
        save_metro_properties(metro, out_dir)

    LOGGER.info("HomeHarvest collection finished for %s metros", len(TOP_25_METRO_AREAS))


if __name__ == "__main__":
    run()
