"""Download TIGER/Line geospatial shapefiles for states present in HomeHarvest data."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import zipfile

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

INPUT_GLOB = "data/raw/homeharvest/*.parquet"
BASE_URL_TEMPLATE = "https://www2.census.gov/geo/tiger/TIGER{year}/{folder}/{filename}"

DATASET_CONFIG = {
    "roads": {
        "folder": "PRISECROADS",
        "filename_template": "tl_{year}_{state_fips}_prisecroads.zip",
    },
}

STATE_ABBR_TO_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09", "DE": "10",
    "DC": "11", "FL": "12", "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18", "IA": "19",
    "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27",
    "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33", "NJ": "34", "NM": "35",
    "NY": "36", "NC": "37", "ND": "38", "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44",
    "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53",
    "WV": "54", "WI": "55", "WY": "56", "PR": "72",
}

STATE_NAME_TO_ABBR = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "district of columbia": "DC",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID", "illinois": "IL",
    "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
    "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
    "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC", "south dakota": "SD",
    "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA",
    "washington": "WA", "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY", "puerto rico": "PR",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download TIGER/Line shapefiles by state")
    parser.add_argument("--year", type=int, default=2023, help="TIGER year, e.g. 2023")
    parser.add_argument(
        "--include-county-boundaries",
        action="store_true",
        help="Also download county boundary shapefiles",
    )
    return parser.parse_args()


def read_property_data() -> pd.DataFrame:
    files = sorted(project_root().glob(INPUT_GLOB))
    if not files:
        raise FileNotFoundError(f"No property parquet files found at {INPUT_GLOB}")
    frames = [pd.read_parquet(path) for path in files]
    df = pd.concat(frames, ignore_index=True)
    LOGGER.info("Loaded %s property rows from %s files", len(df), len(files))
    return df


def normalize_state_to_fips(value: object) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None

    upper = raw.upper()
    if upper in STATE_ABBR_TO_FIPS:
        return STATE_ABBR_TO_FIPS[upper]

    lower = raw.lower()
    if lower in STATE_NAME_TO_ABBR:
        abbr = STATE_NAME_TO_ABBR[lower]
        return STATE_ABBR_TO_FIPS.get(abbr)

    # Allow already-FIPS-like values (e.g., "5" or "05").
    if raw.isdigit() and 1 <= len(raw) <= 2:
        return raw.zfill(2)

    return None


def extract_region_summary(df: pd.DataFrame) -> tuple[list[str], list[str], dict[str, float]]:
    if "state" not in df.columns:
        raise ValueError("Expected 'state' column in property data.")

    state_fips = sorted(
        {
            fips
            for fips in (normalize_state_to_fips(v) for v in df["state"].dropna().unique().tolist())
            if fips
        }
    )
    counties = []
    if "county" in df.columns:
        counties = sorted({str(v).strip() for v in df["county"].dropna().unique().tolist() if str(v).strip()})

    county_fips = []
    if "fips_code" in df.columns:
        county_fips = sorted(
            {
                str(v).strip().split(".")[0].zfill(5)
                for v in df["fips_code"].dropna().tolist()
                if str(v).strip() and str(v).strip().split(".")[0].isdigit()
            }
        )

    bbox = {}
    if "latitude" in df.columns and "longitude" in df.columns:
        lat = pd.to_numeric(df["latitude"], errors="coerce")
        lon = pd.to_numeric(df["longitude"], errors="coerce")
        valid = lat.notna() & lon.notna()
        if valid.any():
            bbox = {
                "min_lat": float(lat[valid].min()),
                "max_lat": float(lat[valid].max()),
                "min_lon": float(lon[valid].min()),
                "max_lon": float(lon[valid].max()),
            }

    return state_fips, county_fips, bbox


def build_download_url(year: int, dataset_key: str, state_fips: str) -> tuple[str, str]:
    config = DATASET_CONFIG[dataset_key]
    state_fips = str(state_fips).zfill(2)
    filename = config["filename_template"].format(year=year, state_fips=state_fips)
    url = BASE_URL_TEMPLATE.format(year=year, folder=config["folder"], filename=filename)
    return url, filename


def build_areawater_url(year: int, county_fips: str) -> tuple[str, str]:
    county_fips = str(county_fips).zfill(5)
    filename = f"tl_{year}_{county_fips}_areawater.zip"
    url = BASE_URL_TEMPLATE.format(year=year, folder="AREAWATER", filename=filename)
    return url, filename


def build_arealm_url(year: int, state_fips: str) -> tuple[str, str]:
    state_fips = str(state_fips).zfill(2)
    filename = f"tl_{year}_{state_fips}_arealm.zip"
    url = BASE_URL_TEMPLATE.format(year=year, folder="AREALM", filename=filename)
    return url, filename


def build_county_boundaries_url(year: int) -> tuple[str, str]:
    filename = f"tl_{year}_us_county.zip"
    url = BASE_URL_TEMPLATE.format(year=year, folder="COUNTY", filename=filename)
    return url, filename


def extracted_shapefile_exists(target_dir: Path) -> bool:
    return any(target_dir.glob("*.shp"))


def download_file(url: str, destination: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with destination.open("wb") as out:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    out.write(chunk)


def unzip_file(zip_path: Path, target_dir: Path) -> None:
    ensure_dir(target_dir)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(target_dir)


def ensure_dataset_for_state(year: int, dataset_key: str, state_fips: str) -> None:
    state_fips = str(state_fips).zfill(2)
    state_dir = ensure_dir(project_root() / "data" / "external" / dataset_key / state_fips)
    url, filename = build_download_url(year=year, dataset_key=dataset_key, state_fips=state_fips)
    zip_path = state_dir / filename

    if extracted_shapefile_exists(state_dir):
        LOGGER.info("Skipping %s %s: shapefile already extracted", dataset_key, state_fips)
        return

    if not zip_path.exists():
        LOGGER.info("Downloading %s %s from %s", dataset_key, state_fips, url)
        download_file(url, zip_path)
    else:
        LOGGER.info("Using existing archive %s", zip_path)

    LOGGER.info("Extracting %s to %s", zip_path.name, state_dir)
    unzip_file(zip_path, state_dir)


def ensure_areawater_for_county(year: int, county_fips: str) -> None:
    county_fips = str(county_fips).zfill(5)
    state_fips = county_fips[:2]
    target_dir = ensure_dir(project_root() / "data" / "external" / "water" / state_fips)
    shp_name = f"tl_{year}_{county_fips}_areawater.shp"
    if (target_dir / shp_name).exists():
        LOGGER.info("Skipping water %s: shapefile already extracted", county_fips)
        return

    url, filename = build_areawater_url(year=year, county_fips=county_fips)
    zip_path = target_dir / filename
    if not zip_path.exists():
        LOGGER.info("Downloading water %s from %s", county_fips, url)
        download_file(url, zip_path)
    else:
        LOGGER.info("Using existing archive %s", zip_path)
    LOGGER.info("Extracting %s to %s", zip_path.name, target_dir)
    unzip_file(zip_path, target_dir)


def ensure_parks_for_state(year: int, state_fips: str) -> None:
    state_fips = str(state_fips).zfill(2)
    target_dir = ensure_dir(project_root() / "data" / "external" / "parks" / state_fips)
    shp_name = f"tl_{year}_{state_fips}_arealm.shp"
    if (target_dir / shp_name).exists():
        LOGGER.info("Skipping parks %s: shapefile already extracted", state_fips)
        return

    url, filename = build_arealm_url(year=year, state_fips=state_fips)
    zip_path = target_dir / filename
    if not zip_path.exists():
        LOGGER.info("Downloading parks %s from %s", state_fips, url)
        download_file(url, zip_path)
    else:
        LOGGER.info("Using existing archive %s", zip_path)
    LOGGER.info("Extracting %s to %s", zip_path.name, target_dir)
    unzip_file(zip_path, target_dir)


def ensure_county_boundaries(year: int) -> None:
    target_dir = ensure_dir(project_root() / "data" / "external" / "county_boundaries" / "us")
    shp_name = f"tl_{year}_us_county.shp"
    if (target_dir / shp_name).exists():
        LOGGER.info("Skipping county boundaries: shapefile already extracted")
        return
    url, filename = build_county_boundaries_url(year=year)
    zip_path = target_dir / filename
    if not zip_path.exists():
        LOGGER.info("Downloading county boundaries from %s", url)
        download_file(url, zip_path)
    else:
        LOGGER.info("Using existing archive %s", zip_path)
    LOGGER.info("Extracting %s to %s", zip_path.name, target_dir)
    unzip_file(zip_path, target_dir)


def run(year: int, include_county_boundaries: bool) -> None:
    df = read_property_data()
    state_fips, county_fips, bbox = extract_region_summary(df)
    if not state_fips:
        raise ValueError("No valid state codes found in property data.")

    LOGGER.info("Identified %s unique states: %s", len(state_fips), ", ".join(state_fips))
    LOGGER.info("Identified %s unique county FIPS", len(county_fips))
    if bbox:
        LOGGER.info(
            "Property bounding box lat=[%.4f, %.4f], lon=[%.4f, %.4f]",
            bbox["min_lat"],
            bbox["max_lat"],
            bbox["min_lon"],
            bbox["max_lon"],
        )

    for sf in state_fips:
        try:
            ensure_dataset_for_state(year=year, dataset_key="roads", state_fips=sf)
        except requests.HTTPError as exc:
            LOGGER.warning("Skipping roads %s due to HTTP error: %s", sf, exc)

    if not county_fips:
        LOGGER.warning("No county FIPS found in data; skipping AREAWATER/AREALM downloads.")
    else:
        for cf in county_fips:
            try:
                ensure_areawater_for_county(year=year, county_fips=cf)
            except requests.HTTPError as exc:
                LOGGER.warning("Skipping water %s due to HTTP error: %s", cf, exc)

    for sf in state_fips:
        try:
            ensure_parks_for_state(year=year, state_fips=sf)
        except requests.HTTPError as exc:
            LOGGER.warning("Skipping parks %s due to HTTP error: %s", sf, exc)

    if include_county_boundaries:
        try:
            ensure_county_boundaries(year=year)
        except requests.HTTPError as exc:
            LOGGER.warning("Skipping county boundaries due to HTTP error: %s", exc)

    LOGGER.info("TIGER/Line download pipeline complete.")


if __name__ == "__main__":
    args = parse_args()
    run(year=args.year, include_county_boundaries=args.include_county_boundaries)
