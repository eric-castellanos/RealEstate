"""Fetch MSA-level Census demographics for properties from HomeHarvest raw data."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

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

ACS_YEAR = 2023
ACS_ENDPOINT = f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5"
HOMEHARVEST_GLOB = "data/raw/homeharvest/*.parquet"
OUTPUT_PATH = "data/raw/census/msa_data.parquet"

MSA_COLUMNS = [
    "msa",
    "msa_name",
    "metro",
    "metro_name",
    "metropolitan_area",
    "cbsa_name",
    "cbsa_title",
]

COUNTY_COLUMNS = ["county", "county_name"]
STATE_COLUMNS = ["state", "state_code", "state_abbr"]
CITY_COLUMNS = ["city", "city_name"]

# Fallback for common counties in major US metro areas (minimal lookup).
COUNTY_STATE_TO_MSA = {
    ("new york county", "ny"): "New York-Newark-Jersey City",
    ("los angeles county", "ca"): "Los Angeles-Long Beach-Anaheim",
    ("cook county", "il"): "Chicago-Naperville-Elgin",
    ("dallas county", "tx"): "Dallas-Fort Worth-Arlington",
    ("harris county", "tx"): "Houston-The Woodlands-Sugar Land",
    ("miami-dade county", "fl"): "Miami-Fort Lauderdale-Pompano Beach",
    ("maricopa county", "az"): "Phoenix-Mesa-Chandler",
    ("philadelphia county", "pa"): "Philadelphia-Camden-Wilmington",
    ("san diego county", "ca"): "San Diego-Chula Vista-Carlsbad",
    ("king county", "wa"): "Seattle-Tacoma-Bellevue",
}

CITY_STATE_TO_MSA = {
    ("new york", "ny"): "New York-Newark-Jersey City",
    ("los angeles", "ca"): "Los Angeles-Long Beach-Anaheim",
    ("chicago", "il"): "Chicago-Naperville-Elgin",
    ("dallas", "tx"): "Dallas-Fort Worth-Arlington",
    ("houston", "tx"): "Houston-The Woodlands-Sugar Land",
    ("washington", "dc"): "Washington-Arlington-Alexandria",
    ("miami", "fl"): "Miami-Fort Lauderdale-Pompano Beach",
    ("philadelphia", "pa"): "Philadelphia-Camden-Wilmington",
    ("atlanta", "ga"): "Atlanta-Sandy Springs-Alpharetta",
    ("phoenix", "az"): "Phoenix-Mesa-Chandler",
    ("boston", "ma"): "Boston-Cambridge-Newton",
    ("riverside", "ca"): "Riverside-San Bernardino-Ontario",
    ("san francisco", "ca"): "San Francisco-Oakland-Berkeley",
    ("detroit", "mi"): "Detroit-Warren-Dearborn",
    ("seattle", "wa"): "Seattle-Tacoma-Bellevue",
    ("minneapolis", "mn"): "Minneapolis-St. Paul-Bloomington",
    ("san diego", "ca"): "San Diego-Chula Vista-Carlsbad",
    ("tampa", "fl"): "Tampa-St. Petersburg-Clearwater",
    ("denver", "co"): "Denver-Aurora-Lakewood",
    ("baltimore", "md"): "Baltimore-Columbia-Towson",
    ("charlotte", "nc"): "Charlotte-Concord-Gastonia",
    ("orlando", "fl"): "Orlando-Kissimmee-Sanford",
    ("san antonio", "tx"): "San Antonio-New Braunfels",
    ("portland", "or"): "Portland-Vancouver-Hillsboro",
    ("st. louis", "mo"): "St. Louis",
    ("st louis", "mo"): "St. Louis",
}


def _clean(value: object) -> str:
    return str(value).strip().lower() if value is not None else ""


def _normalize_msa_name(name: str) -> str:
    base = _clean(name)
    if " metro area" in base:
        base = base.replace(" metro area", "")
    if " micropolitan statistical area" in base:
        base = base.replace(" micropolitan statistical area", "")
    if " metropolitan statistical area" in base:
        base = base.replace(" metropolitan statistical area", "")
    if "," in base:
        base = base.split(",")[0].strip()
    return base


def _first_existing(columns: list[str], available: list[str]) -> Optional[str]:
    for column in columns:
        if column in available:
            return column
    return None


def read_homeharvest_data() -> pd.DataFrame:
    paths = sorted((project_root()).glob(HOMEHARVEST_GLOB))
    if not paths:
        raise FileNotFoundError(f"No HomeHarvest parquet files found at {HOMEHARVEST_GLOB}")

    LOGGER.info("Reading %s HomeHarvest parquet files", len(paths))
    frames = [pd.read_parquet(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def extract_msa_for_row(row: pd.Series, columns: list[str]) -> Optional[str]:
    for col in MSA_COLUMNS:
        if col in columns:
            value = row.get(col)
            if pd.notna(value) and _clean(value):
                return str(value).strip()

    county_col = _first_existing(COUNTY_COLUMNS, columns)
    state_col = _first_existing(STATE_COLUMNS, columns)
    if county_col and state_col:
        county = _clean(row.get(county_col))
        state = _clean(row.get(state_col))
        msa = COUNTY_STATE_TO_MSA.get((county, state))
        if msa:
            return msa

    city_col = _first_existing(CITY_COLUMNS, columns)
    if city_col and state_col:
        city = _clean(row.get(city_col))
        state = _clean(row.get(state_col))
        msa = CITY_STATE_TO_MSA.get((city, state))
        if msa:
            return msa

    return None


def extract_unique_msas(properties: pd.DataFrame) -> list[str]:
    columns = list(properties.columns)
    msa_values = properties.apply(lambda row: extract_msa_for_row(row, columns), axis=1)
    unique = sorted({value for value in msa_values.dropna().tolist() if _clean(value)})
    LOGGER.info("Extracted %s unique MSAs from property data", len(unique))
    return unique


def fetch_msa_lookup(api_key: Optional[str]) -> Dict[str, Dict[str, str]]:
    rows = _census_get_json(
        {
            "get": "NAME",
            "for": "metropolitan statistical area/micropolitan statistical area:*",
        },
        api_key=api_key,
    )

    header, data_rows = rows[0], rows[1:]
    name_idx = header.index("NAME")
    code_idx = header.index("metropolitan statistical area/micropolitan statistical area")

    lookup = {}
    for row in data_rows:
        census_name = row[name_idx]
        cbsa_code = row[code_idx]
        lookup[_normalize_msa_name(census_name)] = {
            "cbsa_code": cbsa_code,
            "census_name": census_name,
        }
    return lookup


def _census_get_json(base_params: Dict[str, str], api_key: Optional[str]) -> list[list[str]]:
    params = {
        **base_params,
    }
    if api_key:
        params["key"] = api_key
    response = requests.get(ACS_ENDPOINT, params=params, timeout=30)
    response.raise_for_status()

    if "invalid key" in response.text.lower():
        LOGGER.warning("CENSUS_API_KEY appears invalid; retrying Census API call without key.")
        fallback_response = requests.get(ACS_ENDPOINT, params=base_params, timeout=30)
        fallback_response.raise_for_status()
        return fallback_response.json()

    return response.json()


def resolve_cbsa(msa_name: str, lookup: Dict[str, Dict[str, str]]) -> Optional[Dict[str, str]]:
    norm = _normalize_msa_name(msa_name)
    if norm in lookup:
        return lookup[norm]

    for key, value in lookup.items():
        if norm in key or key in norm:
            return value
    return None


def fetch_msa_demographics(msa_name: str, cbsa_code: str, api_key: Optional[str]) -> Dict[str, object]:
    rows = _census_get_json(
        {
            "get": ",".join(
                [
                    "NAME",
                    "B19013_001E",  # Median household income
                    "B01003_001E",  # Total population
                    "B15003_001E",  # Educational attainment denominator (25+)
                    "B15003_022E",  # Bachelor's
                    "B15003_023E",  # Master's
                    "B15003_024E",  # Professional
                    "B15003_025E",  # Doctorate
                ]
            ),
            "for": f"metropolitan statistical area/micropolitan statistical area:{cbsa_code}",
        },
        api_key=api_key,
    )

    header, values = rows[0], rows[1]
    payload = dict(zip(header, values))

    education_total = pd.to_numeric(payload.get("B15003_001E"), errors="coerce")
    bachelors_or_higher = sum(
        pd.to_numeric(payload.get(col), errors="coerce")
        for col in ["B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E"]
    )
    share = (
        (bachelors_or_higher / education_total)
        if pd.notna(education_total) and education_total not in (0, 0.0)
        else pd.NA
    )

    return {
        "msa": msa_name,
        "cbsa_code": cbsa_code,
        "census_name": payload.get("NAME"),
        "population": pd.to_numeric(payload.get("B01003_001E"), errors="coerce"),
        "median_household_income": pd.to_numeric(payload.get("B19013_001E"), errors="coerce"),
        "education_total_25_plus": education_total,
        "bachelors_or_higher_25_plus": bachelors_or_higher,
        "bachelors_or_higher_share": share,
        "acs_year": ACS_YEAR,
    }


def output_file() -> Path:
    output = project_root() / OUTPUT_PATH
    ensure_dir(output.parent)
    return output


def run() -> None:
    api_key = os.getenv("CENSUS_API_KEY")
    if not api_key:
        LOGGER.warning("CENSUS_API_KEY is not set; proceeding without an API key.")

    properties = read_homeharvest_data()
    unique_msas = extract_unique_msas(properties)
    lookup = fetch_msa_lookup(api_key)

    records = []
    for msa_name in unique_msas:
        cbsa_meta = resolve_cbsa(msa_name, lookup)
        if not cbsa_meta:
            LOGGER.warning("Could not map MSA '%s' to a Census CBSA code", msa_name)
            continue

        LOGGER.info("Fetching Census demographics for %s (%s)", msa_name, cbsa_meta["cbsa_code"])
        record = fetch_msa_demographics(
            msa_name=msa_name,
            cbsa_code=cbsa_meta["cbsa_code"],
            api_key=api_key,
        )
        records.append(record)

    output_df = pd.DataFrame(records)
    if output_df.empty:
        output_df = pd.DataFrame(
            columns=[
                "msa",
                "cbsa_code",
                "census_name",
                "population",
                "median_household_income",
                "education_total_25_plus",
                "bachelors_or_higher_25_plus",
                "bachelors_or_higher_share",
                "acs_year",
            ]
        )

    output_df = output_df.drop_duplicates(subset=["msa"]).set_index("msa").sort_index()
    out = output_file()
    output_df.to_parquet(out)
    LOGGER.info("Saved %s MSA rows to %s", len(output_df), out)


if __name__ == "__main__":
    run()
