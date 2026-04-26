"""Build geospatial features from local shapefiles (no external APIs)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

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
METRIC_CRS = "EPSG:3857"

# Optional direct path overrides.
ROADS_SHP = os.getenv("ROADS_SHP_PATH")
PARKS_SHP = os.getenv("PARKS_SHP_PATH")
COAST_SHP = os.getenv("COAST_SHP_PATH")
COUNTY_SHP = os.getenv("COUNTY_SHP_PATH", "data/external/county_boundaries/us/tl_2023_us_county.shp")

CITY_CENTER_LOOKUP: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("new york", "ny"): (40.7128, -74.0060),
    ("los angeles", "ca"): (34.0522, -118.2437),
    ("chicago", "il"): (41.8781, -87.6298),
    ("dallas", "tx"): (32.7767, -96.7970),
    ("houston", "tx"): (29.7604, -95.3698),
    ("washington", "dc"): (38.9072, -77.0369),
    ("miami", "fl"): (25.7617, -80.1918),
    ("philadelphia", "pa"): (39.9526, -75.1652),
    ("atlanta", "ga"): (33.7490, -84.3880),
    ("phoenix", "az"): (33.4484, -112.0740),
    ("boston", "ma"): (42.3601, -71.0589),
    ("riverside", "ca"): (33.9806, -117.3755),
    ("san francisco", "ca"): (37.7749, -122.4194),
    ("detroit", "mi"): (42.3314, -83.0458),
    ("seattle", "wa"): (47.6062, -122.3321),
    ("minneapolis", "mn"): (44.9778, -93.2650),
    ("san diego", "ca"): (32.7157, -117.1611),
    ("tampa", "fl"): (27.9506, -82.4572),
    ("denver", "co"): (39.7392, -104.9903),
    ("baltimore", "md"): (39.2904, -76.6122),
    ("charlotte", "nc"): (35.2271, -80.8431),
    ("orlando", "fl"): (28.5383, -81.3792),
    ("san antonio", "tx"): (29.4241, -98.4936),
    ("portland", "or"): (45.5152, -122.6784),
    ("st. louis", "mo"): (38.6270, -90.1994),
    ("st louis", "mo"): (38.6270, -90.1994),
}

COASTAL_COUNTY_LOOKUP = {
    ("new york county", "ny"),
    ("kings county", "ny"),
    ("queens county", "ny"),
    ("bronx county", "ny"),
    ("richmond county", "ny"),
    ("los angeles county", "ca"),
    ("orange county", "ca"),
    ("san diego county", "ca"),
    ("san francisco county", "ca"),
    ("alameda county", "ca"),
    ("miami-dade county", "fl"),
    ("broward county", "fl"),
    ("palm beach county", "fl"),
    ("hillsborough county", "fl"),
    ("pinellas county", "fl"),
    ("suffolk county", "ma"),
    ("norfolk county", "ma"),
    ("essex county", "ma"),
    ("king county", "wa"),
    ("snohomish county", "wa"),
    ("multnomah county", "or"),
    ("clatsop county", "or"),
    ("baltimore city", "md"),
    ("baltimore county", "md"),
}

STATE_FIPS_TO_ABBR = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE",
    "11": "DC", "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA",
    "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM",
    "36": "NY", "37": "NC", "38": "ND", "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY", "72": "PR",
}


def _clean(value: object) -> str:
    return str(value).strip().lower() if value is not None else ""


def _pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _resolve_paths() -> tuple[list[Path], list[Path], list[Path]]:
    root = project_root()
    if ROADS_SHP:
        roads = [root / ROADS_SHP]
    else:
        roads = sorted((root / "data" / "external" / "roads").glob("*/*.shp"))

    if PARKS_SHP:
        parks = [root / PARKS_SHP]
    else:
        # Prefer explicit parks dir; fallback to AREALM files if present.
        parks = sorted((root / "data" / "external" / "parks").glob("*/*.shp"))
        if not parks:
            parks = sorted((root / "data" / "external" / "parks").glob("*/*arealm*.shp"))

    if COAST_SHP:
        coast = [root / COAST_SHP]
    else:
        # Fallback: use AREAWATER when explicit coastline path is not provided.
        coast = sorted((root / "data" / "external" / "water").glob("*/*areawater*.shp"))

    return roads, parks, coast


def validate_local_inputs() -> tuple[list[Path], list[Path], list[Path]]:
    roads, parks, coast = _resolve_paths()
    if not roads:
        raise FileNotFoundError(
            "No roads shapefiles found. Provide ROADS_SHP_PATH or download TIGER roads into data/external/roads/*/*.shp."
        )
    if not coast:
        raise FileNotFoundError(
            "No coastline/water shapefiles found. Provide COAST_SHP_PATH or download TIGER AREAWATER into data/external/water/*/*areawater*.shp."
        )
    if not parks:
        LOGGER.warning(
            "No parks shapefiles found. distance_to_nearest_park_m will be NaN. "
            "Set PARKS_SHP_PATH or download parks files into data/external/parks/*/*.shp."
        )
    return roads, parks, coast


def haversine_vectorized_m(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    r = 6_371_000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r * c


def read_property_data() -> pd.DataFrame:
    files = sorted(project_root().glob(INPUT_GLOB))
    if not files:
        raise FileNotFoundError(f"No property parquet files found at {INPUT_GLOB}")
    LOGGER.info("Reading %s property parquet files", len(files))
    frames = []
    for path in files:
        frame = pd.read_parquet(path)
        frame["source_file"] = path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def normalize_roads(roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Keep major roads if known TIGER columns exist; otherwise keep all non-empty geometries.
    out = roads.copy()
    if "RTTYP" in out.columns:
        out = out[out["RTTYP"].isin(["I", "U", "S"])]
    elif "MTFCC" in out.columns:
        out = out[out["MTFCC"].isin(["S1100", "S1200"])]
    out = out[["geometry"]].dropna(subset=["geometry"])
    out = out[~out.geometry.is_empty]
    return out


def normalize_geometry_layer(layer: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = layer[["geometry"]].dropna(subset=["geometry"])
    out = out.explode(index_parts=False)
    out = out[~out.geometry.is_empty]
    return out


def _read_many(paths: list[Path]) -> gpd.GeoDataFrame:
    if not paths:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    frames = []
    for path in paths:
        try:
            frames.append(gpd.read_file(path))
        except Exception as exc:
            LOGGER.warning("Failed to read %s: %s", path, exc)
    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)


def _normalize_county_name_for_lookup(name: str) -> str:
    n = _clean(name)
    suffixes = [
        " county",
        " parish",
        " borough",
        " census area",
        " municipality",
        " city and borough",
        " city",
    ]
    if any(n.endswith(s) for s in suffixes):
        return n
    return f"{n} county"


def build_coastline_from_counties(counties: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if counties.empty or "NAME" not in counties.columns or "STATEFP" not in counties.columns:
        return gpd.GeoDataFrame(geometry=[], crs=counties.crs if not counties.empty else "EPSG:4326")

    work = counties.copy()
    work["state_abbr"] = work["STATEFP"].astype(str).str.zfill(2).map(STATE_FIPS_TO_ABBR).str.lower()
    work["county_key"] = work["NAME"].astype(str).map(_normalize_county_name_for_lookup)
    mask = [
        (ck, sa) in COASTAL_COUNTY_LOOKUP
        for ck, sa in zip(work["county_key"].tolist(), work["state_abbr"].tolist())
    ]
    coastal_counties = work.loc[mask]
    if coastal_counties.empty:
        LOGGER.warning("No coastal counties matched lookup table in county boundaries shapefile.")
        return gpd.GeoDataFrame(geometry=[], crs=work.crs)

    coast_geom = coastal_counties.boundary.explode(index_parts=False)
    coast = gpd.GeoDataFrame(geometry=coast_geom, crs=work.crs)
    coast = coast[coast.geometry.notna() & ~coast.geometry.is_empty]
    LOGGER.info("Derived coastline proxy from %s coastal counties", len(coastal_counties))
    return coast


def load_layers(
    roads_paths: list[Path], parks_paths: list[Path], coast_paths: list[Path]
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    roads = _read_many(roads_paths)
    parks = _read_many(parks_paths)
    coast = _read_many(coast_paths)
    counties = gpd.GeoDataFrame()
    county_path = project_root() / COUNTY_SHP
    if county_path.exists():
        try:
            counties = gpd.read_file(county_path)
        except Exception as exc:
            LOGGER.warning("Failed to read county boundaries from %s: %s", county_path, exc)

    roads = normalize_roads(roads)
    parks = normalize_geometry_layer(parks) if not parks.empty else parks
    coast = normalize_geometry_layer(coast) if not coast.empty else coast
    if not counties.empty:
        if counties.crs is None:
            counties = counties.set_crs("EPSG:4326")
        derived = build_coastline_from_counties(counties)
        if not derived.empty:
            coast = derived

    if roads.crs is None:
        roads = roads.set_crs("EPSG:4326")
    if parks.crs is None:
        parks = parks.set_crs("EPSG:4326")
    if coast.crs is None:
        coast = coast.set_crs("EPSG:4326")

    roads = roads.to_crs(METRIC_CRS)
    parks = parks.to_crs(METRIC_CRS)
    coast = coast.to_crs(METRIC_CRS)

    LOGGER.info(
        "Loaded layers: roads=%s parks=%s coastline=%s",
        len(roads),
        len(parks),
        len(coast),
    )
    return roads, parks, coast


def nearest_distance_series(
    points: gpd.GeoDataFrame, targets: gpd.GeoDataFrame
) -> pd.Series:
    if points.empty or targets.empty:
        return pd.Series(np.nan, index=points.index, dtype="float64")
    joined = gpd.sjoin_nearest(
        points[["geometry"]],
        targets[["geometry"]],
        how="left",
        distance_col="distance_m",
    )
    # If equidistant ties create duplicates, keep the min distance per point.
    return joined.groupby(joined.index)["distance_m"].min().reindex(points.index)


def compute_city_center_distance(df: pd.DataFrame, lat_col: str, lon_col: str) -> pd.Series:
    lookup_df = pd.DataFrame(
        [
            {"city_key": city, "state_key": state, "center_lat": coords[0], "center_lon": coords[1]}
            for (city, state), coords in CITY_CENTER_LOOKUP.items()
        ]
    )

    base = df.copy()
    base["city_key"] = base["city"].map(_clean) if "city" in base.columns else ""
    base["state_key"] = base["state"].map(_clean) if "state" in base.columns else ""
    merged = base.merge(lookup_df, on=["city_key", "state_key"], how="left")

    out = pd.Series(np.nan, index=df.index, dtype="float64")
    valid = (
        merged[lat_col].notna()
        & merged[lon_col].notna()
        & merged["center_lat"].notna()
        & merged["center_lon"].notna()
    )
    if valid.any():
        out.loc[valid] = haversine_vectorized_m(
            merged.loc[valid, lat_col].astype(float).to_numpy(),
            merged.loc[valid, lon_col].astype(float).to_numpy(),
            merged.loc[valid, "center_lat"].astype(float).to_numpy(),
            merged.loc[valid, "center_lon"].astype(float).to_numpy(),
        )
    return out


def compute_is_coastal_county(df: pd.DataFrame) -> pd.Series:
    county = df["county"].map(_clean) if "county" in df.columns else pd.Series("", index=df.index)
    state = df["state"].map(_clean) if "state" in df.columns else pd.Series("", index=df.index)
    key = list(zip(county, state))
    return pd.Series([k in COASTAL_COUNTY_LOOKUP for k in key], index=df.index, dtype="bool")


def overwrite_raw_files(df: pd.DataFrame) -> None:
    raw_dir = project_root() / "data" / "raw" / "homeharvest"
    ensure_dir(raw_dir)
    for source_file, group in df.groupby("source_file", sort=True):
        out_path = raw_dir / source_file
        out_df = group.drop(columns=["source_file"]).reset_index(drop=True)
        out_df.to_parquet(out_path, index=False)
        LOGGER.info("Overwrote %s with %s rows", out_path, len(out_df))


def run() -> None:
    roads_paths, parks_paths, coast_paths = validate_local_inputs()
    df = read_property_data()

    lat_col = _pick_column(df, ["latitude", "lat"])
    lon_col = _pick_column(df, ["longitude", "lon", "lng"])
    if not lat_col or not lon_col:
        raise ValueError("No latitude/longitude columns found in input data.")

    roads, parks, coast = load_layers(roads_paths, parks_paths, coast_paths)

    df["distance_to_city_center_m"] = compute_city_center_distance(df, lat_col, lon_col)
    df["distance_to_nearest_highway_m"] = np.nan
    df["distance_to_nearest_park_m"] = np.nan
    df["distance_to_coastline_m"] = np.nan

    valid_mask = df[lat_col].notna() & df[lon_col].notna()
    valid_points = gpd.GeoDataFrame(
        df.loc[valid_mask].copy(),
        geometry=gpd.points_from_xy(df.loc[valid_mask, lon_col], df.loc[valid_mask, lat_col]),
        crs="EPSG:4326",
    ).to_crs(METRIC_CRS)

    if not valid_points.empty:
        LOGGER.info("Computing nearest highway distance")
        road_dist = nearest_distance_series(valid_points, roads)
        df.loc[road_dist.index, "distance_to_nearest_highway_m"] = road_dist.to_numpy()

        if parks.empty:
            LOGGER.info("Skipping nearest park distance: no parks layer loaded")
        else:
            LOGGER.info("Computing nearest park distance")
            park_dist = nearest_distance_series(valid_points, parks)
            df.loc[park_dist.index, "distance_to_nearest_park_m"] = park_dist.to_numpy()

        LOGGER.info("Computing nearest coastline distance")
        coast_dist = nearest_distance_series(valid_points, coast)
        df.loc[coast_dist.index, "distance_to_coastline_m"] = coast_dist.to_numpy()

    df["is_coastal_county"] = compute_is_coastal_county(df)

    overwrite_raw_files(df)
    LOGGER.info("Geospatial feature overwrite complete for %s raw files", df['source_file'].nunique())


if __name__ == "__main__":
    run()
