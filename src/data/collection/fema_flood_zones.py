"""Append FEMA NFHL flood-zone classifications to HomeHarvest raw property files."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterable, Optional

import geopandas as gpd
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
FEMA_NFHL_MAPSERVER = os.getenv(
    "FEMA_NFHL_MAPSERVER_URL",
    "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer",
)
FEMA_LAYER_ID = os.getenv("FEMA_NFHL_LAYER_ID")
FEMA_PAGE_SIZE = int(os.getenv("FEMA_NFHL_PAGE_SIZE", "2000"))
FEMA_ID_CHUNK_SIZE = int(os.getenv("FEMA_NFHL_ID_CHUNK_SIZE", "100"))


def _pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _layer_url(base_url: str, layer_id: int) -> str:
    return f"{base_url.rstrip('/')}/{layer_id}"


def _get_json(session: requests.Session, url: str, timeout: int = 30) -> dict:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    try:
        payload = response.json()
    except requests.exceptions.JSONDecodeError as exc:
        raise RuntimeError(f"Non-JSON response from {url}: {response.text[:200]}") from exc
    if "error" in payload:
        raise RuntimeError(f"ArcGIS error from {url}: {payload['error']}")
    return payload


def resolve_layer_id(base_url: str, session: requests.Session) -> int:
    """Resolve NFHL layer containing FLD_ZONE if layer id is not explicitly set."""
    if FEMA_LAYER_ID is not None and str(FEMA_LAYER_ID).strip():
        return int(FEMA_LAYER_ID)

    metadata = _get_json(session, f"{base_url}?f=pjson", timeout=30)
    layers = metadata.get("layers", [])
    for layer in layers:
        lid = layer["id"]
        info = _get_json(session, f"{base_url}/{lid}?f=pjson", timeout=30)
        field_names = {f["name"] for f in info.get("fields", [])}
        if "FLD_ZONE" in field_names:
            LOGGER.info("Auto-selected FEMA layer %s (%s)", lid, layer.get("name", "unknown"))
            return int(lid)

    raise RuntimeError("Could not find an NFHL layer with FLD_ZONE field.")


def query_nfhl_geojson_for_bbox(
    layer_url: str,
    bbox: tuple[float, float, float, float],
    session: requests.Session,
) -> gpd.GeoDataFrame:
    """Query FEMA NFHL polygons intersecting bbox via ArcGIS IDs + chunked feature fetch."""
    xmin, ymin, xmax, ymax = bbox
    id_params = {
        "where": "1=1",
        "geometry": f"{xmin},{ymin},{xmax},{ymax}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "returnIdsOnly": "true",
        "f": "json",
    }
    id_resp = session.get(f"{layer_url}/query", params=id_params, timeout=60)
    id_resp.raise_for_status()
    id_payload = id_resp.json()
    object_ids = id_payload.get("objectIds", []) or []
    if not object_ids:
        return gpd.GeoDataFrame(columns=["FLD_ZONE", "geometry"], geometry="geometry", crs="EPSG:4326")

    def fetch_chunk(ids: list[int]) -> list[dict]:
        if not ids:
            return []
        q_params = {
            "objectIds": ",".join(map(str, ids)),
            "outFields": "FLD_ZONE",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
        }
        response = session.post(f"{layer_url}/query", data=q_params, timeout=60)
        if response.status_code >= 500 and len(ids) > 1:
            mid = len(ids) // 2
            left = fetch_chunk(ids[:mid])
            right = fetch_chunk(ids[mid:])
            return left + right
        if response.status_code >= 500 and len(ids) == 1:
            LOGGER.warning("Skipping problematic FEMA objectId %s due to server error", ids[0])
            return []
        response.raise_for_status()
        payload = response.json()
        return payload.get("features", [])

    all_features: list[dict] = []
    for start in range(0, len(object_ids), FEMA_ID_CHUNK_SIZE):
        chunk = object_ids[start : start + FEMA_ID_CHUNK_SIZE]
        all_features.extend(fetch_chunk(chunk))

    if not all_features:
        return gpd.GeoDataFrame(columns=["FLD_ZONE", "geometry"], geometry="geometry", crs="EPSG:4326")

    gdf = gpd.GeoDataFrame.from_features(all_features, crs="EPSG:4326")
    if "FLD_ZONE" not in gdf.columns:
        gdf["FLD_ZONE"] = pd.NA
    gdf = gdf[["FLD_ZONE", "geometry"]].dropna(subset=["geometry"])
    return gdf


def get_flood_zone_for_point(
    lat: float,
    lon: float,
    layer_url: str,
    session: requests.Session,
) -> Optional[str]:
    """Reusable single-point helper (main pipeline uses batch spatial joins)."""
    bbox = (lon, lat, lon, lat)
    polygons = query_nfhl_geojson_for_bbox(layer_url, bbox, session)
    if polygons.empty:
        return None
    points = gpd.GeoDataFrame(
        [{"geometry": gpd.points_from_xy([lon], [lat])[0]}],
        geometry="geometry",
        crs="EPSG:4326",
    )
    join = gpd.sjoin(points, polygons, how="left", predicate="within")
    value = join["FLD_ZONE"].iloc[0] if not join.empty else None
    return None if pd.isna(value) else str(value)


def enrich_file(path: Path, layer_url: str, session: requests.Session) -> None:
    df = pd.read_parquet(path)
    lat_col = _pick_column(df, ["latitude", "lat"])
    lon_col = _pick_column(df, ["longitude", "lon", "lng"])
    if not lat_col or not lon_col:
        LOGGER.warning("Skipping %s: missing lat/lon columns", path.name)
        return

    df["flood_zone"] = pd.NA
    valid = df[lat_col].notna() & df[lon_col].notna()
    if not valid.any():
        LOGGER.warning("Skipping %s: no valid coordinates", path.name)
        ensure_dir(path.parent)
        df.to_parquet(path, index=False)
        return

    points = gpd.GeoDataFrame(
        df.loc[valid].copy(),
        geometry=gpd.points_from_xy(df.loc[valid, lon_col], df.loc[valid, lat_col]),
        crs="EPSG:4326",
    )
    xmin, ymin, xmax, ymax = points.total_bounds
    LOGGER.info("Querying FEMA NFHL for %s (bbox=[%.4f, %.4f, %.4f, %.4f])", path.name, xmin, ymin, xmax, ymax)

    polygons = query_nfhl_geojson_for_bbox(layer_url, (xmin, ymin, xmax, ymax), session)
    if polygons.empty:
        LOGGER.info("No FEMA flood polygons intersecting %s", path.name)
        ensure_dir(path.parent)
        df.to_parquet(path, index=False)
        return

    joined = gpd.sjoin(points, polygons, how="left", predicate="within")
    zones = joined.groupby(joined.index)["FLD_ZONE"].first().reindex(points.index)
    df.loc[zones.index, "flood_zone"] = zones.astype("string")

    ensure_dir(path.parent)
    df.to_parquet(path, index=False)
    LOGGER.info("Overwrote %s with flood_zone for %s rows", path, int(valid.sum()))


def run() -> None:
    files = sorted(project_root().glob(INPUT_GLOB))
    if not files:
        raise FileNotFoundError(f"No property parquet files found at {INPUT_GLOB}")

    with requests.Session() as session:
        layer_id = resolve_layer_id(FEMA_NFHL_MAPSERVER, session)
        layer_url = _layer_url(FEMA_NFHL_MAPSERVER, layer_id)
        LOGGER.info("Using FEMA NFHL layer URL: %s", layer_url)

        for path in files:
            enrich_file(path, layer_url, session)

    LOGGER.info("FEMA flood-zone enrichment complete for %s files", len(files))


if __name__ == "__main__":
    run()
