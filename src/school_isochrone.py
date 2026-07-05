"""Valhalla pedestrian isochrone generation for the school VRU zone (Step B)

Design overview:
  1. school_origins()  — normalise OSM amenity=school to a single point
  2. snap_origins()    — snap onto the nearest network (kept separate from the origin)
  3. build_and_cache() — parallel calls to the Valhalla /isochrone API →
                         corridor buffer → save as GeoParquet

Reproducibility strategy:
  If data/processed/school_isochrones_{country}.parquet is already committed,
  Valhalla never needs to be started. build_and_cache() skips the API calls
  when the parquet already exists (idempotent).

Asymmetric design:
  URBAN  : isochrone only
  RURAL  : isochrone union RURAL_FLOOR_M circular buffer (safety-side floor)

Fallbacks (all safety-side):
  (1) snap_dist_m > MAX_SNAP_M (off-network)
  (2) Valhalla response error / timeout
  (3) area is still 0 after applying the corridor
  -> all fall back to the RURAL_FLOOR_M circular buffer (source='buffer_fallback')
"""

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import nearest_points, unary_union
from shapely.strtree import STRtree

sys.path.insert(0, "src")
from exposure_signals import BUFFER_CRS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Module constants (sensitivity-analysis targets)
# --------------------------------------------------------------------------- #
ISOCHRONE_MIN_URBAN = 3       # urban isochrone time (minutes)
ISOCHRONE_MIN_RURAL = 5       # rural isochrone time (minutes)
RURAL_FLOOR_M = 200           # rural safety-side floor circular-buffer radius (m)
ISOCHRONE_CORRIDOR_M = 15     # corridor width against degenerate geometry (m)
MAX_SNAP_M = 50               # maximum snap distance (m)
PARALLEL_WORKERS = 16         # number of parallel Valhalla API requests
VALHALLA_URL = "http://localhost:8003/isochrone"
REQUEST_TIMEOUT = 30          # per-request timeout (seconds)

PROCESSED_DIR = Path("data/processed")

# --------------------------------------------------------------------------- #
# school_origins
# --------------------------------------------------------------------------- #

def school_origins(country: str) -> gpd.GeoDataFrame:
    """Normalise amenity=school POIs to a single isochrone origin point.

    - Point       : used as-is
    - Polygon etc.: representative_point() (guaranteed to be inside the polygon)

    Also fetches land_use (URBAN/RURAL) via sjoin_nearest against segments_v_safe.parquet.
    Defaults to RURAL if no segments exist (safety-side).

    Returns
    -------
    GeoDataFrame (geometry = origin_geometry, EPSG:4326)
      columns: school_id, origin_geometry, land_use
    """
    pois_path = PROCESSED_DIR / f"osm_pois_{country}.parquet"
    pois = gpd.read_parquet(pois_path)
    schools = pois[pois["amenity"] == "school"].copy()

    if schools.empty:
        log.warning("%s: 0 amenity=school POIs found", country)
        return gpd.GeoDataFrame(
            columns=["school_id", "origin_geometry", "land_use"],
            geometry="origin_geometry",
            crs="EPSG:4326",
        )

    def _to_point(geom):
        if geom is None or geom.is_empty:
            return None
        if geom.geom_type == "Point":
            return geom
        return geom.representative_point()

    schools["origin_geometry"] = schools["geometry"].apply(_to_point)
    schools = schools[schools["origin_geometry"].notna()].copy()
    schools["school_id"] = schools.index.astype(str)

    # Fetch land_use from segments_v_safe.parquet via sjoin_nearest
    segs_path = PROCESSED_DIR / "segments_v_safe.parquet"
    land_use_series = pd.Series("RURAL", index=schools.index, name="land_use")
    try:
        segs = gpd.read_parquet(segs_path, columns=["country", "land_use", "geometry"])
        segs_c = segs[segs["country"] == country][["land_use", "geometry"]].copy()
        if not segs_c.empty:
            pts = gpd.GeoDataFrame(
                {"school_id": schools["school_id"]},
                geometry=schools["origin_geometry"].values,
                crs="EPSG:4326",
                index=schools.index,
            )
            joined = gpd.sjoin_nearest(pts, segs_c, how="left")
            # sjoin_nearest can return duplicate rows for multiple equidistant matches; keep only the first
            joined = joined[~joined.index.duplicated(keep="first")]
            land_use_series = joined["land_use"].fillna("RURAL")
    except Exception as exc:
        log.warning("failed to fetch land_use (falling back to RURAL): %s", exc)

    schools["land_use"] = land_use_series.values

    gdf = gpd.GeoDataFrame(
        {"school_id": schools["school_id"].values,
         "land_use": schools["land_use"].values,
         "origin_geometry": schools["origin_geometry"].values},
        geometry="origin_geometry",
        crs="EPSG:4326",
    )
    log.info("%s: %d schools (URBAN %d / RURAL %d)",
             country, len(gdf),
             (gdf["land_use"] == "URBAN").sum(),
             (gdf["land_use"] == "RURAL").sum())
    return gdf


# --------------------------------------------------------------------------- #
# snap_origins
# --------------------------------------------------------------------------- #

def snap_origins(origins: gpd.GeoDataFrame, country: str) -> gpd.GeoDataFrame:
    """Snap each school point onto the nearest network (kept as a separate layer).

    Snap target: nearest point on osm_roads_{country}.parquet union
    osm_pedestrian_ways_{country}.parquet (same source as the pbf).

    Added columns:
      snapped_geometry  : the snapped point (WGS84)
      snap_dist_m       : snap distance (m, after UTM projection)
      snapped           : snap_dist_m <= MAX_SNAP_M

    origin_geometry is kept (for QA / traceability).
    """
    origins = origins.copy()
    crs = BUFFER_CRS[country]

    # load the road network
    way_gdfs = []
    for name in ("osm_roads", "osm_pedestrian_ways"):
        p = PROCESSED_DIR / f"{name}_{country}.parquet"
        if p.exists():
            try:
                gdf = gpd.read_parquet(p, columns=["geometry"])
                way_gdfs.append(gdf)
            except Exception as exc:
                log.warning("skipping load of %s: %s", p.name, exc)

    if not way_gdfs:
        log.warning("%s: no road network found -> no snapping", country)
        origins["snapped_geometry"] = origins["origin_geometry"]
        origins["snap_dist_m"] = float("inf")
        origins["snapped"] = False
        return origins

    ways = pd.concat([g[["geometry"]] for g in way_gdfs], ignore_index=True)
    ways_utm = ways.to_crs(crs)
    way_geoms = ways_utm.geometry.tolist()
    tree = STRtree(way_geoms)

    # project school points to UTM
    schools_utm = origins.set_geometry("origin_geometry").to_crs(crs)

    snap_pts_utm = []
    snap_dists = []

    log.info("%s: %d snap-target road segments, %d schools", country, len(way_geoms), len(schools_utm))
    for pt in schools_utm["origin_geometry"]:
        nearest_idx = tree.nearest(pt)
        nearest_geom = way_geoms[nearest_idx]
        snap_pt, _ = nearest_points(nearest_geom, pt)
        snap_dist = pt.distance(snap_pt)
        snap_pts_utm.append(snap_pt)
        snap_dists.append(snap_dist)

    # UTM snap points -> WGS84
    snap_gdf_utm = gpd.GeoDataFrame({"geometry": snap_pts_utm}, crs=crs)
    snap_gdf_wgs84 = snap_gdf_utm.to_crs("EPSG:4326")

    origins["snapped_geometry"] = snap_gdf_wgs84.geometry.values
    origins["snap_dist_m"] = snap_dists
    origins["snapped"] = [d <= MAX_SNAP_M for d in snap_dists]

    n_snapped = sum(origins["snapped"])
    log.info("%s: snap succeeded %d / %d (MAX_SNAP_M=%dm)",
             country, n_snapped, len(origins), MAX_SNAP_M)
    log.info("%s: snap_dist_m quantiles — p50=%.1fm p90=%.1fm p99=%.1fm",
             country,
             pd.Series(snap_dists).quantile(0.50),
             pd.Series(snap_dists).quantile(0.90),
             pd.Series(snap_dists).quantile(0.99))
    return origins


# --------------------------------------------------------------------------- #
# Valhalla API
# --------------------------------------------------------------------------- #

def _fetch_single_isochrone(
    pt: Point,
    minutes: float,
    valhalla_url: str,
) -> Polygon | None:
    """Call the Valhalla /isochrone API for a single point and return a Shapely Polygon.

    Returns None on failure (the caller decides the fallback).
    search_cutoff is set to MAX_SNAP_M to suppress mis-snapping.
    """
    payload = {
        "locations": [{"lat": pt.y, "lon": pt.x}],
        "costing": "pedestrian",
        "costing_options": {"pedestrian": {"use_ferry": 0, "use_living_streets": 1}},
        "contours": [{"time": minutes}],
        "polygons": True,
        "denoise": 1.0,
        "generalize": 0,  # vertex reduction is done via simplify() in to_zone
    }
    try:
        resp = requests.post(valhalla_url, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        fc = resp.json()
        features = fc.get("features", [])
        if not features:
            return None
        geom_dict = features[0].get("geometry")
        if geom_dict is None:
            return None
        from shapely.geometry import shape
        geom = shape(geom_dict)
        return geom if not geom.is_empty else None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# to_zone
# --------------------------------------------------------------------------- #

def to_zone(
    iso_geom,           # Polygon / LineString / None from Valhalla (WGS84)
    fallback_pt: Point,  # WGS84: snapped point or original school point
    land_use: str,
    crs: str,           # UTM CRS
) -> tuple[Polygon, str]:
    """Normalise the Valhalla response geometry to a "corridor polygon with area > 0 (WGS84)".

    Returns
    -------
    (polygon_wgs84, source)
      source: 'isochrone' | 'buffer_fallback'
    """
    floor_pt_utm = (
        gpd.GeoDataFrame({"geometry": [fallback_pt]}, crs="EPSG:4326")
        .to_crs(crs)
        .geometry[0]
    )

    if iso_geom is None or iso_geom.is_empty:
        # full fallback: floor buffer
        zone_utm = floor_pt_utm.buffer(RURAL_FLOOR_M)
        zone_wgs84 = (
            gpd.GeoDataFrame({"geometry": [zone_utm]}, crs=crs)
            .to_crs("EPSG:4326")
            .geometry[0]
        )
        return zone_wgs84, "buffer_fallback"

    # project the Valhalla response to UTM and turn it into a corridor
    iso_utm = (
        gpd.GeoDataFrame({"geometry": [iso_geom]}, crs="EPSG:4326")
        .to_crs(crs)
        .geometry[0]
    )
    iso_utm = iso_utm.buffer(ISOCHRONE_CORRIDOR_M)

    # fall back if the area is still 0 after the corridor (e.g. collapsed to a point geometry)
    if iso_utm.is_empty or iso_utm.area <= 0:
        zone_utm = floor_pt_utm.buffer(RURAL_FLOOR_M)
        zone_wgs84 = (
            gpd.GeoDataFrame({"geometry": [zone_utm]}, crs=crs)
            .to_crs("EPSG:4326")
            .geometry[0]
        )
        return zone_wgs84, "buffer_fallback"

    # RURAL: isochrone union floor buffer (safety-side)
    if land_use == "RURAL":
        floor_utm = floor_pt_utm.buffer(RURAL_FLOOR_M)
        iso_utm = unary_union([iso_utm, floor_utm])

    # simplify by 2m (vertex reduction; minor impact on intersection-test precision)
    iso_utm = iso_utm.simplify(2)

    # convert back to WGS84
    zone_wgs84 = (
        gpd.GeoDataFrame({"geometry": [iso_utm]}, crs=crs)
        .to_crs("EPSG:4326")
        .geometry[0]
    )
    return zone_wgs84, "isochrone"


# --------------------------------------------------------------------------- #
# build_and_cache
# --------------------------------------------------------------------------- #

def _process_one_school(args: tuple) -> dict:
    """Per-school processing, executed in the thread pool."""
    row, valhalla_url, crs = args
    school_id = row["school_id"]
    land_use = row["land_use"]
    origin_pt = row["origin_geometry"]
    snapped = row["snapped"]
    snap_pt = row.get("snapped_geometry")
    snap_dist = row.get("snap_dist_m", float("inf"))
    minutes = ISOCHRONE_MIN_URBAN if land_use == "URBAN" else ISOCHRONE_MIN_RURAL

    # Valhalla is always queried with origin_pt.
    # Because Valhalla internally snaps against the whole PBF (including residential
    # streets and footways), an isochrone is often still obtained even when our
    # pre-snap failed.
    # snap_dist_m / snapped are kept only for QA / traceability.
    iso_geom = _fetch_single_isochrone(origin_pt, minutes, valhalla_url)

    # fallback point: the snap point if one exists, otherwise the origin
    fallback_pt = snap_pt if (snap_pt is not None) else origin_pt
    zone_geom, source = to_zone(iso_geom, fallback_pt, land_use, crs)

    return {
        "school_id": school_id,
        "geometry": zone_geom,
        "origin_geometry": origin_pt,
        "snapped_geometry": snap_pt,
        "snap_dist_m": snap_dist,
        "snapped": snapped,
        "minutes": minutes,
        "land_use": land_use,
        "source": source,
    }


def build_and_cache(
    country: str,
    valhalla_url: str = VALHALLA_URL,
    force: bool = False,
) -> gpd.GeoDataFrame:
    """Generate school isochrones and save them to GeoParquet (idempotent).

    Reads and returns the parquet if it already exists (force=True regenerates it).
    If Valhalla is not running, entries are treated as snap-failed or fall back.

    Output schema:
      school_id, geometry(Polygon WGS84), origin_geometry, snapped_geometry,
      snap_dist_m, snapped, minutes, land_use, source
    """
    out_path = PROCESSED_DIR / f"school_isochrones_{country}.parquet"
    meta_path = PROCESSED_DIR / f"school_isochrones_{country}_meta.json"

    if out_path.exists() and not force:
        log.info("%s: loading cached isochrones: %s", country, out_path)
        return gpd.read_parquet(out_path)

    crs = BUFFER_CRS[country]

    # Step 1: build school origin points
    origins = school_origins(country)
    if origins.empty:
        log.warning("%s: 0 schools found, saving an empty parquet", country)
        empty = gpd.GeoDataFrame(
            columns=["school_id", "geometry", "origin_geometry", "snapped_geometry",
                     "snap_dist_m", "snapped", "minutes", "land_use", "source"],
            geometry="geometry", crs="EPSG:4326",
        )
        empty.to_parquet(out_path)
        return empty

    # Step 2: snap onto the network
    origins = snap_origins(origins, country)

    # Step 3: parallel Valhalla isochrone fetch + corridor conversion
    args_list = [
        (row, valhalla_url, crs)
        for _, row in origins.iterrows()
    ]

    log.info("%s: starting Valhalla isochrone fetch (%d schools, %d workers)",
             country, len(args_list), PARALLEL_WORKERS)

    results = []
    fallback_count = 0
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures = {executor.submit(_process_one_school, a): i for i, a in enumerate(args_list)}
        for done_count, future in enumerate(as_completed(futures), 1):
            try:
                r = future.result()
                results.append(r)
                if r["source"] == "buffer_fallback":
                    fallback_count += 1
            except Exception as exc:
                log.error("error processing school: %s", exc)
            if done_count % 500 == 0:
                log.info("  %d / %d done (%d fallbacks)",
                         done_count, len(args_list), fallback_count)

    log.info("%s: isochrone generation complete — isochrone %d / buffer_fallback %d",
             country,
             len(results) - fallback_count,
             fallback_count)

    # Step 4: build GeoDataFrame and save
    result_gdf = gpd.GeoDataFrame(results, geometry="geometry", crs="EPSG:4326")

    # origin_geometry / snapped_geometry columns hold WGS84 Points (separate from the geometry column)
    # GeoParquet doesn't support multiple geometry columns, so auxiliary columns are saved as WKT strings
    for col in ("origin_geometry", "snapped_geometry"):
        result_gdf[col] = result_gdf[col].apply(
            lambda g: g.wkt if g is not None else None
        )

    result_gdf.to_parquet(out_path)
    log.info("%s: saved parquet: %s (%.1f MB)",
             country, out_path, out_path.stat().st_size / 1e6)

    # metadata JSON
    meta = {
        "country": country,
        "ISOCHRONE_MIN_URBAN": ISOCHRONE_MIN_URBAN,
        "ISOCHRONE_MIN_RURAL": ISOCHRONE_MIN_RURAL,
        "RURAL_FLOOR_M": RURAL_FLOOR_M,
        "ISOCHRONE_CORRIDOR_M": ISOCHRONE_CORRIDOR_M,
        "MAX_SNAP_M": MAX_SNAP_M,
        "PARALLEL_WORKERS": PARALLEL_WORKERS,
        "valhalla_url": valhalla_url,
        "n_schools": len(result_gdf),
        "n_isochrone": len(result_gdf) - fallback_count,
        "n_buffer_fallback": fallback_count,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log.info("%s: saved meta JSON: %s", country, meta_path)

    return result_gdf


def load_cached_isochrones(country: str) -> gpd.GeoDataFrame | None:
    """Read the cached isochrone parquet. Returns None if it doesn't exist."""
    p = PROCESSED_DIR / f"school_isochrones_{country}.parquet"
    if not p.exists():
        return None
    return gpd.read_parquet(p)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:  # noqa: C901
    global MAX_SNAP_M
    parser = argparse.ArgumentParser(
        description="Generate and cache the school VRU zone using Valhalla pedestrian isochrones"
    )
    parser.add_argument(
        "--country",
        choices=["thailand", "maharashtra"],
        default=None,
        help="target country (both if omitted)",
    )
    parser.add_argument(
        "--valhalla-url",
        default=VALHALLA_URL,
        help=f"Valhalla /isochrone endpoint (default: {VALHALLA_URL})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="regenerate even if the parquet already exists",
    )
    parser.add_argument(
        "--max-snap-m",
        type=float,
        default=MAX_SNAP_M,
        help=f"maximum snap distance in m (default: {MAX_SNAP_M})",
    )
    args = parser.parse_args()
    MAX_SNAP_M = args.max_snap_m

    countries = [args.country] if args.country else ["thailand", "maharashtra"]
    for country in countries:
        log.info("=== %s ===", country)
        build_and_cache(country, valhalla_url=args.valhalla_url, force=args.force)


if __name__ == "__main__":
    main()
