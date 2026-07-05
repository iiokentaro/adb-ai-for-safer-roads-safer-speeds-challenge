"""Extract OSM POI / pedestrian-way / crossing signals once per country and
cache to data/processed/, so the expensive pbf parse (tens of minutes over
a near-country-wide bbox) never has to re-run.

One pyrosm call per country pulls POIs, pedestrian ways, and crossings
together (same underlying parse pass) instead of three separate calls.

★ Grade-separated crossings are not at-grade VRU/vehicle conflict points.★
`highway=crossing` is a node tagged on a footway; the bridge/tunnel/layer
tag that says "this is a skywalk/underpass, not an at-grade crossing" lives
on the *footway way*, not the node. A crossing node sitting on a footway
tagged bridge=yes (common in Bangkok at busy intersections) means
pedestrians go OVER the road, not across it -- counting it as exposure
would be backwards. We exclude any crossing node within 3m of a
grade-separated pedestrian way before it's used in §1a/§1b.
"""

import json
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pyrosm import OSM
from shapely.geometry import Point

sys.path.insert(0, "src")
from poi_categories import (
    MAPILLARY_BOOL_COLS,
    MAPILLARY_JSON_PATHS,
    OSM_BOOL_COLS,
    SEGMENT_BOOL_COLS,
    mapillary_flags,
)
from schema import load_target

warnings.filterwarnings("ignore", category=UserWarning)

PROCESSED_DIR = "data/processed"

PBF_PATHS = {
    "thailand": "data/external/thailand-260621.osm.pbf",
    "maharashtra": "data/external/western-zone-260621.osm.pbf",
}

CUSTOM_FILTER = {
    "amenity": ["school", "marketplace", "hospital"],
    "shop": True,
    "highway": ["bus_stop", "footway", "path", "pedestrian", "steps", "crossing"],
}
TAGS_AS_COLUMNS = ["amenity", "shop", "highway", "bridge", "tunnel", "layer", "crossing", "footway"]

# Way tags that mean "grade-separated" (pedestrian path goes over/under the
# road, so a crossing on it is not an at-grade vehicle/VRU conflict point).
GRADE_SEPARATED_BRIDGE_VALUES = {"yes", "boardwalk", "viaduct", "movable", "construction"}
GRADE_SEPARATED_TUNNEL_VALUES = {"yes", "building_passage", "covered", "culvert"}


def _country_bbox(country: str) -> list[float]:
    target = load_target()
    bounds = target[target["country"] == country].total_bounds
    return list(bounds)


def extract_raw(country: str) -> gpd.GeoDataFrame:
    bbox = _country_bbox(country)
    osm = OSM(PBF_PATHS[country], bounding_box=bbox)
    return osm.get_data_by_custom_criteria(
        custom_filter=CUSTOM_FILTER, tags_as_columns=TAGS_AS_COLUMNS, filter_type="keep"
    )


def _is_grade_separated(row) -> bool:
    bridge = row.get("bridge")
    tunnel = row.get("tunnel")
    layer = row.get("layer")
    if bridge in GRADE_SEPARATED_BRIDGE_VALUES:
        return True
    if tunnel in GRADE_SEPARATED_TUNNEL_VALUES:
        return True
    if layer is not None and str(layer) not in ("nan", "None"):
        try:
            if float(str(layer).split(";")[0]) != 0:
                return True
        except ValueError:
            pass
    return False


def split_signals(raw: gpd.GeoDataFrame, crs_for_buffer: str | int) -> dict[str, gpd.GeoDataFrame]:
    pois = raw[raw["amenity"].notna() | raw["shop"].notna() | (raw["highway"] == "bus_stop")].copy()

    pedestrian_ways = raw[
        raw["highway"].isin(["footway", "path", "pedestrian", "steps"]) & (raw.geometry.type == "LineString")
    ].copy()
    pedestrian_ways["grade_separated"] = pedestrian_ways.apply(_is_grade_separated, axis=1)

    crossings_all = raw[
        ((raw["highway"] == "crossing") | (raw["footway"] == "crossing")) & (raw.geometry.type == "Point")
    ].copy()

    crossings_at_grade = crossings_all
    grade_sep_ways = pedestrian_ways[pedestrian_ways["grade_separated"]]
    if len(grade_sep_ways) and len(crossings_all):
        buffered = grade_sep_ways.to_crs(crs_for_buffer).copy()
        buffered["geometry"] = buffered.geometry.buffer(3)
        crossings_proj = crossings_all.to_crs(crs_for_buffer)
        hits = gpd.sjoin(crossings_proj, buffered[["geometry"]], predicate="within")
        at_grade_mask = ~crossings_all.index.isin(hits.index.unique())
        crossings_at_grade = crossings_all[at_grade_mask].copy()

    n_excluded = len(crossings_all) - len(crossings_at_grade)
    print(f"  excluded {n_excluded} / {len(crossings_all)} crossing nodes as grade-separated (bridge/tunnel/layer)")

    return {"pois": pois, "pedestrian_ways": pedestrian_ways, "crossings_at_grade": crossings_at_grade}


# UTM zone used for the 3m grade-separation buffer (dominant zone per country
# confirmed via geometry.py; precision to the metre doesn't matter here,
# just "is this projected in metres").
BUFFER_CRS = {"thailand": "EPSG:32647", "maharashtra": "EPSG:32643"}


def extract_and_cache(country: str) -> dict[str, gpd.GeoDataFrame]:
    raw = extract_raw(country)
    signals = split_signals(raw, BUFFER_CRS[country])
    for name, gdf in signals.items():
        gdf.to_parquet(f"{PROCESSED_DIR}/osm_{name}_{country}.parquet")
    return signals


def load_cached_signals(country: str) -> dict[str, gpd.GeoDataFrame]:
    return {
        name: gpd.read_parquet(f"{PROCESSED_DIR}/osm_{name}_{country}.parquet")
        for name in ("pois", "pedestrian_ways", "crossings_at_grade")
    }


# dwithin distance per segment land_use (metres, applied after UTM projection).
POI_BUFFER_M = {"URBAN": 200, "RURAL": 400}
CROSSING_BUFFER_M = 25


def _tag_osm_pois(pois: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    pois = pois.copy()
    pois["source"] = "osm"
    pois["is_school"] = pois["amenity"] == "school"
    pois["is_hospital"] = pois["amenity"] == "hospital"
    pois["is_marketplace"] = pois["amenity"] == "marketplace"
    pois["is_shop"] = pois["shop"].notna()
    pois["is_bus_stop"] = pois["highway"] == "bus_stop"
    for col in MAPILLARY_BOOL_COLS:
        pois[col] = False
    return pois


def _flatten_mapillary_json(country: str) -> gpd.GeoDataFrame:
    json_path = Path(MAPILLARY_JSON_PATHS[country])
    if not json_path.exists():
        return gpd.GeoDataFrame(
            columns=["source", "object_value", *OSM_BOOL_COLS, *MAPILLARY_BOOL_COLS, "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    with json_path.open() as f:
        by_objectid = json.load(f)

    rows: list[dict] = []
    for features in by_objectid.values():
        for feat in features:
            geom = feat.get("geometry") or {}
            coords = geom.get("coordinates")
            if not coords:
                continue
            object_value = feat["object_value"]
            flags = mapillary_flags(object_value)
            rows.append(
                {
                    "source": "mapillary",
                    "object_value": object_value,
                    **{col: False for col in OSM_BOOL_COLS},
                    **flags,
                    "geometry": Point(coords[0], coords[1]),
                }
            )

    if not rows:
        return gpd.GeoDataFrame(
            columns=["source", "object_value", *OSM_BOOL_COLS, *MAPILLARY_BOOL_COLS, "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def load_mapillary_pois(country: str) -> gpd.GeoDataFrame:
    cache_path = Path(f"{PROCESSED_DIR}/mapillary_pois_{country}.parquet")
    if cache_path.exists():
        return gpd.read_parquet(cache_path)

    gdf = _flatten_mapillary_json(country)
    gdf.to_parquet(cache_path)
    return gdf


def _combined_pois(country: str) -> gpd.GeoDataFrame:
    osm = _tag_osm_pois(load_cached_signals(country)["pois"])
    map_pois = load_mapillary_pois(country)
    if len(map_pois) == 0:
        return osm
    if len(osm) == 0:
        return map_pois
    return pd.concat([osm, map_pois], ignore_index=True)


def _apply_dwithin_hits(
    gdf: gpd.GeoDataFrame,
    seg_idx: pd.Index,
    joined: gpd.GeoDataFrame,
) -> None:
    if joined.empty:
        return

    osm_hits = joined[joined["source"] == "osm"]
    map_hits = joined[joined["source"] == "mapillary"]

    if not osm_hits.empty:
        osm_agg = osm_hits.groupby(osm_hits.index)[OSM_BOOL_COLS].max()
        gdf.loc[osm_agg.index, OSM_BOOL_COLS] = osm_agg.astype(bool).values
        osm_cat = osm_agg.sum(axis=1).astype(int)
        gdf.loc[osm_cat.index, "osm_poi_category_count"] = osm_cat.values

    if not map_hits.empty:
        map_agg = map_hits.groupby(map_hits.index)[MAPILLARY_BOOL_COLS].max()
        gdf.loc[map_agg.index, MAPILLARY_BOOL_COLS] = map_agg.astype(bool).values
        map_vru = map_agg["map_is_pedestrian"] | map_agg["map_is_bicycle"] | map_agg["map_is_school"]
        gdf.loc[map_vru.index, "is_mapillary_vru"] = map_vru.values

    counts = joined.groupby(joined.index).size()
    gdf.loc[counts.index, "poi_count"] = counts.values

    # Combined segment-facing bool flags (OR across sources).
    gdf.loc[seg_idx, "is_school"] = (
        gdf.loc[seg_idx, "is_school"] | gdf.loc[seg_idx, "map_is_school"]
    )
    gdf.loc[seg_idx, "is_hospital"] = (
        gdf.loc[seg_idx, "is_hospital"] | gdf.loc[seg_idx, "map_is_hospital"]
    )
    gdf.loc[seg_idx, "is_pedestrian"] = gdf.loc[seg_idx, "map_is_pedestrian"]
    gdf.loc[seg_idx, "is_bicycle"] = gdf.loc[seg_idx, "map_is_bicycle"]


# road_class values that are access-controlled by definition: a Mapillary
# VRU detection (school-zone sign, crosswalk marking, bicycle marking) within
# the dwithin distance does not imply an at-grade conflict there, the same
# reasoning safe_speed.py's motorway override already applies.
MAPILLARY_VRU_EXCLUDED_ROAD_CLASSES = ["motorway", "trunk"]


def _apply_isochrone_school_hits(
    gdf: gpd.GeoDataFrame,
    isochrones: gpd.GeoDataFrame,
    country: str,
) -> None:
    """Overwrite is_school with the isochrone intersection (Step C).

    URBAN: reset the dwithin-derived is_school to isochrone-only.
    RURAL: keep the dwithin result (floor) and OR it with the isochrone.
    osm_poi_category_count (exposure axis) is left unchanged from the dwithin result.
    """
    mask_country = gdf["country"] == country
    if not mask_country.any() or isochrones.empty:
        return

    for land_use in ("URBAN", "RURAL"):
        mask = mask_country & (gdf["land_use"] == land_use)
        if not mask.any():
            continue

        iso_lu = isochrones[isochrones["land_use"] == land_use]
        if iso_lu.empty:
            continue

        # URBAN: reset the dwithin-derived is_school (replaced with isochrone-only)
        if land_use == "URBAN":
            gdf.loc[mask, "is_school"] = False

        # intersects sjoin between the isochrone polygon and the segments
        iso_utm = iso_lu[["geometry"]].to_crs(BUFFER_CRS[country])
        segs_utm = gdf.loc[mask, ["geometry"]].to_crs(BUFFER_CRS[country])
        joined = gpd.sjoin(segs_utm, iso_utm, predicate="intersects")
        if not joined.empty:
            gdf.loc[joined.index.unique(), "is_school"] = True


def add_poi_proximity(
    gdf: gpd.GeoDataFrame,
    use_isochrone: bool = True,
) -> gpd.GeoDataFrame:
    """Attach POI proximity via UTM dwithin (no segment buffer polygons).

    OSM POIs and Mapillary map_features points are joined at 200 m (urban) or
    400 m (rural). Category bools are aggregated with max(); poi_count is the
    total hit count across both sources.

    When use_isochrone=True (default) and
    data/processed/school_isochrones_{country}.parquet exists:
      - URBAN: is_school is replaced from dwithin with the isochrone intersect.
      - RURAL: dwithin result (floor) union isochrone (safety-side union).
    If the parquet does not exist, falls back to the legacy dwithin (backward compatible).
    osm_poi_category_count (exposure axis) always keeps the dwithin result.

    `is_mapillary_vru` (school-zone sign / crosswalk marking / bicycle marking
    detected within the dwithin distance) is the Mapillary-only sub-signal used
    by exposure_level. `is_vru` is the source-agnostic VRU trigger that drives
    V_safe: `is_mapillary_vru OR is_school`. Both are forced to False for
    motorway/trunk and for any segment with `is_separated==True`.
    """
    gdf = gdf.copy()
    for col in SEGMENT_BOOL_COLS:
        gdf[col] = False
    for col in MAPILLARY_BOOL_COLS:
        gdf[col] = False
    gdf["poi_count"] = 0
    gdf["osm_poi_category_count"] = 0
    gdf["is_mapillary_vru"] = False
    gdf["is_vru"] = False

    poi_cols = ["source", "geometry", *OSM_BOOL_COLS, *MAPILLARY_BOOL_COLS]

    for country in gdf["country"].unique():
        pois = _combined_pois(country)
        if len(pois) == 0:
            continue

        for land_use, dist_m in POI_BUFFER_M.items():
            mask = (gdf["country"] == country) & (gdf["land_use"] == land_use)
            if not mask.any():
                continue
            seg_idx = gdf.index[mask]
            segs = gdf.loc[mask, ["geometry"]].to_crs(BUFFER_CRS[country])
            pois_utm = pois[poi_cols].to_crs(BUFFER_CRS[country])
            joined = gpd.sjoin(segs, pois_utm, predicate="dwithin", distance=dist_m)
            _apply_dwithin_hits(gdf, seg_idx, joined)

    # Overwrite is_school if an isochrone is available (osm_poi_category_count is unchanged)
    if use_isochrone:
        for country in gdf["country"].unique():
            iso_path = Path(f"data/processed/school_isochrones_{country}.parquet")
            if iso_path.exists():
                try:
                    isochrones = gpd.read_parquet(iso_path)
                    _apply_isochrone_school_hits(gdf, isochrones, country)
                except Exception as exc:
                    warnings.warn(
                        f"{country}: failed to load isochrone, keeping dwithin: {exc}",
                        stacklevel=2,
                    )

    # is_vru is the source-agnostic VRU trigger that actually drives V_safe
    # (safe_speed.classify_collision_type): a Mapillary VRU detection OR an OSM
    # amenity=school node within the dwithin distance. OSM schools are included
    # because a school almost never grade-separates from a non-access-controlled
    # road, and Mapillary coverage is incomplete -- so an OSM school node is
    # treated as concrete at-grade VRU evidence here, the same way as a Mapillary
    # school-zone sign. is_mapillary_vru is kept unchanged as the Mapillary-only
    # exposure sub-signal (exposure_level.URBAN_SIGNALS); the OSM school is also
    # already represented there via osm_poi_category_count (double counting is
    # accepted, mirroring the existing Mapillary VRU behaviour).
    gdf["is_vru"] = gdf["is_mapillary_vru"] | gdf["is_school"]

    if "is_separated" in gdf.columns:
        not_at_grade = gdf["road_class"].isin(MAPILLARY_VRU_EXCLUDED_ROAD_CLASSES) | gdf["is_separated"]
        gdf.loc[not_at_grade, "is_mapillary_vru"] = False
        gdf.loc[not_at_grade, "is_vru"] = False

    return gdf


add_poi_count = add_poi_proximity


def add_crossing_signal(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """has_crossing/crossing_count are urban-only signals (OSM crossing tagging
    is near-zero in rural areas); leaving rural at 0 here is intentional and
    is NOT treated as "low exposure" downstream (see exposure_level.py)."""
    gdf = gdf.copy()
    gdf["has_crossing"] = False
    gdf["crossing_count"] = 0

    for country in gdf["country"].unique():
        crossings = load_cached_signals(country)["crossings_at_grade"]
        mask = gdf["country"] == country
        if len(crossings) == 0 or not mask.any():
            continue
        crossings_utm = crossings.to_crs(BUFFER_CRS[country])

        segs = gdf.loc[mask, ["geometry"]].to_crs(BUFFER_CRS[country]).copy()
        segs["geometry"] = segs.geometry.buffer(CROSSING_BUFFER_M)
        joined = gpd.sjoin(crossings_utm[["geometry"]], segs, predicate="within")
        counts = joined.groupby("index_right").size()
        gdf.loc[counts.index, "crossing_count"] = counts.values
        gdf.loc[counts.index, "has_crossing"] = True

    return gdf


if __name__ == "__main__":
    for country in PBF_PATHS:
        print(f"--- {country} ---")
        signals = extract_and_cache(country)
        for name, gdf in signals.items():
            print(f"{name}: {len(gdf)}")
        print()
