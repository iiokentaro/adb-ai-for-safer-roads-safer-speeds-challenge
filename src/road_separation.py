"""Derive is_separated from OSM road-network tags (user-specified criteria,
not road_class as a proxy):

A way is separated if ANY of:
  (1) highway in {motorway, trunk} AND oneway == 'yes'
  (2) lanes:divided == 'yes'
  (3) dual_carriageway == 'yes'

(2)/(3) were empirically absent from a Bangkok sample check -- this dataset
doesn't seem to carry those tags -- but are still implemented in case they
appear elsewhere (e.g. Maharashtra) or in future pbf updates.

Our Overture segments don't 1:1 match OSM ways (different source, different
digitization), so each segment is matched to the OSM road ways running
alongside it within a buffer, and is_separated is True only if EVERY
matched way satisfies the criteria above -- one non-separated stretch
within an aggregated segment makes the whole segment is_separated=False
(per the user's explicit instruction; conservative/safe-side by construction
like the rest of the exposure pipeline).
"""

import sys
import warnings

import geopandas as gpd
from pyrosm import OSM

sys.path.insert(0, "src")
from exposure_signals import (
    BUFFER_CRS,
    GRADE_SEPARATED_BRIDGE_VALUES,
    GRADE_SEPARATED_TUNNEL_VALUES,
    PBF_PATHS,
    _country_bbox,
)
from schema import load_target

warnings.filterwarnings("ignore", category=UserWarning)

PROCESSED_DIR = "data/processed"
ROAD_HIGHWAY_VALUES = ["motorway", "trunk", "primary", "secondary"]
MATCH_BUFFER_M = 15  # tolerance for Overture-vs-OSM digitization offset
MATCH_OVERLAP_FRACTION = 0.5  # >= this fraction of the OSM way's length must fall in the segment's corridor


def extract_road_network(country: str) -> gpd.GeoDataFrame:
    bbox = _country_bbox(country)
    osm = OSM(PBF_PATHS[country], bounding_box=bbox)
    roads = osm.get_data_by_custom_criteria(
        custom_filter={"highway": ROAD_HIGHWAY_VALUES},
        tags_as_columns=["highway", "oneway"],
        filter_type="keep",
    )
    return roads[roads.geometry.type == "LineString"].copy()


def _is_separated_way(row) -> bool:
    if row.get("highway") in ("motorway", "trunk") and row.get("oneway") == "yes":
        return True
    tags = row.get("tags")
    if isinstance(tags, str):
        import json

        try:
            tags = json.loads(tags)
        except ValueError:
            tags = {}
    if isinstance(tags, dict):
        if tags.get("lanes:divided") == "yes":
            return True
        if tags.get("dual_carriageway") == "yes":
            return True
    return False


def _is_grade_separated_way(row) -> bool:
    """A way is grade-separated (bridge/flyover or tunnel/underpass) if its
    `tags` JSON carries bridge/tunnel/layer values meaning "physically above
    or below the surrounding network" -- same value sets exposure_signals.py
    uses for pedestrian-way crossings, reused here for junction_speed_cap.py's
    at-grade assumption (a flyover passing over an at-grade junction is not
    itself an at-grade side-impact conflict point)."""
    tags = row.get("tags")
    if isinstance(tags, str):
        import json

        try:
            tags = json.loads(tags)
        except ValueError:
            tags = {}
    if not isinstance(tags, dict):
        return False
    if tags.get("bridge") in GRADE_SEPARATED_BRIDGE_VALUES:
        return True
    if tags.get("tunnel") in GRADE_SEPARATED_TUNNEL_VALUES:
        return True
    layer = tags.get("layer")
    if layer is not None and str(layer) not in ("nan", "None"):
        try:
            if float(str(layer).split(";")[0]) != 0:
                return True
        except ValueError:
            pass
    return False


def cache_road_network(country: str) -> gpd.GeoDataFrame:
    roads = extract_road_network(country)
    roads["is_separated_way"] = roads.apply(_is_separated_way, axis=1)
    roads["is_grade_separated_way"] = roads.apply(_is_grade_separated_way, axis=1)
    roads.to_parquet(f"{PROCESSED_DIR}/osm_roads_{country}.parquet")
    return roads


def add_is_separated(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf["is_separated"] = False
    # high: an OSM way was actually matched and inspected. low: no OSM road
    # of any class fell within the buffer, so is_separated=False here is a
    # blind default, not a confirmed "not separated" -- distinguishing this
    # matters because is_separated=False drives motorway segments to
    # head_on (70 km/h); without this flag the scoring cannot tell "genuinely
    # undivided" apart from "OSM just didn't have a matching way" (30% of
    # Thailand's motorway segments, per README notes).
    gdf["is_separated_confidence"] = "low"
    # ANY matched way grade-separated (bridge/tunnel/layer!=0) -> the segment is
    # grade_separated=True (safe-side: a structure is involved somewhere along
    # it, so junction_speed_cap.py should not assume an at-grade conflict).
    # Computed live from each cached osm_roads_{country}.parquet's `tags` JSON
    # column rather than read from a precomputed column, so this works against
    # parquets built before is_grade_separated_way existed (no re-extraction
    # needed -- see road_separation.py module docstring / plan doc).
    gdf["is_grade_separated"] = False

    for country in gdf["country"].unique():
        roads = gpd.read_parquet(f"{PROCESSED_DIR}/osm_roads_{country}.parquet")
        if len(roads) == 0:
            continue
        crs = BUFFER_CRS[country]
        roads_utm = roads.to_crs(crs).copy()
        roads_utm["way_length"] = roads_utm.geometry.length
        roads_utm["is_grade_separated_way"] = roads_utm.apply(_is_grade_separated_way, axis=1)

        mask = gdf["country"] == country
        segs = gdf.loc[mask, ["geometry"]].to_crs(crs).copy()
        segs["corridor"] = segs.geometry.buffer(MATCH_BUFFER_M)

        corridors = gpd.GeoDataFrame(segs[["corridor"]], geometry="corridor", crs=crs)
        joined = gpd.sjoin(roads_utm, corridors, predicate="intersects")

        separated_results = {}
        matched_results = {}
        grade_separated_results = {}
        for seg_idx, group in joined.groupby("index_right"):
            corridor = corridors.loc[seg_idx, "corridor"]
            matched_any = False
            all_separated = True
            grade_separated_any = False
            for _, way in group.iterrows():
                overlap_len = way.geometry.intersection(corridor).length
                if way["way_length"] == 0 or overlap_len / way["way_length"] < MATCH_OVERLAP_FRACTION:
                    continue
                matched_any = True
                if not way["is_separated_way"]:
                    all_separated = False
                if way["is_grade_separated_way"]:
                    grade_separated_any = True
            separated_results[seg_idx] = matched_any and all_separated
            matched_results[seg_idx] = matched_any
            grade_separated_results[seg_idx] = grade_separated_any

        for seg_idx, is_sep in separated_results.items():
            gdf.loc[seg_idx, "is_separated"] = is_sep
        for seg_idx, matched in matched_results.items():
            gdf.loc[seg_idx, "is_separated_confidence"] = "high" if matched else "low"
        for seg_idx, grade_sep in grade_separated_results.items():
            gdf.loc[seg_idx, "is_grade_separated"] = grade_sep

    return gdf


if __name__ == "__main__":
    for country in PBF_PATHS:
        print(f"--- {country} ---")
        roads = cache_road_network(country)
        print(f"road ways: {len(roads)}, is_separated_way=True: {roads['is_separated_way'].sum()}")

    target = load_target()
    target = add_is_separated(target)
    print()
    print(target.groupby(["country", "road_class"])["is_separated"].sum())
    print()
    print("total is_separated segments:", target["is_separated"].sum(), "/", len(target))
