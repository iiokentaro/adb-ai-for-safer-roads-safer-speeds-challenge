"""P1-0: Stage-2 refinement — clip long segments at influence-zone boundaries
so that V_safe is localized to the parts of a segment that are actually near a
VRU detection (Mapillary VRU or OSM amenity=school) or a junction node.

Design (two-stage pipeline):
- Stage 1 (existing pipeline): whole-segment spatial join flags is_vru and
  near_junction, then assigns a single V_safe to the entire segment.
- Stage 2 (this module): for segments where Stage 1 raised either flag, clips the
  geometry at the influence-zone boundary into an "influenced" portion and a
  "non-influenced" portion. Stage 1 geometry-driven functions are then re-run on the
  resulting geometries; influenced children get the lower V_safe, non-influenced
  children retain the original higher V_safe.

Performance — dwithin pre-filter + group-level closing:
  (1) The influence polygon is built from only the POIs/junctions that are within the
      influence radius of the *target* segments (those that Stage 1 flagged), not the
      full country-wide set. gpd.sjoin(predicate="dwithin") is used as a pre-filter
      before buffering, so unary_union operates on a small local subset.
  (2) The morphological closing (buffer +25 m / -25 m) is computed ONCE per
      (country, land_use) group, not once per segment. This is the critical fix: the
      closed polygon is reused for every segment in the group's intersection/difference
      calls. Moving the closing out of the per-segment loop eliminates what was the
      dominant bottleneck.

  Why a buffer polygon is still needed (not just dwithin):
  Stage 1 uses dwithin as a pure true/false predicate. Stage 2 must *cut* the
  geometry at the influence boundary, which requires an actual polygon — dwithin
  produces a boolean, not a geometry. The buffer/unary_union is therefore unavoidable
  for the intersection/difference operations, but restricting it to nearby points and
  computing the closing once per group makes it fast.

Sliver safety-side closing (Safe System precautionary principle):
  Short non-influence gaps (< MIN_PIECE_M) between two influence zones along a segment
  are absorbed into the influence zone rather than being left as independent "safe"
  pieces. This is implemented via morphological closing on the influence polygon
  (buffer out by MIN_PIECE_M/2, then back in), computed once per (country, land_use)
  group. Any gap shorter than MIN_PIECE_M between two influence buffers is reclassified
  as "influenced" (lower V_safe). Rationale: when a gap is too short to confidently
  classify as outside an influence zone, classify it as influenced — "fail safe"
  under the Safe System precautionary principle. This behaviour is documented here
  and in README.md.

Isochrone splits are exempt from MIN_PIECE_M (no minimum-clip-length gate):
  build_influence_polygon_near() returns a (buffer_poly, iso_poly) pair instead of a
  single unioned polygon. buffer_poly covers Mapillary VRU buffers, junction buffers,
  and the circular-fallback school buffer (used only when no isochrone parquet exists
  for that country) — the MIN_PIECE_M gate and morphological closing still apply here,
  unchanged. iso_poly covers only actual school-isochrone polygons; for these, a
  segment is split as soon as it intersects the isochrone, however small the overlap
  (down to the sub-metre level), because the isochrone boundary is itself a considered
  travel-time estimate of the school VRU zone — an overlap of any size means part of
  the segment genuinely falls inside that zone, so it must not inherit the parent's
  higher V_safe. A segment fully contained in the isochrone (non-influenced remainder
  ~0) is correctly left unsplit. This avoids the previous over-triggering where a
  school isochrone grazing a long rural segment by a few metres forced the entire
  segment to 30 km/h.

sample_size_total length-proportional allocation:
  Each child segment receives: child_SST = parent_SST * (child_len / parent_len).
  Since delta_fatal_abs = (delta_fatal_percent / 100) * sample_size_total and
  delta_fatal_percent is an intensive quantity, this preserves the parent's
  delta_fatal_abs total across children. This is a proportional allocation for
  benefit-index computation, NOT a physical redistribution of probe samples.
  See README.md and priority_lists.py for context.

segment_id normalisation:
  All rows get segment_id converted to "{country}_{original_objectid}" (string).
  Children get "{country}_{parent_objectid}-{k}". This guarantees global uniqueness
  even though Thailand and Maharashtra both use OBJECTID sequences starting from 1.
"""

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import GeometryCollection, LineString, MultiLineString
from shapely.ops import unary_union

sys.path.insert(0, "src")
from exposure_signals import (
    BUFFER_CRS,
    POI_BUFFER_M,
    _tag_osm_pois,
    load_cached_signals,
    load_mapillary_pois,
)
from junction_speed_cap import JUNCTION_BUFFER_M, load_cached_junctions

# Minimum length (metres) for an independent segment piece after clipping.
MIN_PIECE_M = 50

# Non-geometry columns copied verbatim from parent to each child.
# sample_size_total is excluded — it is length-proportionally allocated.
# All geometry-driven signal columns are re-computed by the Stage-1 functions.
INHERIT_COLS = [
    "segment_id", "country", "road_class", "land_use", "urban_pc",
    "speed_limit", "median_speed", "f85_speed",
    "analysis_status", "exclude_from_speedspi", "data_quality_flag",
]

_MAPILLARY_VRU_FLAGS = ["map_is_pedestrian", "map_is_bicycle", "map_is_school"]


def build_influence_polygon_near(
    target_segs_utm: gpd.GeoDataFrame,
    country: str,
    land_use: str,
):
    """Build the UTM influence polygons from only the POIs/junctions near target_segs_utm.

    Uses gpd.sjoin(predicate="dwithin") to restrict buffering to the few points that
    are actually within the influence radius of any target segment, avoiding the
    country-wide unary_union bottleneck.

    target_segs_utm: segments projected to BUFFER_CRS[country].
    Returns a (buffer_poly, iso_poly) tuple, each the unary_union of the relevant
    geometries or None if none exist:
      - buffer_poly: Mapillary VRU buffers + junction buffers + circular-fallback
        school buffers (i.e. every influence source that is NOT an actual isochrone
        polygon). MIN_PIECE_M gating still applies to this source.
      - iso_poly: union of actual school-isochrone polygons only (when the isochrone
        parquet was readable). MIN_PIECE_M gating does NOT apply to this source —
        see refine_influenced_segments().
    """
    crs = BUFFER_CRS[country]
    poi_radius = POI_BUFFER_M[land_use]
    segs_geom = target_segs_utm[["geometry"]]
    buffer_polys: list = []
    iso_polys: list = []

    # Mapillary VRU points pre-filtered to those near target segments.
    map_pois = load_mapillary_pois(country)
    if len(map_pois) > 0:
        vru_mask = pd.Series(False, index=map_pois.index)
        for flag in _MAPILLARY_VRU_FLAGS:
            if flag in map_pois.columns:
                vru_mask = vru_mask | (map_pois[flag] == True)  # noqa: E712
        vru_pts = map_pois[vru_mask]
        if len(vru_pts) > 0:
            vru_utm = vru_pts[["geometry"]].to_crs(crs)
            nearby = gpd.sjoin(vru_utm, segs_geom, predicate="dwithin", distance=poi_radius)
            local_vru = vru_utm.loc[nearby.index.unique()]
            if len(local_vru) > 0:
                buffer_polys.extend(local_vru.geometry.buffer(poi_radius).tolist())

    # School VRU zone: use the isochrone polygon if the isochrone parquet exists,
    # otherwise fall back to the existing circular buffer (backward compatible).
    # Isochrone polygons are already corridor/floor-unioned, so no buffering is needed.
    # Isochrone-derived polygons go into iso_polys; the circular fallback goes into
    # buffer_polys (the fallback is not an isochrone, so the MIN_PIECE_M gate still applies).
    iso_path = Path(f"data/processed/school_isochrones_{country}.parquet")
    if iso_path.exists():
        try:
            isochrones = gpd.read_parquet(iso_path)
            if len(isochrones) > 0:
                iso_utm = isochrones[["geometry"]].to_crs(crs)
                nearby_iso = gpd.sjoin(iso_utm, segs_geom, predicate="intersects")
                local_iso = iso_utm.loc[nearby_iso.index.unique()]
                if len(local_iso) > 0:
                    iso_polys.extend(local_iso.geometry.tolist())
        except Exception:
            # Fall back to the circular buffer if loading fails.
            iso_path = None  # handled by the else branch below
    if not iso_path or not iso_path.exists():
        # Fallback: circular buffer (backward compatible when no isochrone exists).
        osm_pois = _tag_osm_pois(load_cached_signals(country)["pois"])
        if len(osm_pois) > 0 and "is_school" in osm_pois.columns:
            school_pts = osm_pois[osm_pois["is_school"] == True]  # noqa: E712
            if len(school_pts) > 0:
                school_utm = school_pts[["geometry"]].to_crs(crs)
                nearby = gpd.sjoin(school_utm, segs_geom, predicate="dwithin", distance=poi_radius)
                local_school = school_utm.loc[nearby.index.unique()]
                if len(local_school) > 0:
                    buffer_polys.extend(local_school.geometry.buffer(poi_radius).tolist())

    # Junction nodes pre-filtered to those near target segments.
    junctions = load_cached_junctions(country)
    if len(junctions) > 0:
        junc_utm = junctions[["geometry"]].to_crs(crs)
        nearby = gpd.sjoin(junc_utm, segs_geom, predicate="dwithin", distance=JUNCTION_BUFFER_M)
        local_junc = junc_utm.loc[nearby.index.unique()]
        if len(local_junc) > 0:
            buffer_polys.extend(local_junc.geometry.buffer(JUNCTION_BUFFER_M).tolist())

    buffer_poly = unary_union(buffer_polys) if buffer_polys else None
    iso_poly = unary_union(iso_polys) if iso_polys else None
    return buffer_poly, iso_poly


def _extract_linestrings(geom) -> list:
    """Normalise a Shapely geometry to a list of non-empty LineStrings."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return [g for g in geom.geoms if isinstance(g, LineString) and not g.is_empty]
    if isinstance(geom, GeometryCollection):
        result = []
        for g in geom.geoms:
            if isinstance(g, LineString) and not g.is_empty:
                result.append(g)
            elif isinstance(g, MultiLineString):
                result.extend(gg for gg in g.geoms if isinstance(gg, LineString) and not gg.is_empty)
        return result
    return []


def _make_child(parent_row: dict, child_geom_4326, child_len_m: float,
                parent_len_m: float, child_id: str) -> dict:
    """Inherit all parent columns and override geometry-specific fields."""
    child = dict(parent_row)
    child["geometry"] = child_geom_4326
    child["segment_id"] = child_id
    child["parent_section_id"] = parent_row["segment_id"]  # already normalised string
    child["shape_length"] = child_len_m
    if parent_len_m > 0 and pd.notna(parent_row.get("sample_size_total")):
        child["sample_size_total"] = parent_row["sample_size_total"] * (child_len_m / parent_len_m)
    return child


def refine_influenced_segments(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Stage-2 refinement: split influenced segments at the influence-zone boundary.

    For every segment where is_vru or near_junction is True, clips the
    geometry into an "influenced" portion and a "non-influenced" portion.
    For buffer-derived influence (Mapillary VRU / junction / circular-fallback school
    buffer), a segment is only split when *both* portions are at least MIN_PIECE_M
    long. For isochrone-derived influence (actual school-isochrone polygons), this
    minimum-length gate does NOT apply — any non-trivial overlap causes a split
    (see module docstring, "Isochrone splits are exempt from MIN_PIECE_M").

    Performance: the morphological closing (safety-side sliver absorption) is computed
    ONCE per (country, land_use) group and reused for all segments in that group.
    The influence polygon itself is built from only the nearby POIs/junctions
    (dwithin pre-filter), not the full country-wide set.

    segment_id normalisation: all rows get "{country}_{original_objectid}" to ensure
    global uniqueness (Thailand and Maharashtra share OBJECTID sequences from 1).
    Children get "{country}_{parent_objectid}-{k}".
    parent_section_id is set for split children; NA for all other rows.
    sample_size_total is length-proportionally allocated for children.
    All other geometry-driven columns are inherited and overwritten by the caller's
    Stage-1 re-run.
    """
    gdf = gdf.copy()

    # Normalise segment_id to globally unique "{country}_{objectid}" strings.
    gdf["segment_id"] = gdf["country"] + "_" + gdf["segment_id"].astype(str)
    if "parent_section_id" not in gdf.columns:
        gdf["parent_section_id"] = pd.NA

    target_mask = (gdf["is_vru"] == True) | (gdf["near_junction"] == True)  # noqa: E712
    if not target_mask.any():
        return gdf

    target_gdf = gdf.loc[target_mask]
    non_target_gdf = gdf.loc[~target_mask]

    split_parent_indices: set = set()
    new_child_dicts: list = []

    for (country, land_use), group in target_gdf.groupby(["country", "land_use"]):
        crs = BUFFER_CRS[country]

        # Project the whole group to UTM once.
        group_utm = group[["geometry"]].to_crs(crs)

        # Build influence polygons from nearby POIs/junctions only (dwithin pre-filter).
        # buffer_poly: Mapillary VRU / junction / circular-fallback buffers (MIN_PIECE_M
        # gating applies). iso_poly: actual school-isochrone polygons (no length gating —
        # see module docstring "isochrone" section).
        buffer_poly, iso_poly = build_influence_polygon_near(group_utm, country, land_use)
        if buffer_poly is None and iso_poly is None:
            continue

        # Morphological closing (safety-side sliver absorption) applies only to the
        # buffer-derived polygon: isochrones are already corridor/floor-unioned and
        # need no closing. Computed ONCE per group, not per segment.
        closed_buffer_poly = (
            buffer_poly.buffer(MIN_PIECE_M / 2).buffer(-MIN_PIECE_M / 2)
            if buffer_poly is not None else None
        )
        cut_poly = unary_union([p for p in (closed_buffer_poly, iso_poly) if p is not None])

        for idx, row in group.iterrows():
            line_utm = group_utm.loc[idx, "geometry"]
            parent_len = line_utm.length
            if parent_len < 1e-6:
                continue

            influenced_parts = _extract_linestrings(line_utm.intersection(cut_poly))
            non_influenced_parts = _extract_linestrings(line_utm.difference(cut_poly))

            influenced_len = sum(p.length for p in influenced_parts)
            non_influenced_len = sum(p.length for p in non_influenced_parts)

            # Isochrone-derived influence: no minimum-length gate. Split whenever both
            # sides are non-trivial; a segment fully inside the isochrone (non-influenced
            # side ~0) is correctly left unsplit (whole segment stays at the lower V_safe).
            seg_hits_iso = (iso_poly is not None) and line_utm.intersects(iso_poly)
            if seg_hits_iso:
                if influenced_len < 1e-9 or non_influenced_len < 1e-9:
                    continue
            else:
                # Buffer-derived (Mapillary VRU / junction / circular fallback):
                # keep the original minimum-length gate.
                if influenced_len < MIN_PIECE_M or non_influenced_len < MIN_PIECE_M:
                    continue

            split_parent_indices.add(idx)
            parent_dict = row.to_dict()
            parent_id_str = str(row["segment_id"])  # already "{country}_{objectid}"

            k = 0
            for part_utm in influenced_parts:
                if part_utm.length < 1e-9:
                    continue
                part_4326 = gpd.GeoSeries([part_utm], crs=crs).to_crs("EPSG:4326").iloc[0]
                new_child_dicts.append(
                    _make_child(parent_dict, part_4326, part_utm.length, parent_len,
                                f"{parent_id_str}-{k}")
                )
                k += 1

            for part_utm in non_influenced_parts:
                if part_utm.length < 1e-9:
                    continue
                part_4326 = gpd.GeoSeries([part_utm], crs=crs).to_crs("EPSG:4326").iloc[0]
                new_child_dicts.append(
                    _make_child(parent_dict, part_4326, part_utm.length, parent_len,
                                f"{parent_id_str}-{k}")
                )
                k += 1

    if not new_child_dicts:
        return gdf

    unsplit_target = target_gdf.loc[~target_gdf.index.isin(split_parent_indices)]
    children_gdf = gpd.GeoDataFrame(new_child_dicts, geometry="geometry", crs="EPSG:4326")

    n_parents = len(split_parent_indices)
    n_children = len(children_gdf)
    print(f"[segment_localization] split {n_parents} segments into {n_children} children "
          f"(net +{n_children - n_parents} rows)")

    combined = pd.concat([non_target_gdf, unsplit_target, children_gdf], ignore_index=True)
    return gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")
