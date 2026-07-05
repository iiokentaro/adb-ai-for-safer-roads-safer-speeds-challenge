"""Post-hoc V_safe cap near OSM junction nodes (signal-controlled
intersections and marked minor junctions), applied *after* `safe_speed.add_v_safe`
has already produced its road_class/is_separated/is_mapillary_vru-based V_safe.

★ Why this is a separate, later step rather than another classify_collision_type branch ★
Safe System guidance sets ~50 km/h as the survivable-impact threshold for the
side-impact (right-angle, vehicle-vehicle) crash type characteristic of
intersections -- a distinct physical conflict from the pedestrian (30) and
head-on (70) cases `safe_speed.py` already covers. It is a *cap*, not a
floor: a segment already at or below 50 (e.g. `pedestrian`, 30) is left
untouched, since that is already more conservative.

★ Exclusions: motorway and grade-separated segments ★
The cap fires regardless of plain `is_separated` (a divided road's median
does not stop a signal-controlled intersection from being an at-grade
side-impact conflict point) -- unlike the `is_mapillary_vru` signal
(exposure_signals.py), which is masked for `is_separated`/motorway/trunk
because a *nearby* detection doesn't imply an at-grade conflict *on* the
segment; a junction node directly *on* or beside the segment is not the
same kind of signal. But two cases mean a junction within JUNCTION_BUFFER_M
is *not* actually an at-grade conflict for this segment, so the cap is
excluded for them:
- `road_class == 'motorway'`: fully access-controlled -- any junction within
  the buffer is a grade-separated ramp/interchange, not an at-grade
  crossing motorway traffic itself passes through.
- `is_grade_separated == True` (road_separation.py; the segment's matched
  OSM way(s) carry bridge/tunnel/layer!=0): a flyover/underpass passing
  near or over a junction node is not at the same grade as that junction.
Both exclusions gate `near_junction` itself (not just the cap), so an
excluded segment is also not pulled into segment_localization.py's Stage 2
influence-zone split on the junction's account (it can still split on
`is_vru`).

★ Tags used / data source ★
- `highway=traffic_signals` (node): signal-controlled intersection.
- `junction=yes` (way): a marked, non-roundabout junction.
`junction=roundabout` is deliberately excluded -- a roundabout is itself a
traffic-calming/speed-reducing design, not the uncontrolled side-impact-risk
case this cap targets.

Source: `data/external/osm_junctions_{country}.geojson`, an Overpass Turbo
export (given input, not re-derived from the `.osm.pbf` files this project
otherwise parses via pyrosm). Every row already satisfies
`highway=='traffic_signals' or junction=='yes'` -- some rows additionally
carry `junction=roundabout`/`intersection` as a secondary tag alongside a
matching primary one, which is fine; what matters is the row matched on
`highway` or `junction=='yes'`, not what else is tagged on it.

★ Granularity ★
P1-0 (long-segment POI localisation, segment_localization.py) is implemented
as a two-stage pipeline in build_v_safe.py. Stage 1 runs this function on
whole segments and marks near_junction; Stage 2 clips near-junction segments at
the 300 m buffer boundary so that the cap applies only to the influenced
portion. Segments that do not reach the minimum split size (50 m per part)
are still capped whole-segment as before.
"""

import sys
import warnings
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, "src")
from exposure_signals import BUFFER_CRS
from safe_speed import V_SAFE_TABLE

warnings.filterwarnings("ignore", category=UserWarning)

EXTERNAL_DIR = "data/external"
PROCESSED_DIR = "data/processed"
JUNCTION_BUFFER_M = 300
JUNCTION_V_SAFE_CAP = V_SAFE_TABLE["side_impact"]  # 50 km/h, single source of truth with safe_speed.py
COUNTRIES = ["thailand", "maharashtra"]


def extract_junctions(country: str) -> gpd.GeoDataFrame:
    """Read + filter the given Overpass Turbo export. Every row already
    matches `highway=='traffic_signals' or junction=='yes'` in practice, but
    the filter is applied explicitly so this stays correct even if the
    source export is regenerated with a looser query."""
    gdf = gpd.read_file(f"{EXTERNAL_DIR}/osm_junctions_{country}.geojson")
    keep = (gdf["highway"] == "traffic_signals") | (gdf["junction"] == "yes")
    return gdf.loc[keep, ["highway", "junction", "geometry"]].copy()


def cache_junctions(country: str) -> gpd.GeoDataFrame:
    junctions = extract_junctions(country)
    junctions.to_parquet(f"{PROCESSED_DIR}/osm_junctions_{country}.parquet")
    return junctions


def load_cached_junctions(country: str) -> gpd.GeoDataFrame:
    cache_path = Path(f"{PROCESSED_DIR}/osm_junctions_{country}.parquet")
    if not cache_path.exists():
        return cache_junctions(country)
    return gpd.read_parquet(cache_path)


def add_junction_speed_cap(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Cap v_safe to JUNCTION_V_SAFE_CAP (50) for segments within
    JUNCTION_BUFFER_M (300m) of a cached junction feature, excluding
    motorway and grade-separated segments (see module docstring). Must run
    after `safe_speed.add_v_safe` and `road_separation.add_is_separated`
    (for `is_grade_separated`). Only ever lowers v_safe (min()), and only
    relabels collision_type/v_safe_basis for segments it actually changes.
    """
    gdf = gdf.copy()
    gdf["near_junction"] = False

    for country in gdf["country"].unique():
        junctions = load_cached_junctions(country)
        mask = gdf["country"] == country
        if len(junctions) == 0 or not mask.any():
            continue
        crs = BUFFER_CRS[country]
        junctions_utm = junctions.to_crs(crs)[["geometry"]]
        segs = gdf.loc[mask, ["geometry"]].to_crs(crs)
        joined = gpd.sjoin(segs, junctions_utm, predicate="dwithin", distance=JUNCTION_BUFFER_M)
        gdf.loc[joined.index.unique(), "near_junction"] = True

    # Motorway (fully access-controlled -- any nearby junction is a
    # grade-separated interchange) and grade-separated segments (flyover/
    # underpass, road_separation.add_is_separated) are not at-grade side-impact
    # conflicts, so they are excluded from near_junction itself -- not just the
    # cap -- so they also don't get pulled into the Stage 2 influence-zone
    # split on the junction's account (segment_localization.py).
    exclude = (gdf["road_class"] == "motorway") | (gdf.get("is_grade_separated") == True)  # noqa: E712
    gdf.loc[exclude, "near_junction"] = False

    capped = gdf["near_junction"] & (gdf["v_safe"] > JUNCTION_V_SAFE_CAP)
    gdf.loc[capped, "v_safe"] = JUNCTION_V_SAFE_CAP
    gdf.loc[capped, "collision_type"] = "side_impact"
    gdf.loc[capped, "v_safe_basis"] = "side_impact:junction_buffer"

    return gdf


if __name__ == "__main__":
    for country in COUNTRIES:
        print(f"--- {country} ---")
        junctions = cache_junctions(country)
        print(f"junction features: {len(junctions)}")
        if len(junctions):
            print(junctions.geometry.type.value_counts())
            print(junctions[["highway", "junction"]].apply(lambda s: s.value_counts(dropna=False).to_dict()))

    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    from build_v_safe import build

    target, _, _ = build()
    print()
    print(f"capped to {JUNCTION_V_SAFE_CAP}km/h near a junction: "
          f"{(target['v_safe_basis'] == 'side_impact:junction_buffer').sum()} / {len(target)}")
