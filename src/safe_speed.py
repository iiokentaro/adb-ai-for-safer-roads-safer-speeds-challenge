"""Collision-type classification and the V_safe lookup table.

★ Structural guarantee ★
`classify_collision_type` and `compute_v_safe` take no `speed_limit` or
observed-speed argument anywhere in their signatures. V_safe cannot be a
function of the current posted limit even by accident -- it isn't in scope
to use.

Fixed speed thresholds are human injury-tolerance values from the Safe
System approach (WHO Global Status Report on Road Safety / iRAP star
rating star-rating speed bands), not fitted or learned from this dataset.

★ `exposure_level` does not drive this decision ★
Earlier versions classified "pedestrian"/"side_impact" off `exposure_level`
(a population/POI/crossing density tertile). That composite is now reserved
for prioritization only (safety_score.py, priority_lists.py) -- it is a proxy
for "how many people are around", not direct evidence of an at-grade VRU
conflict point. The "pedestrian" collision type here instead requires a
concrete VRU detection: `is_vru` (a Mapillary-detected school-zone sign,
crosswalk marking, or bicycle marking, OR an OSM amenity=school node, within
the dwithin distance -- see exposure_signals.py), which is itself already
forced False for motorway/trunk and for any segment with `is_separated==True`
(exposure_signals.add_poi_proximity). OSM schools are included because a school
almost never grade-separates from a non-access-controlled road and Mapillary
coverage is incomplete. `side_impact` (50 km/h) is no longer
assigned here at all -- it is applied as a post-hoc cap by
junction_speed_cap.py, run after `add_v_safe`, for segments near an OSM
junction node (highway=traffic_signals / junction=yes).
"""

import sys

import pandas as pd

sys.path.insert(0, "src")

# WHO Safe System injury-tolerance thresholds
V_SAFE_TABLE = {
    "pedestrian": 30,
    "side_impact": 50,
    "head_on": 70,
}
# "separated" gets a road_class-dependent value instead of one constant.
SEPARATED_V_SAFE_BY_ROAD_CLASS = {
    "motorway": 100,
    "trunk": 90,
    "primary": 80,
    "secondary": 80,
}
SEPARATED_V_SAFE_DEFAULT = 80

# RoadClass is itself an Overture estimate, not ground truth (same caveat as
# LandUse/SpeedLimit throughout this project). The motorway override below
# only fires because access control makes VRU conflict physically
# implausible -- but that reasoning collapses if the "motorway" tag is
# wrong. A genuinely access-controlled road essentially never has an
# observed 85th-percentile speed this low; below it, trust the exposure
# signal instead (the safe direction: it can only raise V_safe down toward
# 30, never assume away a real VRU conflict). 16/169 (9.5%) of motorway
# segments fall below this in practice, including 13 that are exactly 0 for
# SpeedLimit/MedianSpeed/F85 together -- a separate placeholder-looking data
# issue, not just a borderline speed.
MOTORWAY_TAG_SUSPECT_F85_THRESHOLD = 50


def classify_collision_type(road_class, is_separated: bool, is_vru: bool, motorway_tag_suspect: bool = False) -> str:
    """Physical separation and a concrete VRU detection are the primary
    drivers, consistent with Safe System theory: collision type is about
    who can collide with whom, not the road's legal classification or
    ambient population density.

    The one exception: motorway is access-controlled by definition (no
    legal pedestrian/cyclist access). Without this override, motorway had
    the highest "pedestrian collision type" rate of any road_class (47.9%)
    under the old exposure_level-based rule, which didn't reflect a
    real-world conflict. trunk is deliberately excluded from this
    particular override: unlike motorway, it isn't reliably access-controlled
    in this dataset (Thai/Indian "trunk" often carries roadside
    pedestrian/informal activity) -- but `is_vru` is itself already
    forced False for trunk upstream (exposure_signals.py), so trunk never
    reaches the `is_vru` branch below either way.

    The motorway override doesn't fire if `motorway_tag_suspect` -- see
    MOTORWAY_TAG_SUSPECT_F85_THRESHOLD.
    """
    if road_class == "motorway" and not motorway_tag_suspect:
        return "separated" if is_separated else "head_on"
    if is_vru:
        return "pedestrian"
    return "separated" if is_separated else "head_on"


def compute_v_safe(collision_type: str, road_class) -> tuple[int, str]:
    """Returns (v_safe, basis_string). basis_string records exactly which
    rule fired, for segment-level explainability."""
    if collision_type == "separated":
        v_safe = SEPARATED_V_SAFE_BY_ROAD_CLASS.get(road_class, SEPARATED_V_SAFE_DEFAULT)
        return v_safe, f"separated:{road_class}"
    return V_SAFE_TABLE[collision_type], collision_type


def add_v_safe(gdf, is_separated_col="is_separated"):
    gdf = gdf.copy()
    if is_separated_col not in gdf.columns:
        # Safe default if road_separation.add_is_separated wasn't run first:
        # don't assume separation.
        gdf[is_separated_col] = False

    gdf["motorway_tag_suspect"] = (gdf["road_class"] == "motorway") & (
        gdf["f85_speed"] < MOTORWAY_TAG_SUSPECT_F85_THRESHOLD
    )

    collision_types = [
        classify_collision_type(rc, sep, vru, suspect)
        for rc, sep, vru, suspect in zip(
            gdf["road_class"], gdf[is_separated_col], gdf["is_vru"], gdf["motorway_tag_suspect"]
        )
    ]
    gdf["collision_type"] = collision_types

    results = [
        compute_v_safe(ct, rc) for ct, rc in zip(gdf["collision_type"], gdf["road_class"])
    ]
    gdf["v_safe"] = [r[0] for r in results]
    gdf["v_safe_basis"] = [r[1] for r in results]
    return gdf


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    from exposure_level import add_exposure_level, apply_rural_safety_margin
    from exposure_signals import add_crossing_signal, add_poi_proximity
    from pop_density import add_pop_density
    from road_separation import add_is_separated
    from schema import load_target

    target = load_target()
    target = add_is_separated(target)  # add_poi_proximity needs is_separated to mask is_vru/is_mapillary_vru
    target = add_pop_density(target)
    target = add_poi_proximity(target)
    target = add_crossing_signal(target)
    target = add_exposure_level(target)  # kept for prioritization only, not read by add_v_safe anymore
    target, thresholds = apply_rural_safety_margin(target)
    target = add_v_safe(target)
    print("rural safety-margin thresholds by country:", thresholds)
    print()

    print(target.groupby(["road_class", "land_use"])["v_safe"].describe())
    print()
    print("collision_type distribution:")
    print(target["collision_type"].value_counts())
