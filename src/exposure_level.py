"""Discretize VRU exposure into high/medium/low, separately for urban and
rural systems, then apply the rural safety margin.

★ `exposure_level` no longer drives V_safe ★
As of the is_mapillary_vru redesign, `exposure_level` is used only for
prioritization (safety_score.py's 35%-weight exposure axis, priority_lists.py's
rank_within_environment) -- never to pick the recommended speed itself. The
collision-type / V_safe decision (safe_speed.py) now runs on `is_vru` and
`is_separated` directly, not on this composite tertile. `is_vru` is the
source-agnostic VRU trigger (Mapillary VRU OR OSM amenity=school);
`is_mapillary_vru` (Mapillary-only) is kept here as an URBAN signal. The OSM
school is also present in this composite via `osm_poi_category_count`, so it
contributes to both axes (double counting is accepted, mirroring is_mapillary_vru).

★ Why percentile-rank-then-average, not a raw sum ★
pop_density and poi_count are on different scales and both heavily
right-skewed; summing raw values lets whichever signal has the larger
numbers dominate. Each signal is converted to its percentile rank *within
its own land_use system* first (so urban and rural never share a scale),
then averaged. This keeps every signal monotonically increasing the
composite (more population/POIs/crossings never lowers exposure) while
keeping urban's rich signal set from being compared on the same axis as
rural's sparse one -- the systematic-bias problem documented in PIPELINE.md.
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

URBAN_SIGNALS = [
    "pop_density",
    "poi_count",
    "osm_poi_category_count",
    "is_mapillary_vru",
    "crossing_count",
]
RURAL_SIGNALS = ["pop_density", "poi_count", "osm_poi_category_count"]

LEVELS = ["low", "medium", "high"]


def _percentile_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, method="average")


def _composite_and_level(gdf, mask, signals):
    sub = gdf.loc[mask]
    ranks = pd.concat([_percentile_rank(sub[s]) for s in signals], axis=1)
    composite = ranks.mean(axis=1)
    # qcut with duplicate edges dropped can collapse to <3 bins on heavily
    # tied data (e.g. many rural segments at poi_count==0); fall back to
    # rank-based tertiles which always produce 3 groups.
    try:
        level = pd.qcut(composite, q=3, labels=LEVELS)
    except ValueError:
        level = pd.qcut(composite.rank(method="first"), q=3, labels=LEVELS)
    return composite, level


def add_exposure_level(gdf):
    """Percentile rank + tertile split is computed within each (country,
    land_use) cell independently -- four cells, not two. Pooling both
    countries within a single land_use system (as an earlier version did)
    reintroduces exactly the cross-population scale mismatch this module's
    own docstring warns about for urban-vs-rural:
    Thailand and Maharashtra's pop_density/poi_count distributions differ
    enough (see apply_rural_safety_margin's per-country threshold below)
    that a pooled tertile would let one country's larger raw numbers
    dominate the other's percentile ranks."""
    gdf = gdf.copy()
    gdf["exposure_composite"] = np.nan
    gdf["exposure_level"] = pd.Categorical([None] * len(gdf), categories=LEVELS, ordered=True)

    for land_use, signals in [("URBAN", URBAN_SIGNALS), ("RURAL", RURAL_SIGNALS)]:
        for country in gdf["country"].unique():
            mask = (gdf["land_use"] == land_use) & (gdf["country"] == country)
            if not mask.any():
                continue
            composite, level = _composite_and_level(gdf, mask, signals)
            gdf.loc[mask, "exposure_composite"] = composite
            gdf.loc[mask, "exposure_level"] = level.values

    return gdf


def apply_rural_safety_margin(gdf, quantile=0.75):
    """Rural segments with substantial roadside population but no detected
    crossing structure (true for nearly all rural segments) are
    "exposure unknown, potentially high risk" -- not "low exposure". Raise
    them to at least medium, mark confidence low.

    Threshold is the 75th percentile of *that country's rural* pop_density,
    computed separately per country -- not rural+urban combined, and not
    one threshold shared across countries. Thailand and Maharashtra rural
    pop_density distributions differ enough (medians 4.9 vs 13.8) that a
    combined threshold uplifted 51% of Maharashtra's rural segments but
    only 11% of Thailand's: the same absolute-value-across-systems bias
    that motivates the separate urban/rural exposure tracks, just one level
    down. Returns thresholds as a dict keyed by country for inspection.
    """
    gdf = gdf.copy()
    gdf["exposure_confidence"] = "high"

    rural_mask = gdf["land_use"] == "RURAL"
    thresholds = {}

    for country in gdf.loc[rural_mask, "country"].unique():
        country_rural_mask = rural_mask & (gdf["country"] == country)
        threshold = gdf.loc[country_rural_mask, "pop_density"].quantile(quantile)
        thresholds[country] = threshold

        uncertain_mask = country_rural_mask & (gdf["pop_density"] >= threshold) & (~gdf["has_crossing"])
        gdf.loc[uncertain_mask, "exposure_confidence"] = "low"
        levels_as_int = gdf["exposure_level"].cat.codes
        medium_code = LEVELS.index("medium")
        needs_raise = uncertain_mask & (levels_as_int < medium_code)
        gdf.loc[needs_raise, "exposure_level"] = "medium"

    return gdf, thresholds


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    from exposure_signals import add_crossing_signal, add_poi_proximity
    from pop_density import add_pop_density
    from road_separation import add_is_separated
    from schema import load_target

    target = load_target()
    target = add_is_separated(target)  # add_poi_proximity needs is_separated to mask is_vru/is_mapillary_vru
    target = add_pop_density(target)
    target = add_poi_proximity(target)
    target = add_crossing_signal(target)
    target = add_exposure_level(target)
    target, threshold = apply_rural_safety_margin(target)

    print("rural pop_density safety-margin threshold:", threshold)
    print()
    print(target.groupby(["country", "land_use"])["exposure_level"].value_counts())
    print()
    print("exposure_confidence by land_use:")
    print(target.groupby("land_use")["exposure_confidence"].value_counts())
