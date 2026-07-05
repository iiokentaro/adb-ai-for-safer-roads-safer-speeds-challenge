"""FINAL_SPRINT_PLAN.md P0-1: speed-to-fatality Power Model benefit estimate,
with road-environment-specific exponents (Cameron & Elvik 2010 / Elvik 2009
Table 8, reproduced in cameron_and_elvik.json).

  delta_fatal_percent = (1 - (V_post / V_op) ** p) * 100

This turns the priority list from "where the limit is too high" into "how
many fewer fatal crashes this segment would see if operating speed
converged on V_safe" -- without inventing a new speed model: V_op/V_post
reuse exactly the columns already produced by the pipeline (`median_speed`, `v_safe`).

★ Category choice: fatal_accidents, not fatalities ★
Elvik (2009) Table 8 has six severity categories. This project commits to
"fatal_accidents" (crashes with >=1 death, i.e. the rate per crash) rather
than "fatalities" (deaths per person) and says so explicitly here and in the
README -- reporting "the Power Model exponent"
without naming which of the two categories is in use overstates precision
that doesn't exist. fatal_accidents is the more conservative/standard choice
in the road-safety literature for this kind of segment-level screening.

★ Environment assignment mirrors safe_speed.py's motorway override ★
Elvik (2009)'s freeway exponents cluster with the rural/high-speed group,
not with land_use=='URBAN' -- a motorway segment tagged URBAN (e.g. an urban
ring road) still gets the rural_freeway exponent, consistent with the
access-controlled override already applied when v_safe itself was derived
(safe_speed.py classify_collision_type).

★ V_post = v_safe, not 0 and not speed_limit ★
The benefit answers "if operating speed converged on the Safe System speed
(v_safe)", not "if the posted limit were enforced" -- the two differ exactly
on the segments review_track.py already flags as plausibility=low, where the
posted limit itself is the thing in question.

★ Negative/zero values are not errors -- they are the "no benefit" signal ★
Segments already operating at or below v_safe (median_speed <= v_safe) get
delta_fatal_percent <= 0 by construction: there is no proposed speed
reduction to claim a benefit for. These are deliberately left negative
rather than clipped to 0 -- clipping would hide the FINAL_SPRINT_PLAN.md
P0-1 sanity check ("these segments must read as no-benefit, not as
reduction candidates") behind a wall of identical zeros.

★ Range, not a single point estimate ★
power_exponent_ci_low/high and delta_fatal_percent_ci_low/high redo the same
formula at Elvik's reported 95% CI bounds for the exponent (urban
fatal_accidents: 0.3-4.9 -- very wide). FINAL_SPRINT_PLAN.md P0-1 item 4
asks for this range to travel alongside the point estimate by default, not
be computed only on request, precisely so a single number is never
presented as a precise, certain benefit.
"""

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

EXPONENT_TABLE_PATH = "src/cameron_and_elvik.json"
CATEGORY = "fatal_accidents"  # see module docstring: crashes-with-a-death, not deaths-per-person


def load_exponents(category: str = CATEGORY, path: str = EXPONENT_TABLE_PATH) -> dict:
    with open(path, encoding="utf-8-sig") as f:
        table = json.load(f)
    return table["exponents"][category]


def environment_for(gdf: pd.DataFrame) -> pd.Series:
    """rural_freeway for RURAL land_use OR any motorway (regardless of land_use,
    per cameron_and_elvik.json's motorway_handling note); urban_residential
    for everything else (i.e. URBAN, non-motorway)."""
    is_rural_freeway = (gdf["road_class"] == "motorway") | (gdf["land_use"] == "RURAL")
    return pd.Series(np.where(is_rural_freeway, "rural_freeway", "urban_residential"), index=gdf.index)


def _delta_fatal_percent(v_op: pd.Series, v_post: pd.Series, p: pd.Series) -> pd.Series:
    return (1 - (v_post / v_op) ** p) * 100


def add_fatal_reduction(gdf, exponents: dict | None = None) -> pd.DataFrame:
    gdf = gdf.copy()
    exponents = exponents or load_exponents()

    new_columns = [
        "power_environment_used", "power_exponent_used",
        "power_exponent_ci_low", "power_exponent_ci_high",
        "delta_fatal_percent", "delta_fatal_percent_ci_low", "delta_fatal_percent_ci_high",
    ]
    for col in new_columns:
        gdf[col] = pd.NA

    has_flag_col = "data_quality_flag" in gdf.columns
    valid_mask = gdf["data_quality_flag"].isna() if has_flag_col else pd.Series(True, index=gdf.index)
    valid = gdf.loc[valid_mask]

    environment = environment_for(valid)
    p_best = environment.map({env: exponents[env]["best"] for env in exponents}).astype(float)
    p_ci_low = environment.map({env: exponents[env]["ci95"][0] for env in exponents}).astype(float)
    p_ci_high = environment.map({env: exponents[env]["ci95"][1] for env in exponents}).astype(float)

    v_op = valid["median_speed"]
    v_post = valid["v_safe"]

    gdf.loc[valid_mask, "power_environment_used"] = environment.values
    gdf.loc[valid_mask, "power_exponent_used"] = p_best.values
    gdf.loc[valid_mask, "power_exponent_ci_low"] = p_ci_low.values
    gdf.loc[valid_mask, "power_exponent_ci_high"] = p_ci_high.values
    gdf.loc[valid_mask, "delta_fatal_percent"] = _delta_fatal_percent(v_op, v_post, p_best).values
    gdf.loc[valid_mask, "delta_fatal_percent_ci_low"] = _delta_fatal_percent(v_op, v_post, p_ci_low).values
    gdf.loc[valid_mask, "delta_fatal_percent_ci_high"] = _delta_fatal_percent(v_op, v_post, p_ci_high).values

    for col in ["power_exponent_used", "power_exponent_ci_low", "power_exponent_ci_high",
                "delta_fatal_percent", "delta_fatal_percent_ci_low", "delta_fatal_percent_ci_high"]:
        gdf[col] = pd.to_numeric(gdf[col])

    return gdf


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    from build_v_safe import build

    target, _, _ = build()  # build() already runs add_fatal_reduction as part of the pipeline
    valid = target[target["data_quality_flag"].isna()]

    print(f"exponent category used: {CATEGORY}")
    print(valid.groupby("power_environment_used")[
        ["power_exponent_used", "power_exponent_ci_low", "power_exponent_ci_high"]
    ].first())

    print("\n=== delta_fatal_percent by environment (n, mean, median) ===")
    print(valid.groupby("power_environment_used")["delta_fatal_percent"].agg(["count", "mean", "median"]))

    no_benefit = valid["delta_fatal_percent"] <= 0
    print(f"\nsegments already at/below v_safe (median_speed <= v_safe, no benefit to claim): "
          f"{no_benefit.sum()} / {len(valid)} ({no_benefit.mean():.1%})")

    print("\n=== sanity: urban should read lower (more conservative) than rural for the same speed gap ===")
    print(valid.groupby("power_environment_used")["delta_fatal_percent"].describe())

    print("\n=== priority list (FINAL_SPRINT_PLAN.md P0-1 'review_needed' segments) ===")
    from review_track import REVIEW_NEEDED

    review = valid[valid["review_track"] == REVIEW_NEEDED]
    print(f"n={len(review)}")
    print(review[["delta_fatal_percent", "delta_fatal_percent_ci_low", "delta_fatal_percent_ci_high"]].describe())

    print("\n=== top 5 by delta_fatal_percent ===")
    top5 = review.nlargest(5, "delta_fatal_percent")
    print(top5[["segment_id", "country", "road_class", "land_use", "median_speed", "v_safe",
                "power_environment_used", "power_exponent_used",
                "delta_fatal_percent", "delta_fatal_percent_ci_low", "delta_fatal_percent_ci_high"]]
          .to_string(index=False))
