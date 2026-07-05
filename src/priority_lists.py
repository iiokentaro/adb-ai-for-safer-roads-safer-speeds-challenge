"""FINAL_SPRINT_PLAN.md P1-1b: split the single combined priority list into
two environment-specific lists (`priority_urban` / `priority_rural`) instead
of ranking every segment -- both countries, both land-use systems, motorway
and local roads -- on one scale.

★ Why a structural split, not a bias correction ★
exposure_level.py already builds urban and rural exposure on separate scales
(by design: a rural segment's "high" exposure is not measured the same way
as an urban one's -- see exposure_level.py). Funnelling both back into a single ranked
list re-introduces exactly the cross-scale comparison the two-system design
was meant to avoid, and whether rural segments are systematically pushed
down that combined list because they're genuinely safer or because the two
exposure scales don't commute is not something this dataset can verify
(there's no external ground truth to test it against). The fix is
structural: never put them on one ranked list at all. `priority_class`
itself -- and the score behind it -- is unchanged; only the *output lists*
split this way.

★ Why the split follows power_environment_used, not raw land_use ★
A handful of motorway segments are tagged land_use=='URBAN' (2 of 3,662
priority-list segments here) but are access-controlled like any other
motorway -- the same exception safe_speed.py and fatality_reduction.py
already apply. Reusing `power_environment_used` (rather than inventing a
second environment split with different edge-case handling) keeps a
motorway's intervention story consistent end to end: it was already grouped
with the rural/high-speed cohort for its V_safe collision-type override and
its Power Model exponent, so it stays there for which list it appears in
and what kind of intervention (signage/self-enforcing road design vs.
crosswalks/traffic calming) gets recommended for it.

★ Why `rank_within_environment` is also computed per country ★
The two output files (`priority_urban.csv` / `priority_rural.csv`) already
keep urban and rural off one ranked scale (above). But within each file,
an earlier version still ranked Thailand and Maharashtra together as one
pool -- the same cross-population risk safety_score.py's own docstring
documents for `priority_class` (Thailand's systematically larger
misalignment would crowd Maharashtra out of low rank numbers, the
"who gets acted on first" question this rank exists to answer). The rank
is therefore computed within each (country, power_environment_used) cell
independently; both countries still appear in the same CSV (sorted by
`rank_within_environment`, so the two countries' rank-1 rows interleave at
the top), but neither's `rank_within_environment` value is influenced by
the other's score distribution.

★ Why `rank_within_environment` excludes the confidence axis ★
`safety_score.py`'s 0.50/0.35/0.15 weighting folds data confidence into the
score that decides `priority_class` (the gate for being on a list at all --
left untouched here). But ranking *within* a list by that same score would
let a rural safety-margin segment (high exposure, *because* the underlying
signal is too thin to read directly -- exposure_level.py's
apply_rural_safety_margin) drop a few places purely for being
less-measured, not less dangerous -- contradicting the project's own
"structural absence is itself the danger signal" stance on rural exposure.
`rank_within_environment` instead re-derives just the misalignment+exposure
portion of the score, renormalized to sum to 1 once confidence is dropped
(0.50/0.35 -> 0.5882/0.4118) -- the same two weights, just no longer
diluted by a third axis that was never supposed to move a segment down the
list. Confidence is preserved as a plain-language flag (`confidence_note`)
instead, never as a rank penalty.

★ What `delta_fatal_abs` is, and what it deliberately is not ★
Neither country's dataset carries a crash or fatality count column --
there is no observed baseline to subtract from to report literally "N fewer
deaths." `delta_fatal_abs = (delta_fatal_percent / 100) * sample_size_total`
is a traffic-volume-weighted index: it lets a small percentage reduction on
a heavily-travelled urban road and a large percentage reduction on a
lightly-travelled rural road be compared by something other than rank
percentile within their own list (FINAL_SPRINT_PLAN.md P1-1b's "common
currency" for cross-environment budget decisions). It is NOT a predicted
body count, and must never be presented as one -- `sample_size_total` is a
probe-sample count (a traffic-volume proxy), not a crash history. Same sign
convention as `delta_fatal_percent` (fatality_reduction.py): segments
already at/below v_safe carry a non-positive value and are not reduction
candidates.
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from safety_score import EXPOSURE_POINTS, MISALIGNMENT_CAP_KMH, WEIGHT_EXPOSURE, WEIGHT_MISALIGNMENT  # noqa: E402

ENVIRONMENTS = ["urban_residential", "rural_freeway"]
ON_LIST_CLASSES = ["Top Priority", "Priority", "Watch"]

LIST_COLUMNS = [
    "segment_id", "country", "road_class", "land_use", "power_environment_used",
    "rank_within_environment", "speed_limit", "median_speed", "v_safe", "misalignment",
    "exposure_level", "confidence_level", "confidence_note", "speedlimit_plausibility",
    "delta_fatal_percent", "delta_fatal_abs", "sample_size_total",
    "priority_class", "review_track", "score_explanation", "street_image_link",
]


def add_priority_environment_rank(gdf: pd.DataFrame) -> pd.DataFrame:
    gdf = gdf.copy()
    has_flag_col = "data_quality_flag" in gdf.columns
    valid_mask = gdf["data_quality_flag"].isna() if has_flag_col else pd.Series(True, index=gdf.index)
    on_list = valid_mask & gdf["priority_class"].isin(ON_LIST_CLASSES)

    misalignment_norm = gdf["misalignment_magnitude"].clip(upper=MISALIGNMENT_CAP_KMH) / MISALIGNMENT_CAP_KMH
    exposure_norm = gdf["exposure_level"].astype(str).map(EXPOSURE_POINTS).astype(float)
    weight_total = WEIGHT_MISALIGNMENT + WEIGHT_EXPOSURE
    rank_key = (WEIGHT_MISALIGNMENT * misalignment_norm + WEIGHT_EXPOSURE * exposure_norm) / weight_total

    gdf["rank_within_environment"] = pd.NA
    for env in ENVIRONMENTS:
        for country in gdf["country"].unique():
            mask = on_list & (gdf["power_environment_used"] == env) & (gdf["country"] == country)
            if not mask.any():
                continue
            ranks = rank_key.loc[mask].rank(ascending=False, method="min")
            gdf.loc[mask, "rank_within_environment"] = ranks.values

    gdf["delta_fatal_abs"] = pd.NA
    gdf.loc[valid_mask, "delta_fatal_abs"] = (
        gdf.loc[valid_mask, "delta_fatal_percent"] / 100 * gdf.loc[valid_mask, "sample_size_total"]
    )

    gdf["confidence_note"] = pd.NA
    gdf.loc[on_list & (gdf["confidence_level"] == "low"), "confidence_note"] = "Exposure uncertain -- field verification recommended"

    gdf["rank_within_environment"] = pd.to_numeric(gdf["rank_within_environment"])
    gdf["delta_fatal_abs"] = pd.to_numeric(gdf["delta_fatal_abs"])
    return gdf


def write_priority_environment_lists(gdf: pd.DataFrame, out_dir: str = "outputs") -> tuple[str, str]:
    # MISALIGNMENT_CAP_KMH/EXPOSURE_POINTS saturate at the top of the rank_key
    # (safety_score.py's own documented cap, not something introduced here),
    # so many segments legitimately tie at rank_within_environment==1.
    # delta_fatal_percent breaks ties for row order only -- it never changes
    # the stored rank_within_environment value itself.
    sort_cols = ["rank_within_environment", "delta_fatal_percent"]
    sort_order = [True, False]
    on_list = gdf["rank_within_environment"].notna()
    urban = gdf[on_list & (gdf["power_environment_used"] == "urban_residential")].sort_values(
        sort_cols, ascending=sort_order
    )
    rural = gdf[on_list & (gdf["power_environment_used"] == "rural_freeway")].sort_values(
        sort_cols, ascending=sort_order
    )

    urban_path = f"{out_dir}/priority_urban.csv"
    rural_path = f"{out_dir}/priority_rural.csv"
    urban[LIST_COLUMNS].to_csv(urban_path, index=False)
    rural[LIST_COLUMNS].to_csv(rural_path, index=False)
    return urban_path, rural_path


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    from build_v_safe import build

    target, _, _ = build()  # build() already runs add_priority_environment_rank as part of the pipeline
    valid = target[target["data_quality_flag"].isna()]
    on_list = valid["rank_within_environment"].notna()
    pri = valid[on_list]

    print(f"priority list (both environments combined): n={len(pri)}")
    print(pri.groupby("power_environment_used").size())

    print("\n=== confirm: confidence does not move rank (low-confidence segments present at rank 1) ===")
    for env in ENVIRONMENTS:
        sub = pri[pri["power_environment_used"] == env]
        rank1 = sub[sub["rank_within_environment"] == 1]
        low_conf_at_rank1 = (rank1["confidence_level"] == "low").sum()
        print(f"{env}: {low_conf_at_rank1}/{len(rank1)} segments tied at rank 1 are confidence_level=='low' "
              f"(non-zero is expected -- confidence must not auto-demote rank)")

    print("\n=== delta_fatal_abs: cross-environment comparability check ===")
    print(pri.groupby("power_environment_used")["delta_fatal_abs"].describe())

    urban_path, rural_path = write_priority_environment_lists(target)

    print("\n=== top 5 of each environment list (as written to CSV, ties broken by delta_fatal_percent) ===")
    cols = ["segment_id", "country", "road_class", "land_use", "rank_within_environment",
            "delta_fatal_percent", "delta_fatal_abs", "confidence_note"]
    for path in [urban_path, rural_path]:
        print(f"\n-- {path} --")
        print(pd.read_csv(path)[cols].head(5).to_string(index=False))
    print(f"\nsaved {urban_path}")
    print(f"saved {rural_path}")
