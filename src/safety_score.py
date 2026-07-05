"""Speed Safety Score -- a transparent, explainable one-dimensional priority
score over three axes: misalignment, exposure, and confidence.

★ Three axes, weighted sum, no black box ★
(a) misalignment  -- `misalignment_magnitude` (main "too high" direction
    only), the policy gap this whole deliverable is about.
(b) exposure       -- `exposure_level` (high/medium/low). Same gap, more
    people exposed, higher priority -- this is what lets two segments with
    an identical km/h gap rank differently.
(c) confidence     -- `exposure_confidence`, `is_separated_confidence`, and
    `speedlimit_plausibility` integrated into one `confidence_level`
    (high/medium/low) by counting how many of the three read "low". This is
    the first point `speedlimit_plausibility` feeds the score itself
    (review_track.py then uses it again, on its own, to split the list).

Weights (WEIGHT_MISALIGNMENT=0.50, WEIGHT_EXPOSURE=0.35,
WEIGHT_CONFIDENCE=0.15): misalignment gets the largest share because it's
the deliverable's main axis; exposure is weighted
second so it can meaningfully separate segments that tie on misalignment,
without ever letting a low-confidence reading outweigh the two substantive
axes. These are stated assumptions, not fitted values -- sensitivity_analysis.py
tests them under alternative weightings.

★ Why percentile-based class cutoffs, not fixed score thresholds ★
Empirically, "high misalignment AND high exposure" alone already covers
~3,300/14,711 segments (22%) -- the underlying problem this dataset
surfaces is genuinely widespread, not a rare-event hunt. A government
agency can't act on thousands at once,
so `safety_score`'s top class is deliberately defined as the most severe
slice *by percentile* (top ~2%) rather than "anyone above some score" --
the score still ranks everyone the same transparent way, the class
boundary is just chosen to match what's actionable. PRIORITY_CLASS_QUANTILES
below is the one knob; sensitivity_analysis.py re-runs this under weight
perturbation to check the resulting list is not an artifact of this choice.

★ Why the percentile cutoff is computed per (country, land_use), not over
the pooled population ★
Thailand and Maharashtra are two separately funded projects (different
budget lines), each needing its own "top N segments to act on" -- not a
combined ranking where one country's systematically larger misalignment
crowds the other out entirely. That's exactly what a single pooled
percentile did in practice: Maharashtra's score distribution sits lower
across the board, so a global top-3%/10%/15% cut put 0 Maharashtra
segments in Top Priority/Priority and only 69/3,577 in Watch -- the other 98% read
as "No Issue" purely from being compared against Thailand, not because
Maharashtra has no relative problem segments of its own.

The same cross-population risk exists one level down between urban and
rural within a single country: `exposure_level.py` already computes its
own tertile separately per (country, land_use) for exactly this reason
(URBAN_SIGNALS/RURAL_SIGNALS differ, and the two systems' score
distributions aren't comparable on one scale). Pooling urban+rural back
together here, after keeping them apart everywhere upstream, would let
whichever land_use happens to score higher in a given country crowd the
other out of that country's own top-3%/10%/15% -- the same artifact this
docstring already documents for country-pooling, one level down. So the
threshold is computed within each of the four (country, land_use) cells
independently. `safety_score` itself (the continuous 0-100 value) is still
computed identically and (country, land_use)-independently; only the
*priority_class* cutoff is taken within each cell's own score distribution,
so each of the four cells gets its own top-3%/10%/15% triage list.
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

MISALIGNMENT_CAP_KMH = 60  # >=60 km/h over V_safe is already "as severe as it gets" for ranking purposes
WEIGHT_MISALIGNMENT = 0.50
WEIGHT_EXPOSURE = 0.35
WEIGHT_CONFIDENCE = 0.15

EXPOSURE_POINTS = {"low": 0.0, "medium": 0.5, "high": 1.0}
CONFIDENCE_COLUMNS = ["exposure_confidence", "is_separated_confidence", "speedlimit_plausibility"]

# Cumulative top-down shares of the *valid* population, computed
# separately within each country (see module docstring).
# top: most severe top 3% -- the literal "act on this first" list.
# priority: next 7% (top 10% cumulative) -- the follow-up pipeline.
# watch: next 10% (top 20% cumulative) -- monitor, no action yet.
PRIORITY_CLASS_QUANTILES = {
    "Top Priority": 0.97,
    "Priority": 0.90,
    "Watch": 0.80,
}
PRIORITY_CLASSES_ORDERED = ["Top Priority", "Priority", "Watch", "No Issue"]
DATA_QUALITY_CLASS = "Data Quality Issue (Excluded)"

CONFIDENCE_LEVELS = ["high", "medium", "low"]


def _confidence_level(low_count: pd.Series) -> pd.Series:
    return pd.Series(
        np.select([low_count == 0, low_count == 1], ["high", "medium"], default="low"),
        index=low_count.index,
    )


def _confidence_label_en(level: str) -> str:
    return {"high": "High", "medium": "Medium", "low": "Low"}[level]


def _exposure_label_en(level: str) -> str:
    return {"high": "High", "medium": "Medium", "low": "Low"}[level]


def _explain(misalignment_magnitude, exposure_level, confidence_level) -> str:
    if misalignment_magnitude <= 0:
        gap = "Posted speed limit does not exceed the safe speed (V_safe)"
    else:
        gap = f"Posted speed limit is {misalignment_magnitude:.0f} km/h above the safe speed (V_safe)"
    return (
        f"{gap}. VRU exposure: {_exposure_label_en(exposure_level)}. "
        f"Data confidence: {_confidence_label_en(confidence_level)}."
    )


def add_safety_score(
    gdf,
    weight_misalignment=WEIGHT_MISALIGNMENT,
    weight_exposure=WEIGHT_EXPOSURE,
    weight_confidence=WEIGHT_CONFIDENCE,
):
    """Weight overrides exist solely for sensitivity_analysis.py's weight-
    sensitivity test -- the pipeline itself always calls this with the
    documented defaults above."""
    gdf = gdf.copy()
    has_flag_col = "data_quality_flag" in gdf.columns
    valid_mask = gdf["data_quality_flag"].isna() if has_flag_col else pd.Series(True, index=gdf.index)
    valid = gdf.loc[valid_mask]

    low_count = (valid[CONFIDENCE_COLUMNS] == "low").sum(axis=1)
    confidence_level = _confidence_level(low_count)

    misalignment_norm = valid["misalignment_magnitude"].clip(upper=MISALIGNMENT_CAP_KMH) / MISALIGNMENT_CAP_KMH
    exposure_norm = valid["exposure_level"].astype(str).map(EXPOSURE_POINTS).astype(float)
    confidence_norm = confidence_level.map({"high": 1.0, "medium": 2 / 3, "low": 1 / 3})

    score = 100 * (
        weight_misalignment * misalignment_norm
        + weight_exposure * exposure_norm
        + weight_confidence * confidence_norm
    )

    priority_class = pd.Series(DATA_QUALITY_CLASS, index=gdf.index, dtype=object)
    valid_class = pd.Series("No Issue", index=valid.index, dtype=object)
    thresholds = {}
    for country in valid["country"].unique():
        for land_use in valid.loc[valid["country"] == country, "land_use"].unique():
            cell_mask = (valid["country"] == country) & (valid["land_use"] == land_use)
            cell_thresholds = {label: score[cell_mask].quantile(q) for label, q in PRIORITY_CLASS_QUANTILES.items()}
            thresholds[(country, land_use)] = cell_thresholds
            # Iterate from the least exclusive cutoff up, so a later (stricter) match overwrites it.
            for label in ["Watch", "Priority", "Top Priority"]:
                valid_class.loc[cell_mask & (score >= cell_thresholds[label])] = label

    explanation = pd.Series(
        [
            "Speed data (SpeedLimit/MedianSpeed/F85) are all zero; excluded from scoring (data quality issue)."
        ]
        * len(gdf),
        index=gdf.index,
        dtype=object,
    )
    valid_explanation = [
        _explain(mag, level, conf)
        for mag, level, conf in zip(valid["misalignment_magnitude"], valid["exposure_level"], confidence_level)
    ]

    gdf["confidence_level"] = pd.NA
    gdf.loc[valid_mask, "confidence_level"] = confidence_level.values
    gdf["safety_score"] = pd.NA
    gdf.loc[valid_mask, "safety_score"] = score.values
    gdf["safety_score"] = pd.to_numeric(gdf["safety_score"])
    priority_class.loc[valid_mask] = valid_class.values
    gdf["priority_class"] = pd.Categorical(
        priority_class, categories=PRIORITY_CLASSES_ORDERED + [DATA_QUALITY_CLASS], ordered=True
    )
    explanation.loc[valid_mask] = valid_explanation
    gdf["score_explanation"] = explanation

    return gdf, thresholds


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    from build_v_safe import build

    target, _, thresholds = build()  # build() already runs add_safety_score as part of the pipeline
    valid = target[target["data_quality_flag"].isna()]

    print("priority class score thresholds (per country, land_use):")
    for cell, cell_thresholds in thresholds.items():
        print(f"  {cell}: {{{', '.join(f'{k}: {round(v, 1)}' for k, v in cell_thresholds.items())}}}")
    print()
    print("priority_class distribution by country, land_use:")
    print(valid.groupby(["country", "land_use"])["priority_class"].value_counts().unstack().reindex(columns=PRIORITY_CLASSES_ORDERED))
    print()
    print("=== sanity: top class composition ===")
    top = valid[valid["priority_class"] == "Top Priority"]
    print(f"n={len(top)}")
    print("by country:\n", top["country"].value_counts())
    print("exposure_level in top class:\n", top["exposure_level"].value_counts())
    print("motorway segments in top class:", (top["road_class"] == "motorway").sum())
    print("low-exposure segments in top class:", (top["exposure_level"] == "low").sum())
    print()
    print("=== sample explanations ===")
    for s in top["score_explanation"].head(3):
        print(" -", s)
