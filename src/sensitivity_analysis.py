"""Sensitivity analysis, answering the organizers' Methodology Warning directly.

Three independent questions, the first two phrased as "if we'd made a
different defensible choice, would the Top Priority (top-priority) list have come
out basically the same list of segments?", the third (FINAL_SPRINT_PLAN.md
P1-1 item 3) phrased as "how wide is the P0-1 benefit estimate once its
exponent's own reported uncertainty is carried through?":

1. **Sample-size robustness.** `sample_size_total` is right-skewed and its
   bottom quartile is flagged as a "low confidence" candidate threshold
   (README §3). If the top-priority list is
   largely segments whose extreme misalignment/exposure reading is an
   artifact of a thin sample, restricting to only the well-sampled segments
   and recomputing should shrink/reshuffle the list a lot. It shouldn't if
   the conclusion is real.
2. **Weight robustness.** `safety_score.py`'s 0.50/0.35/0.15 weighting is a
   stated assumption (disclosed openly, not hidden). Re-running the same pipeline under a handful of other
   defensible weightings and checking whether the top-priority list stays
   close to the baseline is the direct test of "the conclusion doesn't
   hinge on our specific weight choice."
3. **Power Model exponent sensitivity.** `fatality_reduction.py` already
   carries Cameron & Elvik (2010)'s 95% CI for the exponent through to
   `delta_fatal_percent_ci_low/high` on every segment -- there's no separate
   formula to re-derive here. This function only aggregates those
   already-computed columns for the segments the P0-1 benefit headline is
   actually about (`review_track == "Review Needed"`, restricted to
   `delta_fatal_percent > 0` -- see fatality_reduction.py's docstring on why
   non-positive segments are excluded from benefit framing), so the README
   can report a range instead of a single point estimate.

All three reuse pipeline output unmodified (1-2 call `add_safety_score`
directly; 3 reads `add_fatal_reduction`'s output columns) -- there's no
separate sensitivity-only logic to keep in sync with the real pipeline.
"""

import sys
import warnings

import pandas as pd

sys.path.insert(0, "src")

from review_track import REVIEW_NEEDED  # noqa: E402
from safety_score import add_safety_score  # noqa: E402

SAMPLE_SIZE_LOW_QUANTILE = 0.25  # bottom-quartile "low confidence" candidate threshold

WEIGHT_SCENARIOS = {
    "baseline (0.50/0.35/0.15)": (0.50, 0.35, 0.15),
    "equal weights (0.33/0.33/0.34)": (1 / 3, 1 / 3, 1 / 3),
    "misalignment-heavy (0.70/0.20/0.10)": (0.70, 0.20, 0.10),
    "exposure-heavy (0.30/0.55/0.15)": (0.30, 0.55, 0.15),
    "confidence-heavy (0.40/0.30/0.30)": (0.40, 0.30, 0.30),
}


def _top_segment_ids(gdf: pd.DataFrame) -> set:
    return set(gdf.loc[gdf["priority_class"] == "Top Priority", "segment_id"])


def _overlap_stats(baseline_top: set, other_top: set) -> dict:
    inter = baseline_top & other_top
    union = baseline_top | other_top
    return {
        "n_baseline": len(baseline_top),
        "n_other": len(other_top),
        "n_overlap": len(inter),
        "recall_of_baseline": len(inter) / len(baseline_top) if baseline_top else float("nan"),
        "jaccard": len(inter) / len(union) if union else float("nan"),
    }


def sample_size_robustness(valid: pd.DataFrame) -> dict:
    baseline_top = _top_segment_ids(valid)

    threshold = valid["sample_size_total"].quantile(SAMPLE_SIZE_LOW_QUANTILE)
    high_sample = valid[valid["sample_size_total"] >= threshold].copy()
    # Drop the columns add_safety_score recreates, so re-running it on the
    # filtered population can't accidentally read stale values.
    high_sample = high_sample.drop(
        columns=["confidence_level", "safety_score", "priority_class", "score_explanation"]
    )
    recomputed, _ = add_safety_score(high_sample)
    recomputed_top = _top_segment_ids(recomputed)

    n_excluded_from_baseline_top = len(baseline_top - set(high_sample["segment_id"]))

    stats = _overlap_stats(baseline_top, recomputed_top)
    stats["sample_size_threshold"] = threshold
    stats["n_excluded_low_sample_segments"] = len(valid) - len(high_sample)
    stats["n_baseline_top_excluded_as_low_sample"] = n_excluded_from_baseline_top
    return stats


def weight_robustness(valid: pd.DataFrame) -> pd.DataFrame:
    base = valid.drop(columns=["confidence_level", "safety_score", "priority_class", "score_explanation"])
    baseline_top = None
    rows = []
    for label, (w_m, w_e, w_c) in WEIGHT_SCENARIOS.items():
        recomputed, _ = add_safety_score(base, weight_misalignment=w_m, weight_exposure=w_e, weight_confidence=w_c)
        top = _top_segment_ids(recomputed)
        if baseline_top is None:
            baseline_top = top  # first entry in dict is the baseline scenario
        stats = _overlap_stats(baseline_top, top)
        stats["scenario"] = label
        rows.append(stats)
    return pd.DataFrame(rows).set_index("scenario")


def power_model_sensitivity(valid: pd.DataFrame) -> pd.DataFrame:
    review_candidates = valid[(valid["review_track"] == REVIEW_NEEDED) & (valid["delta_fatal_percent"] > 0)]
    return review_candidates.groupby("power_environment_used").agg(
        n=("delta_fatal_percent", "count"),
        ci_low_mean=("delta_fatal_percent_ci_low", "mean"),
        point_estimate_mean=("delta_fatal_percent", "mean"),
        ci_high_mean=("delta_fatal_percent_ci_high", "mean"),
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    df = pd.read_parquet("data/processed/segments_v_safe.parquet")
    valid = df[df["data_quality_flag"].isna()].copy()

    print("=== 1. sample-size robustness ===")
    s = sample_size_robustness(valid)
    print(f"sample_size_total threshold (25th pct): {s['sample_size_threshold']:.0f}")
    print(f"segments excluded as low-sample: {s['n_excluded_low_sample_segments']} / {len(valid)}")
    print(f"baseline Top Priority (n={s['n_baseline']}) segments excluded outright as low-sample: "
          f"{s['n_baseline_top_excluded_as_low_sample']}")
    print(f"recomputed-on-high-sample-only Top Priority: n={s['n_other']}")
    print(f"overlap: {s['n_overlap']} segments "
          f"(recall of baseline={s['recall_of_baseline']:.1%}, jaccard={s['jaccard']:.1%})")

    print("\n=== 2. weight robustness ===")
    w = weight_robustness(valid)
    print(w[["n_baseline", "n_other", "n_overlap", "recall_of_baseline", "jaccard"]].to_string(
        formatters={"recall_of_baseline": "{:.1%}".format, "jaccard": "{:.1%}".format}
    ))

    print("\n=== 3. Power Model exponent sensitivity ===")
    p = power_model_sensitivity(valid)
    print(p.round(1).to_string())
