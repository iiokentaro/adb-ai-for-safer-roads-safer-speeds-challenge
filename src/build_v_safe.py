"""End-to-end pipeline: connect exposure signals and V_safe, write
data/processed/segments_v_safe.parquet, and run sanity checks that do not
use speed_limit to validate V_safe itself.
"""

import sys
import warnings

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, "src")
from exposure_level import add_exposure_level, apply_rural_safety_margin
from exposure_signals import add_crossing_signal, add_poi_proximity
from fatality_reduction import add_fatal_reduction
from junction_speed_cap import add_junction_speed_cap
from misalignment import add_misalignment
from pop_density import add_pop_density
from priority_lists import add_priority_environment_rank, write_priority_environment_lists
from priority_map import build_priority_map, plot_static_summary, write_geo_outputs
from review_track import add_review_track, write_lists
from road_separation import add_is_separated
from safe_speed import add_v_safe
from safety_score import add_safety_score
from schema import load_target
from segment_localization import refine_influenced_segments
from speedlimit_plausibility import add_speedlimit_plausibility

warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_PATH = "data/processed/segments_v_safe.parquet"


def _apply_geometry_signals(target):
    """Run all geometry-driven signal functions in order.

    Called twice in build(): once for Stage 1 (whole segments) and once for
    Stage 2 (after refine_influenced_segments splits influenced segments). Running
    the same functions on child geometries ensures that influenced children get the
    lower V_safe and non-influenced children retain the original higher V_safe.
    Non-split rows produce identical values on the second pass (same geometry).
    """
    target = add_is_separated(target)
    target = add_pop_density(target)
    target = add_poi_proximity(target)
    target = add_crossing_signal(target)
    target = add_exposure_level(target)  # prioritization only; not read by add_v_safe
    target, rural_thresholds = apply_rural_safety_margin(target)
    target = add_v_safe(target)
    target = add_junction_speed_cap(target)  # post-hoc 50km/h cap near OSM junction nodes
    return target, rural_thresholds


def build() -> tuple[pd.DataFrame, dict]:
    target = load_target()

    # Stage 1: whole-segment signal computation (is_vru, near_junction, v_safe).
    # is_separated moves ahead of add_poi_proximity: it doesn't depend on
    # pop_density/POI/exposure columns, and add_poi_proximity now needs it
    # to mask is_vru (and is_mapillary_vru) for separated/motorway/trunk segments.
    target, rural_thresholds = _apply_geometry_signals(target)

    # Stage 2: clip influenced segments at their influence-zone boundary, then re-run
    # the same geometry-driven functions so each child segment gets its own V_safe.
    # Segments not touched by any VRU/junction influence are not modified.
    target = refine_influenced_segments(target)
    target, rural_thresholds = _apply_geometry_signals(target)

    target = add_speedlimit_plausibility(target)
    target = add_misalignment(target)
    target = add_fatal_reduction(target)
    target, score_thresholds = add_safety_score(target)
    target = add_review_track(target)
    target = add_priority_environment_rank(target)
    return target, rural_thresholds, score_thresholds


def sanity_checks(gdf: pd.DataFrame) -> None:
    print("=== v_safe by road_class x land_use (no speed_limit involved) ===")
    print(gdf.groupby(["road_class", "land_use"])["v_safe"].median().unstack())
    urban_median = gdf.loc[gdf["land_use"] == "URBAN", "v_safe"].median()
    rural_median = gdf.loc[gdf["land_use"] == "RURAL", "v_safe"].median()
    print(f"\nurban median v_safe={urban_median}, rural median v_safe={rural_median} "
          f"({'OK: urban <= rural' if urban_median <= rural_median else 'CHECK: urban > rural'})")

    print("\n=== data_quality_flag: retained but excluded from misalignment scoring ===")
    invalid = gdf["data_quality_flag"] == "invalid_speed"
    print(f"{invalid.sum()} / {len(gdf)} segments flagged invalid_speed "
          f"(speed_limit/median_speed/f85_speed all exactly 0)")
    print(gdf.loc[invalid, "road_class"].value_counts())

    print("\n=== diagnostic only: v_safe vs f85_speed (not used to change v_safe; excludes invalid_speed) ===")
    usable = gdf[~invalid]
    over = (usable["f85_speed"] > usable["v_safe"]).sum()
    print(f"segments where observed F85 > v_safe (operating speed exceeds safe speed): "
          f"{over} / {len(usable)} ({over / len(usable):.1%})")

    print("\n=== basis / confidence flags present ===")
    for col in ["v_safe_basis", "exposure_confidence", "speedlimit_plausibility", "is_separated_confidence"]:
        print(f"{col}: {gdf[col].notna().mean():.1%} non-null")

    print("\n=== misalignment (main axis, SpeedLimit - V_safe) ===")
    too_high = usable["misalignment"] > 0
    print(f"too-high direction (SpeedLimit > V_safe, review priority): {too_high.sum()} / {len(usable)} "
          f"({too_high.mean():.1%})")
    print("by road_class:")
    print(usable.groupby("road_class")["misalignment"].apply(lambda s: (s > 0).mean()))
    print("by land_use:")
    print(usable.groupby("land_use")["misalignment"].apply(lambda s: (s > 0).mean()))

    print("\n=== axis agreement: top decile by misalignment_magnitude vs by operating_gap ===")
    top_n = max(1, len(usable) // 10)
    top_misalignment = set(usable.nlargest(top_n, "misalignment_magnitude").index)
    top_operating_gap = set(usable.nlargest(top_n, "operating_gap").index)
    overlap = top_misalignment & top_operating_gap
    print(f"overlap: {len(overlap)} / {top_n} ({len(overlap) / top_n:.1%}) -- "
          f"limit too high AND actually speeding, the highest-confidence priority segments")

    print("\n=== Speed Safety Score / priority_class ===")
    from safety_score import PRIORITY_CLASSES_ORDERED
    print(usable["priority_class"].value_counts().reindex(PRIORITY_CLASSES_ORDERED))
    top = usable[usable["priority_class"] == "Top Priority"]
    print(f"\nTop Priority (n={len(top)}): exposure_level=low count={int((top['exposure_level'] == 'low').sum())}, "
          f"motorway count={int((top['road_class'] == 'motorway').sum())} "
          f"(should both be 0/near-0 -- low-exposure rural motorway must not dominate the top class)")

    print("\n=== review_track: plausibility-based split of the priority list ===")
    from review_track import FIELD_CHECK_NEEDED, REVIEW_NEEDED
    review = usable[usable["review_track"] == REVIEW_NEEDED]
    field_check = usable[usable["review_track"] == FIELD_CHECK_NEEDED]
    print(f"Review Needed (plausibility=high): {len(review)}")
    print(f"Field Verification Needed (plausibility=low):  {len(field_check)}")
    print(f"ratio: {len(review)} : {len(field_check)} ({len(review) / max(1, len(field_check)):.1f} : 1)")
    print(usable[usable["review_track"].notna()].groupby("priority_class")["review_track"].value_counts())

    print("\n=== motorway segments where separation couldn't be confirmed (no OSM match) ===")
    moto_uncertain = gdf[(gdf["road_class"] == "motorway") & (gdf["is_separated_confidence"] == "low")]
    print(f"{len(moto_uncertain)} segments resolved to collision_type="
          f"{moto_uncertain['collision_type'].unique().tolist()} via the safe default, not a confirmed reading")

    print("\n=== is_vru masking (must be 0 for motorway/trunk and for is_separated==True) ===")
    bad_mask = gdf["is_vru"] & (gdf["road_class"].isin(["motorway", "trunk"]) | gdf["is_separated"])
    print(f"is_vru True where it should have been masked to False: {bad_mask.sum()} (should be 0)")
    print(f"is_vru True overall: {gdf['is_vru'].sum()} / {len(gdf)} "
          f"(of which is_mapillary_vru: {gdf['is_mapillary_vru'].sum()}, OSM-school-added: "
          f"{(gdf['is_vru'] & ~gdf['is_mapillary_vru']).sum()})")
    print(f"collision_type=='pedestrian' segments driven by is_vru: "
          f"{(gdf['collision_type'] == 'pedestrian').sum()} (should equal is_vru True count above, "
          f"since pedestrian is no longer reachable via exposure_level)")

    print("\n=== junction speed cap (50km/h within 300m of highway=traffic_signals / junction=yes, "
          "excluding motorway and grade-separated segments) ===")
    capped = gdf["v_safe_basis"] == "side_impact:junction_buffer"
    print(f"segments capped to 50km/h: {capped.sum()} / {len(gdf)} ({capped.mean():.1%})")
    print(f"near a junction (motorway/grade-separated already excluded from near_junction) but "
          f"already <=50 before the cap (left untouched): "
          f"{(gdf['near_junction'] & ~capped).sum()}")


def plot_map(gdf, out_path="outputs/v_safe_map.png"):
    # exposure_confidence=='low' is drawn as a grey halo underneath the
    # v_safe-colored line (wider, plotted first), not a competing line
    # color -- every line's own color still comes from the same RdYlGn
    # scale as the legend; nothing falls outside it.
    cmap, vmin, vmax = "RdYlGn", 30, 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, country in zip(axes, ["thailand", "maharashtra"]):
        sub = gdf[gdf["country"] == country]

        low_conf = sub[sub["exposure_confidence"] == "low"]
        low_conf.plot(ax=ax, color="grey", linewidth=2.0,
                      label="rural safety-margin applied (confidence=low)")

        sub.plot(column="v_safe", ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0.6)

        ax.set_title(f"{country} V_safe (n={len(sub)})")
        ax.set_aspect("equal")
        ax.legend(loc="lower left", fontsize=6)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, ax=axes, label="V_safe (km/h)", shrink=0.6)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    target, rural_thresholds, score_thresholds = build()
    print("rural safety-margin thresholds:", rural_thresholds)
    print("safety_score priority-class thresholds (per country, land_use):")
    for cell, cell_thresholds in score_thresholds.items():
        print(f"  {cell}: {{{', '.join(f'{k}: {round(v, 1)}' for k, v in cell_thresholds.items())}}}")
    print()

    target.to_parquet(OUTPUT_PATH)
    print(f"saved {OUTPUT_PATH} ({len(target)} rows)\n")

    sanity_checks(target)
    plot_map(target)

    review_path, field_check_path = write_lists(target)
    print(f"\nsaved {review_path}")
    print(f"saved {field_check_path}")

    urban_path, rural_path = write_priority_environment_lists(target)
    print(f"saved {urban_path}")
    print(f"saved {rural_path}")

    priority_map = build_priority_map(target)
    priority_map.save("outputs/priority_map.html")
    print("saved outputs/priority_map.html")

    geojson_path, gpkg_path = write_geo_outputs(target)
    print(f"saved {geojson_path}")
    print(f"saved {gpkg_path}")

    png_path = plot_static_summary(target)
    print(f"saved {png_path}")
