"""Fast-path reproduction for reviewers cloning this repository.

`build_v_safe.py` is the *full* pipeline -- it starts from `schema.load_target()`,
which needs the raw GeoJSON (`data/raw/*.geojson`, too large for git, gitignored)
and WorldPop rasters (`data/external/*.tif`, ~2GB combined, also gitignored and
fetched via `fetch_worldpop.py`). A reviewer cloning this repo does not have
either and should not be expected to download ~2GB to see the deliverables.

`data/processed/segments_v_safe.parquet` -- unlike the raw/external inputs --
*is* committed to git: it's the fully-computed output of the entire pipeline
(V_safe, exposure, misalignment, safety_score, priority_class, review_track,
all already baked in). Every downstream artifact (map, GeoJSON/GPKG, static
PNG, priority CSV lists, sensitivity analysis) is derived purely from that
one file's columns -- none of it touches `schema.load_target()`, rasters,
or OSM extraction. So this script reproduces every deliverable from a clean
clone in well under a minute, with zero large external downloads.

Re-deriving `segments_v_safe.parquet` itself from scratch is the *full* path
(`README.md`'s "Full reproduction path"): copy `QGIS/ADB_Innovation_*.geojson` into
`data/raw/` (identical content, just committed under a different path because
of git size limits -- verified byte-identical), run
`src/fetch_worldpop.py`, then `python src/build_v_safe.py`. That path is for
someone who wants to verify the computation itself, not someone checking the
deliverables.
"""

import sys
import warnings

sys.path.insert(0, "src")

warnings.filterwarnings("ignore", category=UserWarning)

INPUT_PATH = "data/processed/segments_v_safe.parquet"


def main():
    import geopandas as gpd

    from priority_lists import write_priority_environment_lists
    from priority_map import build_priority_map, plot_static_summary, write_geo_outputs
    from review_track import FIELD_CHECK_NEEDED, REVIEW_NEEDED, write_lists
    from sensitivity_analysis import power_model_sensitivity, sample_size_robustness, weight_robustness

    gdf = gpd.read_parquet(INPUT_PATH)
    print(f"loaded {INPUT_PATH} ({len(gdf)} rows) -- no raw GeoJSON / rasters / OSM extraction needed")

    valid = gdf[gdf["data_quality_flag"].isna()]
    print(f"\npriority_class by country:")
    print(valid.groupby("country")["priority_class"].value_counts().unstack())

    review_path, field_check_path = write_lists(gdf)
    print(f"\nsaved {review_path} ({(valid['review_track'] == REVIEW_NEEDED).sum()} rows)")
    print(f"saved {field_check_path} ({(valid['review_track'] == FIELD_CHECK_NEEDED).sum()} rows)")

    urban_path, rural_path = write_priority_environment_lists(gdf)
    on_list = valid["rank_within_environment"].notna()
    print(f"saved {urban_path} ({(on_list & (valid['power_environment_used'] == 'urban_residential')).sum()} rows)")
    print(f"saved {rural_path} ({(on_list & (valid['power_environment_used'] == 'rural_freeway')).sum()} rows)")

    fmap = build_priority_map(gdf)
    fmap.save("outputs/priority_map.html")
    print("saved outputs/priority_map.html")

    geojson_path, gpkg_path = write_geo_outputs(gdf)
    print(f"saved {geojson_path}")
    print(f"saved {gpkg_path}")

    png_path = plot_static_summary(gdf)
    print(f"saved {png_path}")

    print("\n=== sensitivity analysis ===")
    s = sample_size_robustness(valid)
    print(f"sample-size robustness: recall={s['recall_of_baseline']:.1%}, jaccard={s['jaccard']:.1%}")
    w = weight_robustness(valid)
    print(w[["n_other", "recall_of_baseline", "jaccard"]])
    p = power_model_sensitivity(valid)
    print(p.round(1))


if __name__ == "__main__":
    main()
