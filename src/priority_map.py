"""Priority-segment deliverable map.

Three outputs, all built from the same `priority_class` / `review_track`
columns so they can never drift from each other or from the review lists in
review_track.py:
- an interactive Folium/Leaflet HTML map (primary deliverable, clickable
  popups explain each segment in plain language)
- GeoJSON + GeoPackage (Phase 2 is ESRI-based; these load directly into
  ArcGIS without going through the interactive map at all)
- a static PNG (fallback for a reviewer who can't open the HTML)

★ Display category: map_class (splits "No Issue") ★
The data column priority_class is never modified. For display only, the valid
"No Issue" tier is split into two map_class values via derive_map_class():
- Aligned     (misalignment <= 0): the posted limit is not above V_safe -- nothing to
  lower. This is genuinely "aligned", not just "low priority".
- Low Priority (misalignment > 0): a gap exists but the composite score is low.
Separating them stops "No Issue" from being misread as "limit already equals
V_safe" (see README). Aligned/Low Priority are rendered without popups (the bulk of the
network) to keep the HTML light; Top Priority/Priority/Watch get full popups.

② All five valid categories (Top Priority/Priority/Watch/Low Priority/Aligned) are shown by default
at load. Data Quality Issue stays toggle-on.

③ V_safe-driving point sources are plotted as clustered markers
(folium.plugins.MarkerCluster): Mapillary VRU detections, OSM school nodes, and
junction nodes -- the features that actually localize each segment's V_safe.

④ Two continuous-color speed layers (off by default): current posted
`speed_limit` vs recommended `v_safe`, one FeatureGroup per country per field.
Both use the same RdYlGn/30-100km/h scale as v_safe_map.png so switching
between them (or eyeballing both at once) reads as "did the color change".
These are popup-free -- they're a pure choropleth, and each already duplicates
the full network once per field, so no popup fields are serialized at all.
Unlike the priority-class layers, both speed layers include
data_quality_flag=='invalid_speed' rows too (excluded elsewhere on the map):
in the current-SpeedLimit layer they're drawn black (speed_limit==0 there is
a data artifact, not a real posted limit -- see SPEED_LAYER_FIELDS), while in
the V_safe layer they're colored normally, since V_safe is computed
independently of speed_limit and is real for every row.

⑤ 300m junction buffer (off by default): one true circle
(folium.Circle/Leaflet L.circle, not a shapely.buffer() polygon) per cached
junction node, radius=JUNCTION_BUFFER_M -- the exact same zone
junction_speed_cap.py's dwithin check uses to cap V_safe to 50km/h. A real
circle needs only center+radius, so this stays light even at ~10,700 points
combined, unlike a buffered polygon which would need dozens of vertices per
point to look round.

★ No dedicated review_track (triage) layer -- avoiding double-embedding ★
An earlier version added a second, cross-cutting FeatureGroup pair for
Review Needed/Field Verification Needed, but every row in it was already present verbatim (same
geometry, same popup fields) inside that row's country x map_class layer --
a full second copy of ~8,500 rows' worth of GeoJSON for a view that adds no
information the dashArray styling below doesn't already show. Removed; the
solid/dashed line style on the existing layers is the only place review_track
is now rendered.

Geometry for the interactive map is rendered at full resolution (no Douglas-Peucker
simplification). Coordinates are rounded to COORD_ROUND_DECIMALS (~1.1 m) only to
cut HTML weight; the GeoJSON/GPKG exports are also unaffected.
Popup field values are rounded (see _sanitize_for_geojson) before being turned
into strings, since a float's full repr adds bytes with no reading value.

★ Visual encoding ★
Color = map_class (Top Priority=red, Priority=orange, Watch=yellow, Low Priority=yellow-green,
Aligned=saturated green, Data Quality Issue=dark grey). Line style = review_track
(Review Needed=solid, Field Verification Needed=dashed). Points colored by source type.

The layer tree (folium.plugins.TreeLayerControl) groups overlays by country, by
map_class within country, a speed-layer group, and a separate POI/junction
point group.
"""

import sys

import branca.colormap as cm
import folium
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from folium.plugins import FastMarkerCluster, TreeLayerControl

sys.path.insert(0, "src")
from exposure_signals import _tag_osm_pois, load_cached_signals, load_mapillary_pois
from junction_speed_cap import JUNCTION_BUFFER_M, load_cached_junctions

# Display-only split of the "No Issue" tier (the underlying priority_class column is
# never changed): Aligned = the posted limit is not above V_safe (misalignment <= 0,
# nothing to lower), Low Priority = a gap exists (misalignment > 0) but the composite score
# is low. Splitting them keeps "no action needed" from being read as "limit already
# matches V_safe" -- see README.
MAP_ALIGNED = "Aligned"
MAP_LOW_PRIORITY = "Low Priority"

PRIORITY_COLORS = {
    "Top Priority": "#d73027",
    "Priority": "#fc8d59",
    "Watch": "#fee08b",
    MAP_LOW_PRIORITY: "#a6d96a",  # yellow-green: gap exists but low priority
    MAP_ALIGNED: "#4dd0e1",       # light blue: limit not above V_safe
    "No Issue": "#d9d9d9",          # legacy fallback (should not appear once map_class is set)
    "Data Quality Issue (Excluded)": "#636363",
}
# All line categories are rendered at the same weight; color alone distinguishes them.
LINE_WEIGHT = 2
PRIORITY_WEIGHT = {cls: LINE_WEIGHT for cls in [
    "Top Priority", "Priority", "Watch", MAP_LOW_PRIORITY, MAP_ALIGNED, "No Issue",
    "Data Quality Issue (Excluded)",
]}
# The five valid priority tiers shown by default (② all categories on at load).
# Aligned/Low Priority are rendered without popups to keep the HTML light (they
# are the bulk of the network, ~"no action needed").
MAP_CLASSES_VALID = ["Top Priority", "Priority", "Watch", MAP_LOW_PRIORITY, MAP_ALIGNED]
POPUP_CLASSES = {"Top Priority", "Priority", "Watch"}
LIGHT_CLASSES = {MAP_ALIGNED, MAP_LOW_PRIORITY}

# V_safe-driving point sources plotted as clustered markers (③).
POINT_STYLES = {
    "mapillary_vru": {"color": "#762a83", "label": "Mapillary VRU Detection (signs/markings)"},
    "osm_school": {"color": "#1b7837", "label": "OSM School Node"},
    "junction": {"color": "#2166ac", "label": "Junction Node"},
}
_MAPILLARY_VRU_FLAGS = ["map_is_pedestrian", "map_is_bicycle", "map_is_school"]

COORD_ROUND_DECIMALS = 5  # ~1.1m; rounds rendered line coords to cut HTML weight (display only)

# ④ Continuous speed layers (current SpeedLimit vs recommended V_safe). Same
# vmin/vmax as build_v_safe.py:plot_map's RdYlGn scale for v_safe_map.png,
# so the two interactive layers and the static V_safe map all read the same way.
SPEED_COLOR_VMIN, SPEED_COLOR_VMAX = 30, 100
# label -> (field, flag_missing). flag_missing=True draws data_quality_flag==
# 'invalid_speed' rows black instead of by value -- only meaningful for the
# current-SpeedLimit layer (those rows' speed_limit==0 is a data artifact, not
# a real posted limit); v_safe is computed independently of speed_limit and is
# always real, so those same rows get colored normally in that layer.
SPEED_LAYER_FIELDS = {
    "Current Speed Limit (SpeedLimit)": ("speed_limit", True),
    "Recommended Safe Speed (V_safe)": ("v_safe", False),
}

# Popup-enabled layers only. delta_fatal_percent_ci_low/high and
# power_environment_used were dropped: rarely-read supporting detail that cost
# 3 extra fields x every popup-enabled feature; power_environment_used is
# implied by road_class/land_use already shown, and the CI bounds are a
# secondary refinement of delta_fatal_percent (kept).
POPUP_FIELDS = [
    "score_explanation", "priority_class", "review_track",
    "speed_limit", "v_safe", "misalignment", "exposure_level",
    "confidence_level", "speedlimit_plausibility", "road_class", "land_use",
    "delta_fatal_percent",
    "country", "street_image_link",
]
POPUP_ALIASES = [
    "Explanation", "Priority Class", "Review Track",
    "Speed Limit (km/h)", "Safe Speed V_safe (km/h)", "Gap (km/h)", "VRU Exposure Level",
    "Data Confidence", "Speed Limit Plausibility", "Road Class", "Land Use",
    "Estimated Fatal Crash Reduction (point estimate, %)",
    "Country", "Field-check coordinates (lon1,lat1,lon2,lat2)",
]

# Numeric fields get rounded before being turned into popup strings (see
# _sanitize_for_geojson) -- a raw float repr (e.g. delta_fatal_percent's
# "91.744665...") is bytes of precision nobody reads in a popup.
POPUP_ROUND_DECIMALS = {
    "speed_limit": 0, "v_safe": 0, "misalignment": 1, "delta_fatal_percent": 1,
}


def derive_map_class(gdf: gpd.GeoDataFrame) -> pd.Series:
    """Display-only category: splits the valid "No Issue" tier into Aligned / Low Priority.

    priority_class is left untouched in the data; this only drives map color/layers.
    - Top Priority / Priority / Watch / Data Quality Issue: unchanged.
    - No Issue & misalignment <= 0  -> Aligned      (posted limit not above V_safe).
    - No Issue & misalignment >  0  -> Low Priority (gap exists but low composite score).
    - No Issue & misalignment is NA -> Low Priority (conservative: don't label as Aligned).
    """
    pc = gdf["priority_class"].astype(str)
    mis = pd.to_numeric(gdf["misalignment"], errors="coerce")
    out = pc.copy()
    no_issue = pc == "No Issue"
    out = out.mask(no_issue & (mis <= 0), MAP_ALIGNED)
    out = out.mask(no_issue & (mis > 0), MAP_LOW_PRIORITY)
    out = out.mask(no_issue & mis.isna(), MAP_LOW_PRIORITY)
    return out


def _sanitize_for_geojson(gdf: gpd.GeoDataFrame, fields: list[str]) -> gpd.GeoDataFrame:
    """folium/Leaflet renders fields as raw JS; pd.NA / NaN in object columns
    breaks the popup template, so every field actually shown is forced to a
    plain string with an explicit placeholder for missing values.

    Fields listed in POPUP_ROUND_DECIMALS are rounded to that precision first --
    done here (once, on the numeric value) rather than after stringifying, so
    "91.744665..." becomes "91.7" instead of a long float repr, shaving bytes
    off every popup-enabled feature.
    """
    sub = gdf[fields + ["geometry"]].copy()
    for col in fields:
        if col in POPUP_ROUND_DECIMALS:
            decimals = POPUP_ROUND_DECIMALS[col]
            rounded = pd.to_numeric(sub[col], errors="coerce").round(decimals)
            sub[col] = rounded.astype("Int64") if decimals == 0 else rounded
        sub[col] = sub[col].astype(object)
        sub[col] = sub[col].where(sub[col].notna(), "(unknown)").astype(str)
    return sub


def _style_function(feature):
    props = feature["properties"]
    cls = props.get("map_class", props.get("priority_class"))
    style = {
        "color": PRIORITY_COLORS.get(cls, "#999999"),
        "weight": PRIORITY_WEIGHT.get(cls, LINE_WEIGHT),
        "opacity": 0.5 if cls in LIGHT_CLASSES else 0.85,
    }
    if props.get("review_track") == "Field Verification Needed":
        style["dashArray"] = "6,6"
    return style


def _layer_for(gdf, label, with_popup, show=False):
    fg = folium.FeatureGroup(name=label, show=show)
    if len(gdf) == 0:
        return fg
    fields = list(POPUP_FIELDS) if with_popup else ["priority_class", "review_track"]
    if "map_class" not in fields:
        fields = fields + ["map_class"]
    sanitized = _sanitize_for_geojson(gdf, fields)
    gj = folium.GeoJson(
        sanitized,
        style_function=_style_function,
        popup=folium.GeoJsonPopup(fields=POPUP_FIELDS, aliases=POPUP_ALIASES, max_width=320) if with_popup else None,
    )
    gj.add_to(fg)
    return fg


SPEED_MISSING_COLOR = "#000000"  # black: no real current speed-limit value to color by


def _speed_style_function(field, colormap, flag_missing: bool):
    """Style-function factory for a continuous speed field (speed_limit / v_safe).

    Values are clipped to [SPEED_COLOR_VMIN, SPEED_COLOR_VMAX] before coloring --
    same clipping the static v_safe_map.png colorbar implicitly applies via
    vmin/vmax -- so an occasional out-of-range value doesn't blow out the scale.

    flag_missing=True (current-SpeedLimit layer only): rows with
    data_quality_flag=='invalid_speed' have speed_limit==0 as a data artifact
    (schema.has_invalid_zero_speeds -- speed_limit/median_speed/f85_speed all
    exactly 0), not a real posted limit, so they're drawn black instead of
    being colored (or silently dropped, as they previously were).
    """
    def _style(feature):
        props = feature["properties"]
        if flag_missing and props.get("data_quality_flag") == "invalid_speed":
            return {"color": SPEED_MISSING_COLOR, "weight": LINE_WEIGHT, "opacity": 0.85}
        try:
            val = float(props.get(field))
        except (TypeError, ValueError):
            color = "#999999"
        else:
            val = min(max(val, SPEED_COLOR_VMIN), SPEED_COLOR_VMAX)
            color = colormap(val)
        return {"color": color, "weight": LINE_WEIGHT, "opacity": 0.85}
    return _style


def _speed_layer_for(gdf, label, field, colormap, flag_missing: bool, show=False) -> folium.FeatureGroup:
    """Popup-free choropleth for one speed field. Only the fields the style
    function actually reads are kept in the serialized properties (`field`,
    plus `data_quality_flag` when flag_missing) -- no popup means nothing else
    needs to travel with the feature."""
    fg = folium.FeatureGroup(name=label, show=show)
    if len(gdf) == 0:
        return fg
    fields = [field, "data_quality_flag"] if flag_missing else [field]
    sanitized = _sanitize_for_geojson(gdf, fields)
    gj = folium.GeoJson(
        sanitized,
        style_function=_speed_style_function(field, colormap, flag_missing),
    )
    gj.add_to(fg)
    return fg


def _vru_points(country: str) -> gpd.GeoDataFrame:
    map_pois = load_mapillary_pois(country)
    if len(map_pois) == 0:
        return map_pois
    mask = pd.Series(False, index=map_pois.index)
    for flag in _MAPILLARY_VRU_FLAGS:
        if flag in map_pois.columns:
            mask = mask | (map_pois[flag] == True)  # noqa: E712
    return map_pois[mask]


def _school_points(country: str) -> gpd.GeoDataFrame:
    osm = _tag_osm_pois(load_cached_signals(country)["pois"])
    if len(osm) == 0 or "is_school" not in osm.columns:
        return osm.iloc[0:0]
    return osm[osm["is_school"] == True]  # noqa: E712


def _point_layer(points_gdf, label, color, show) -> folium.FeatureGroup:
    """Clustered point layer using FastMarkerCluster.

    FastMarkerCluster serializes only a [[lat, lon], ...] array plus one JS
    callback, which is far lighter than one folium marker per point (tens of
    thousands of VRU/junction points would otherwise bloat the HTML)."""
    fg = folium.FeatureGroup(name=label, show=show)
    if points_gdf is None or len(points_gdf) == 0:
        return fg
    pts = points_gdf
    if pts.crs is not None and str(pts.crs).upper() != "EPSG:4326":
        pts = pts.to_crs("EPSG:4326")
    coords = []
    for geom in pts.geometry:
        if geom is None or geom.is_empty:
            continue
        # representative_point() for any non-Point geometry (some OSM POIs are ways).
        pt = geom if geom.geom_type == "Point" else geom.representative_point()
        coords.append([round(pt.y, COORD_ROUND_DECIMALS), round(pt.x, COORD_ROUND_DECIMALS)])
    if not coords:
        return fg
    callback = (
        "function (row) {"
        f"  return L.circleMarker(new L.LatLng(row[0], row[1]), "
        f"{{radius: 3, color: '{color}', fillColor: '{color}', fillOpacity: 0.7, weight: 1}});"
        "}"
    )
    FastMarkerCluster(data=coords, callback=callback).add_to(fg)
    return fg


def _junction_buffer_layer(points_gdf, label, color, radius_m, show=False) -> folium.FeatureGroup:
    """True-circle junction buffer (Leaflet L.circle: center + radius in
    meters), one folium.Circle per junction node -- the exact JUNCTION_BUFFER_M
    zone junction_speed_cap.py's dwithin check caps V_safe within, drawn as a
    geometrically accurate circle rather than a shapely.buffer() polygon (which
    would need dozens of vertices per point to look round, for ~10,700 points
    combined across both countries)."""
    fg = folium.FeatureGroup(name=label, show=show)
    if points_gdf is None or len(points_gdf) == 0:
        return fg
    pts = points_gdf
    if pts.crs is not None and str(pts.crs).upper() != "EPSG:4326":
        pts = pts.to_crs("EPSG:4326")
    for geom in pts.geometry:
        if geom is None or geom.is_empty:
            continue
        pt = geom if geom.geom_type == "Point" else geom.representative_point()
        folium.Circle(
            location=[round(pt.y, COORD_ROUND_DECIMALS), round(pt.x, COORD_ROUND_DECIMALS)],
            radius=radius_m,
            color=color,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.08,
        ).add_to(fg)
    return fg


def _round_geometry(geom):
    """Round line coordinates to COORD_ROUND_DECIMALS (~1m) to cut HTML weight.
    Display-only; never applied to the GeoJSON/GPKG exports."""
    if geom is None or geom.is_empty:
        return geom
    return shapely.transform(geom, lambda a: np.round(a, COORD_ROUND_DECIMALS))


def build_priority_map(gdf: gpd.GeoDataFrame) -> folium.Map:
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.apply(_round_geometry)
    gdf["map_class"] = derive_map_class(gdf)

    valid = gdf[gdf["data_quality_flag"].isna()]
    invalid = gdf[gdf["data_quality_flag"].notna()]

    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    fmap = folium.Map(location=center, zoom_start=4, tiles="cartodbpositron")

    country_tree = []
    for country in ["thailand", "maharashtra"]:
        country_valid = valid[valid["country"] == country]
        children = []
        for cls in MAP_CLASSES_VALID:
            sub = country_valid[country_valid["map_class"] == cls]
            # ② all five categories shown by default; Aligned/Low Priority stay popup-less to
            # keep the HTML light (they are the bulk of the network).
            fg = _layer_for(sub, f"{country}: {cls} (n={len(sub)})",
                            with_popup=(cls in POPUP_CLASSES), show=True)
            fg.add_to(fmap)
            children.append({"label": f"{cls} (n={len(sub)})", "layer": fg})
        country_tree.append({"label": f"{country} (n={len(country_valid)})", "children": children, "collapsed": True})

    # No separate triage (review_track) layer here: every one of those rows is
    # already rendered above inside its country x map_class group, with the
    # same dashArray (Review Needed=solid / Field Verification Needed=dashed) styling. A cross-cutting
    # copy would just re-embed the same ~8,500 rows' geometry+properties a
    # second time for a view that adds no new information (see module docstring).

    # ③ V_safe-driving point sources (Mapillary VRU / OSM school / junctions),
    # clustered. Shown by default so reviewers can see what localized each V_safe.
    point_children = []
    buffer_children = []
    for country in ["thailand", "maharashtra"]:
        vru = _vru_points(country)
        school = _school_points(country)
        junc = load_cached_junctions(country)
        ckids = []
        for key, pts in (("mapillary_vru", vru), ("osm_school", school), ("junction", junc)):
            style = POINT_STYLES[key]
            n = 0 if pts is None else len(pts)
            fg = _point_layer(pts, f"{country}: {style['label']} (n={n})", style["color"], show=True)
            fg.add_to(fmap)
            ckids.append({"label": f"{style['label']} (n={n})", "layer": fg})
        point_children.append({"label": f"{country} points", "children": ckids, "collapsed": True})

        # ⑤ 300m junction buffer (true circle, not a buffered polygon) -- the
        # exact JUNCTION_BUFFER_M zone junction_speed_cap.py caps V_safe within.
        # Off by default: ~10,700 circles combined would otherwise dominate the
        # initial view if shown alongside the priority-class layers.
        n_junc = 0 if junc is None else len(junc)
        buf_fg = _junction_buffer_layer(junc, f"{country} (n={n_junc})",
                                         POINT_STYLES["junction"]["color"], JUNCTION_BUFFER_M, show=False)
        buf_fg.add_to(fmap)
        buffer_children.append({"label": f"{country} (n={n_junc})", "layer": buf_fg})
    country_tree.append({"label": "POI/Junctions (V_safe-driving features, points)", "children": point_children, "collapsed": True})
    country_tree.append({
        "label": f"Junction buffer (radius {JUNCTION_BUFFER_M}m circle, same zone as the V_safe cap)",
        "children": buffer_children,
        "collapsed": True,
    })

    # ④ Continuous speed layers: current SpeedLimit vs recommended V_safe, same
    # RdYlGn/30-100km/h scale so toggling one then the other reads as "did the
    # color change on this segment". Off by default (each covers the full
    # network -- including data_quality_flag rows, see SPEED_LAYER_FIELDS --
    # on top of the priority-class layers already shown at load).
    speed_colormap = cm.linear.RdYlGn_11.scale(SPEED_COLOR_VMIN, SPEED_COLOR_VMAX)
    speed_colormap.caption = "Speed (km/h, red=low / green=high, black=no current speed-limit data)"
    speed_children = []
    for label, (field, flag_missing) in SPEED_LAYER_FIELDS.items():
        field_children = []
        for country in ["thailand", "maharashtra"]:
            sub = gdf[gdf["country"] == country]
            fg = _speed_layer_for(sub, f"{country} (n={len(sub)})", field, speed_colormap,
                                   flag_missing, show=False)
            fg.add_to(fmap)
            field_children.append({"label": f"{country} (n={len(sub)})", "layer": fg})
        speed_children.append({"label": f"{label} (n={len(gdf)})", "children": field_children, "collapsed": True})
    country_tree.append({
        "label": "Speed layers (current speed limit vs recommended V_safe, color=km/h)",
        "children": speed_children,
        "collapsed": True,
    })
    speed_colormap.add_to(fmap)

    dq_label = f"Data Quality Issue (Excluded, n={len(invalid)})"
    dq_fg = _layer_for(invalid, dq_label, with_popup=False, show=False)
    dq_fg.add_to(fmap)
    country_tree.append({"label": dq_label, "layer": dq_fg})

    TreeLayerControl(overlay_tree=country_tree).add_to(fmap)

    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: white; padding: 10px 14px; border: 1px solid #999;
                border-radius: 4px; font-size: 13px; line-height: 1.5;">
      <b>Priority class</b><br>
      <span style="color:#d73027;">━━</span> Top Priority&nbsp;&nbsp;
      <span style="color:#fc8d59;">━━</span> Priority&nbsp;&nbsp;
      <span style="color:#fee08b;">━━</span> Watch<br>
      <span style="color:#a6d96a;">━━</span> Low Priority (gap exists, low score)&nbsp;&nbsp;
      <span style="color:#4dd0e1;">━━</span> Aligned (speed limit &le; V_safe)<br>
      <b>V_safe-driving features (points)</b><br>
      <span style="color:#762a83;">●</span> Mapillary VRU&nbsp;&nbsp;
      <span style="color:#1b7837;">●</span> OSM School&nbsp;&nbsp;
      <span style="color:#2166ac;">●</span> Junction<br>
      <b>Review track</b><br>
      Solid = Review Needed (SpeedLimit record is plausible)&nbsp;&nbsp;
      Dashed = Field Verification Needed (SpeedLimit record looks unreliable)<br>
      <b>Speed layers</b> (hidden by default, toggle via the layer tree): Current Speed Limit (SpeedLimit) / Recommended Safe Speed (V_safe).
      Color follows the colorbar (km/h) at bottom right.<br>
      <span style="color:#000000;">━━</span> Current-speed-limit layer only: no current speed-limit data (`data_quality_flag='invalid_speed'`)<br>
      <b>Junction buffer</b> (hidden by default, toggle via the layer tree): 300m-radius circle centered on each junction node.
      <span style="color:#2166ac;">○</span> Same zone used to cap V_safe at 50km/h.
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    return fmap


def write_geo_outputs(gdf: gpd.GeoDataFrame, out_dir: str = "outputs") -> tuple[str, str]:
    export = gdf.copy()
    for col in ["priority_class", "review_track", "data_quality_flag"]:
        export[col] = export[col].astype(object)
        export[col] = export[col].where(export[col].notna(), None)

    geojson_path = f"{out_dir}/segments_priority.geojson"
    gpkg_path = f"{out_dir}/segments_priority.gpkg"
    export.to_file(geojson_path, driver="GeoJSON")
    export.to_file(gpkg_path, driver="GPKG", layer="segments_priority")
    return geojson_path, gpkg_path


def plot_static_summary(gdf: gpd.GeoDataFrame, out_path: str = "outputs/priority_map_static.png") -> str:
    labels = ["Top Priority", "Priority", "Watch", MAP_LOW_PRIORITY, MAP_ALIGNED, "Data Quality Issue (Excluded)"]
    cmap = mcolors.ListedColormap([PRIORITY_COLORS[label] for label in labels])
    gdf = gdf.copy()
    gdf["map_class"] = derive_map_class(gdf).astype(str)
    gdf["map_class"] = gdf["map_class"].where(gdf["map_class"].isin(labels), labels[-1])
    code = gdf["map_class"].map({label: i for i, label in enumerate(labels)})

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, country in zip(axes, ["thailand", "maharashtra"]):
        sub_mask = gdf["country"] == country
        gdf[sub_mask].plot(ax=ax, color=cmap(code[sub_mask] / (len(labels) - 1)), linewidth=0.7)
        ax.set_title(f"{country} map_class (n={sub_mask.sum()})")
        ax.set_aspect("equal")

    handles = [plt.Line2D([0], [0], color=PRIORITY_COLORS[label], lw=3, label=label) for label in labels]
    fig.legend(handles=handles, loc="lower center", ncol=len(labels), fontsize=8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    import geopandas as gpd

    gdf = gpd.read_parquet("data/processed/segments_v_safe.parquet")

    fmap = build_priority_map(gdf)
    html_path = "outputs/priority_map.html"
    fmap.save(html_path)
    print(f"saved {html_path}")

    geojson_path, gpkg_path = write_geo_outputs(gdf)
    print(f"saved {geojson_path}")
    print(f"saved {gpkg_path}")

    png_path = plot_static_summary(gdf)
    print(f"saved {png_path}")

    print("\n=== weight check ===")
    import os
    print(f"HTML size: {os.path.getsize(html_path) / 1e6:.1f} MB")

    print("\n=== geographic sanity check: Top Priority by road_class / land_use ===")
    valid = gdf[gdf["data_quality_flag"].isna()]
    top = valid[valid["priority_class"] == "Top Priority"]
    print(top.groupby(["road_class", "land_use"]).size())
