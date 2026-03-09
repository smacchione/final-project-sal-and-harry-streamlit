from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import altair as alt
import streamlit as st

# -----------------------------------
# settings
# -----------------------------------
st.set_page_config(page_title="Chicago ZIP Crime Dashboard", layout="wide")
alt.data_transformers.disable_max_rows()

# -----------------------------------
# base path + relative file paths
# -----------------------------------
BASE_PATH = Path(__file__).resolve().parent

DATA_DIR = BASE_PATH / "data"
RAW_DIR = DATA_DIR / "raw-data"
DERIVED_DIR = DATA_DIR / "derived-data"

CRIME_PATH = DERIVED_DIR / "Crime.csv"
ZIP_PATH = RAW_DIR / "zip_bound.geojson"

# -----------------------------------
# helper functions
# -----------------------------------
@st.cache_data
def load_crime_data(path):
    crime = pd.read_csv(path)

    # date cleanup
    if "Date" in crime.columns:
        crime["Date"] = pd.to_datetime(crime["Date"], errors="coerce")

    # zip cleanup
    if "Zip Code" not in crime.columns:
        raise ValueError("Crime file must contain a 'Zip Code' column.")

    crime["Zip Code"] = (
        crime["Zip Code"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .str.zfill(5)
    )

    crime = crime[crime["Zip Code"].notna()]
    crime = crime[crime["Zip Code"].str.len() == 5]

    # boolean cleanup
    if "Arrest" in crime.columns:
        crime["Arrest"] = (
            crime["Arrest"]
            .astype(str)
            .str.lower()
            .map({"true": 1, "false": 0, "1": 1, "0": 0})
        )

    if "Domestic" in crime.columns:
        crime["Domestic"] = (
            crime["Domestic"]
            .astype(str)
            .str.lower()
            .map({"true": 1, "false": 0, "1": 1, "0": 0})
        )

    return crime


@st.cache_data
def load_zip_geometries(path):
    zips = gpd.read_file(path)

    zip_candidates = [
        "zip",
        "zipcode",
        "zip_code",
        "zip5",
        "zip_5",
        "postalcode",
        "zip code",
        "zcta5ce10",
        "zip5ce10"
    ]

    found = None
    lower_map = {col.lower(): col for col in zips.columns}

    for candidate in zip_candidates:
        if candidate in lower_map:
            found = lower_map[candidate]
            break

    if found is None:
        raise ValueError(
            f"Could not find a ZIP column in the boundary file. Columns found: {zips.columns.tolist()}"
        )

    zips["zip_join"] = (
        zips[found]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .str.zfill(5)
    )

    # ensure geographic CRS for Altair/Vega-Lite
    if zips.crs is not None and str(zips.crs) != "EPSG:4326":
        zips = zips.to_crs(epsg=4326)

    # keep only valid geometries
    zips = zips[zips.geometry.notna()].copy()
    zips["geometry"] = zips.geometry.buffer(0)

    return zips


def build_zip_level_table(crime):
    contextual_metrics = [
        "Poor Mental Health",
        "Lacking Social/Emotional Support",
        "Childhood Opportunity Index",
        "Hardship Index",
        "Behavioral Health-related Hospitalizations"
    ]
    contextual_metrics = [c for c in contextual_metrics if c in crime.columns]

    agg_dict = {"ID": "count"}

    if "Population" in crime.columns:
        agg_dict["Population"] = "max"
    if "Arrest" in crime.columns:
        agg_dict["Arrest"] = "mean"
    if "Domestic" in crime.columns:
        agg_dict["Domestic"] = "mean"

    for c in contextual_metrics:
        agg_dict[c] = "mean"

    zip_stats = (
        crime.dropna(subset=["Zip Code"])
        .groupby("Zip Code", as_index=False)
        .agg(agg_dict)
        .rename(columns={"ID": "incidents"})
    )

    if "Population" in zip_stats.columns:
        zip_stats["crime_per_1000"] = (zip_stats["incidents"] / zip_stats["Population"]) * 1000
    else:
        zip_stats["crime_per_1000"] = np.nan

    if "Arrest" in zip_stats.columns:
        zip_stats["arrest_rate"] = zip_stats["Arrest"] * 100
        zip_stats = zip_stats.drop(columns=["Arrest"])

    if "Domestic" in zip_stats.columns:
        zip_stats["domestic_rate"] = zip_stats["Domestic"] * 100
        zip_stats = zip_stats.drop(columns=["Domestic"])

    # rename long columns for Altair friendliness
    rename_map = {
        "Zip Code": "zip_code",
        "Population": "population",
        "Poor Mental Health": "poor_mental_health",
        "Lacking Social/Emotional Support": "lack_social_support",
        "Childhood Opportunity Index": "child_opportunity_index",
        "Hardship Index": "hardship_index",
        "Behavioral Health-related Hospitalizations": "behavioral_hosp"
    }
    zip_stats = zip_stats.rename(columns=rename_map)

    metric_options = [
        "incidents",
        "crime_per_1000",
        "arrest_rate",
        "domestic_rate",
        "poor_mental_health",
        "lack_social_support",
        "child_opportunity_index",
        "hardship_index",
        "behavioral_hosp"
    ]
    metric_options = [m for m in metric_options if m in zip_stats.columns]

    return zip_stats, metric_options


def make_map_gdf(zips, zip_stats):
    map_gdf = zips.merge(zip_stats, left_on="zip_join", right_on="zip_code", how="left")
    return map_gdf


def metric_label(metric_name):
    labels = {
        "incidents": "Incidents",
        "crime_per_1000": "Incidents per 1,000",
        "arrest_rate": "Arrest Rate (%)",
        "domestic_rate": "Domestic Rate (%)",
        "poor_mental_health": "Poor Mental Health",
        "lack_social_support": "Lacking Social/Emotional Support",
        "child_opportunity_index": "Childhood Opportunity Index",
        "hardship_index": "Hardship Index",
        "behavioral_hosp": "Behavioral Health-related Hospitalizations"
    }
    return labels.get(metric_name, metric_name)


def make_zip_map(map_gdf, selected_metric, slider_value):
    plot_gdf = map_gdf.copy()

    valid = plot_gdf[selected_metric].notna()

    if valid.sum() == 0:
        plot_gdf["selected_zip"] = False
        selected_zip = None
        selected_metric_value = None
    else:
        diffs = (plot_gdf.loc[valid, selected_metric] - slider_value).abs()
        selected_idx = diffs.idxmin()
        selected_zip = plot_gdf.loc[selected_idx, "zip_join"]
        selected_metric_value = plot_gdf.loc[selected_idx, selected_metric]
        plot_gdf["selected_zip"] = plot_gdf["zip_join"] == selected_zip

    plot_gdf["metric_value"] = plot_gdf[selected_metric]

# make a JSON-safe copy for Altair
    plot_gdf_json = plot_gdf.copy()

# drop datetime columns that break GeoJSON serialization
    datetime_cols = plot_gdf_json.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    plot_gdf_json = plot_gdf_json.drop(columns=datetime_cols)

    geojson = json.loads(plot_gdf_json.to_json())

    base = (
        alt.Chart(alt.Data(values=geojson["features"]))
        .mark_geoshape(stroke="white", strokeWidth=1)
        .encode(
            color=alt.Color(
                "properties.metric_value:Q",
                title=metric_label(selected_metric),
                scale=alt.Scale(scheme="blues")
            ),
            opacity=alt.condition(
                "datum.properties.selected_zip",
                alt.value(1.0),
                alt.value(0.25)
            ),
            tooltip=[
                alt.Tooltip("properties.zip_join:N", title="ZIP"),
                alt.Tooltip("properties.incidents:Q", title="Incidents", format=",.0f"),
                alt.Tooltip("properties.crime_per_1000:Q", title="Incidents per 1,000", format=".2f"),
                alt.Tooltip("properties.arrest_rate:Q", title="Arrest Rate (%)", format=".1f"),
                alt.Tooltip("properties.domestic_rate:Q", title="Domestic Rate (%)", format=".1f"),
                alt.Tooltip("properties.population:Q", title="Population", format=",.0f"),
                alt.Tooltip("properties.poor_mental_health:Q", title="Poor Mental Health", format=".1f"),
                alt.Tooltip("properties.lack_social_support:Q", title="Lacking Social/Emotional Support", format=".1f"),
                alt.Tooltip("properties.child_opportunity_index:Q", title="Childhood Opportunity Index", format=".2f"),
                alt.Tooltip("properties.hardship_index:Q", title="Hardship Index", format=".1f"),
                alt.Tooltip("properties.behavioral_hosp:Q", title="Behavioral Health Hospitalizations", format=".1f")
            ]
        )
        .properties(
            width=850,
            height=700,
            title=f"Chicago ZIP Codes — {metric_label(selected_metric)}"
        )
        .project(type="mercator")
    )

    highlight = (
        alt.Chart(alt.Data(values=geojson["features"]))
        .transform_filter("datum.properties.selected_zip == true")
        .mark_geoshape(
            filled=False,
            stroke="black",
            strokeWidth=3
        )
        .properties(width=850, height=700)
        .project(type="mercator")
    )

    return base + highlight, selected_zip, selected_metric_value


# -----------------------------------
# load data
# -----------------------------------
crime = load_crime_data(CRIME_PATH)
zip_shapes = load_zip_geometries(ZIP_PATH)
zip_stats, metric_options = build_zip_level_table(crime)
map_gdf = make_map_gdf(zip_shapes, zip_stats)

# -----------------------------------
# UI
# -----------------------------------
st.title("Chicago ZIP-Level Crime and Social Conditions")
st.caption(
    "Choose a ZIP-level metric, then move the slider to highlight the ZIP whose value is closest to that point."
)

if len(metric_options) == 0:
    st.error("No usable metric columns were found in the ZIP-level data.")
    st.stop()

default_metric = "crime_per_1000" if "crime_per_1000" in metric_options else metric_options[0]

left_col, right_col = st.columns([1, 2])

with left_col:
    selected_metric = st.selectbox(
        "Choose a ZIP-level metric",
        metric_options,
        index=metric_options.index(default_metric),
        format_func=metric_label
    )

    metric_series = map_gdf[selected_metric].dropna()

    if metric_series.empty:
        st.error(f"No non-missing values found for {metric_label(selected_metric)}.")
        st.stop()

    slider_min = float(metric_series.min())
    slider_max = float(metric_series.max())
    slider_default = float(metric_series.median())

    slider_value = st.slider(
        f"Highlight the ZIP closest to this {metric_label(selected_metric)} value",
        min_value=slider_min,
        max_value=slider_max,
        value=slider_default
    )

    st.markdown("### How to read this")
    st.write(
        "The fill color shows the selected metric across ZIP codes. "
        "The slider acts as a scrubber across that metric’s range. "
        "The ZIP whose value is closest to the slider is highlighted, while the others fade."
    )

# build and show map
zip_map, selected_zip, selected_metric_value = make_zip_map(
    map_gdf=map_gdf,
    selected_metric=selected_metric,
    slider_value=slider_value
)

with right_col:
    st.altair_chart(zip_map, use_container_width=True)

# selected ZIP summary
st.markdown("### Selected ZIP snapshot")

if selected_zip is not None:
    selected_row = zip_stats.loc[zip_stats["zip_code"] == selected_zip].copy()

    if not selected_row.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Selected ZIP", selected_zip)
        c2.metric("Metric", metric_label(selected_metric))
        c3.metric("Value", f"{selected_metric_value:,.2f}")

        display_cols = [
            c for c in [
                "zip_code",
                "incidents",
                "population",
                "crime_per_1000",
                "arrest_rate",
                "domestic_rate",
                "poor_mental_health",
                "lack_social_support",
                "child_opportunity_index",
                "hardship_index",
                "behavioral_hosp"
            ]
            if c in selected_row.columns
        ]

        st.dataframe(selected_row[display_cols], use_container_width=True, hide_index=True)

# ranking table
st.markdown("### ZIP ranking for selected metric")
rank_table = (
    zip_stats[["zip_code", selected_metric]]
    .dropna()
    .sort_values(selected_metric, ascending=False)
    .reset_index(drop=True)
)
rank_table = rank_table.rename(columns={
    "zip_code": "ZIP",
    selected_metric: metric_label(selected_metric)
})
st.dataframe(rank_table, use_container_width=True, hide_index=True)

# optional debugging block
with st.expander("Debug info"):
    st.write("crime shape:", crime.shape)
    st.write("zip_shapes shape:", zip_shapes.shape)
    st.write("zip_stats shape:", zip_stats.shape)
    st.write("map_gdf shape:", map_gdf.shape)
    st.write("zip_shapes columns:", zip_shapes.columns.tolist())
    st.write("map_gdf columns:", map_gdf.columns.tolist())
    st.write("matched ZIP rows:", map_gdf["zip_code"].notna().sum())
    st.write("unmatched ZIP rows:", map_gdf["zip_code"].isna().sum())
    st.write("geometry null count:", map_gdf.geometry.isna().sum())
    st.write("geometry types:", map_gdf.geom_type.value_counts())
    st.write(map_gdf.drop(columns="geometry").head())