from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import altair as alt
import streamlit as st

st.set_page_config(page_title="Chicago Crime Dashboards", layout="wide")
alt.data_transformers.disable_max_rows()

# -----------------------------------
# Sidebar Page Selector
# -----------------------------------
page = st.sidebar.radio("Select Page", ["ZIP-Level Map Dashboard", "Monthly Violent Crime Dashboard"])

# -----------------------------------
# Define Base Paths (relative to main folder)
# -----------------------------------
BASE_PATH = Path(__file__).resolve().parent  # main folder
DATA_DIR = BASE_PATH / "data"
RAW_DIR = DATA_DIR / "raw-data"
DERIVED_DIR = DATA_DIR / "derived-data"

CRIME_PATH = DERIVED_DIR / "Crime.csv"
ZIP_PATH = RAW_DIR / "zip_bound.geojson"

# -----------------------------------
# ------------------- Page 1: Sal's ZIP-Level Map Dashboard -------------------
# -----------------------------------
if page == "ZIP-Level Map Dashboard":

    @st.cache_data
    def load_crime_data(path):
        crime = pd.read_csv(path)

        if "Date" in crime.columns:
            crime["Date"] = pd.to_datetime(crime["Date"], errors="coerce")

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
            "zip", "zipcode", "zip_code", "zip5", "zip_5",
            "postalcode", "zip code", "zcta5ce10", "zip5ce10"
        ]

        found = None
        lower_map = {col.lower(): col for col in zips.columns}
        for candidate in zip_candidates:
            if candidate in lower_map:
                found = lower_map[candidate]
                break
        if found is None:
            raise ValueError(f"Could not find a ZIP column in the boundary file. Columns found: {zips.columns.tolist()}")

        zips["zip_join"] = (
            zips[found].astype(str)
            .str.extract(r"(\d+)", expand=False)
            .str.zfill(5)
        )

        if zips.crs is not None and str(zips.crs) != "EPSG:4326":
            zips = zips.to_crs(epsg=4326)

        zips = zips[zips.geometry.notna()].copy()
        zips["geometry"] = zips.geometry.buffer(0)

        return zips

    def metric_definition(metric_name):
        definitions = {
            "incidents": "Raw count of violent-crime incidents in the ZIP code from 2021-2025.",
            "crime_per_1000": "Violent-crime rate per 1,000 residents in the ZIP code.",
            "arrest_rate": "Percent of incidents in the ZIP code that resulted in an arrest.",
            "domestic_rate": "Percent of incidents in the ZIP code flagged as domestic incidents.",
            "poor_mental_health": "Percent of resident adults aged 18 and older who report 14 or more days during the past 30 days during which their mental health was poor.",
            "lack_social_support": "Percent of adults who report rarely or never getting the social and emotional support they need.",
            "child_opportunity_index": "Child Opportunity Index score. Higher values indicate greater neighborhood opportunity for children.",
            "hardship_index": "Hardship Index score. Higher values indicate greater socioeconomic hardship.",
            "behavioral_hosp": "Behavioral health-related hospitalization rate from the source data; higher values indicate more hospitalizations."
        }
        return definitions.get(metric_name, "Definition unavailable for this metric.")

    def build_zip_level_table(crime):
        contextual_metrics = [
            "Poor Mental Health", "Lacking Social/Emotional Support",
            "Childhood Opportunity Index", "Hardship Index",
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

        rename_map = {
            "Zip Code": "zip_code", "Population": "population",
            "Poor Mental Health": "poor_mental_health",
            "Lacking Social/Emotional Support": "lack_social_support",
            "Childhood Opportunity Index": "child_opportunity_index",
            "Hardship Index": "hardship_index",
            "Behavioral Health-related Hospitalizations": "behavioral_hosp"
        }
        zip_stats = zip_stats.rename(columns=rename_map)

        metric_options = [
            "incidents", "crime_per_1000", "arrest_rate", "domestic_rate",
            "poor_mental_health", "lack_social_support", "child_opportunity_index",
            "hardship_index", "behavioral_hosp"
        ]
        metric_options = [m for m in metric_options if m in zip_stats.columns]

        return zip_stats, metric_options

    def make_map_gdf(zips, zip_stats):
        return zips.merge(zip_stats, left_on="zip_join", right_on="zip_code", how="left")

    def metric_label(metric_name):
        labels = {
            "incidents": "Incidents", "crime_per_1000": "Incidents per 1,000",
            "arrest_rate": "Arrest Rate (%)", "domestic_rate": "Domestic Rate (%)",
            "poor_mental_health": "Poor Mental Health",
            "lack_social_support": "Lacking Social/Emotional Support",
            "child_opportunity_index": "Childhood Opportunity Index",
            "hardship_index": "Hardship Index",
            "behavioral_hosp": "Behavioral Health-related Hospitalizations"
        }
        return labels.get(metric_name, metric_name)

    def make_zip_map(map_gdf, selected_metric, slider_value, highlight_enabled=True):
        plot_gdf = map_gdf.copy()
        if highlight_enabled:
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
        else:
    # Highlighting disabled: no ZIP is selected
            plot_gdf["selected_zip"] = False
            selected_zip = None
            selected_metric_value = None

# Set Altair opacity depending on highlight setting
        opacity_value = (
            alt.condition("datum.properties.selected_zip", alt.value(1.0), alt.value(0.25))
            if highlight_enabled else alt.value(1.0)
        )
        plot_gdf["metric_value"] = plot_gdf[selected_metric]

        plot_gdf_json = plot_gdf.copy()
        datetime_cols = plot_gdf_json.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
        plot_gdf_json = plot_gdf_json.drop(columns=datetime_cols)
        geojson = json.loads(plot_gdf_json.to_json())

        base = (
            alt.Chart(alt.Data(values=geojson["features"]))
            .mark_geoshape(stroke="white", strokeWidth=1)
            .encode(
                color=alt.Color("properties.metric_value:Q", title=metric_label(selected_metric),
                                scale=alt.Scale(scheme="blues")),
                opacity=opacity_value,
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
            .properties(width=850, height=700, title=f"Chicago ZIP Codes — {metric_label(selected_metric)}")
            .project(type="mercator")
        )

        highlight = (
            alt.Chart(alt.Data(values=geojson["features"]))
            .transform_filter("datum.properties.selected_zip == true")
            .mark_geoshape(filled=False, stroke="black", strokeWidth=3)
            .properties(width=850, height=700)
            .project(type="mercator")
        )

        return base + highlight, selected_zip, selected_metric_value

    # -----------------------------------
    # Load Data
    # -----------------------------------
    crime = load_crime_data(CRIME_PATH)
    zip_shapes = load_zip_geometries(ZIP_PATH)
    zip_stats, metric_options = build_zip_level_table(crime)
    map_gdf = make_map_gdf(zip_shapes, zip_stats)

    # -----------------------------------
    # UI
    # -----------------------------------
    st.title("Chicago ZIP-Level Crime and Social Conditions")
    st.caption("Choose a ZIP-level metric, then move the slider to highlight the ZIP whose value is closest to that point.")

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

        highlight_enabled = st.checkbox(
            "Enable ZIP highlight for slider",
            value=True,
            help="When enabled, the ZIP closest to the slider value is highlighted and others are faded. "
                "When disabled, all ZIPs use normal heatmap coloring."
)

        st.markdown("### Metric definition")
        st.info(metric_definition(selected_metric))

        st.markdown("### How to read this")
        st.write(
            "All crime data is from 2021-2025. Survey measurements are from 2023. "
            "The fill color shows the selected metric across ZIP codes. "
            "The slider acts as a scrubber across that metric’s range. "
            "The ZIP whose value is closest to the slider is highlighted, while the others fade."
        )

    zip_map, selected_zip, selected_metric_value = make_zip_map(
        map_gdf=map_gdf,
        selected_metric=selected_metric,
        slider_value=slider_value,
        highlight_enabled=highlight_enabled
    )

    with right_col:
        st.altair_chart(zip_map, use_container_width=True)

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
                    "zip_code","incidents","population","crime_per_1000","arrest_rate",
                    "domestic_rate","poor_mental_health","lack_social_support",
                    "child_opportunity_index","hardship_index","behavioral_hosp"
                ] if c in selected_row.columns
            ]
            st.dataframe(selected_row[display_cols], use_container_width=True, hide_index=True)

    st.markdown("### ZIP ranking for selected metric")
    rank_table = zip_stats[["zip_code", selected_metric]].dropna().sort_values(selected_metric, ascending=False).reset_index(drop=True)
    rank_table = rank_table.rename(columns={"zip_code": "ZIP", selected_metric: metric_label(selected_metric)})
    st.dataframe(rank_table, use_container_width=True, hide_index=True)

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

# -----------------------------------
# ------------------- Page 2: Monthly Violent Crime Dashboard -------------------
# -----------------------------------
elif page == "Monthly Violent Crime Dashboard":

    @st.cache_data
    def load_crime_data_monthly():
        df = pd.read_csv(CRIME_PATH, parse_dates=["Date"])
        return df

    crime_data = load_crime_data_monthly()

    st.title("Violent Crime Rates in Chicago (2021-2025)")

    crime_options = ["All Violent Crime", "HOMICIDE", "CRIMINAL SEXUAL ASSAULT", "ASSAULT", "BATTERY"]
    selected_crime = st.selectbox("Select Crime Type", crime_options)

    violent_types = ["HOMICIDE","CRIMINAL SEXUAL ASSAULT","ASSAULT","BATTERY"]
    violent_df = crime_data[crime_data["Primary Type"].isin(violent_types)].copy()

    if selected_crime != "All Violent Crime":
        violent_df = violent_df[violent_df["Primary Type"] == selected_crime]

    violent_df["YearMonth"] = violent_df["Date"].dt.to_period("M")

    zip_totals = (
        violent_df.groupby("Zip Code")
        .agg(total_crimes_5yr=("Primary Type", "count"),
             population=("Population", "first"))
        .reset_index()
    )
    zip_totals["crime_rate_5yr"] = (zip_totals["total_crimes_5yr"] / zip_totals["population"] * 1000)
    baseline_df = zip_totals[["Zip Code", "crime_rate_5yr"]]

    if "slider_count" not in st.session_state:
        st.session_state.slider_count = 1
    max_sliders = 5

    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.slider_count < max_sliders:
            if st.button("➕ Add Percentile"):
                st.session_state.slider_count += 1
                st.rerun()
    with col2:
        if st.session_state.slider_count > 1:
            if st.button("➖ Remove Last Percentile"):
                st.session_state.slider_count -= 1
                st.rerun()

    percentiles = []
    previous_value = 0
    st.write("### Percentile Thresholds")
    for i in range(st.session_state.slider_count):
        value = st.slider(
            f"Percentile #{i+1}",
            min_value=previous_value,
            max_value=100,
            value=st.session_state.get(f"slider_{i}", min(previous_value+10,100)),
            key=f"slider_{i}"
        )
        percentiles.append(value)
        previous_value = value

    percentile_bounds = sorted(percentiles) + [100]

    zip_groups = []
    lower_rate = -float("inf")
    for p in percentile_bounds:
        threshold = baseline_df["crime_rate_5yr"].quantile(p/100)
        zips = baseline_df[(baseline_df["crime_rate_5yr"] > lower_rate) & (baseline_df["crime_rate_5yr"] <= threshold)]["Zip Code"].tolist()
        zip_groups.append(zips)
        lower_rate = threshold

    def compute_group_monthly_rate(df, zip_list):
        group_df = df[df["Zip Code"].isin(zip_list)]
        if group_df.empty:
            return pd.DataFrame()
        monthly_population = (
            group_df.groupby(["YearMonth", "Zip Code"])
            .agg(zip_population=("Population","first"), monthly_crimes=("Primary Type","count"))
            .reset_index()
        )
        monthly_summary = monthly_population.groupby("YearMonth").agg(
            total_crimes=("monthly_crimes","sum"),
            total_population=("zip_population","sum")
        ).reset_index()
        monthly_summary["crime_rate_per_1000"] = monthly_summary["total_crimes"]/monthly_summary["total_population"]*1000
        monthly_summary["YearMonth"] = monthly_summary["YearMonth"].dt.to_timestamp()
        return monthly_summary

    plot_dfs = []
    for idx, zips in enumerate(zip_groups):
        df_grp = compute_group_monthly_rate(violent_df, zips)
        if df_grp.empty:
            continue
        df_grp["Group"] = f"P{idx+1}: {percentile_bounds[idx]}%"
        plot_dfs.append(df_grp)

    if plot_dfs:
        plot_df = pd.concat(plot_dfs)
        color_palette = ["#0b3d91","#1e5ebf","#5b8fd1","#a4c3e0","#d0e1f2","#e0efff"]
        num_groups = len(plot_dfs)

        chart = alt.Chart(plot_df).mark_line(point=True).encode(
            x=alt.X("YearMonth:T", axis=alt.Axis(title="Month", format="%b %Y", tickCount="month")),
            y=alt.Y("crime_rate_per_1000:Q", title="Crime Rate Per 1,000 Residents"),
            color=alt.Color("Group:N",
                            scale=alt.Scale(domain=plot_df["Group"].unique(), range=color_palette[:num_groups]),
                            legend=alt.Legend(title="Percentile Group")),
            tooltip=["YearMonth:T","crime_rate_per_1000:Q","total_crimes:Q","Group:N"]
        ).properties(
            title=f"Monthly Crime Rate by Zip Crime Percentile ({selected_crime})",
            width=700, height=400
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No data available for selected groups.")

    st.markdown("""
    The chart above graphs the monthly violent crime rate from 2021–2025. 
    Use the adjustment tools to subset by type of crime or to split the data into percentile groups by crime rate.

    For example, setting the first slider at **20%** and a second slider at **60%** would create three lines:
    - the monthly crime rate for the **20% of zip codes with the lowest crime rate**
    - the monthly crime rate for zip codes in the **20%–60% range**
    - the monthly crime rate for the remaining 40% of zip codes with the highest crime rates
    """)