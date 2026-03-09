import streamlit as st
import pandas as pd
import geopandas as gpd
import altair as alt

st.set_page_config(
    page_title="Chicago Crime and Health Dashboard",
    layout="wide"
)

# -------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------

page = st.sidebar.radio(
    "Navigate",
    ["Crime Trends", "Chicago Indicator Maps"]
)

# =================================================
# PAGE 1 — CRIME TRENDS
# =================================================

if page == "Crime Trends":

    st.header("Interactive Crime Trends in Chicago (2021–2025)")

    @st.cache_data
    def load_crime_data():

        Crime = pd.read_csv("data/raw-data/InitialCrimes.csv")

        Crime = Crime.dropna(subset=["Latitude", "Longitude"])

        Crime_gdf = gpd.GeoDataFrame(
            Crime,
            geometry=gpd.points_from_xy(Crime['Longitude'], Crime['Latitude']),
            crs="EPSG:4326"
        )

        zip_codes = gpd.read_file("data/raw-data/ZC/ZC.shp")
        zip_codes = zip_codes.to_crs(Crime_gdf.crs)

        Crime_with_zip = gpd.sjoin(
            Crime_gdf,
            zip_codes,
            how="left",
            predicate="within"
        )

        Crime["Zip Code"] = Crime_with_zip["zip"]
        Crime = Crime.dropna(subset=["Zip Code"])

        Crime["Date"] = pd.to_datetime(Crime["Date"])

        Population = pd.read_csv("data/raw-data/PopulationZC.csv")

        Crime["Zip Code"] = Crime["Zip Code"].astype(str)
        Population["GEOID"] = Population["GEOID"].astype(str)

        Crime = Crime.merge(
            Population[["GEOID", "POP_2019-2023"]],
            left_on="Zip Code",
            right_on="GEOID",
            how="left"
        )

        Crime = Crime.drop(columns=["GEOID"])
        Crime = Crime.rename(columns={"POP_2019-2023": "Population"})

        Crime["Population"] = (
            Crime["Population"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .astype(float)
        )

        Crime["Primary Type"] = Crime["Primary Type"].replace({
            "CRIM SEXUAL ASSAULT": "CRIMINAL SEXUAL ASSAULT"
        })

        return Crime


    Crime = load_crime_data()

    metric = st.selectbox(
        "Metric",
        ["Total Crime", "Crime Rate"]
    )

    crime_type = st.selectbox(
        "Crime Type",
        ["All CRIME", "HOMICIDE", "CRIMINAL SEXUAL ASSAULT", "ASSAULT", "BATTERY"]
    )

    split_quintiles = st.checkbox("Split Zip Codes into Quintiles")

    if crime_type != "All CRIME":
        subset = Crime[Crime["Primary Type"] == crime_type].copy()
    else:
        subset = Crime.copy()

    if not split_quintiles:

        monthly = (
            subset
            .set_index("Date")
            .resample("M")
            .size()
            .reset_index(name="CrimeCount")
        )

        if metric == "Total Crime":

            chart = alt.Chart(monthly).mark_line(color="#08306B").encode(
                x=alt.X("Date:T", title="Month"),
                y=alt.Y("CrimeCount:Q", title="Number of Crimes")
            )

        else:

            total_population = (
                Crime[["Zip Code", "Population"]]
                .drop_duplicates()["Population"]
                .sum()
            )

            monthly["CrimeRate"] = monthly["CrimeCount"] / total_population * 1000

            chart = alt.Chart(monthly).mark_line(color="#08306B").encode(
                x=alt.X("Date:T", title="Month"),
                y=alt.Y("CrimeRate:Q", title="Crime Rate (per 1,000 Residents)")
            )

    else:

        zip_counts = (
            subset.groupby("Zip Code")
            .size()
            .reset_index(name="TotalCrimes")
        )

        zip_pop = Crime[["Zip Code", "Population"]].drop_duplicates()

        zip_stats = zip_pop.merge(zip_counts, on="Zip Code", how="left")

        zip_stats["TotalCrimes"] = zip_stats["TotalCrimes"].fillna(0)

        if metric == "Total Crime":
            zip_stats["Metric"] = zip_stats["TotalCrimes"]
        else:
            zip_stats["Metric"] = zip_stats["TotalCrimes"] / zip_stats["Population"] * 1000

        zip_stats["Quintile"] = pd.qcut(
            zip_stats["Metric"],
            5,
            labels=["Q1 Lowest", "Q2", "Q3", "Q4", "Q5 Highest"]
        )

        subset = subset.merge(
            zip_stats[["Zip Code", "Quintile"]],
            on="Zip Code",
            how="left"
        )

        monthly = (
            subset
            .set_index("Date")
            .groupby("Quintile")
            .resample("M")
            .size()
            .reset_index(name="CrimeCount")
        )

        chart = alt.Chart(monthly).mark_line().encode(
            x=alt.X("Date:T", title="Month"),
            y=alt.Y("CrimeCount:Q", title="Number of Crimes"),
            color=alt.Color(
                "Quintile:N",
                scale=alt.Scale(range=[
                    "#08306B",
                    "#2171B5",
                    "#6BAED6",
                    "#BDD7E7",
                    "#C6DBEF"
                ]),
                title="Zip Code Quintile"
            )
        )

    st.altair_chart(chart, use_container_width=True)


# =================================================
# PAGE 2 — CHICAGO INDICATOR HEATMAPS
# =================================================

if page == "Chicago Indicator Maps":
    import streamlit as st
    import pandas as pd
    import geopandas as gpd
    import altair as alt
    import os

    st.set_page_config(page_title="Health & Crime Heatmaps", layout="wide")
    st.header("Chicago Zip Code Heatmaps")

    # -----------------------------
    # Load derived zip_data
    # -----------------------------
    zip_data_path = os.path.join("data", "derived-data", "zip_data.csv")
    zip_data = pd.read_csv(zip_data_path)

    # -----------------------------
    # Dropdown for variable selection
    # -----------------------------
    variable_map = {
        "Crime Rate": "CrimeRate",
        "% Reporting Poor Mental Health": "Poor Mental Health",
        "% Reporting a Lack of Social/Emotional Support": "Lacking Social/Emotional Support",
        "Behavioral Health-Related Hospitalizations": "Behavioral Health-related Hospitalizations",
        "Childhood Opportunity Index": "Childhood Opportunity Index",
        "Social Vulnerability Index": "Social Vulnerability Index",
        "Hardship Index": "Hardship Index"
    }

    selected_label = st.selectbox("Select a variable to plot", list(variable_map.keys()))
    selected_var = variable_map[selected_label]

    # -----------------------------
    # Ensure numeric and drop NA
    # -----------------------------
    zip_data[selected_var] = pd.to_numeric(zip_data[selected_var], errors="coerce")
    zip_data = zip_data.dropna(subset=[selected_var])

    # -----------------------------
    # Load Chicago ZIP shapefile
    # -----------------------------
    zip_shapefile_path = os.path.join("data", "raw-data", "ZC", "ZC.shp")
    zip_shapes = gpd.read_file(zip_shapefile_path)
    zip_shapes["zip"] = zip_shapes["zip"].astype(str).str.strip()
    zip_data["Zip Code"] = zip_data["Zip Code"].astype(str).str.strip()

    merged = zip_shapes.merge(zip_data, left_on="zip", right_on="Zip Code", how="left")

    # -----------------------------
    # Cap outliers for color scale
    # -----------------------------
    # Avoid extreme outliers flattening the color gradient
    max_val = merged[selected_var].quantile(0.95)
    min_val = merged[selected_var].min()

    # -----------------------------
    # Flatten column names for Altair
    # -----------------------------
    merged_alt = merged.copy()
    merged_alt.columns = [c.replace(" ", "_") for c in merged_alt.columns]
    selected_var_alt = selected_var.replace(" ", "_")

    # -----------------------------
    # Create Altair heatmap
    # -----------------------------
    chart = alt.Chart(alt.Data(values=merged_alt.__geo_interface__["features"])).mark_geoshape(
        stroke='black',
        strokeWidth=1
    ).encode(
        color=alt.Color(
            f"properties.{selected_var_alt}:Q",
            scale=alt.Scale(scheme="blues", domain=[min_val, max_val]),
            title=selected_label
        ),
        tooltip=[
            alt.Tooltip("properties.zip:N", title="ZIP Code"),
            alt.Tooltip(f"properties.{selected_var_alt}:Q", title=selected_label)
        ]
    ).properties(
        width=700,
        height=700,
        title=f"{selected_label} across Chicago by ZIP Code"
    ).project("mercator")

    st.altair_chart(chart, use_container_width=True)

    # Clip CrimeRate to avoid outlier blowout
    if selected_var == "CrimeRate":
        merged[selected_var] = merged[selected_var].clip(upper=150)  # clip top 3 outliers

    # Convert to GeoJSON-like dictionary for Altair
    map_json = merged.__geo_interface__

    # Altair geoshape heatmap
    chart = alt.Chart(alt.Data(values=map_json["features"])).mark_geoshape(
        stroke='black',
        strokeWidth=1
    ).encode(
        color=alt.Color(
            f"properties.{selected_var}:Q",
            scale=alt.Scale(scheme="blues", type="quantile"),
            title=selected_label
        ),
        tooltip=[
            alt.Tooltip("properties.zip:N", title="ZIP Code"),
            alt.Tooltip(f"properties.{selected_var}:Q", title=selected_label)
        ]
    ).properties(
        width=700,
        height=700,
        title=f"{selected_label} Across Chicago by Zip Code"
    ).project("mercator")

    st.altair_chart(chart, use_container_width=True)
