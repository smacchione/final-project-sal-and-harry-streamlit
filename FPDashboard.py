import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import altair as alt
import json

alt.data_transformers.disable_max_rows()

# ========================
# Load Data
# ========================
@st.cache_data
def load_data():
    # Load raw crime data
    Crime = pd.read_csv("data/Crimes.csv")
    Crime["Date"] = pd.to_datetime(Crime["Date"])

    # Convert "SEXUAL ASSAULT" variants
    Crime["Primary Type"] = Crime["Primary Type"].replace(
        ["CRIM SEXUAL ASSAULT", "CRIMINAL SEXUAL ASSAULT"],
        "SEXUAL ASSAULT"
    )

    # Create GeoDataFrame for spatial join
    geometry = [Point(xy) for xy in zip(Crime["Longitude"], Crime["Latitude"])]
    crimes_gdf = gpd.GeoDataFrame(Crime, geometry=geometry, crs="EPSG:4326")

    # Load census tracts shapefile
    tracts = gpd.read_file("data/Census_20Tracts/Census_Tracts.shp").to_crs(crimes_gdf.crs)
    tracts["CENSUS_T_1"] = tracts["CENSUS_T_1"].astype(str)

    # Spatial join: assign each crime to a tract
    crimes_with_tracts = gpd.sjoin(
        crimes_gdf,
        tracts[["CENSUS_T_1", "geometry"]],
        how="left",
        predicate="within"
    )

    Crime = pd.DataFrame(crimes_with_tracts)  # convert back to DataFrame
    Crime["CENSUS_T_1"] = Crime["CENSUS_T_1"].astype(str)

    # Load Population
    Pop2023 = pd.read_csv("data/Population.csv")
    Pop2023["GEOID"] = Pop2023["GEOID"].astype(str)
    Crime = Crime.merge(
        Pop2023[["GEOID","Population"]],
        left_on="CENSUS_T_1",
        right_on="GEOID",
        how="left"
    ).drop(columns=[col for col in Crime.columns if "GEOID" in col], errors="ignore")

    Crime["Population"] = pd.to_numeric(Crime["Population"].astype(str).str.replace(",", "").str.strip(), errors="coerce")

    # Load PMH, SES, HDX, COI
    PMH2023 = pd.read_csv("data/PMH2023.csv")
    SES2023 = pd.read_csv("data/SES2023.csv")
    HI2023  = pd.read_csv("data/HI2023.csv")
    CHI2023 = pd.read_csv("data/CHI2023.csv")

    for df, new_col in zip([PMH2023, SES2023, HI2023, CHI2023],
                           ["PMH", "SES", "HDX", "COI"]):
        df["GEOID"] = df["GEOID"].astype(str)
        col_name = df.columns[-1]  # last column has the metric
        Crime = Crime.merge(
            df[["GEOID", col_name]].rename(columns={col_name: new_col}),
            left_on="CENSUS_T_1",
            right_on="GEOID",
            how="left"
        ).drop(columns=["GEOID"], errors="ignore")
        Crime[new_col] = pd.to_numeric(Crime[new_col], errors="coerce").fillna(0)

    # Subsets
    Battery = Crime[Crime['Primary Type'] == 'BATTERY']
    Assault = Crime[Crime['Primary Type'] == 'ASSAULT']
    SA = Crime[Crime['Primary Type'] == 'SEXUAL ASSAULT']
    Homicide = Crime[Crime['Primary Type'] == 'HOMICIDE']

    return Crime, Battery, Assault, SA, Homicide, tracts

# ========================
# Helper Functions
# ========================

def plot_time_series(df, title=None):
    monthly = df.groupby(pd.Grouper(key="Date", freq="M")).size().reset_index(name="count")
    chart = (
        alt.Chart(monthly)
        .mark_line(point=True)
        .encode(
            x=alt.X("Date:T", title="Month"),
            y=alt.Y("count:Q", title="Number of Crimes"),
            tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("count:Q")]
        )
        .properties(title=title, width=700, height=400)
    )
    return chart

def plot_quintile_crime_rates(df, rate_per=1000, title=None):
    # Compute tract totals
    tract_counts = df.groupby("CENSUS_T_1").size().reset_index(name="total_crimes")
    tract_counts["quintile"] = pd.qcut(tract_counts["total_crimes"], 5,
                                       labels=["Q1 (Lowest)","Q2","Q3","Q4","Q5 (Highest)"])
    df = df.merge(tract_counts[["CENSUS_T_1","quintile"]], on="CENSUS_T_1", how="left")
    df["month"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    quintile_time = df.groupby(["month","quintile"]).size().reset_index(name="count")
    chart = (
        alt.Chart(quintile_time)
        .mark_line()
        .encode(
            x=alt.X("month:T", title="Month"),
            y=alt.Y("count:Q", title="Number of Crimes"),
            color=alt.Color("quintile:N", title="Census Tract Quintile")
        )
        .properties(width=700, height=400, title=title)
    )
    return chart

def plot_chicago_heatmap(gdf, var, color_scheme='reds', title=None):
    import json
    gdf_json = json.loads(gdf.to_json())
    if title is None:
        title = f'Chicago {var} by Census Tract (2023)'
    chart = alt.Chart(alt.Data(values=gdf_json['features'])).mark_geoshape().encode(
        color=alt.Color(f'properties.{var}:Q', scale=alt.Scale(scheme=color_scheme), title=var.replace("_"," ")),
        tooltip=[
            alt.Tooltip('properties.CENSUS_T_1:N', title='Census Tract'),
            alt.Tooltip(f'properties.{var}:Q', title=var.replace("_"," "))
        ]
    ).properties(width=700, height=700, title=title).project('mercator')
    return chart

# ========================
# Streamlit App
# ========================

st.title("Chicago Crime Dashboard (2021–2025)")

Crime, Battery, Assault, SA, Homicide, tracts = load_data()

# --- Section 1: Crime Counts Over Time ---
st.header("Crime Counts Over Time")
crime_options = {
    "All Violent Crimes": Crime,
    "Battery": Battery,
    "Assault": Assault,
    "Sexual Assault": SA,
    "Homicide": Homicide
}
selected_crime = st.selectbox("Select Crime Type", list(crime_options.keys()))
st.altair_chart(plot_time_series(crime_options[selected_crime], title=f"{selected_crime} Over Time"), use_container_width=True)

# --- Section 2: Quintile Plots ---
st.header("Census Tract Quintile Trends")
quintile_options = {
    "Battery": Battery,
    "Assault": Assault,
    "Sexual Assault": SA,
    "Homicide": Homicide
}
selected_quintile = st.selectbox("Select Crime Type for Quintiles", list(quintile_options.keys()))
st.altair_chart(plot_quintile_crime_rates(quintile_options[selected_quintile],
                                         title=f"{selected_quintile} Monthly Counts by Quintile"),
                use_container_width=True)

# --- Section 3: Chicago Heatmaps ---
st.header("Chicago Census-Tract Heatmaps (2023)")

Crime2023 = Crime[Crime["Date"].dt.year == 2023].copy()

# Load tract-level variables
PMH2023 = pd.read_csv("data/PMH2023.csv")
SES2023 = pd.read_csv("data/SES2023.csv")
HI2023 = pd.read_csv("data/HI2023.csv")
CHI2023 = pd.read_csv("data/CHI2023.csv")

# Merge all tract variables
for df, col in zip([PMH2023, SES2023, HI2023, CHI2023], ["PMH", "SES", "HDX", "COI"]):
    df["GEOID"] = df["GEOID"].astype(str)
    Crime2023["CENSUS_T_1"] = Crime2023["CENSUS_T_1"].astype(str)
    if col in ["PMH", "SES"]:
        Crime2023 = Crime2023.merge(df[["GEOID", f"{col}_2023"]], left_on="CENSUS_T_1", right_on="GEOID", how="left")
        Crime2023.rename(columns={f"{col}_2023": col}, inplace=True)
    else:
        Crime2023 = Crime2023.merge(df[["GEOID", df.columns[1]]], left_on="CENSUS_T_1", right_on="GEOID", how="left")
        Crime2023.rename(columns={df.columns[1]: col}, inplace=True)

Crime2023 = Crime2023.drop(columns=[c for c in Crime2023.columns if "GEOID" in c], errors="ignore")

# Clean Population
Crime2023["Population"] = (
    Crime2023["Population"].astype(str).str.replace(",", "", regex=False).str.strip()
)
Crime2023["Population"] = pd.to_numeric(Crime2023["Population"], errors="coerce")
Crime2023 = Crime2023.dropna(subset=["Population"])

# Aggregate tract-level variables
tract_data = (
    Crime2023.groupby("CENSUS_T_1")
    .agg(
        total_crimes=("CENSUS_T_1", "size"),
        Population=("Population", "first"),
        PMH=("PMH", "first"),
        SES=("SES", "first"),
        HDX=("HDX", "first"),
        COI=("COI", "first"),
    )
    .reset_index()
)

# Compute crime rate
tract_data["crime_rate"] = (tract_data["total_crimes"] / tract_data["Population"]) * 1000

# Load shapefile
gdf = gpd.read_file("data/Census_20Tracts/Census_Tracts.shp")
gdf = gdf.to_crs(epsg=4326)
gdf["CENSUS_T_1"] = gdf["CENSUS_T_1"].astype(str)

# Merge shapefile with tract data
gdf = gdf.merge(tract_data, on="CENSUS_T_1", how="left")
for col in ['PMH','SES','HDX','COI','crime_rate']:
    gdf[col] = gdf[col].fillna(0)

# Convert to JSON for Altair
gdf_json = json.loads(gdf.to_json())

# Dropdown mapping
dropdown_options = {
    "Self-Reported Poor Mental Health (%)": "PMH",
    "Self-Reported Lacking Social/Emotional Support": "SES",
    "Child Opportunity Index": "COI",
    "Hardship Index": "HDX",
    "Crime Rate per 1,000 Residents": "crime_rate"
}

selected_var = st.selectbox(
    "Select variable to plot",
    options=list(dropdown_options.keys())
)

# Map colors for each variable
color_schemes = {
    "PMH": "blues",
    "SES": "purples",
    "COI": "greens",
    "HDX": "oranges",
    "crime_rate": "reds"
}

# Function to plot
def plot_chicago_heatmap(var, title=None):
    if title is None:
        title = f"Chicago {var} by Census Tract (2023)"
    chart = alt.Chart(alt.Data(values=gdf_json['features'])).mark_geoshape().encode(
        color=alt.Color(
            f'properties.{var}:Q',
            scale=alt.Scale(scheme=color_schemes.get(var, "reds")),
            title=var.replace('_',' ')
        ),
        tooltip=[
            alt.Tooltip('properties.CENSUS_T_1:N', title='Census Tract'),
            alt.Tooltip(f'properties.{var}:Q', title=var.replace('_',' '))
        ]
    ).properties(
        width=700,
        height=700,
        title=title
    ).project('mercator')
    return chart

# Render selected plot
st.altair_chart(plot_chicago_heatmap(dropdown_options[selected_var], selected_var), use_container_width=True)