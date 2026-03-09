import streamlit as st
import pandas as pd
import geopandas as gpd
import altair as alt

st.title("Chicago Crime Heatmap Test")

# Load zip CSV
zip_data = pd.read_csv("data/derived-data/zip_data.csv")
zip_data["Zip Code"] = zip_data["Zip Code"].astype(str).str.strip()
zip_data["CrimeRate"] = pd.to_numeric(zip_data["CrimeRate"], errors="coerce")

# Load shapefile
map_data = gpd.read_file("data/raw-data/ZC/ZC.shp")
map_data["zip"] = map_data["zip"].astype(str).str.strip()
map_data = map_data.to_crs("EPSG:4326")

# Merge
merged = map_data.merge(zip_data, left_on="zip", right_on="Zip Code", how="left")

st.write("Merged head", merged[["zip", "CrimeRate"]].head(10))

# Only continue if merge succeeded
if merged["CrimeRate"].notna().any():
    chart = alt.Chart(alt.Data(values=merged.__geo_interface__["features"])).mark_geoshape(
        stroke='black'
    ).encode(
        color=alt.Color("properties.CrimeRate:Q", scale=alt.Scale(scheme="blues")),
        tooltip=[
            alt.Tooltip("properties.zip:N", title="ZIP Code"),
            alt.Tooltip("properties.CrimeRate:Q", title="Crime Rate")
        ]
    ).properties(width=700, height=700).project("mercator")

    st.altair_chart(chart, use_container_width=True)
else:
    st.error("Merge failed: all CrimeRate values are NaN")

