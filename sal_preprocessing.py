import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt 
import os
import seaborn as sns 
import geopandas as gpd
from shapely.geometry import Point

#Crime Data Loading
base_path = os.path.dirname(os.path.abspath(__file__))
Crime = pd.read_csv(os.path.join(base_path, "data/raw-data/Crimes_-_2001_to_Present_20260228.csv"))

#Tracts Crossover File Data Loading
tracts = gpd.read_file(os.path.join(base_path, "data/raw-data/Census_Tracts_20260228/geo_export_89c4b171-2783-4715-9b7d-e49f753f00f0.shp"))

#SVI Data Loading
svi = pd.read_csv(os.path.join(base_path, "data/raw-data/illinois.csv"))

### ~~~ Cleaning and Merging Data ~~~ ###
#svi clean
svi_cook = svi[svi["COUNTY"] == "Cook County"] 
svi_cook = svi_cook[["LOCATION", "FIPS", "RPL_THEMES"]] 
svi_cook = svi_cook.rename(columns = {"RPL_THEMES" : "SVI"}) 
svi_cook

#crime + tracts spatial merge
geometry = [Point(xy) for xy in zip(Crime["Longitude"], Crime["Latitude"])] 
crimes_gdf = gpd.GeoDataFrame(Crime, geometry=geometry, crs="EPSG:4326")
tracts = tracts.to_crs(crimes_gdf.crs)
crimes_with_tracts = gpd.sjoin(crimes_gdf, tracts, how="left", predicate="within")
crimes_with_tracts

#convert FIPS to census tract 
svi_cook["tract_short"] = svi_cook["FIPS"].astype(str).str[5:]

#merging crime and svi data 
crimes_merge = crimes_with_tracts.merge(svi_cook, left_on = "census_tra", right_on = "tract_short", how = "left")
crimes_merge.head()

#specifying crime type categories
