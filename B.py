import streamlit as st
import pandas as pd
import altair as alt

# -----------------------------
# Title
# -----------------------------
st.title("Violent Crime Rates in Chicago (2021-2025)")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_crime_data():
    df = pd.read_csv("data/derived-data/Crime.csv", parse_dates=["Date"])
    return df

crime_data = load_crime_data()

# -----------------------------
# Crime Type Selector
# -----------------------------
crime_options = [
    "All Violent Crime",
    "HOMICIDE",
    "CRIMINAL SEXUAL ASSAULT",
    "ASSAULT",
    "BATTERY",
]

selected_crime = st.selectbox("Select Crime Type", crime_options)

violent_types = [
    "HOMICIDE",
    "CRIMINAL SEXUAL ASSAULT",
    "ASSAULT",
    "BATTERY",
]

violent_df = crime_data[crime_data["Primary Type"].isin(violent_types)].copy()

if selected_crime != "All Violent Crime":
    violent_df = violent_df[violent_df["Primary Type"] == selected_crime]

violent_df["YearMonth"] = violent_df["Date"].dt.to_period("M")

# -----------------------------
# Compute 5-Year Zip Crime Rates
# -----------------------------
zip_totals = (
    violent_df.groupby("Zip Code")
    .agg(
        total_crimes_5yr=("Primary Type", "count"),
        population=("Population", "first"),
    )
    .reset_index()
)

zip_totals["crime_rate_5yr"] = (
    zip_totals["total_crimes_5yr"] / zip_totals["population"] * 1000
)

baseline_df = zip_totals[["Zip Code", "crime_rate_5yr"]]

# -----------------------------
# Slider Count State
# -----------------------------
if "slider_count" not in st.session_state:
    st.session_state.slider_count = 1

max_sliders = 5

# -----------------------------
# Add / Remove Percentile Buttons
# -----------------------------
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

# -----------------------------
# Render Sliders
# -----------------------------
percentiles = []
previous_value = 0

st.write("### Percentile Thresholds")

for i in range(st.session_state.slider_count):

    value = st.slider(
        f"Percentile #{i+1}",
        min_value=previous_value,
        max_value=100,
        value=st.session_state.get(f"slider_{i}", min(previous_value + 10, 100)),
        key=f"slider_{i}",
    )

    percentiles.append(value)
    previous_value = value

# -----------------------------
# Percentile Boundaries
# -----------------------------
percentile_bounds = sorted(percentiles) + [100]

# -----------------------------
# Create Zip Groups
# -----------------------------
zip_groups = []

lower_rate = -float("inf")

for p in percentile_bounds:

    threshold = baseline_df["crime_rate_5yr"].quantile(p / 100)

    zips = baseline_df[
        (baseline_df["crime_rate_5yr"] > lower_rate)
        & (baseline_df["crime_rate_5yr"] <= threshold)
    ]["Zip Code"].tolist()

    zip_groups.append(zips)

    lower_rate = threshold

# -----------------------------
# Monthly Crime Rate Function
# -----------------------------
def compute_group_monthly_rate(df, zip_list):

    group_df = df[df["Zip Code"].isin(zip_list)]

    if group_df.empty:
        return pd.DataFrame()

    monthly_population = (
        group_df.groupby(["YearMonth", "Zip Code"])
        .agg(
            zip_population=("Population", "first"),
            monthly_crimes=("Primary Type", "count"),
        )
        .reset_index()
    )

    monthly_summary = (
        monthly_population.groupby("YearMonth")
        .agg(
            total_crimes=("monthly_crimes", "sum"),
            total_population=("zip_population", "sum"),
        )
        .reset_index()
    )

    monthly_summary["crime_rate_per_1000"] = (
        monthly_summary["total_crimes"]
        / monthly_summary["total_population"]
        * 1000
    )

    monthly_summary["YearMonth"] = monthly_summary["YearMonth"].dt.to_timestamp()

    return monthly_summary


# -----------------------------
# Build Plot Data
# -----------------------------
plot_dfs = []

for idx, zips in enumerate(zip_groups):

    df_grp = compute_group_monthly_rate(violent_df, zips)

    if df_grp.empty:
        continue

    df_grp["Group"] = f"P{idx+1}: {percentile_bounds[idx]}%"

    plot_dfs.append(df_grp)

# -----------------------------
# Plot
# -----------------------------
if plot_dfs:

    plot_df = pd.concat(plot_dfs)

    color_palette = [
        "#0b3d91",
        "#1e5ebf",
        "#5b8fd1",
        "#a4c3e0",
        "#d0e1f2",
        "#e0efff",
    ]

    num_groups = len(plot_dfs)

    chart = alt.Chart(plot_df).mark_line(point=True).encode(

        x=alt.X(
            "YearMonth:T",
            axis=alt.Axis(title="Month", format="%b %Y", tickCount="month"),
        ),

        y=alt.Y(
            "crime_rate_per_1000:Q",
            title="Crime Rate Per 1,000 Residents",
        ),

        color=alt.Color(
            "Group:N",
            scale=alt.Scale(
                domain=plot_df["Group"].unique(),
                range=color_palette[:num_groups],
            ),
            legend=alt.Legend(title="Percentile Group"),
        ),

        tooltip=[
            "YearMonth:T",
            "crime_rate_per_1000:Q",
            "total_crimes:Q",
            "Group:N",
        ],
    ).properties(
        title=f"Monthly Crime Rate by Zip Crime Percentile ({selected_crime})",
        width=700,
        height=400,
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
- the monthly crime rate for the **remaining 40% of zip codes with the highest crime rates**
""")



