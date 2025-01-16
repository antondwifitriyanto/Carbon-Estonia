import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import pydeck as pdk
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Set up Streamlit app
st.set_page_config(page_title="Carbon Stock Simulation Tool", page_icon="🌍", layout="wide")

# Add custom theme
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f8f5;
    }
    .stHeader {
        color: #006400;
        font-family: Arial, sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add logo
st.sidebar.image("logo.png", use_container_width=True, width=100)

# Page title and description
st.title("Carbon Stock Simulation and Scenario Analysis")
st.markdown(
    """
    This tool simulates carbon stock changes based on IPCC equations and guidelines, 
    incorporating the Carbon Budget Model of the Canadian Forest Sector (CBM-CFS3) framework.
    Use the inputs below to customize your analysis.
    """
)

# Input section
st.sidebar.header("Input Parameters")

# Baseline inputs
initial_carbon_stock = st.sidebar.number_input(
    "Initial Carbon Stock (tons/ha):", min_value=0.0, value=50.0, step=1.0
)
carbon_sequestration_rate = st.sidebar.number_input(
    "Annual Carbon Sequestration Rate (tons/ha/year):", min_value=0.0, value=2.0, step=0.1
)
carbon_loss_rate = st.sidebar.number_input(
    "Annual Carbon Loss Rate (tons/ha/year):", min_value=0.0, value=0.5, step=0.1
)

# Scenario settings
time_horizon = st.sidebar.selectbox("Time Horizon (years):", options=[5, 10, 20], index=1)
simulation_areas = st.sidebar.number_input(
    "Area Covered (hectares):", min_value=1, value=100, step=1
)

# Calculations
def calculate_annual_stock(initial_stock, sequestration, loss, years):
    """Calculate carbon stock over a given time horizon."""
    annual_stock = [initial_stock]
    for year in range(1, years + 1):
        new_stock = annual_stock[-1] + sequestration - loss
        annual_stock.append(new_stock)
    return annual_stock

def calculate_total_emissions(annual_stock, area):
    """Calculate total emissions reduction and sequestration potential."""
    total_sequestration = sum([stock * area for stock in annual_stock])
    return total_sequestration

# Simulate annual carbon stock
annual_stock = calculate_annual_stock(
    initial_carbon_stock, carbon_sequestration_rate, carbon_loss_rate, time_horizon
)
total_carbon_sequestered = calculate_total_emissions(annual_stock, simulation_areas)

# Visualization
st.header("Simulation Results")
st.markdown(
    f"**Initial Carbon Stock:** {initial_carbon_stock} tons/ha\n"
    f"**Sequestration Rate:** {carbon_sequestration_rate} tons/ha/year\n"
    f"**Loss Rate:** {carbon_loss_rate} tons/ha/year\n"
    f"**Time Horizon:** {time_horizon} years\n"
    f"**Total Area Covered:** {simulation_areas} hectares"
)
st.markdown(
    f"**Total Carbon Sequestered:** {total_carbon_sequestered:.2f} tons over {time_horizon} years"
)

# Line chart for annual stock changes
st.subheader("Annual Carbon Stock Changes")
fig, ax = plt.subplots()
years = np.arange(0, time_horizon + 1)
ax.plot(years, annual_stock, marker="o", color="green")
ax.set_title("Annual Carbon Stock Changes")
ax.set_xlabel("Year")
ax.set_ylabel("Carbon Stock (tons/ha)")
ax.grid(True)
st.pyplot(fig)

# Scenario Analysis
st.header("Scenario Analysis")
st.markdown(
    "This section compares carbon stock dynamics under different sequestration and loss rates."
)

# Scenario comparison inputs
scenarios = [
    {"name": "Optimistic", "sequestration": 3.0, "loss": 0.3},
    {"name": "Conservative", "sequestration": 1.5, "loss": 0.7},
    {"name": "Business-as-Usual", "sequestration": 2.0, "loss": 0.5},
]

scenario_data = {}

for scenario in scenarios:
    scenario_name = scenario["name"]
    scenario_stock = calculate_annual_stock(
        initial_carbon_stock, scenario["sequestration"], scenario["loss"], time_horizon
    )
    scenario_data[scenario_name] = scenario_stock

# Plot scenarios
fig, ax = plt.subplots()
for scenario_name, stock in scenario_data.items():
    ax.plot(years, stock, label=f"{scenario_name} Scenario")

ax.set_title("Scenario Analysis: Carbon Stock Dynamics")
ax.set_xlabel("Year")
ax.set_ylabel("Carbon Stock (tons/ha)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Map Visualization
st.header("Geospatial Simulation with 3D Visualization")

# Load GeoJSON data
geojson_path = "tuimoisa.geojson"
gdf = gpd.read_file(geojson_path)

# Add a new column for visualization (example: carbon stock density per area)
gdf["carbon_density"] = np.random.uniform(50, 200, size=len(gdf))

# Create a Pydeck 3D map layer
layer = pdk.Layer(
    "GeoJsonLayer",
    gdf,
    pickable=True,
    extruded=True,
    get_fill_color="[255 - carbon_density, 255 - carbon_density, 200]",
    get_elevation="carbon_density",
)

# Set the map style and view state
view_state = pdk.ViewState(
    latitude=gdf.geometry.centroid.y.mean(),
    longitude=gdf.geometry.centroid.x.mean(),
    zoom=10,
    pitch=50,
)

# Render the map
r = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    map_style="mapbox://styles/mapbox/outdoors-v11",
)

st.pydeck_chart(r)

# Machine Learning Prediction
st.header("Machine Learning Prediction")

st.markdown(
    "**Expected Outcome:** Accurate quantification of CO2 reduction and carbon storage potential.\n"
    "Validated models and tools for future applications.\n"
    "Stakeholder-ready materials to showcase sustainability impacts."
)

# Prepare data for prediction
data = pd.DataFrame({
    "Year": np.arange(0, time_horizon + 1),
    "CarbonStock": annual_stock
})

# Train a simple linear regression model
X = data["Year"].values.reshape(-1, 1)
y = data["CarbonStock"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Predict future carbon stock
future_years = np.arange(time_horizon + 1, time_horizon + 6).reshape(-1, 1)
future_predictions = model.predict(future_years)

# Display predictions
st.subheader("Predicted Carbon Stock for Future Years")
prediction_data = pd.DataFrame({
    "Year": future_years.flatten(),
    "Predicted Carbon Stock (tons/ha)": future_predictions
})
st.dataframe(prediction_data)

# Plot future predictions
fig, ax = plt.subplots()
ax.plot(data["Year"], data["CarbonStock"], label="Observed", marker="o")
ax.plot(future_years, future_predictions, label="Predicted", linestyle="--", marker="x")
ax.set_title("Observed vs Predicted Carbon Stock")
ax.set_xlabel("Year")
ax.set_ylabel("Carbon Stock (tons/ha)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.success("Simulation, scenario analysis, and machine learning prediction completed!")
