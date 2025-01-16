import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Carbon Sequestration Tool", page_icon="🌱", layout="wide")

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

# Title and description
st.title("Carbon Sequestration and CO2 Emission Reduction Tool")
st.markdown(
    """
    This tool helps estimate the potential CO2 reduction and carbon sequestration benefits 
    of using soil improvement products. Input your data below to see the results.
    """
)

# Section: Input Data
st.header("1. Input Baseline Data")

col1, col2 = st.columns(2)

with col1:
    baseline_carbon = st.number_input("Baseline Soil Carbon Level (tons/ha):", min_value=0.0, step=0.1, value=10.0)
    fertilizer_use = st.number_input("Fertilizer Use Reduction (%):", min_value=0, max_value=100, step=1, value=20)

with col2:
    pesticide_use = st.number_input("Pesticide Use Reduction (%):", min_value=0, max_value=100, step=1, value=15)
    yield_increase = st.number_input("Yield Increase (%):", min_value=0, max_value=100, step=1, value=10)

area_size = st.number_input("Target Area Size (ha):", min_value=1, step=1, value=100)

time_horizon = st.selectbox("Time Horizon (years):", [5, 10, 20])

# Section: Modeling and Analysis
st.header("2. Modeling and Analysis")

# Example calculations
def calculate_carbon_sequestration(baseline, reduction_rate, years, area):
    sequestration_rate = 0.5  # Example rate of carbon sequestration in tons/ha/year
    total_sequestration = (baseline + reduction_rate * years) * sequestration_rate * area
    return total_sequestration

def calculate_co2_reduction(fertilizer_reduction, pesticide_reduction, area):
    co2_reduction_factor = 0.3  # Example CO2 reduction factor (tons CO2/ha)
    total_reduction = (fertilizer_reduction + pesticide_reduction) * co2_reduction_factor * area / 100
    return total_reduction

# Perform calculations
carbon_sequestration = calculate_carbon_sequestration(baseline_carbon, yield_increase / 100, time_horizon, area_size)
co2_reduction = calculate_co2_reduction(fertilizer_use, pesticide_use, area_size)

# Display results
st.write(f"**Estimated Carbon Sequestration Over {time_horizon} Years:** {carbon_sequestration:.2f} tons")
st.write(f"**Estimated CO2 Emissions Reduction:** {co2_reduction:.2f} tons")

# Section: Visualization
st.header("3. Visualization")

# Create a chart for carbon sequestration over time
years = np.arange(1, time_horizon + 1)
carbon_data = [calculate_carbon_sequestration(baseline_carbon, yield_increase / 100, y, area_size) for y in years]

fig, ax = plt.subplots()
ax.plot(years, carbon_data, marker="o", color="green")
ax.set_title("Carbon Sequestration Over Time", color="#006400")
ax.set_xlabel("Years", color="#006400")
ax.set_ylabel("Total Carbon Sequestration (tons)", color="#006400")
st.pyplot(fig)

# Section: Stakeholder Presentation
st.header("4. Stakeholder Presentation")
st.markdown(
    """
    Use this tool to showcase the benefits of your soil improvement product to stakeholders. 
    Customize the data inputs and share the visualized results to demonstrate potential impacts.
    """
)

st.success("Tool ready for demonstration!")
