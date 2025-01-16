
# Carbon Stock Simulation Tool

## Overview
This is a web application built using **Streamlit** that simulates carbon stock changes based on IPCC equations and guidelines. The application incorporates the Carbon Budget Model of the Canadian Forest Sector (CBM-CFS3) framework for scenario analysis and machine learning predictions.

![App Screenshot](Screenshot%202025-01-16%20094311.png)

## Features
- **User-Friendly Interface:** Easily input parameters for carbon sequestration simulations.
- **Scenario Analysis:** Compare different carbon stock dynamics under varying conditions.
- **Geospatial Simulation:** Visualize results with 3D maps using **Pydeck**.
- **Machine Learning Integration:** Predict future carbon stocks with linear regression models.

## Requirements
Ensure you have the following Python dependencies installed:
- `streamlit`
- `numpy`
- `pandas`
- `matplotlib`
- `geopandas`
- `pydeck`
- `scikit-learn`

Install all dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/antondwifitriyanto/carbon-stock-simulation.git
   ```
2. Navigate to the project directory:
   ```bash
   cd carbon-stock-simulation
   ```
3. Run the application:
   ```bash
   streamlit run app2.py
   ```
4. Access the application at `http://localhost:8501` in your browser.

## Project Structure
- `app2.py`: Main application script.
- `requirements.txt`: Python dependencies for the project.
- `Screenshot 2025-01-16 094311.png`: Example screenshot of the application.

## Screenshot
![Screenshot of the App](Screenshot%202025-01-16%20094311.png)

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contribution
Contributions are welcome! Please fork this repository and submit a pull request for review.

## Contact
For any questions or suggestions, feel free to contact [anton.dwi@binus.ac.id].
