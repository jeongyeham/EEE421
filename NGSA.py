# MIT License
#
# Copyright (c) 2024 Yihan Ding, Dingyue Hu, Yichao Yang, Bohan Cao.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import joblib
import pandas as pd
from pymoo.core.problem import Problem
import math
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import plotly.express as px

# Costs associated with different energy efficiency improvements
wall_insulation_costs = [1000, 5000, 12000, 14000]
heating_system_costs = [2000, 5000, 6500, 7000]
glazing_costs = [1500, 2000, 2500, 4000]
pv_costs = [0, 800, 1000, 1200, 1600, 2400, 2800, 3600, 4000]

# Lifetimes for each system
wall_insulation_lifetime = 30
heating_system_lifetime = 16
glazing_lifetime = 20
pv_lifetime = 10

# Discount rate and electricity tariff for economic calculations
Discount_rate = 0.05
Tariff = 0.25


def region_model_load(model_path):
    """
    Load the predictive model from the specified file path.

    Parameters:
    - model_path: Path to the saved model file.

    Returns:
    - Loaded model object.
    """
    return joblib.load(model_path)


def region_model_predict(feature_names, model):
    """
    Predict energy consumption based on the input features.

    Parameters:
    - feature_names: A DataFrame of feature values.
    - model: The predictive model.

    Returns:
    - Predicted value for the input data.
    """
    return model.predict(feature_names)


def predict_current_energy_consumption():
    """
    Predict the current energy consumption for the first building in the dataset.

    Returns:
    - Current energy consumption as a single float value.
    """
    model = region_model_load('./model.joblib')
    feature_columns = ['FLOOR_LEVEL', 'FLOOR_ENERGY_EFF', 'GLAZED_TYPE',
                       'WALLS_ENERGY_EFF', 'ROOF_ENERGY_EFF', 'MAINHEAT_ENERGY_EFF',
                       'MAINHEATC_ENERGY_EFF', 'LIGHTING_ENERGY_EFF']
    all_feature_data = pd.read_csv('./group_data.csv')
    X = pd.DataFrame(all_feature_data.iloc[0, all_feature_data.columns.get_indexer(feature_columns)]).T
    current_energy_consumption = region_model_predict(X, model).item()
    return current_energy_consumption


def calculate_epv(kWp, latitude, orientation, tilt, annual_solar_radiation):
    """
    Calculate the annual electricity production of a photovoltaic (PV) system.

    Parameters:
    - kWp: Installed peak power of the PV module (kWp).
    - latitude: Latitude of the location (degrees).
    - orientation: Orientation of the PV array (degrees from North).
    - tilt: Tilt angle of the PV array (degrees from horizontal).
    - annual_solar_radiation: Annual solar radiation (kWh/m²).

    Returns:
    - Annual electricity production (kWh).
    """
    # Constants for calculation
    k1 = 26.3
    k2 = -38.5
    k3 = 14.8
    k4 = -16.5
    k5 = 27.3
    k6 = -11.9
    k7 = -1.06
    k8 = 0.0872
    k9 = -0.191

    # Convert orientation to radians
    orientation_rad = math.radians(orientation)

    # Factors for converting horizontal to inclined solar flux
    A = (k1 * math.sin(tilt / 2) ** 2 + k2 * math.sin(tilt / 2) + k3 * math.sin(tilt / 2))
    B = (k4 * math.sin(tilt / 2) ** 2 + k5 * math.sin(tilt / 2) + k6 * math.sin(tilt / 2))
    C = (k7 * math.sin(tilt / 2) ** 2 + k8 * math.sin(tilt / 2) + k9 * math.sin(tilt / 2) + 1)

    R_horizontal_to_inclined = A * math.cos(latitude - math.radians(180)) + B * math.cos(
        2 * (latitude - math.radians(180))) + C

    # Annual electricity production
    Epv = 0.8 * kWp * annual_solar_radiation * R_horizontal_to_inclined

    return Epv


def calculate_annual_solar_radiation(region, orientation):
    """
    Calculate the annual solar radiation for a given region and surface orientation.

    Parameters:
    - region: The region number corresponding to solar radiation data (e.g., from SAP Table U3).
    - orientation: The orientation of the surface (N, NE, E, SE, S, SW, W, NW).

    Returns:
    - annual_solar_radiation: The annual solar radiation (kWh/m²).
    """
    # Solar radiation data for each region, from SAP Table U3
    solar_radiation_data = {
        1: [26, 54, 96, 150, 192, 200, 189, 157, 115, 66, 33, 21],  # UK average
        2: [30, 56, 98, 157, 195, 217, 203, 173, 127, 73, 39, 24],  # Thames
        3: [32, 59, 104, 170, 208, 231, 216, 182, 133, 77, 41, 25],  # South East England
        4: [35, 62, 109, 172, 209, 235, 217, 185, 138, 80, 44, 27],  # Southern England
        5: [36, 63, 110, 174, 210, 233, 204, 182, 136, 78, 44, 28],  # South West England
        6: [32, 59, 105, 167, 201, 226, 206, 175, 130, 74, 40, 25],  # Severn Wales / Severn England
        7: [28, 55, 97, 153, 191, 208, 194, 163, 121, 69, 35, 23],  # Midlands
        8: [24, 51, 95, 152, 191, 203, 186, 152, 115, 65, 31, 20],  # West Pennines Wales / West Pennines England
        9: [23, 51, 95, 157, 200, 203, 194, 156, 113, 62, 30, 19],  # North West England / South West Scotland
        10: [23, 50, 92, 151, 200, 196, 187, 153, 111, 61, 30, 18],  # Borders Scotland / Borders England
        11: [25, 51, 95, 152, 196, 198, 190, 156, 115, 64, 32, 20],  # North East England
        12: [26, 54, 96, 150, 192, 200, 189, 157, 115, 66, 33, 21],  # East Pennines
        13: [30, 58, 101, 165, 203, 220, 206, 173, 128, 74, 39, 24],  # East Anglia
        14: [29, 57, 104, 164, 205, 220, 199, 167, 120, 68, 35, 22],  # Wales
        15: [19, 46, 88, 148, 196, 193, 185, 150, 101, 55, 25, 15],  # West Scotland
        16: [21, 46, 89, 146, 198, 191, 183, 150, 106, 57, 27, 15],  # East Scotland
        17: [19, 45, 89, 143, 194, 188, 177, 144, 101, 54, 25, 14],  # North East Scotland
        18: [17, 43, 85, 145, 189, 185, 170, 139, 98, 51, 22, 12],  # Highland
        19: [16, 41, 87, 155, 205, 206, 185, 148, 101, 51, 21, 11],  # Western Isles
        20: [14, 39, 84, 143, 205, 201, 178, 145, 100, 50, 19, 9],  # Orkney
        21: [12, 34, 79, 135, 196, 190, 168, 144, 90, 46, 16, 7],  # Shetland
        22: [24, 52, 96, 155, 201, 198, 183, 150, 107, 61, 30, 18]  # Northern Ireland
    }

    # Calculate the average solar radiation for the given region
    annual_solar_radiation = sum(solar_radiation_data[region]) / 12  # Average monthly solar radiation

    # Adjust for surface orientation (simplified approach)
    orientation_factors = {
        'N': 0.9,
        'NE': 0.95,
        'E': 1.0,
        'SE': 1.05,
        'S': 1.1,
        'SW': 1.05,
        'W': 1.0,
        'NW': 0.95
    }

    # Adjust the solar radiation based on orientation
    annual_solar_radiation *= orientation_factors[orientation]

    return annual_solar_radiation


class EnergyCostOptimization(Problem):
    """
    This class defines an energy cost optimization problem using NSGA-II.

    It optimizes the energy consumption and the net present value (NPV) of improvements
    to energy efficiency systems in a building, such as insulation, heating, glazing, and solar panels.
    """

    def __init__(self):
        """
        Initializes the optimization problem with specific variables and objectives.
        """
        super().__init__(n_var=4,  # Number of decision variables
                         n_obj=2,  # Number of objectives (cost and energy consumption)
                         n_constr=0,  # Number of constraints
                         xl=np.array([3, 1, 0, 0]),  # Lower bounds of decision variables
                         xu=np.array([3, 3, 3, 7]))  # Upper bounds of decision variables

        self.model = region_model_load('./model.joblib')  # Load the predictive model
        self.feature_columns = ['FLOOR_LEVEL', 'FLOOR_ENERGY_EFF', 'GLAZED_TYPE',
                                'WALLS_ENERGY_EFF', 'ROOF_ENERGY_EFF', 'MAINHEAT_ENERGY_EFF',
                                'MAINHEATC_ENERGY_EFF', 'LIGHTING_ENERGY_EFF']  # Feature columns used in prediction
        self.all_feature_data = pd.read_csv('./group_data.csv')  # Load the dataset
        self.X = pd.DataFrame(self.all_feature_data.iloc[0, self.all_feature_data.columns.get_indexer(
            self.feature_columns)]).T  # Extract features for the first building
        self.current_energy_consumption = predict_current_energy_consumption()  # Get the current energy consumption for comparison

    def _evaluate(self, X, out, *args):
        """
        Evaluate the objective function for each solution in X.

        Parameters:
        - X: Array of solutions (each row is a solution with decision variable values).
        - out: Dictionary to store objective values for each solution.
        """
        # Lists to store the objective values (NPV and energy consumption)
        npvs = []
        energies = []
        constrains = []  # Not used, but could be added in the future

        # Evaluate each solution in X
        for x in X:
            # Extract decision variables from the solution
            wall_level = int(x[0])
            heat_level = int(x[1])
            glazing_level = int(x[2])
            pv_count = int(x[3])

            # Update feature data with the current decision variables and predict energy consumption
            updated_X = self.X.copy()  # Deep copy to avoid modifying original data
            updated_X['WALLS_ENERGY_EFF'] = wall_level
            energy_predictions = region_model_predict(updated_X, self.model).item()
            energy_saving_transfer_benifit_per_year_wall = ((
                                                                    energy_predictions - self.current_energy_consumption) * Tariff) * \
                                                           self.all_feature_data['TOTAL_FLOOR_AREA']

            updated_X = self.X.copy()
            updated_X['MAINHEAT_ENERGY_EFF'] = heat_level
            energy_predictions = region_model_predict(updated_X, self.model).item()
            energy_saving_transfer_benifit_per_year_heat = ((
                                                                    energy_predictions - self.current_energy_consumption) * Tariff) * \
                                                           self.all_feature_data['TOTAL_FLOOR_AREA']

            updated_X = self.X.copy()
            updated_X['GLAZED_TYPE'] = glazing_level
            energy_predictions = region_model_predict(updated_X, self.model).item()
            energy_saving_transfer_benifit_per_year_glazing = ((
                                                                       energy_predictions - self.current_energy_consumption) * Tariff) * \
                                                              self.all_feature_data['TOTAL_FLOOR_AREA']

            # Calculate the total cost of the systems
            wall_cost = wall_insulation_costs[wall_level]
            heat_cost = heating_system_costs[heat_level]
            glazing_cost = glazing_costs[glazing_level]
            pv_cost = pv_costs[pv_count]

            total_inital_cost = wall_cost + heat_cost + glazing_cost + pv_cost

            # Calculate discounted benefits for each system
            discounted_wall_benifit = 0
            for year in range(1, wall_insulation_lifetime + 1):
                discounted_wall_benifit += energy_saving_transfer_benifit_per_year_wall / ((1 + Discount_rate) ** year)

            discounted_heat_benifit = 0
            for year in range(1, heating_system_lifetime + 1):
                discounted_heat_benifit += energy_saving_transfer_benifit_per_year_heat / ((1 + Discount_rate) ** year)

            discounted_glazing_benifit = 0
            for year in range(1, glazing_lifetime + 1):
                discounted_glazing_benifit += energy_saving_transfer_benifit_per_year_glazing / (
                        (1 + Discount_rate) ** year)

            discounted_pv_benifit = 0
            kWp = 0.32 * pv_count  # PV system size
            latitude = 51.5  # Latitude for Severn Wales
            orientation = 135  # South orientation
            tilt = 45  # 45-degree tilt
            annual_solar_radiation = calculate_annual_solar_radiation(14, 'SW')
            for year in range(1, pv_lifetime + 1):
                discounted_pv_benifit += calculate_epv(kWp, latitude, orientation, tilt,
                                                       annual_solar_radiation) * Tariff / ((1 + Discount_rate) ** year)

            # Calculate total discounted benefit
            total_discounted_device_benifit = (
                    discounted_wall_benifit + discounted_heat_benifit + discounted_glazing_benifit + discounted_pv_benifit)

            # Total cost: discounted benefits - initial cost
            total_npv = total_discounted_device_benifit - total_inital_cost

            # Calculate total energy savings
            total_energy_saving = ((energy_predictions - self.current_energy_consumption) * self.all_feature_data[
                'TOTAL_FLOOR_AREA'] + calculate_epv(kWp, latitude, orientation, tilt, annual_solar_radiation)) / \
                                  self.all_feature_data['TOTAL_FLOOR_AREA']

            # Append results
            npvs.append(-total_npv)
            energies.append(-total_energy_saving)

        # Store the results in the output dictionary for NSGA-II
        out["F"] = np.column_stack([npvs, energies])


if __name__ == "__main__":
    """
    Main entry point for running the NSGA-II optimization algorithm on the energy cost problem.
    """
    # Create an optimization problem instance
    problem = EnergyCostOptimization()

    # Define the NSGA-II algorithm
    algorithm = NSGA2(pop_size=100)

    # Run the optimization process
    res = minimize(problem,
                   algorithm,
                   ('n_gen', 20),  # Run for 20 generations
                   verbose=False)

    # Print Pareto-optimal solutions
    print("Pareto-optimal solutions:")
    for sol in res.F:
        print(f"NPV: {-sol[0]:.2f}, Energy: {-sol[1]:.2f}")

    # Print decision variable assignments for the optimal solutions
    print("Pareto-optimal solutions (Decision Variables):")
    for i, sol in enumerate(res.X):
        wall_level = int(sol[0])
        heat_level = int(sol[1])
        glazing_level = int(sol[2])
        pv_count = int(sol[3])

    print(f"Solution {i + 1}: Wall Insulation Level: {wall_level}, Heating System Level: {heat_level}, "
          f"Glazing Level: {glazing_level}, PV Count: {pv_count}")

    # Create a DataFrame to store the results
    solutions = []
    for i, (decision_vars, objectives) in enumerate(zip(res.X, res.F)):
        wall_level = int(decision_vars[0])
        heat_level = int(decision_vars[1])
        glazing_level = int(decision_vars[2])
        pv_count = int(decision_vars[3])

        cost = -objectives[0]
        energy = -objectives[1]  # Convert to positive value for energy saving

        solutions.append({
            "Solution #": i + 1,
            "Wall Insulation Level": wall_level,
            "Heating System Level": heat_level,
            "Glazing Level": glazing_level,
            "PV Count": pv_count,
            "NPV": round(cost, 2),
            "Energy Saving": round(energy, 2)
        })

    # Convert to DataFrame
    df_solutions = pd.DataFrame(solutions)

    # Create an interactive plot using Plotly
    fig = px.scatter(
        df_solutions,
        x="NPV",
        y="Energy Saving",
        hover_data=["Solution #", "Wall Insulation Level", "Heating System Level", "Glazing Level", "PV Count"],
        title="Pareto Front - NPV vs Energy Saving",
        labels={"NPV": "Total NPV", "Energy Saving": "Energy Saving (kWh/mm2)"},
        template="plotly_dark"
    )

    # Customize the plot
    fig.update_traces(marker=dict(size=10, color='red'), selector=dict(mode='markers'))

    # Show the plot
    fig.show()

    # Save the plot as an HTML file
    fig.write_html("./pareto_front.html")
    print("Interactive plot saved as 'pareto_front.html'.")
