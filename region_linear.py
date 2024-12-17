# MIT License
#
# Copyright (c) 2024 Yichao Yang, Dingyue Hu, Yihan Ding, Bohan Cao.
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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt

def region_model_save(model_path, model):
    """
    Save the trained model to the specified path.

    Parameters:
    - model_path: The file path where the model will be saved.
    - model: The trained machine learning model to be saved.
    """
    joblib.dump(model, model_path)

# Load the dataset
data = pd.read_csv('./Training data.csv')

# Define the feature columns and target column
feature_columns = ['FLOOR_LEVEL', 'FLOOR_ENERGY_EFF', 'GLAZED_TYPE',
                   'WALLS_ENERGY_EFF', 'ROOF_ENERGY_EFF', 'MAINHEAT_ENERGY_EFF',
                   'MAINHEATC_ENERGY_EFF', 'LIGHTING_ENERGY_EFF']
target_column = 'ENERGY_CONSUMPTION_CURRENT'

# Assign the feature matrix (X) and target vector (y)
X = data[feature_columns]
y = data[target_column]

# Handle missing values by removing rows with missing values
X = X.dropna()
y = y.dropna()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define bins to partition the target variable for regional linearization
bins = np.percentile(y_train, [25, 50, 70, 85, 95, 100])

# Initialize lists to store models, regions, and linear expressions
models = []
regions = []
linear_expressions = []

# Perform regional linearization by fitting a separate linear model for each region
for i in range(len(bins)):

    # Define the mask for selecting data points in the current region
    if i == 0:
        region_mask = y_train <= bins[i]
    else:
        region_mask = (y_train > bins[i - 1]) & (y_train <= bins[i])

    # Assign data points to the current region
    X_train_region = X_train[region_mask]
    y_train_region = y_train[region_mask]

    # Fit a linear regression model to the data in the current region
    model = LinearRegression()
    model.fit(X_train_region, y_train_region)

    # Append the model and region bounds
    models.append(model)
    regions.append((bins[i - 1] if i > 0 else None, bins[i]))

    # Create a string representation of the linear model for the current region
    coefficients = model.coef_
    intercept = model.intercept_
    linear_expr = f"y = {intercept:.2f}"
    for j, coef in enumerate(coefficients, start=1):
        linear_expr += f" + {coef:.2f}*x{j}"
    linear_expressions.append(linear_expr)

# Evaluate the performance of the model on the test set
y_pred_test = np.zeros_like(y_test)
for i, model in enumerate(models):
    lower_bound = regions[i][0] if regions[i][0] is not None else -np.inf
    upper_bound = regions[i][1] if i < len(regions) - 1 else np.inf
    region_mask = (y_test >= lower_bound) & (y_test < upper_bound)
    y_pred_test[region_mask] = model.predict(X_test[region_mask])

# Calculate the R-squared value to assess the model's performance
r_squared = r2_score(y_test, y_pred_test)

# Plot the predicted vs actual energy consumption
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, color='blue', label='Predicted vs Actual')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', lw=2, label='Ideal Fit')
plt.xlabel('Actual Energy Consumption')
plt.ylabel('Predicted Energy Consumption')
plt.title('Fit of the Regional Linear Models')
plt.legend()
plt.show()

# Save the trained model to a file
region_model_save('./model.joblib', model)

# Output the R-squared value and the linear expressions for each region
print(f"R^2: {r_squared}")
for i, expr in enumerate(linear_expressions, start=1):
    print(f"Region {i}: {expr}")

def region_model_load(model_path):
    """
    Load the model from the specified file path.

    Parameters:
    - model_path: The file path where the model is stored.

    Returns:
    - The loaded machine learning model.
    """
    return joblib.load(model_path)

def region_model_predict(feature_names, model):
    """
    Predict the target variable for the given features using the provided model.

    Parameters:
    - feature_names: The feature matrix for which predictions are to be made.
    - model: The trained model used to make predictions.

    Returns:
    - The predicted values based on the input features.
    """
    return model.predict(feature_names)
