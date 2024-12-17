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

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset from a CSV file
df = pd.read_csv('./Training data.csv')

# Separate the features and target variable
features = df.drop('ENERGY_CONSUMPTION_CURRENT', axis=1)  # All columns except the target variable
target = df['ENERGY_CONSUMPTION_CURRENT']  # The target variable for prediction

# Calculate the Pearson correlation between each feature and the target variable
correlation = features.corrwith(target)  # Compute correlation with the target variable
correlation_sorted = correlation.sort_values(ascending=False)  # Sort the correlation values in descending order

# Print the sorted correlation values
print(correlation_sorted)

# Plot the correlation coefficients as a bar chart
correlation_sorted.plot(kind='bar')  # Create a bar plot for the correlation values
plt.title('Feature Correlation with Target Variable')  # Set the title of the plot
plt.xlabel('Features')  # Label for the x-axis
plt.ylabel('Correlation Coefficient')  # Label for the y-axis
plt.xticks(rotation=45, fontsize=7, ha='right')  # Rotate the x-axis labels for better readability
plt.tight_layout()  # Automatically adjust layout to avoid label overlap
plt.show()  # Display the plot
