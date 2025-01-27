import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score

# Constants
L0 = 25  # Gauge length in mm
d = 10   # Diameter in mm
A0 = np.pi * np.square(d / 2)  # Cross-sectional area in mm^2
print("---------------Iron-----------------")

# Read and process data
file_path_iron = 'Lab2_CastIron.csv'
data_iron_initial = pd.read_csv(file_path_iron)
data_iron_initial.columns = data_iron_initial.columns.str.strip()

# Filter and process the data
data_iron = data_iron_initial[data_iron_initial['Compressive strain'] >= 0.004].reset_index(drop=True)
data_iron['Stress (MPa)'] = data_iron['Standard force'] / A0
data_iron['Compressive strain'] = data_iron['Compressive strain'] / L0

# Find the start and end indices
end_index = data_iron['Stress (MPa)'].idxmax()
tolerance = 1e-6
start_index_list = data_iron.index[(np.abs(data_iron['Compressive strain'] - 0.006) < tolerance)]

if len(start_index_list) > 0:
    start_index = start_index_list[0]  # Use the first matching index
else:
    print("No data points found near strain = 0.006")
    start_index = None

# Ensure valid indices for linear fit
if start_index is not None and start_index < end_index:
    # Slice the data for the linear fit
    x_data = data_iron['Compressive strain'].iloc[start_index:end_index + 1]
    y_data = data_iron['Stress (MPa)'].iloc[start_index:end_index + 1]

    # Perform the linear fit
    coefficients = np.polyfit(x_data, y_data, 1)  # Linear fit (degree 1)
    linear_fit = np.poly1d(coefficients)

    # Calculate R^2 score
    y_pred = linear_fit(x_data)
    r2 = r2_score(y_data, y_pred)

    # Print the results
    print(f"Linear fit equation: y = {coefficients[0]:.3f}x + {coefficients[1]:.3f}")

    # Plot the data and the linear fit
    plt.figure(figsize=(8, 6))
    plt.plot(data_iron['Compressive strain'], data_iron['Stress (MPa)'])
    plt.plot(x_data, y_pred, '-', label=f'Linear fit, Elastic Modulus = {coefficients[0]:.3f} MPa', color='red')
    plt.xlabel('Compressive Strain')
    plt.ylabel('Stress (MPa)')
    plt.title('Linear Fit for Stress vs Compressive Strain')
    plt.legend()
    plt.grid(True)
    plt.savefig('images/iron/elastic_modulus_iron.png')
    
else:
    print("Invalid indices for linear fitting.")
