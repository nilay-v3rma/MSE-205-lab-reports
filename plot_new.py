import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Constants
L0 = 83  # Gauge length in mm
t = 3.36  # Thickness in mm
w = 12.87  # Width in mm
A0 = t * w  # Cross-sectional area in mm^2

# Read the CSV file
file_path_aluminum = 'MSE205_Aluminum.csv'
data_aluminum = pd.read_csv(file_path_aluminum)
data_aluminum.columns = data_aluminum.columns.str.strip()

# Remove rows with negative strain values
data_aluminum = data_aluminum[data_aluminum['Strain'] >= 0]

# Calculate stress
data_aluminum['Stress (MPa)'] = data_aluminum['Standard force'] / A0

# Debugging: Print strain range
print(f"Strain range: {data_aluminum['Strain'].min()} to {data_aluminum['Strain'].max()}")

# 0.2% offset strain
offset_strain = 0.002

# Fit linear region (elastic region)
elastic_region = data_aluminum.iloc[:100]  # Selecting the first 100 data points
print(f"Elastic region size: {len(elastic_region)}")

# Perform linear fit
linear_fit = np.polyfit(elastic_region['Strain'], elastic_region['Stress (MPa)'], 1)
slope, intercept = linear_fit
print(f"Linear fit: slope = {slope:.2f}, intercept = {intercept:.2f}")

# Limit the offset line to match the stress range
max_stress = data_aluminum['Stress (MPa)'].max()
data_aluminum['Offset Line'] = np.minimum(
    slope * (data_aluminum['Strain'] - offset_strain) + intercept,
    max_stress
)

# Find the intersection of the offset line and stress-strain curve
intersection = data_aluminum[np.isclose(data_aluminum['Stress (MPa)'], data_aluminum['Offset Line'], atol=0.09)]
if not intersection.empty:
    intersection_point = intersection.iloc[0]
    yield_stress = intersection_point['Stress (MPa)']
    yield_strain = intersection_point['Strain']
    print(f"Yield point: Stress = {yield_stress:.2f} MPa, Strain = {yield_strain:.5f}")
else:
    print("No intersection found. Adjust the tolerance or check the data.")

# Plot the stress-strain curve with the yield point and bounded offset line
plt.figure(figsize=(8, 6))
plt.plot(data_aluminum['Strain'], data_aluminum['Stress (MPa)'], label='Aluminum', color='green')
plt.plot(data_aluminum['Strain'], data_aluminum['Offset Line'], '--', label='0.2% Offset Line', color='red')
if not intersection.empty:
    plt.scatter(yield_strain, yield_stress, color='blue', label=f"Yield Point: {yield_stress:.2f} MPa")
plt.xlabel('Strain')
plt.ylabel('Stress (MPa)')
plt.title('Engineering Stress-Strain Curve with Yield Point')
plt.legend()
plt.grid()
plt.savefig('yield_point_refined.png')
plt.show()
