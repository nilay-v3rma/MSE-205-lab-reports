import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Constants
L0 = 83  # Gauge length in mm
t = 3.36  # Thickness in mm
w = 12.87  # Width in mm
A0 = t * w  # Cross-sectional area in mm^2

#--------------------Aluminum----------------------#
# Read the CSV file
file_path_aluminum = 'MSE205_Aluminum.csv'
data_aluminum = pd.read_csv(file_path_aluminum)  # Creating the Pandas DataFrame

data_aluminum.columns = data_aluminum.columns.str.strip()  # Removing trailing white spaces

# Calculate stress and strain
data_aluminum['Stress (MPa)'] = data_aluminum['Standard force'] / A0  # Stress in MPa (load (N)/ initial area (mm^2))
data_aluminum['Strain2'] = (data_aluminum['Grip to grip separat'] - L0) / L0  # Strain (grip to grip seperation (mm) - gauge length (mm)/ gauge length (mm))

# Calculate true stress
data_aluminum['True Stress (MPa)'] = data_aluminum['Stress (MPa)'] * (1 + data_aluminum['Strain'])

#--------------------Stainless Steel----------------------#
# Read the CSV file
file_path_steel = 'MSE205_Stainless Steel.csv'
data_steel = pd.read_csv(file_path_steel)  # Creating the Pandas DataFrame

data_steel.columns = data_steel.columns.str.strip()  # Removing trailing white spaces

# Calculate stress and strain
data_steel['Stress (MPa)'] = data_steel['Standard force'] / A0  # Stress in MPa (load (N)/ initial area (mm^2))
data_steel['Strain2'] = (data_steel['Grip to grip separat'] - L0) / L0  # Strain (grip to grip seperation (mm) - gauge length (mm)/ gauge length (mm))

# Calculate true stress
data_steel['True Stress (MPa)'] = data_steel['Stress (MPa)'] * (1 + data_steel['Strain'])

#--------------------Plotting----------------------#
plt.figure(figsize=(8, 6))
plt.plot(data_aluminum['Strain'], data_aluminum['Stress (MPa)'], label='Aluminum', color='green')
plt.xlabel('Strain')
plt.ylabel('Stress (MPa)')
plt.title('Engineering Stress-Strain Curve')
plt.legend()
plt.grid()
plt.savefig('stress_strain_onlyAluminum.png')

# plt.figure(figsize=(8, 6))
# plt.plot(data_aluminum['Strain'], data_aluminum['Stress (MPa)'], label='Aluminum', color='blue')
# plt.plot(data_steel['Strain'], data_steel['Stress (MPa)'], label='Stainless Steel', color='green')
# plt.xlabel('Strain')
# plt.ylabel('Stress (MPa)')
# plt.title('Engineering Stress-Strain Curve')
# plt.legend()
# plt.grid()
# plt.savefig('stress_strain_curve.png')
#
# plt.figure(figsize=(8, 6))
# plt.plot(data_aluminum['Strain'], data_aluminum['Stress (MPa)'], label='Engineering Aluminum', color='blue')
# plt.plot(data_aluminum['Strain'], data_aluminum['True Stress (MPa)'], label='True Aluminum', color='red')
# plt.xlabel('Strain')
# plt.ylabel('Stress (MPa)')
# plt.title('True Stress-Strain Curve')
# plt.legend()
# plt.grid()
# plt.savefig('stress_strain_aluminum.png')
#
# plt.figure(figsize=(8, 6))
# plt.plot(data_aluminum['Strain2'], data_aluminum['Stress (MPa)'], label='Aluminum2', color='red')
# plt.plot(data_steel['Strain2'], data_steel['Stress (MPa)'], label='Stainless Steel2', color='darkgreen')
# plt.xlabel('Strain')
# plt.ylabel('Stress (MPa)')
# plt.title('Stress-Strain Curve')
# plt.legend()
# plt.grid()
# plt.savefig('stress_strain_gripgrip.png')
#
# plt.figure(figsize=(8, 6))
# plt.plot(data_steel['Strain'], data_steel['Stress (MPa)'], label='Engineering Steel', color='blue')
# plt.plot(data_steel['Strain'], data_steel['True Stress (MPa)'], label='True Stainless Steel', color='red')
# plt.xlabel('Strain')
# plt.ylabel('Stress (MPa)')
# plt.title('Stress-Strain Curve')
# plt.legend()
# plt.grid()
# plt.savefig('stress_strain_steel.png')
