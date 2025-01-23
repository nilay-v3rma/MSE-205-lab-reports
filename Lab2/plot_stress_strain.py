import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Constants
L0 = 25  # Gauge length in mm
d = 10 # diameter in mm
A0 = np.pi * np.square(d/2) # Cross-sectional area in mm^2

#--------------------Aluminum----------------------#
# Read the CSV file
file_path_aluminum = 'Lab2_Aluminum.csv'
data_aluminum = pd.read_csv(file_path_aluminum)  # Creating the Pandas DataFrame

data_aluminum.columns = data_aluminum.columns.str.strip()  # Removing trailing white spaces
data_aluminum = data_aluminum[data_aluminum['Compressive strain'] >= 0]  # Ensure non-negative strain values

# Calculate engineering stress and strain
data_aluminum['Stress (MPa)'] = data_aluminum['Standard force'] / A0  # Stress in MPa (load (N)/ initial area (mm^2))
data_aluminum['Compressive strain'] = data_aluminum['Compressive strain'] / L0  # Strain (normalized)

# Additional strain column using grip-to-grip separation
data_aluminum['Strain2'] = (data_aluminum['Absolute crosshead t'] - L0) / L0

# Calculate true stress
data_aluminum['True Stress (MPa)'] = data_aluminum['Stress (MPa)'] * (1 + data_aluminum['Compressive strain'])

# Clean True Stress: Remove decreasing values
true_stress_aluminum = data_aluminum['True Stress (MPa)'].copy()
max_index = true_stress_aluminum.idxmax()  # Find the index of maximum True Stress
true_stress_aluminum.iloc[max_index + 1:] = np.nan  # Remove all values after the maximum index
data_aluminum['Corrected True Stress (MPa)'] = true_stress_aluminum

#--------------------Casr Iron----------------------#
# Read the CSV file
file_path_iron = 'Lab2_CastIron.csv'
data_iron = pd.read_csv(file_path_iron)  # Creating the Pandas DataFrame

data_iron.columns = data_iron.columns.str.strip()  # Removing trailing white spaces
data_iron = data_iron[data_iron['Compressive strain'] >= 0]

# Calculate stress and strain
data_iron['Stress (MPa)'] = data_iron['Standard force'] / A0  # Stress in MPa (load (N)/ initial area (mm^2))
data_iron['Compressive strain'] = data_iron['Compressive strain'] / L0 #calculate strain
data_iron['Strain2'] = (data_iron['Absolute crosshead t'] - L0) / L0  # Strain (grip to grip seperation (mm) - gauge length (mm)/ gauge length (mm))

# Calculate true stress
data_iron['True Stress (MPa)'] = data_iron['Stress (MPa)'] * (1 + data_iron['Compressive strain'])

# Clean True Stress: Remove decreasing values
true_stress_iron = data_iron['True Stress (MPa)'].copy()
max_index = true_stress_iron.idxmax()  # Find the index of maximum True Stress
true_stress_iron.iloc[max_index + 1:] = np.nan  # Remove all values after the maximum index
data_iron['Corrected True Stress (MPa)'] = true_stress_iron


#--------------------Plotting----------------------#
#mix engineering stress strain
# plt.figure(figsize=(8, 6))
# plt.plot(data_aluminum['Strain'], data_aluminum['Stress (MPa)'], label='Aluminum', color='blue')
# plt.plot(data_steel['Strain'], data_steel['Stress (MPa)'], label='Stainless Steel', color='green')
# plt.xlabel('Strain')
# plt.ylabel('Stress (MPa)')
# plt.title('Engineering Stress-Strain Curve')
# plt.legend()
# plt.grid()
# plt.savefig('images/engineering_stress_strain.png')

# #mix engineering stress strain using gripgrip
# plt.figure(figsize=(8, 6))
# plt.plot(data_aluminum['Strain2'], data_aluminum['Stress (MPa)'], label='Aluminum', color='blue')
# plt.plot(data_steel['Strain2'], data_steel['Stress (MPa)'], label='Stainless Steel', color='green')
# plt.xlabel('Strain')
# plt.ylabel('Stress (MPa)')
# plt.title('Engineering Stress-Strain Curve using grip to grip seperation')
# plt.legend()
# plt.grid()
# plt.savefig('images/engineering_stress_strain_gripgrip.png')

# #mix true stress strain
# plt.figure(figsize=(8, 6))
# plt.plot(data_aluminum['Strain'], data_aluminum['Corrected True Stress (MPa)'], label='Aluminum', color='blue')
# plt.plot(data_steel['Strain'], data_steel['Corrected True Stress (MPa)'], label='Stainless Steel', color='green')
# plt.xlabel('Strain')
# plt.ylabel('True Stress (MPa)')
# plt.title('True Stress-Strain Curve')
# plt.legend()
# plt.grid()
# plt.savefig('images/true_stress_strain.png')

# #mix true stress strain using gripgrip
# plt.figure(figsize=(8, 6))
# plt.plot(data_aluminum['Strain2'], data_aluminum['Corrected True Stress (MPa)'], label='Aluminum', color='blue')
# plt.plot(data_steel['Strain2'], data_steel['Corrected True Stress (MPa)'], label='Stainless Steel', color='green')
# plt.xlabel('Strain')
# plt.ylabel('True Stress (MPa)')
# plt.title('True Stress-Strain Curve using grip to grip seperation')
# plt.legend()
# plt.grid()
# plt.savefig('images/true_stress_strain_gripgrip.png')

#aluminum engineering stress strain
plt.figure(figsize=(8, 6))
plt.plot(data_aluminum['Compressive strain'], data_aluminum['Stress (MPa)'], label='Aluminum', color='blue')
plt.xlabel('Strain')
plt.ylabel('Stress (MPa)')
plt.title('Engineering Stress-Strain Curve for Aluminum')
plt.legend()
plt.grid()
plt.savefig('images/aluminum/engineering_stress_strain.png')

# #aluminum engineering stress strain gripgrip
# plt.figure(figsize=(8, 6))
# plt.plot(data_aluminum['Strain2'], data_aluminum['Stress (MPa)'], label='Aluminum', color='blue')
# plt.xlabel('Strain')
# plt.ylabel('Stress (MPa)')
# plt.title('Engineering Stress-Strain Curve for Aluminum using Grip to Grip Seperation')
# plt.legend()
# plt.grid()
# plt.savefig('images/aluminum/engineering_stress_strain_gripgrip.png')

# #aluminum true stress strain
# plt.figure(figsize=(10, 8))
# plt.plot(data_aluminum['Strain'], data_aluminum['Corrected True Stress (MPa)'], 
#          label='True Stress-Strain Curve', color='blue', linewidth=2)
# plt.xlabel('Strain')
# plt.ylabel('True Stress (MPa)')
# plt.title('True Stress-Strain Curve for Aluminum')
# plt.legend()
# plt.grid()
# plt.savefig('images/aluminum/true_stress_strain.png')
# plt.close()

# #aluminum true stress strain gripgrip
# plt.figure(figsize=(10, 8))
# plt.plot(data_aluminum['Strain2'], data_aluminum['Corrected True Stress (MPa)'], 
#          label='True Stress-Strain Curve', color='blue', linewidth=2)
# plt.xlabel('Strain')
# plt.ylabel('True Stress (MPa)')
# plt.title('True Stress-Strain Curve for Aluminum using Grip to Grip Seperation')
# plt.legend()
# plt.grid()
# plt.savefig('images/aluminum/true_stress_strain_gripgrip.png')
# plt.close()

#steel engineering stress strain
plt.figure(figsize=(8, 6))
plt.plot(data_iron['Compressive strain'], data_iron['Stress (MPa)'], label='Cast Iron', color='blue')
plt.xlabel('Strain')
plt.ylabel('Stress (MPa)')
plt.title('Engineering Stress-Strain Curve for Cast Iron')
plt.legend()
plt.grid()
plt.savefig('images/iron/engineering_stress_strain.png')

# #steel engineering stress strain gripgrip
# plt.figure(figsize=(8, 6))
# plt.plot(data_steel['Strain2'], data_steel['Stress (MPa)'], label='Stainless Steel', color='blue')
# plt.xlabel('Strain')
# plt.ylabel('Stress (MPa)')
# plt.title('Engineering Stress-Strain Curve for Steel using Grip to Grip Seperation')
# plt.legend()
# plt.grid()
# plt.savefig('images/steel/engineering_stress_strain_gripgrip.png')

# #steel true stress strain
# plt.figure(figsize=(10, 8))
# plt.plot(data_steel['Strain'], data_steel['Corrected True Stress (MPa)'], 
#          label='True Stress-Strain Curve', color='blue', linewidth=2)
# plt.xlabel('Strain')
# plt.ylabel('True Stress (MPa)')
# plt.title('True Stress-Strain Curve for Steel')
# plt.legend()
# plt.grid()
# plt.savefig('images/steel/true_stress_strain.png')
# plt.close()

# #steel true stress strain gripgrip
# plt.figure(figsize=(10, 8))
# plt.plot(data_steel['Strain2'], data_steel['Corrected True Stress (MPa)'], 
#          label='True Stress-Strain Curve', color='blue', linewidth=2)
# plt.xlabel('Strain')
# plt.ylabel('True Stress (MPa)')
# plt.title('True Stress-Strain Curve for Steel Using Grip to Grip Seperation')
# plt.legend()
# plt.grid()
# plt.savefig('images/steel/true_stress_strain_gripgrip.png')
# plt.close()
