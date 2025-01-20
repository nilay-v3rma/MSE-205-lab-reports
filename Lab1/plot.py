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
data_aluminum = data_aluminum[data_aluminum['Strain'] >= 0]  # Ensure non-negative strain values

# Calculate engineering stress and strain
data_aluminum['Stress (MPa)'] = data_aluminum['Standard force'] / A0  # Stress in MPa (load (N)/ initial area (mm^2))
data_aluminum['Strain'] = data_aluminum['Strain'] / L0  # Strain (normalized)

# Additional strain column using grip-to-grip separation
data_aluminum['Strain2'] = (data_aluminum['Grip to grip separat'] - L0) / L0

# Calculate true stress
data_aluminum['True Stress (MPa)'] = data_aluminum['Stress (MPa)'] * (1 + data_aluminum['Strain'])

# Clean True Stress: Remove decreasing values
true_stress_aluminum = data_aluminum['True Stress (MPa)'].copy()
max_index = true_stress_aluminum.idxmax()  # Find the index of maximum True Stress
true_stress_aluminum.iloc[max_index + 1:] = np.nan  # Remove all values after the maximum index
data_aluminum['Corrected True Stress (MPa)'] = true_stress_aluminum

#--------------------Stainless Steel----------------------#
# Read the CSV file
file_path_steel = 'MSE205_Stainless Steel.csv'
data_steel = pd.read_csv(file_path_steel)  # Creating the Pandas DataFrame

data_steel.columns = data_steel.columns.str.strip()  # Removing trailing white spaces
data_steel = data_steel[data_steel['Strain'] >= 0]

# Calculate stress and strain
data_steel['Stress (MPa)'] = data_steel['Standard force'] / A0  # Stress in MPa (load (N)/ initial area (mm^2))
data_steel['Strain'] = data_steel['Strain'] / L0 #calculate strain
data_steel['Strain2'] = (data_steel['Grip to grip separat'] - L0) / L0  # Strain (grip to grip seperation (mm) - gauge length (mm)/ gauge length (mm))

# Calculate true stress
data_steel['True Stress (MPa)'] = data_steel['Stress (MPa)'] * (1 + data_steel['Strain'])

# Clean True Stress: Remove decreasing values
true_stress_steel = data_steel['True Stress (MPa)'].copy()
max_index = true_stress_steel.idxmax()  # Find the index of maximum True Stress
true_stress_steel.iloc[max_index + 1:] = np.nan  # Remove all values after the maximum index
data_steel['Corrected True Stress (MPa)'] = true_stress_steel

#--------------------Plotting----------------------#
#mix engineering stress strain
plt.figure(figsize=(8, 6))
plt.plot(data_aluminum['Strain'], data_aluminum['Stress (MPa)'], label='Aluminum', color='blue')
plt.plot(data_steel['Strain'], data_steel['Stress (MPa)'], label='Stainless Steel', color='green')
plt.xlabel('Strain')
plt.ylabel('Stress (MPa)')
plt.title('Engineering Stress-Strain Curve')
plt.legend()
plt.grid()
plt.savefig('images/engineering_stress_strain.png')

#mix engineering stress strain using gripgrip
plt.figure(figsize=(8, 6))
plt.plot(data_aluminum['Strain2'], data_aluminum['Stress (MPa)'], label='Aluminum', color='blue')
plt.plot(data_steel['Strain2'], data_steel['Stress (MPa)'], label='Stainless Steel', color='green')
plt.xlabel('Strain')
plt.ylabel('Stress (MPa)')
plt.title('Engineering Stress-Strain Curve using grip to grip seperation')
plt.legend()
plt.grid()
plt.savefig('images/engineering_stress_strain_gripgrip.png')

#mix true stress strain
plt.figure(figsize=(8, 6))
plt.plot(data_aluminum['Strain'], data_aluminum['Corrected True Stress (MPa)'], label='Aluminum', color='blue')
plt.plot(data_steel['Strain'], data_steel['Corrected True Stress (MPa)'], label='Stainless Steel', color='green')
plt.xlabel('Strain')
plt.ylabel('True Stress (MPa)')
plt.title('True Stress-Strain Curve')
plt.legend()
plt.grid()
plt.savefig('images/true_stress_strain.png')

#mix true stress strain using gripgrip
plt.figure(figsize=(8, 6))
plt.plot(data_aluminum['Strain2'], data_aluminum['Corrected True Stress (MPa)'], label='Aluminum', color='blue')
plt.plot(data_steel['Strain2'], data_steel['Corrected True Stress (MPa)'], label='Stainless Steel', color='green')
plt.xlabel('Strain')
plt.ylabel('True Stress (MPa)')
plt.title('True Stress-Strain Curve using grip to grip seperation')
plt.legend()
plt.grid()
plt.savefig('images/true_stress_strain_gripgrip.png')

#aluminum engineering stress strain
plt.figure(figsize=(8, 6))
plt.plot(data_aluminum['Strain'], data_aluminum['Stress (MPa)'], label='Aluminum', color='blue')
plt.xlabel('Strain')
plt.ylabel('Stress (MPa)')
plt.title('Engineering Stress-Strain Curve for Aluminum')
plt.legend()
plt.grid()
plt.savefig('images/aluminum/engineering_stress_strain.png')

#aluminum engineering stress strain gripgrip
plt.figure(figsize=(8, 6))
plt.plot(data_aluminum['Strain2'], data_aluminum['Stress (MPa)'], label='Aluminum', color='blue')
plt.xlabel('Strain')
plt.ylabel('Stress (MPa)')
plt.title('Engineering Stress-Strain Curve for Aluminum using Grip to Grip Seperation')
plt.legend()
plt.grid()
plt.savefig('images/aluminum/engineering_stress_strain_gripgrip.png')

#aluminum true stress strain
plt.figure(figsize=(10, 8))
plt.plot(data_aluminum['Strain'], data_aluminum['Corrected True Stress (MPa)'], 
         label='True Stress-Strain Curve', color='blue', linewidth=2)
plt.xlabel('Strain')
plt.ylabel('True Stress (MPa)')
plt.title('True Stress-Strain Curve for Aluminum')
plt.legend()
plt.grid()
plt.savefig('images/aluminum/true_stress_strain.png')
plt.close()

#aluminum true stress strain gripgrip
plt.figure(figsize=(10, 8))
plt.plot(data_aluminum['Strain2'], data_aluminum['Corrected True Stress (MPa)'], 
         label='True Stress-Strain Curve', color='blue', linewidth=2)
plt.xlabel('Strain')
plt.ylabel('True Stress (MPa)')
plt.title('True Stress-Strain Curve for Aluminum using Grip to Grip Seperation')
plt.legend()
plt.grid()
plt.savefig('images/aluminum/true_stress_strain_gripgrip.png')
plt.close()

#steel engineering stress strain
plt.figure(figsize=(8, 6))
plt.plot(data_steel['Strain'], data_steel['Stress (MPa)'], label='Stainless Steel', color='blue')
plt.xlabel('Strain')
plt.ylabel('Stress (MPa)')
plt.title('Engineering Stress-Strain Curve for Steel')
plt.legend()
plt.grid()
plt.savefig('images/steel/engineering_stress_strain.png')

#steel engineering stress strain gripgrip
plt.figure(figsize=(8, 6))
plt.plot(data_steel['Strain2'], data_steel['Stress (MPa)'], label='Stainless Steel', color='blue')
plt.xlabel('Strain')
plt.ylabel('Stress (MPa)')
plt.title('Engineering Stress-Strain Curve for Steel using Grip to Grip Seperation')
plt.legend()
plt.grid()
plt.savefig('images/steel/engineering_stress_strain_gripgrip.png')

#steel true stress strain
plt.figure(figsize=(10, 8))
plt.plot(data_steel['Strain'], data_steel['Corrected True Stress (MPa)'], 
         label='True Stress-Strain Curve', color='blue', linewidth=2)
plt.xlabel('Strain')
plt.ylabel('True Stress (MPa)')
plt.title('True Stress-Strain Curve for Steel')
plt.legend()
plt.grid()
plt.savefig('images/steel/true_stress_strain.png')
plt.close()

#steel true stress strain gripgrip
plt.figure(figsize=(10, 8))
plt.plot(data_steel['Strain2'], data_steel['Corrected True Stress (MPa)'], 
         label='True Stress-Strain Curve', color='blue', linewidth=2)
plt.xlabel('Strain')
plt.ylabel('True Stress (MPa)')
plt.title('True Stress-Strain Curve for Steel Using Grip to Grip Seperation')
plt.legend()
plt.grid()
plt.savefig('images/steel/true_stress_strain_gripgrip.png')
plt.close()
