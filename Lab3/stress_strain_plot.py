import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Constants
span_length = 90  # in mm
width = 25        # in mm
thickness = 3     # in mm

#-------------------- Aluminum ----------------------#
# Read the CSV file
file_path_aluminum = 'Flexure_Al_csv.csv'
data_aluminum = pd.read_csv(file_path_aluminum)

# Clean the data
data_aluminum.columns = data_aluminum.columns.str.strip()  # Remove trailing white spaces
data_aluminum = data_aluminum[data_aluminum['Deformation'] >= 0]  # Ensure non-negative deformation values

# Flexural Stress
data_aluminum['Flexure Stress'] = (3 * data_aluminum['Standard force'] * span_length) / (2 * width * thickness**2)

# Flexural Strain
data_aluminum['Flexure Strain'] = (6 * thickness * data_aluminum['Deformation']) / (span_length**2)

# index of the maximum flexural stress
max_index = data_aluminum['Flexure Stress'].idxmax()

# Selecting a few indexes before the max value to avoid fluctuations
adjusted_index = max(0, max_index - 35)

# Extract key parameters
max_flexure_stress = data_aluminum.loc[:adjusted_index, 'Flexure Stress'].max()
failure_strain = data_aluminum.loc[adjusted_index, 'Flexure Strain']

# Linear Elastic Region (Highlight this)
linear_region = data_aluminum[data_aluminum['Flexure Strain'] <= 0.0018]

# Flexural Modulus (Slope of the linear region)
flexural_modulus = np.polyfit(linear_region['Flexure Strain'], linear_region['Flexure Stress'], 1)[0]

# table of parameters
parameters = {
    'Parameter': ['Max Flexural Stress (MPa)', 'Failure Strain', 'Flexural Modulus (MPa)'],
    'Value': [max_flexure_stress, failure_strain, flexural_modulus]
}

file_path = 'result values/Aluminum.txt'
with open(file_path, 'w') as file:
    file.write("Flexural Test Results for Aluminum\n")
    file.write("-" * 48 + "\n")
    
    file.write(f"{'Parameter':<30} | {'Value':>15}\n")
    file.write("-" * 48 + "\n")
    
    file.write(f"{'Max Flexural Stress (MPa)':<30} | {max_flexure_stress:>15.3f}\n")
    file.write(f"{'Failure Strain':<30} | {failure_strain:>15.5f}\n")
    file.write(f"{'Flexural Modulus (MPa)':<30} | {flexural_modulus:>15,.3f}\n")  # Adds thousands separator
    
    file.write("-" * 48 + "\n")

print(f"Results saved to {file_path}")


#-------------------- Plot ----------------------#
plt.figure(figsize=(10, 6))

# raw flexural stress-strain curve
plt.plot(data_aluminum['Flexure Strain'], data_aluminum['Flexure Stress'], label='Aluminum', color='royalblue', linewidth=2)

# Highlight the Linear Elastic Region
plt.fill_between(
    linear_region['Flexure Strain'],
    linear_region['Flexure Stress'],
    color='lightblue', alpha=0.5, label='Linear Elastic Region'
)

# Highlight max stress point
plt.scatter(failure_strain, max_flexure_stress, color='red', marker='o', zorder=3, label=f"Max Stress: {max_flexure_stress:.3f} MPa")
plt.scatter(failure_strain, 0, color='red', marker='x', zorder=3, label=f"Failure Strain: {failure_strain:.3f}")

# Add a vertical dotted line between (failure_strain, 0) and (failure_strain, max_flexure_stress)
plt.plot([failure_strain, failure_strain], [0, max_flexure_stress], linestyle='dotted', color='grey', linewidth=1.5)

# Labels and Title
plt.title('Flexural Stress vs. Flexural Strain', fontsize=14, fontweight='bold')
plt.xlabel('Flexural Strain', fontsize=12)
plt.ylabel('Flexural Stress (MPa)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('images/aluminum/FlexuralStress_FlexuralStrain_Aluminum.png', dpi=300, bbox_inches='tight')

#-------------------- Stainless Steel ----------------------#
# Read the CSV file
file_path_steel = 'Flexure_SS_csv.csv'
data_steel = pd.read_csv(file_path_steel)

# Clean the data
data_steel.columns = data_steel.columns.str.strip()  # Remove trailing white spaces
data_steel = data_steel[data_steel['Deformation'] >= 0]  # Ensure non-negative deformation values

# Flexural Stress
data_steel['Flexure Stress'] = (3 * data_steel['Standard force'] * span_length) / (2 * width * thickness**2)

# Flexural Strain
data_steel['Flexure Strain'] = (6 * thickness * data_steel['Deformation']) / (span_length**2)

# index of the maximum flexural stress
max_index = data_steel['Flexure Stress'].idxmax()

# Selecting a few indexes before the max value to avoid fluctuations
adjusted_index = max(0, max_index)  # Adjust the number (e.g., 5) as needed

# key parameters
max_flexure_stress = data_steel.loc[:adjusted_index, 'Flexure Stress'].max()
failure_strain = data_steel.loc[adjusted_index, 'Flexure Strain']

# Linear Elastic Region (Highlight this)
linear_region = data_steel[data_steel['Flexure Strain'] <= 0.0022]

# Flexural Modulus (Slope of this region)
flexural_modulus = np.polyfit(linear_region['Flexure Strain'], linear_region['Flexure Stress'], 1)[0]

# table of parameters
parameters = {
    'Parameter': ['Max Flexural Stress (MPa)', 'Failure Strain', 'Flexural Modulus (MPa)'],
    'Value': [max_flexure_stress, failure_strain, flexural_modulus]
}

file_path = 'result values/Stainless Steel.txt' 
with open(file_path, 'w') as file:
    file.write("Flexural Test Results for Aluminum\n")
    file.write("-" * 48 + "\n")
    
    file.write(f"{'Parameter':<30} | {'Value':>15}\n")
    file.write("-" * 48 + "\n")
    
    file.write(f"{'Max Flexural Stress (MPa)':<30} | {max_flexure_stress:>15.3f}\n")
    file.write(f"{'Failure Strain':<30} | {failure_strain:>15.5f}\n")
    file.write(f"{'Flexural Modulus (MPa)':<30} | {flexural_modulus:>15,.3f}\n")
    
    file.write("-" * 48 + "\n")

print(f"Results saved to {file_path}")

#-------------------- Plot ----------------------#
plt.figure(figsize=(10, 6))

# raw flexural stress-strain curve
plt.plot(data_steel['Flexure Strain'], data_steel['Flexure Stress'], label='Stainless Steel', color='royalblue', linewidth=2)

# Highlight the Linear Elastic Region
plt.fill_between(
    linear_region['Flexure Strain'],
    linear_region['Flexure Stress'],
    color='lightblue', alpha=0.5, label='Linear Elastic Region'
)

# Highlight max stress point
plt.scatter(failure_strain, max_flexure_stress, color='red', marker='o', zorder=3, label=f"Max Stress: {max_flexure_stress:.3f} MPa")
plt.scatter(failure_strain, 0, color='red', marker='x', zorder=3, label=f"Failure Strain: {failure_strain:.3f}")

# Add a vertical dotted line between (failure_strain, 0) and (failure_strain, max_flexure_stress)
plt.plot([failure_strain, failure_strain], [0, max_flexure_stress], linestyle='dotted', color='grey', linewidth=1.5)

# Labels and Title
plt.title('Flexural Stress vs. Flexural Strain', fontsize=14, fontweight='bold')
plt.xlabel('Flexural Strain', fontsize=12)
plt.ylabel('Flexural Stress (MPa)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('images/steel/FlexuralStress_FlexuralStrain_Stainless Steel.png', dpi=300, bbox_inches='tight')

#-------------------- Common Plot ----------------------#
plt.figure(figsize=(10, 6))

# Plot raw flexural stress-strain curve
plt.plot(data_steel['Flexure Strain'], data_steel['Flexure Stress'], label='Stainless Steel', color='royalblue', linewidth=2)
plt.plot(data_aluminum['Flexure Strain'], data_aluminum['Flexure Stress'], label='Aluminum', color='green', linewidth=2)

# Labels and Title
plt.title('Flexural Stress vs. Flexural Strain', fontsize=14, fontweight='bold')
plt.xlabel('Flexural Strain', fontsize=12)
plt.ylabel('Flexural Stress (MPa)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('images/FlexuralStress_FlexuralStrain.png', dpi=300, bbox_inches='tight')
