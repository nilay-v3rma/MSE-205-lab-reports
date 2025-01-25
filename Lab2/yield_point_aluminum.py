import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score

# Constants
L0 = 25  # Gauge length in mm
d = 10 # diameter in mm
A0 = np.pi * np.square(d/2) # Cross-sectional area in mm^2
print("---------------Aluminum-----------------")

# Read and process data (same as before)
file_path_aluminum = 'Lab2_Aluminum.csv'
data_aluminum_initial = pd.read_csv(file_path_aluminum)
data_aluminum_initial.columns = data_aluminum_initial.columns.str.strip()
data_aluminum = data_aluminum_initial[data_aluminum_initial['Compressive strain'] >= 0.006].reset_index(drop=True)
data_aluminum['Stress (MPa)'] = data_aluminum['Standard force'] / A0
data_aluminum['Compressive strain'] = data_aluminum['Compressive strain'] / L0


# Function to find the elastic region automatically
def find_elastic_region(strain, stress, window_size=20, r2_threshold=0.99):
    best_r2 = 0
    best_end_idx = window_size
    
    for i in range(window_size, len(strain)//2):
        current_strain = strain[:i]
        current_stress = stress[:i]
        
        # Fit linear regression
        coeffs = np.polyfit(current_strain, current_stress, 1)
        y_pred = np.polyval(coeffs, current_strain)
        r2 = r2_score(current_stress, y_pred)
        
        if r2 > best_r2:
            best_r2 = r2
            best_end_idx = i
        
        # If R² drops significantly, we've left the elastic region
        if r2 < r2_threshold and best_r2 > r2_threshold:
            break
    
    return best_end_idx

# Find elastic region
strain_data = data_aluminum['Compressive strain'].values
stress_data = data_aluminum['Stress (MPa)'].values
elastic_end_idx = find_elastic_region(strain_data, stress_data)
closest_idx = np.abs(data_aluminum['Compressive strain'].values - 0.013).argmin()
elastic_start_idx = data_aluminum.index[closest_idx]
elastic_region = data_aluminum.iloc[elastic_start_idx:elastic_end_idx]


# Calculate elastic modulus (slope)
elastic_fit = np.polyfit(elastic_region['Compressive strain'], elastic_region['Stress (MPa)'], 1)
E, b = elastic_fit
print(f"Elastic Modulus: {E:.2f} MPa")

# Create offset line
offset_strain = 0.002  # 0.2%
offset_line = lambda x: E * (x - offset_strain) + b

# Find intersection using interpolation
def find_intersection(x, y1, y2, x_range):
    # Interpolate stress-strain curve
    f1 = interp1d(x, y1, kind='linear')
    # Create offset line values
    y2_vals = y2(x_range)
    
    # Find where difference is closest to zero
    diff = f1(x_range) - y2_vals
    intersection_idx = np.argmin(np.abs(diff))
    
    return x_range[intersection_idx], y2_vals[intersection_idx]

# Create x_range for intersection finding
x_range_full = np.linspace(strain_data.min(), strain_data.max(), 10000)
yield_strain, yield_stress = find_intersection(
    strain_data, 
    stress_data, 
    offset_line, 
    x_range_full
)

# Truncated x_range for plotting offset line
x_range_plot = np.linspace(0.002, yield_strain * 1.1, 1000)  # Extend just 10% past yield point

print(f"Yield point: Stress = {yield_stress:.2f} MPa, Strain = {yield_strain:.5f}")

# Calculate Ultimate Tensile Strength (UTS)
max_stress_idx = np.argmax(stress_data)
uts_stress = stress_data[max_stress_idx]
uts_strain = strain_data[max_stress_idx]

print(f"Ultimate Tensile Strength (UTS): {uts_stress:.2f} MPa")
print(f"Uniform Elongation: {uts_strain:.4f}")
print(f"Max stress index: {max_stress_idx}")

# Plotting the UTS point
plt.figure(figsize=(10, 8))
plt.plot(strain_data, stress_data, label='Stress-Strain Curve', color='blue')
plt.plot(elastic_region['Compressive strain'], elastic_region['Stress (MPa)'], 
         color='green', label='Elastic Region')
plt.plot(x_range_plot, offset_line(x_range_plot), '--', 
         label='0.2% Offset Line', color='red')
plt.ylim(bottom=0)
plt.scatter(yield_strain, yield_stress, color='purple', s=100,
           label=f'Yield Point ({yield_stress:.1f} MPa)')
plt.scatter(uts_strain, uts_stress, color='orange', s=100, 
           label=f'UTS ({uts_stress:.1f} MPa)', marker='D')
plt.scatter(uts_strain, 0, color='red', s=100, 
           label=f'Uniform elongation ({uts_strain:.4f})', marker='D')
plt.plot([uts_strain, uts_strain], [uts_stress, 0], '--', color='purple', linewidth=1)

plt.xlabel('Strain')
plt.ylabel('Stress (MPa)')
plt.title('Engineering Stress-Strain Curve with Yield and UTS Points for Aluminum')
plt.legend()
plt.grid(True)
plt.savefig('images/aluminum/yield_and_uts_point_aluminum.png')
plt.close()

# Convert to true stress and true strain
data_aluminum['True Stress (MPa)'] = data_aluminum['Stress (MPa)'] * (1 + data_aluminum['Compressive strain'])
data_aluminum['True Strain'] = np.log(1 + data_aluminum['Compressive strain'])

# Clean True Stress: Remove decreasing values
true_stress_aluminum = data_aluminum['True Stress (MPa)'].copy()
max_index = true_stress_aluminum.idxmax()  # Find the index of maximum True Stress
true_stress_aluminum.iloc[max_index + 1:] = np.nan  # Remove all values after the maximum index
data_aluminum['Corrected True Stress (MPa)'] = true_stress_aluminum

# Convert the engineering yield strain to true strain
yield_true_strain = np.log(1 + yield_strain)

# Filter for plastic region (true strain beyond the yield point)
plastic_region = data_aluminum[data_aluminum['True Strain'] > yield_true_strain].dropna()

# Apply logarithmic transformation for Ludwik-Hollomon relation
log_true_stress = np.log(plastic_region['Corrected True Stress (MPa)'])
log_true_strain = np.log(plastic_region['True Strain'])

# Linear regression on log-log data
coeffs = np.polyfit(log_true_strain, log_true_stress, 1)
n = coeffs[0]  # Strain hardening exponent (slope)
ln_K = coeffs[1]  # ln(K)
K = np.exp(ln_K)  # Strain hardening coefficient

print(f"Strain Hardening Coefficient (K): {K:.2f} MPa")
print(f"Strain Hardening Exponent (n): {n:.2f}")

# Generate fitted stress values using Ludwik-Hollomon relation
fitted_true_stress = K * (plastic_region['True Strain'] ** n)

# Plot the plastic region with the fitted curve
plt.figure(figsize=(10, 8))
plt.plot(data_aluminum['True Strain'], data_aluminum['Corrected True Stress (MPa)'], 
         label='True Stress-Strain Curve', color='blue')
plt.plot(plastic_region['True Strain'], fitted_true_stress, 
         '--', label = f'Fitted Curve: $σ = {K:.2f} ε^{{{n:.2f}}}$', color='red')
plt.scatter(plastic_region['True Strain'], plastic_region['Corrected True Stress (MPa)'], 
            color='green', s=10, label='Plastic Region Data Points')

plt.xlabel('True Strain')
plt.ylabel('True Stress (MPa)')
plt.title('Strain Hardening Fit for Aluminum')
plt.legend()
plt.grid(True)
plt.savefig('images/aluminum/strain_hardening_fit_aluminum.png')
print("-------------------------------------")