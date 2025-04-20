import pandas as pd

# Input and output file names
xls_file = "LS2.xlsx"
csv_file = "LS2.csv"

# Read the .xls file (use sheet_name=None to get all sheets if needed)
df = pd.read_excel(xls_file, sheet_name=0)  # You can specify the sheet name or index
df = df.drop("MeasureDate", axis=1)

# Save to .csv
df.to_csv(csv_file, index=False)

print(f"Converted {xls_file} to {csv_file}")
